from dataclasses import dataclass
import numpy as np
import pandas as pd

FEATURES = ["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "YearBuilt", "Neighborhood"]
TARGET = "SalePrice"
NUMERIC_FEATURES = FEATURES[:-1]
CATEGORICAL_FEATURES = [FEATURES[-1]]

@dataclass(frozen=True)
class QualityReport:
    rows: int
    columns: int
    duplicates: int
    missing: dict[str, int]

def validate_dataset(frame: pd.DataFrame) -> QualityReport:
    required = set(FEATURES + [TARGET])
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    if not pd.api.types.is_numeric_dtype(frame[TARGET]):
        raise TypeError("SalePrice must be numeric")
    if not np.isfinite(frame[TARGET].to_numpy(dtype=float)).all():
        raise ValueError("SalePrice must contain only finite values")
    if (frame[TARGET] <= 0).any():
        raise ValueError("SalePrice must contain only positive values")
    if (frame[NUMERIC_FEATURES] < 0).any().any():
        raise ValueError("Area, quality, and garage features must be non-negative")
    return QualityReport(len(frame), len(frame.columns), int(frame.duplicated().sum()), frame.isna().sum().to_dict())

def quality_summary(frame: pd.DataFrame) -> dict:
    report = validate_dataset(frame)
    return {"rows": report.rows, "columns": report.columns, "duplicates": report.duplicates,
            "missing": report.missing, "numeric": NUMERIC_FEATURES, "categorical": CATEGORICAL_FEATURES}
