import pandas as pd
import pytest

from house_price.data.validation import validate_dataset


def valid_frame():
    return pd.DataFrame({
        "OverallQual": [6], "GrLivArea": [1500], "GarageCars": [2],
        "TotalBsmtSF": [800], "YearBuilt": [2000], "Neighborhood": ["CollgCr"],
        "SalePrice": [180000],
    })


def test_schema_report_accepts_valid_frame():
    report = validate_dataset(valid_frame())
    assert report.rows == 1
    assert report.missing["Neighborhood"] == 0


def test_schema_rejects_missing_required_column():
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataset(valid_frame().drop(columns=["SalePrice"]))


def test_schema_rejects_non_positive_target():
    frame = valid_frame()
    frame.loc[0, "SalePrice"] = 0
    with pytest.raises(ValueError, match="positive"):
        validate_dataset(frame)
