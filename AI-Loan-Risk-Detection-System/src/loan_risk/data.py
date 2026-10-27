"""Origination-only synthetic data, strict ingestion and mature temporal splits."""

import numpy as np
import pandas as pd
from scipy.special import expit

from loan_risk.config import TrainConfig
from loan_risk.contracts import CATEGORICAL, FEATURES, METADATA, NUMERIC, RANGES, TARGET


class DataError(ValueError):
    """Invalid data contract; messages deliberately omit record values."""


def synthetic_data(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    if n < 300:
        raise DataError("At least 300 synthetic rows required")
    rng = np.random.default_rng(seed)
    income = np.clip(rng.lognormal(10.9, 0.55, n), 5000, 1_000_000)
    amount = np.clip(income * rng.uniform(0.08, 0.9, n), 500, 250_000)
    dti = np.clip(rng.beta(2, 5, n) * 1.4, 0, 2)
    utilization = rng.beta(2, 3, n)
    delinquencies = rng.poisson(0.35, n)
    history = rng.choice(["current", "previous_late", "limited"], n, p=[0.7, 0.18, 0.12])
    origination = pd.Timestamp("2010-01-01") + pd.to_timedelta(
        rng.integers(0, 14 * 365, n), unit="D"
    )
    probability = expit(
        -4.2
        + 2.7 * dti
        + 2.1 * utilization
        + 0.55 * delinquencies
        + 1.2 * amount / income
        + 0.45 * (history == "previous_late")
        + 0.10 * (origination.year >= 2021)
    )
    target = rng.binomial(1, probability)
    default_date = pd.Series(origination + pd.to_timedelta(rng.integers(1, 365, n), unit="D"))
    default_date[target == 0] = pd.NaT
    df = pd.DataFrame(
        {
            "annual_income": income,
            "loan_amount": amount,
            "loan_term_months": rng.choice([12, 24, 36, 48, 60], n),
            "employment_years": np.clip(rng.gamma(2, 3, n), 0, 45),
            "debt_to_income": dti,
            "credit_utilization": utilization,
            "credit_history_years": rng.uniform(0, 35, n),
            "monthly_debt_obligations": income / 12 * dti,
            "prior_delinquencies": delinquencies,
            "active_credit_accounts": rng.poisson(5, n),
            "repayment_history": history,
            "loan_id": [f"synthetic-loan-{i}" for i in range(n)],
            "borrower_id": [f"synthetic-borrower-{i}" for i in range(n)],
            "origination_date": origination,
            "observation_end": pd.Timestamp("2025-01-01"),
            "default_date": default_date,
            TARGET: target,
        }
    )
    for column in ["annual_income", "employment_years", "repayment_history"]:
        df.loc[rng.random(n) < 0.025, column] = np.nan
    return df


def validate_features(frame: pd.DataFrame) -> pd.DataFrame:
    if set(frame.columns) != set(FEATURES) or frame.columns.duplicated().any():
        raise DataError("Feature columns must exactly match the feature allowlist")
    result = frame[FEATURES].copy()
    for name in NUMERIC:
        if result[name].map(lambda value: isinstance(value, bool | np.bool_)).any():
            raise DataError(f"Invalid numeric type in {name}")
        try:
            result[name] = pd.to_numeric(result[name], errors="raise").astype(float)
        except (ValueError, TypeError) as exc:
            raise DataError(f"Invalid numeric type in {name}") from exc
        lo, hi = RANGES[name]
        values = result[name].dropna()
        if not np.isfinite(values).all() or not values.between(lo, hi).all():
            raise DataError(f"Invalid range in {name}")
        if name in ("loan_term_months", "prior_delinquencies", "active_credit_accounts"):
            if (values % 1 != 0).any():
                raise DataError(f"Integer values required in {name}")
    for name in CATEGORICAL:
        values = result[name].dropna()
        if not values.map(lambda x: isinstance(x, str) and 1 <= len(x) <= 40).all():
            raise DataError(f"Invalid category in {name}")
        result[name] = result[name].where(result[name].notna(), np.nan)
    return result


def validate_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    expected = set(FEATURES + METADATA + [TARGET])
    if set(frame.columns) != expected or frame.columns.duplicated().any():
        raise DataError("Dataset must exactly match the approved schema")
    df = frame.drop_duplicates().copy()
    duplicate_count = len(frame) - len(df)
    if df["loan_id"].isna().any() or df["loan_id"].duplicated().any():
        raise DataError("Conflicting or missing loan identifiers")
    if df["borrower_id"].isna().any():
        raise DataError("Missing borrower grouping identifiers")
    df[FEATURES] = validate_features(df[FEATURES])
    for column in ["origination_date", "observation_end", "default_date"]:
        try:
            df[column] = pd.to_datetime(df[column], errors="raise", utc=True).dt.tz_localize(None)
        except (ValueError, TypeError) as exc:
            raise DataError(f"Invalid date in {column}") from exc
    if df[["origination_date", "observation_end"]].isna().any().any():
        raise DataError("Missing observation dates")
    if (df.observation_end < df.origination_date).any():
        raise DataError("Observation ends before origination")
    event = df.default_date.notna()
    if (
        (df.default_date[event] <= df.origination_date[event])
        | (df.default_date[event] > df.observation_end[event])
    ).any():
        raise DataError("Default event outside observable interval")
    maturity = df.origination_date + pd.DateOffset(months=12)
    mature = df.observation_end >= maturity
    immature_count = int((~mature).sum())
    df = df.loc[mature].copy()
    if not df[TARGET].isin([0, 1]).all():
        raise DataError("Mature loans require binary targets")
    derived = (df.default_date.notna() & (df.default_date <= maturity.loc[mature])).astype(int)
    if not np.array_equal(derived.to_numpy(), df[TARGET].to_numpy()):
        raise DataError("Target disagrees with first 90+ DPD date within 12 months")
    df[TARGET] = df[TARGET].astype(int)
    return df.sort_values("origination_date").reset_index(drop=True), {
        "exact_duplicates_removed": duplicate_count,
        "immature_excluded": immature_count,
        "mature_rows": len(df),
    }


def temporal_split(df: pd.DataFrame, config: TrainConfig) -> dict[str, pd.DataFrame]:
    train_end = pd.Timestamp(config.train_end)
    val_start, val_end = pd.Timestamp(config.validation_start), pd.Timestamp(config.validation_end)
    test_start = pd.Timestamp(config.test_start)
    if not train_end + pd.DateOffset(months=12) < val_start:
        raise DataError("Training labels must mature before validation starts")
    if not val_end + pd.DateOffset(months=12) < test_start:
        raise DataError("Validation labels must mature before test starts")
    dates = df.origination_date
    splits = {
        "train": df.loc[dates <= train_end].copy(),
        "validation": df.loc[dates.between(val_start, val_end)].copy(),
        "test": df.loc[dates >= test_start].copy(),
    }
    seen: set = set()
    for name, part in splits.items():
        borrowers = set(part.borrower_id)
        if borrowers & seen:
            raise DataError("Borrowers cross temporal boundaries; resolve cohort assignment")
        seen |= borrowers
        if len(part) < 30 or part[TARGET].value_counts().reindex([0, 1], fill_value=0).min() < 5:
            raise DataError(f"Insufficient rows or classes in {name}")
    return splits


def temporal_folds(train: pd.DataFrame, n_splits: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window CV; purge unmatured labels and repeated borrowers per fold."""
    dates = train.origination_date
    boundaries = pd.date_range(
        dates.min(), dates.max() + pd.Timedelta(days=1), periods=n_splits + 3
    )
    folds = []
    for start, end in zip(boundaries[2:-1], boundaries[3:], strict=True):
        valid = (dates >= start) & (dates < end)
        fit = dates + pd.DateOffset(months=12) < start
        fit &= ~train.borrower_id.isin(train.loc[valid, "borrower_id"])
        a, b = np.flatnonzero(fit), np.flatnonzero(valid)
        if min(len(a), len(b)) < 20 or any(train.iloc[x][TARGET].nunique() < 2 for x in [a, b]):
            raise DataError("Insufficient mature history for temporal CV")
        folds.append((a, b))
    return folds
