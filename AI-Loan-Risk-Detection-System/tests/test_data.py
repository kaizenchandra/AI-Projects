import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn.linear_model import LogisticRegression

from loan_risk.config import Settings, TrainConfig
from loan_risk.contracts import FEATURES, TARGET, Borrower
from loan_risk.data import (
    DataError,
    temporal_folds,
    temporal_split,
    validate_dataset,
    validate_features,
)
from loan_risk.features import OriginationFeatures, pipeline


def test_maturity_duplicates_target(dataset):
    frame = pd.concat([dataset, dataset.iloc[[0]]], ignore_index=True)
    valid, report = validate_dataset(frame)
    assert report["exact_duplicates_removed"] == 1
    assert len(valid) == len(dataset)
    frame = dataset.copy()
    frame.loc[0, "observation_end"] = frame.loc[0, "origination_date"]
    frame.loc[0, "default_date"] = pd.NaT
    _, report = validate_dataset(frame)
    assert report["immature_excluded"] == 1
    frame = dataset.copy()
    frame.loc[0, TARGET] = 1 - frame.loc[0, TARGET]
    with pytest.raises(DataError, match="Target disagrees"):
        validate_dataset(frame)


def test_leakage_allowlist(dataset):
    with pytest.raises(DataError):
        validate_features(dataset[FEATURES + [TARGET]])
    with pytest.raises(DataError):
        validate_dataset(dataset.assign(future_payment=100))
    with pytest.raises(DataError):
        validate_dataset(dataset.drop(columns=["annual_income"]))


def test_splits_and_cv_maturity(dataset):
    valid, _ = validate_dataset(dataset)
    splits = temporal_split(valid, TrainConfig())
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not set(splits[a].loan_id) & set(splits[b].loan_id)
            assert not set(splits[a].borrower_id) & set(splits[b].borrower_id)
    train = splits["train"]
    for fit, check in temporal_folds(train, 3):
        assert not set(fit) & set(check)
        assert (train.iloc[fit].origination_date + pd.DateOffset(months=12)).max() < train.iloc[
            check
        ].origination_date.min()
    with pytest.raises(DataError, match="mature"):
        temporal_split(valid, TrainConfig(validation_start="2017-01-01"))
    valid.loc[valid.origination_date >= "2021-01-01", "borrower_id"] = train.borrower_id.iloc[0]
    with pytest.raises(DataError, match="Borrowers"):
        temporal_split(valid, TrainConfig())


def test_feature_math_and_order(request_record):
    frame = pd.DataFrame([request_record])[FEATURES]
    features = OriginationFeatures().fit(frame)
    transformed = features.transform(frame)
    assert transformed.loan_to_income.iloc[0] == pytest.approx(18000 / 65000)
    assert transformed.income_to_obligation.iloc[0] == pytest.approx(65000 / 12 / 1625)
    frame.loc[0, "monthly_debt_obligations"] = 0
    assert features.transform(frame).income_to_obligation.iloc[0] == 100
    frame.loc[0, "annual_income"] = np.nan
    assert np.isnan(features.transform(frame).loan_to_income.iloc[0])
    pd.testing.assert_frame_equal(
        features.transform(frame), features.transform(frame[FEATURES[::-1]])
    )
    frame.loc[0, "annual_income"] = 0
    with pytest.raises(DataError):
        features.transform(frame)


def test_unseen_and_all_missing(dataset):
    frame = dataset[FEATURES].copy()
    frame["employment_years"] = np.nan
    estimator = pipeline(LogisticRegression(max_iter=1000)).fit(frame, dataset[TARGET])
    row = frame.iloc[[0]].copy()
    row["repayment_history"] = "unseen"
    probabilities = estimator.predict_proba(row)
    assert np.isfinite(probabilities).all()
    assert probabilities.sum() == pytest.approx(1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("annual_income", "65000"),
        ("annual_income", float("inf")),
        ("annual_income", float("nan")),
        ("annual_income", -1),
        ("loan_amount", 3_000_000),
        ("loan_term_months", 36.5),
        ("prior_delinquencies", True),
    ],
)
def test_strict_request(request_record, field, value):
    request_record[field] = value
    with pytest.raises(ValidationError):
        Borrower(**request_record)


def test_configuration():
    with pytest.raises(ValidationError):
        TrainConfig(low_risk_cutoff=0.8, high_risk_cutoff=0.1)
    with pytest.raises(ValidationError):
        Settings(concurrency=0)
    with pytest.raises(ValidationError):
        TrainConfig(unknown_option=1)


def test_conflicting_ids_bad_dates_and_numeric_input(dataset):
    duplicate = dataset.iloc[[0]].copy()
    duplicate["loan_amount"] = 1000
    with pytest.raises(DataError, match="Conflicting"):
        validate_dataset(pd.concat([dataset, duplicate]))
    invalid = dataset.copy()
    invalid.loc[0, "observation_end"] = invalid.loc[0, "origination_date"] - pd.Timedelta(days=1)
    with pytest.raises(DataError, match="before origination"):
        validate_dataset(invalid)
    features = dataset[FEATURES].copy()
    features["loan_amount"] = features.loan_amount.astype(str)
    assert validate_features(features).loan_amount.dtype == float
    features.loc[0, "loan_amount"] = "invalid"
    with pytest.raises(DataError, match="numeric"):
        validate_features(features)
    features = dataset[FEATURES].copy()
    features.loc[0, "annual_income"] = np.inf
    with pytest.raises(DataError, match="range"):
        validate_features(features)
