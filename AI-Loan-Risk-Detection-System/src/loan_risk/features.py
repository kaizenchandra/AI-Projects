"""Serializable feature engineering inside the fitted estimator pipeline."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from loan_risk.contracts import CATEGORICAL, FEATURES, NUMERIC
from loan_risk.data import validate_features

ENGINEERED = ["loan_to_income", "income_to_obligation", "monthly_principal_to_income"]


class OriginationFeatures(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        validate_features(X)
        self.feature_names_in_ = np.array(FEATURES, dtype=object)
        self.n_features_in_ = len(FEATURES)
        return self

    def transform(self, X):
        check_is_fitted(self)
        frame = validate_features(X)
        income = frame.annual_income
        # Zero obligations mean no debt; cap ratio at 100 rather than divide by zero.
        frame["loan_to_income"] = (frame.loan_amount / income).clip(upper=100)
        frame["income_to_obligation"] = (
            income / 12 / frame.monthly_debt_obligations.clip(lower=1)
        ).clip(upper=100)
        frame["monthly_principal_to_income"] = (
            frame.loan_amount / frame.loan_term_months / (income / 12)
        ).clip(upper=100)
        return frame

    def get_feature_names_out(self, input_features=None):
        return np.array(FEATURES + ENGINEERED, dtype=object)


def pipeline(estimator) -> Pipeline:
    numeric = Pipeline(
        [
            (
                "impute",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            ),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            (
                "impute",
                SimpleImputer(strategy="constant", fill_value="missing", keep_empty_features=True),
            ),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return Pipeline(
        [
            ("features", OriginationFeatures()),
            (
                "preprocess",
                ColumnTransformer(
                    [
                        ("numeric", numeric, NUMERIC + ENGINEERED),
                        ("categorical", categorical, CATEGORICAL),
                    ]
                ),
            ),
            ("model", estimator),
        ]
    )
