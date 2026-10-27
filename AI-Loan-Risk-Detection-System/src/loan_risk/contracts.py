"""Single feature contract shared by ingestion, training and serving."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1"
# Monetary fields use a single documented currency per model deployment.
RANGES = {
    "annual_income": (1.0, 10_000_000.0),
    "loan_amount": (100.0, 2_000_000.0),
    "loan_term_months": (6.0, 360.0),
    "employment_years": (0.0, 60.0),
    "debt_to_income": (0.0, 3.0),
    "credit_utilization": (0.0, 2.0),
    "credit_history_years": (0.0, 80.0),
    "monthly_debt_obligations": (0.0, 1_000_000.0),
    "prior_delinquencies": (0.0, 100.0),
    "active_credit_accounts": (0.0, 100.0),
}
NUMERIC = list(RANGES)
CATEGORICAL = ["repayment_history"]
FEATURES = NUMERIC + CATEGORICAL
METADATA = ["loan_id", "borrower_id", "origination_date", "observation_end", "default_date"]
TARGET = "default_12m"


class Borrower(BaseModel):
    """Every feature must be supplied; explicit null is allowed for imputation."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    annual_income: Annotated[float | None, Field(ge=1, le=10_000_000)]
    loan_amount: Annotated[float | None, Field(ge=100, le=2_000_000)]
    loan_term_months: Annotated[int | None, Field(ge=6, le=360)]
    employment_years: Annotated[float | None, Field(ge=0, le=60)]
    debt_to_income: Annotated[float | None, Field(ge=0, le=3)]
    credit_utilization: Annotated[float | None, Field(ge=0, le=2)]
    credit_history_years: Annotated[float | None, Field(ge=0, le=80)]
    monthly_debt_obligations: Annotated[float | None, Field(ge=0, le=1_000_000)]
    prior_delinquencies: Annotated[int | None, Field(ge=0, le=100)]
    active_credit_accounts: Annotated[int | None, Field(ge=0, le=100)]
    repayment_history: Annotated[str | None, Field(min_length=1, max_length=40)]


class Prediction(BaseModel):
    request_id: str
    model_version: str
    default_probability: float = Field(ge=0, le=1)
    risk_segment: str
    decision_support_only: bool = True


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    records: list[Borrower] = Field(min_length=1, max_length=100)
