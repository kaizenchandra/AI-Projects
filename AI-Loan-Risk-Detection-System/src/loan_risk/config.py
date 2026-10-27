"""Validated, explicit training and service configuration."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    seed: int = 42
    model_version: str = Field(default="1.0.0", pattern=r"^[A-Za-z0-9_.-]{1,64}$")
    dataset_id: str = "synthetic-v1"
    synthetic: bool = True
    train_end: str = "2016-12-31"
    validation_start: str = "2018-01-01"
    validation_end: str = "2019-12-31"
    test_start: str = "2021-01-01"
    folds: int = Field(default=3, ge=2, le=5)
    search_iterations: int = Field(default=4, ge=1, le=20)
    trees: int = Field(default=100, ge=20, le=500)
    bootstrap_iterations: int = Field(default=100, ge=10, le=1000)
    false_negative_cost: float = Field(default=10_000, gt=0)
    false_positive_cost: float = Field(default=200, ge=0)
    manual_review_cost: float = Field(default=50, ge=0)
    review_capacity: float = Field(default=0.25, gt=0, le=1)
    low_risk_cutoff: float = Field(default=0.10, gt=0, lt=1)
    high_risk_cutoff: float = Field(default=0.25, gt=0, lt=1)

    @model_validator(mode="after")
    def ordered_bands(self):
        if self.low_risk_cutoff >= self.high_risk_cutoff:
            raise ValueError("Risk band cutoffs must be ordered")
        return self


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LOAN_RISK_", extra="ignore")
    model_dir: Path = Path("artifacts/candidate")
    trusted_manifest_sha256: str = Field(default="", pattern=r"^([a-f0-9]{64})?$")
    api_key: SecretStr = SecretStr("")
    allow_candidate: bool = False
    max_batch_size: int = Field(default=100, ge=1, le=100)
    max_body_bytes: int = Field(default=131072, ge=1024, le=1048576)
    concurrency: int = Field(default=2, ge=1, le=16)
    request_timeout_seconds: float = Field(default=10, gt=0, le=120)
