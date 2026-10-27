from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


class Filter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    column: str
    op: Literal["eq", "ne", "gt", "ge", "lt", "le", "contains", "is_null"]
    value: Optional[str] = None


class AnalysisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    filters: list[Filter] = Field(default_factory=list, max_length=10)
    overrides: dict[str, Literal["numeric", "categorical", "text", "temporal", "identifier", "boolean"]] = (
        Field(default_factory=dict)
    )
    correlation: Literal["pearson", "spearman"] = "pearson"
    outlier_method: Literal["iqr", "zscore", "mad"] = "iqr"
    threshold: float = Field(default=1.5, gt=0, le=20)
    bins: int = Field(default=0, ge=0, le=100)
    columns: list[str] = Field(default_factory=list, max_length=20)


class DatasetMetadata(BaseModel):
    id: str
    name: str
    fingerprint: str
    created: float
    rows: int
    cols: int
    sheets: str


class JobStatus(BaseModel):
    id: str
    dataset: str
    state: Literal["queued", "running", "completed", "failed", "cancelled"]
    config: str
    created: float
    error: Optional[str] = None
    duration: Optional[float] = None


class Insight(BaseModel):
    id: str
    type: str
    severity: str
    title: str
    description: str
    affected_columns: list[str]
    evidence: dict
    recommendation: str
    method: str
    limitations: list[str]
    analysis_run_id: str


class AnalysisResult(BaseModel):
    run_id: str
    fingerprint: str
    algorithm_version: str
    created_at: str
    config: AnalysisConfig
    overview: dict
    profiles: list[dict]
    correlations: list[dict]
    associations: list[dict]
    insights: list[Insight]
    preview: list[dict]
    missingness: dict
    methodology: dict
