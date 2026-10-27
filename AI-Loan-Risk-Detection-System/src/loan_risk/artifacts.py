"""Integrity-checked trusted model packages and explicit promotion records.

joblib is executable serialization. A hash is integrity, not a trust source: the
manifest digest MUST arrive through a separately controlled deployment channel.
"""

import hashlib
import importlib.metadata
import io
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import joblib
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    ValidationError,
    field_validator,
    model_validator,
)

from loan_risk.contracts import FEATURES, SCHEMA_VERSION
from loan_risk.reporting import write_json


class ArtifactError(ValueError):
    pass


class Approval(BaseModel):
    """Release attestations bound to exactly one independently trusted candidate."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    reviewer: str = Field(min_length=1, max_length=200, strict=True)
    ticket: str = Field(min_length=1, max_length=200, strict=True)
    approved_at: AwareDatetime
    candidate_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$", strict=True)
    independent_validation: StrictBool
    security_review: StrictBool
    fairness_review: StrictBool
    legal_review: StrictBool
    evaluation_accepted: StrictBool

    @field_validator("approved_at", mode="before")
    @classmethod
    def timestamp_type(cls, value):
        if not isinstance(value, str | datetime):
            raise ValueError("Approval timestamp must be a timezone-aware ISO timestamp")
        return value

    @model_validator(mode="after")
    def completed_reviews(self):
        if self.approved_at > datetime.now(UTC):
            raise ValueError("Approval timestamp cannot be in the future")
        if not all(
            [
                self.independent_validation,
                self.security_review,
                self.fairness_review,
                self.legal_review,
                self.evaluation_accepted,
            ]
        ):
            raise ValueError("All review gates must explicitly be true")
        return self


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def dependencies() -> dict:
    return {
        name: importlib.metadata.version(name)
        for name in ["scikit-learn", "numpy", "pandas", "scipy", "joblib", "loan-risk"]
    }


def package(model, metadata: dict, output: Path) -> str:
    if output.exists():
        raise ArtifactError("Artifact destination already exists; versions are immutable")
    output.mkdir(parents=True)
    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=3)
    payload = buffer.getvalue()
    (output / "model.joblib").write_bytes(payload)
    manifest = {
        **metadata,
        "model_sha256": digest(payload),
        "dependencies": dependencies(),
        "python": platform.python_version(),
        "data_schema_version": SCHEMA_VERSION,
        "feature_schema_version": SCHEMA_VERSION,
        "feature_order": FEATURES,
        "status": "candidate",
    }
    write_json(output / "manifest.json", manifest)
    checksum = digest((output / "manifest.json").read_bytes())
    (output / "manifest.sha256").write_text(checksum + "\n")
    return checksum


def load(output: Path, trusted_manifest_sha256: str, allow_candidate: bool = False):
    try:
        content = (output / "manifest.json").read_bytes()
        if len(trusted_manifest_sha256) != 64 or digest(content) != trusted_manifest_sha256:
            raise ArtifactError("Untrusted or corrupt manifest")
        manifest = json.loads(content)
        if manifest["feature_order"] != FEATURES or any(
            manifest[key] != SCHEMA_VERSION
            for key in ["data_schema_version", "feature_schema_version"]
        ):
            raise ArtifactError("Schema mismatch")
        if manifest["dependencies"] != dependencies():
            raise ArtifactError("Model runtime dependencies differ from training")
        if manifest["python"].split(".")[:2] != platform.python_version().split(".")[:2]:
            raise ArtifactError("Python minor version mismatch")
        if manifest["status"] != "approved" and not allow_candidate:
            raise ArtifactError("Model is not approved for deployment")
        payload = (output / "model.joblib").read_bytes()
        if digest(payload) != manifest["model_sha256"]:
            raise ArtifactError("Corrupt model")
        # Load exactly the bytes hashed above, avoiding path replacement races.
        model = joblib.load(io.BytesIO(payload))
        if list(model.classes_) != [0, 1]:
            raise ArtifactError("Invalid model class ordering")
        return model, manifest
    except ArtifactError:
        raise
    except Exception as exc:
        raise ArtifactError("Model artifact unavailable or invalid") from exc


def promote(source: Path, destination: Path, trusted_hash: str, approval: dict) -> str:
    try:
        validated = Approval.model_validate(approval)
    except ValidationError as exc:
        raise ArtifactError("Invalid governance approval record") from exc
    if validated.candidate_manifest_sha256 != trusted_hash:
        raise ArtifactError("The governance approval does not match the candidate manifest")
    model, manifest = load(source, trusted_hash, allow_candidate=True)
    if manifest["synthetic"]:
        raise ArtifactError("Synthetic demonstration models cannot be production-promoted")
    package(model, {**manifest, "approval": validated.model_dump(mode="json")}, destination)
    promoted = json.loads((destination / "manifest.json").read_text())
    promoted["status"] = "approved"
    write_json(destination / "manifest.json", promoted)
    checksum = digest((destination / "manifest.json").read_bytes())
    (destination / "manifest.sha256").write_text(checksum + "\n")
    return checksum
