"""Regression cases for invalid business inputs and promotion attestations."""

import json

import pandas as pd
import pytest
from pydantic import ValidationError

from loan_risk.artifacts import ArtifactError, promote
from loan_risk.config import TrainConfig
from loan_risk.contracts import FEATURES
from loan_risk.data import DataError, validate_features


@pytest.mark.parametrize(
    "cost", ["false_negative_cost", "false_positive_cost", "manual_review_cost"]
)
def test_nonfinite_business_costs_rejected(cost):
    with pytest.raises(ValidationError):
        TrainConfig(**{cost: float("inf")})


def test_boolean_training_feature_rejected(request_record):
    request_record["annual_income"] = True
    with pytest.raises(DataError, match="numeric"):
        validate_features(pd.DataFrame([request_record])[FEATURES])


def approval_for(checksum):
    return {
        "reviewer": "independent-test-reviewer",
        "ticket": "TEST-123",
        "approved_at": "2026-01-01T00:00:00Z",
        "candidate_manifest_sha256": checksum,
        "independent_validation": True,
        "security_review": True,
        "fairness_review": True,
        "legal_review": True,
        "evaluation_accepted": True,
    }


@pytest.mark.parametrize(
    "change",
    [
        {"reviewer": True},
        {"reviewer": "  "},
        {"approved_at": True},
        {"approved_at": "2026-01-01"},
        {"approved_at": "2099-01-01T00:00:00Z"},
        {"candidate_manifest_sha256": "0" * 64},
        {"legal_review": 1},
        {"security_review": False},
        {"unexpected": "unapproved field"},
    ],
)
def test_invalid_approval_rejected_before_publication(trained, tmp_path, change):
    source, result = trained
    approval = {**approval_for(result["manifest_sha256"]), **change}
    with pytest.raises(ArtifactError, match="governance"):
        promote(source / "candidate", tmp_path / "approved", result["manifest_sha256"], approval)
    assert not (tmp_path / "approved").exists()


def test_valid_approval_still_cannot_promote_synthetic(trained, tmp_path):
    source, result = trained
    with pytest.raises(ArtifactError, match="Synthetic"):
        promote(
            source / "candidate",
            tmp_path / "approved",
            result["manifest_sha256"],
            approval_for(result["manifest_sha256"]),
        )
    assert not (tmp_path / "approved").exists()


def test_approval_template_fails_closed():
    from pathlib import Path

    from loan_risk.artifacts import Approval

    template = json.loads((Path(__file__).parents[1] / "configs/approval.example.json").read_text())
    with pytest.raises(ValidationError):
        Approval.model_validate(template)
