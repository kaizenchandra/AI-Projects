import json
import shutil

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from loan_risk.artifacts import ArtifactError, digest, load, promote
from loan_risk.config import TrainConfig
from loan_risk.contracts import FEATURES, TARGET
from loan_risk.evaluation import choose_threshold, metrics
from loan_risk.explainability import local_explanation
from loan_risk.features import pipeline
from loan_risk.inference import Predictor, segment
from loan_risk.monitoring import delayed_performance, drift, profile


def test_complete_training_and_serialization(trained, request_record):
    output, result = trained
    report = result["report"]
    assert set(report["models"]) == {"logistic", "random_forest", "gradient_boosting"}
    predictor = Predictor(output / "candidate", result["manifest_sha256"], True)
    frame = pd.DataFrame([request_record])[FEATURES]
    predicted = predictor.predict(frame)[0]["default_probability"]
    assert 0 <= predicted <= 1
    assert predicted == predictor.model.predict_proba(frame)[0, 1]
    assert predictor.predict(frame) == predictor.predict(frame)
    explained = local_explanation(predictor.model, frame, predictor.metadata["reference"])
    assert len(explained["top_features"]) == 5
    assert all(x["feature"] in FEATURES for x in explained["top_features"])
    assert (output / "evaluation.png").stat().st_size > 100
    assert (output / "eda.json").is_file()
    assert set(report["importance"]["permutation_raw_features"]) == set(FEATURES)
    for entry in report["models"].values():
        assert entry["selected_calibration"] in ["sigmoid", "isotonic"]
        assert entry["review_policy_candidate"]["review_rate"] <= 0.25
        assert entry["test_metrics"]["brier"] >= 0


def test_reproducible_pipeline(dataset):
    a = pipeline(RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=1))
    b = pipeline(RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=1))
    a.fit(dataset[FEATURES], dataset[TARGET])
    b.fit(dataset[FEATURES], dataset[TARGET])
    np.testing.assert_array_equal(
        a.predict_proba(dataset[FEATURES]), b.predict_proba(dataset[FEATURES])
    )


def test_artifact_integrity_and_approval(trained, tmp_path):
    output, result = trained
    with pytest.raises(ArtifactError, match="not approved"):
        load(output / "candidate", result["manifest_sha256"])
    with pytest.raises(ArtifactError, match="Untrusted"):
        load(output / "candidate", "0" * 64, True)
    target = tmp_path / "tampered"
    shutil.copytree(output / "candidate", target)
    (target / "model.joblib").write_bytes(b"untrusted payload")
    with pytest.raises(ArtifactError, match="Corrupt model"):
        load(target, result["manifest_sha256"], True)
    manifest = json.loads((target / "manifest.json").read_text())
    manifest["feature_order"] = FEATURES[::-1]
    (target / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError, match="Schema"):
        load(target, digest((target / "manifest.json").read_bytes()), True)
    with pytest.raises(ArtifactError, match="governance"):
        promote(output / "candidate", tmp_path / "approved", result["manifest_sha256"], {})
    approval = dict.fromkeys(
        [
            "reviewer",
            "ticket",
            "approved_at",
            "independent_validation",
            "security_review",
            "fairness_review",
            "legal_review",
            "evaluation_accepted",
        ],
        True,
    )
    approval.update(
        {
            "reviewer": "test-reviewer",
            "ticket": "TEST-123",
            "approved_at": "2026-01-01T00:00:00Z",
            "candidate_manifest_sha256": result["manifest_sha256"],
        }
    )
    with pytest.raises(ArtifactError, match="Synthetic"):
        promote(output / "candidate", tmp_path / "approved", result["manifest_sha256"], approval)


def test_threshold_and_segments():
    config = TrainConfig(review_capacity=0.5)
    y, p = [0, 0, 1, 1], [0.05, 0.2, 0.6, 0.9]
    threshold = choose_threshold(y, p, config)
    assert threshold["threshold"] == 0.6
    assert threshold["false_negatives"] == 0
    assert metrics(y, p, 0.6)["confusion_matrix"] == [[2, 0], [0, 2]]
    assert [segment(x, 0.1, 0.25) for x in [0, 0.1, 0.25, 1]] == ["LOW", "MODERATE", "HIGH", "HIGH"]
    with pytest.raises(ValueError):
        segment(float("nan"), 0.1, 0.25)


def test_monitoring(dataset):
    frame = dataset[FEATURES].copy()
    probabilities = np.repeat(0.2, len(frame))
    reference = profile(frame, probabilities)
    assert not drift(reference, frame, probabilities)["investigate"]
    frame["annual_income"] = 9_000_000
    assert drift(reference, frame, probabilities)["investigate"]
    report = delayed_performance(dataset[TARGET], probabilities, ["audit_group"] * len(frame))
    assert "ci95" in report["groups"]["audit_group"]
    assert delayed_performance([0, 1], [0.1, 0.9], ["a", "b"])["groups"]["a"]["suppressed"]


def test_full_training_reproducibility(trained, dataset, tmp_path):
    from loan_risk.training import train

    first_path, first = trained
    second = train(
        dataset,
        tmp_path / "repeat",
        TrainConfig(trees=20, search_iterations=1, folds=2, bootstrap_iterations=10),
    )
    a = Predictor(first_path / "candidate", first["manifest_sha256"], True)
    b = Predictor(tmp_path / "repeat/candidate", second["manifest_sha256"], True)
    np.testing.assert_array_equal(
        a.model.predict_proba(dataset[FEATURES].iloc[:20]),
        b.model.predict_proba(dataset[FEATURES].iloc[:20]),
    )
    for name in first["report"]["models"]:
        assert (
            first["report"]["models"][name]["test_metrics"]
            == second["report"]["models"][name]["test_metrics"]
        )
    with pytest.raises(ValueError, match="already exists"):
        train(dataset, tmp_path / "repeat", TrainConfig())


def test_dependency_mismatch_and_explanation_failure(trained, tmp_path):
    output, result = trained
    target = tmp_path / "incompatible"
    shutil.copytree(output / "candidate", target)
    manifest = json.loads((target / "manifest.json").read_text())
    manifest["dependencies"]["scikit-learn"] = "0.0.0"
    (target / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ArtifactError, match="dependencies"):
        load(target, digest((target / "manifest.json").read_bytes()), True)
    model, manifest = load(output / "candidate", result["manifest_sha256"], True)
    with pytest.raises(ValueError, match="one ordered"):
        local_explanation(model, pd.DataFrame(columns=FEATURES), manifest["reference"])
