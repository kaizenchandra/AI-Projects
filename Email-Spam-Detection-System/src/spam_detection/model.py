"""Training, thresholding, persistence, and interpretable inference."""

from __future__ import annotations

import hashlib
import json
import platform
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .data import DatasetProfile


@dataclass(frozen=True)
class ModelBundle:
    pipeline: Pipeline
    metadata: dict[str, Any]


def build_pipeline(config: dict[str, Any]) -> Pipeline:
    vectorizer = config["vectorizer"]
    model = config["model"]
    return Pipeline(
        [
            (
                "vectorizer",
                CountVectorizer(
                    ngram_range=tuple(vectorizer.get("ngram_range", [1, 2])),
                    min_df=vectorizer.get("min_df", 1),
                    max_df=vectorizer.get("max_df", 1.0),
                    max_features=vectorizer.get("max_features", 50_000),
                    binary=vectorizer.get("binary", False),
                    token_pattern=r"(?u)\b\w[\w$%!?'-]*\b",
                ),
            ),
            (
                "classifier",
                MultinomialNB(
                    alpha=model.get("alpha", 1.0), fit_prior=model.get("fit_prior", True)
                ),
            ),
        ]
    )


def choose_threshold(y_true: np.ndarray, spam_scores: np.ndarray, fpr_budget: float) -> float:
    candidates = np.unique(np.r_[np.linspace(0.01, 0.99, 99), spam_scores])
    feasible: list[tuple[float, float]] = []
    for threshold in candidates:
        predicted = spam_scores >= threshold
        tn, fp, _, _ = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if fp + tn else 1.0
        recall = float(((predicted == 1) & (y_true == 1)).sum() / max((y_true == 1).sum(), 1))
        if fpr <= fpr_budget:
            feasible.append((recall, float(threshold)))
    if feasible:
        return max(feasible)[1]
    return 0.5


def train_bundle(
    texts: list[str], labels: list[str], profile: DatasetProfile, config: dict[str, Any]
) -> ModelBundle:
    seed = int(config["training"].get("random_seed", 42))
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        texts,
        labels,
        test_size=config["training"].get("test_size", 0.2),
        random_state=seed,
        stratify=labels,
    )
    validation_size = config["training"].get("validation_size", 0.2)
    adjusted_validation = validation_size / (1 - config["training"].get("test_size", 0.2))
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=adjusted_validation,
        random_state=seed,
        stratify=y_train_val,
    )
    pipeline = build_pipeline(config)
    pipeline.fit(x_train, y_train)
    spam_index = list(pipeline.classes_).index("spam")
    val_scores = pipeline.predict_proba(x_val)[:, spam_index]
    threshold = choose_threshold(
        np.array([label == "spam" for label in y_val]),
        val_scores,
        float(config["classification"].get("false_positive_rate_budget", 0.05)),
    )
    metadata = {
        "model_id": f"nb-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{secrets.token_hex(4)}",
        "model_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": profile.__dict__,
        "config": config,
        "decision_threshold": threshold,
        "label_mapping": {"ham": 0, "spam": 1},
        "classes": list(pipeline.classes_),
        "python_version": platform.python_version(),
        "sklearn_version": __import__("sklearn").__version__,
        "split_counts": {"train": len(x_train), "validation": len(x_val), "test": len(x_test)},
        "test_data": {"texts": x_test, "labels": y_test},
    }
    return ModelBundle(pipeline, metadata)


def save_bundle(bundle: ModelBundle, output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_metadata = {key: value for key, value in bundle.metadata.items() if key != "test_data"}
    payload = {"pipeline": bundle.pipeline, "metadata": safe_metadata}
    joblib.dump(payload, path)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    safe_metadata["artifact_sha256"] = checksum
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(safe_metadata, indent=2, default=str), encoding="utf-8"
    )
    path.with_suffix(path.suffix + ".sha256").write_text(checksum + "\n", encoding="utf-8")
    return path


def load_bundle(path: str | Path, *, verify_checksum: bool = True) -> ModelBundle:
    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"model artifact not found: {artifact}")
    sidecar = artifact.with_suffix(artifact.suffix + ".sha256")
    if verify_checksum:
        if not sidecar.is_file():
            raise ValueError("model checksum sidecar is missing")
        expected = sidecar.read_text(encoding="utf-8").strip()
        actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if not secrets.compare_digest(expected, actual):
            raise ValueError("model artifact checksum mismatch")
    payload = joblib.load(artifact)  # Only load artifacts from a trusted, controlled store.
    if not isinstance(payload, dict) or not isinstance(payload.get("pipeline"), Pipeline):
        raise ValueError("model artifact has an invalid schema")
    return ModelBundle(payload["pipeline"], payload.get("metadata", {}))


def explain(bundle: ModelBundle, text: str, *, limit: int = 5) -> dict[str, Any]:
    vectorizer: CountVectorizer = bundle.pipeline.named_steps["vectorizer"]
    classifier: MultinomialNB = bundle.pipeline.named_steps["classifier"]
    vector = vectorizer.transform([text])
    names = vectorizer.get_feature_names_out()
    spam_index = list(classifier.classes_).index("spam")
    ham_index = list(classifier.classes_).index("ham")
    contributions = vector.toarray()[0] * (
        classifier.feature_log_prob_[spam_index] - classifier.feature_log_prob_[ham_index]
    )
    present = np.flatnonzero(vector.toarray()[0])
    spam = sorted(
        ((str(names[i]), float(contributions[i])) for i in present if contributions[i] > 0),
        key=lambda x: x[1],
        reverse=True,
    )[:limit]
    ham = sorted(
        ((str(names[i]), float(contributions[i])) for i in present if contributions[i] < 0),
        key=lambda x: x[1],
    )[:limit]
    return {
        "spam_indicators": spam,
        "ham_indicators": ham,
        "model_version": bundle.metadata.get("model_version"),
    }
