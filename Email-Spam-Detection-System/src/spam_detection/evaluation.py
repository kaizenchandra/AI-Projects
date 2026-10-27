"""Leakage-safe metrics and report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .model import ModelBundle


def evaluate(
    bundle: ModelBundle, texts: list[str], labels: list[str], threshold: float | None = None
) -> dict[str, Any]:
    threshold = bundle.metadata["decision_threshold"] if threshold is None else threshold
    classifier = bundle.pipeline.named_steps["classifier"]
    spam_index = list(classifier.classes_).index("spam")
    y_true = np.array([label == "spam" for label in labels])
    scores = bundle.pipeline.predict_proba(texts)[:, spam_index]
    predicted = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[False, True]).ravel()
    report: dict[str, Any] = {
        "sample_count": len(labels),
        "spam_count": int(y_true.sum()),
        "ham_count": int((~y_true).sum()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision_spam": float(precision_score(y_true, predicted, zero_division=0)),
        "recall_spam": float(recall_score(y_true, predicted, zero_division=0)),
        "f1_spam": float(f1_score(y_true, predicted, zero_division=0)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "false_negative_rate": float(fn / max(fn + tp, 1)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "brier_score": float(brier_score_loss(y_true, scores)),
        "classification_report": classification_report(
            y_true, predicted, target_names=["HAM", "SPAM"], output_dict=True, zero_division=0
        ),
    }
    report["pr_auc"] = (
        float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) == 2 else None
    )
    report["roc_auc"] = (
        float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else None
    )
    probability_true, probability_pred = calibration_curve(
        y_true, scores, n_bins=min(10, max(2, len(labels) // 5)), strategy="uniform"
    )
    report["calibration_curve"] = {
        "mean_predicted": probability_pred.tolist(),
        "fraction_positive": probability_true.tolist(),
    }
    return report


def save_report(report: dict[str, Any], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
