"""Probability quality, validation-only review thresholds and uncertainty."""

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from loan_risk.config import TrainConfig


def metrics(y, probabilities, threshold: float = 0.5) -> dict:
    p = np.asarray(probabilities)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid probabilities")
    predicted = p >= threshold
    both = len(np.unique(y)) == 2
    return {
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "average_precision": float(average_precision_score(y, p)) if both else None,
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "threshold": threshold,
        "rows": len(p),
        "default_rate": float(np.mean(y)),
    }


def curves(y, p) -> dict:
    fpr, tpr, _ = roc_curve(y, p)
    precision, recall, _ = precision_recall_curve(y, p)
    observed, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
    return {
        "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "precision_recall": {"precision": precision.tolist(), "recall": recall.tolist()},
        "calibration": {"observed": observed.tolist(), "predicted": predicted.tolist()},
    }


def uncertainty(y, p, iterations: int, seed: int) -> dict:
    """IID bootstrap is appropriate only for this unique-borrower synthetic cohort."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    samples: dict[str, list[float]] = {"roc_auc": [], "average_precision": [], "brier": []}
    for _ in range(iterations):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        scores = metrics(y[idx], p[idx])
        for key in samples:
            samples[key].append(scores[key])
    return {key: np.quantile(values, [0.025, 0.975]).tolist() for key, values in samples.items()}


def choose_threshold(y, p, config: TrainConfig) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    # 1 + epsilon permits a zero-review policy even for p == 1.
    thresholds = np.unique(np.r_[0.0, p, np.nextafter(1.0, 2.0)])
    candidates = []
    for threshold in thresholds:
        alert = p >= threshold
        rate = float(alert.mean())
        if rate > config.review_capacity:
            continue
        fn = int(((y == 1) & ~alert).sum())
        fp = int(((y == 0) & alert).sum())
        cost = (
            fn * config.false_negative_cost
            + fp * config.false_positive_cost
            + alert.sum() * config.manual_review_cost
        ) / len(y)
        candidates.append(
            {
                "threshold": float(threshold),
                "cost_per_loan": float(cost),
                "review_rate": rate,
                "false_negatives": fn,
                "false_positives": fp,
            }
        )
    return min(candidates, key=lambda item: (item["cost_per_loan"], item["review_rate"]))


def sensitivity(y, p, config: TrainConfig) -> list[dict]:
    return [
        {
            "false_negative_cost_multiplier": multiplier,
            **choose_threshold(
                y,
                p,
                config.model_copy(
                    update={"false_negative_cost": config.false_negative_cost * multiplier}
                ),
            ),
        }
        for multiplier in [0.5, 1.0, 2.0]
    ]


def robustness(model, X, y, seed: int) -> dict:
    """Common validation stress scenarios; never used for final-test selection."""
    rng = np.random.default_rng(seed)
    missing = X.mask(rng.random(X.shape) < 0.1)
    unseen = X.copy()
    unseen["repayment_history"] = "unseen_audit_category"
    return {
        "random_10_percent_missing": metrics(y, model.predict_proba(missing)[:, 1]),
        "unseen_repayment_category": metrics(y, model.predict_proba(unseen)[:, 1]),
    }
