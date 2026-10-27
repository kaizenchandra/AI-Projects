"""Aggregate drift and delayed-label evaluation, without raw feature logging."""

import numpy as np
import pandas as pd

from loan_risk.contracts import CATEGORICAL, NUMERIC
from loan_risk.evaluation import metrics, uncertainty


def profile(frame: pd.DataFrame, probabilities) -> dict:
    output: dict = {"numeric": {}, "categorical": {}, "missing": frame.isna().mean().to_dict()}
    for name in NUMERIC:
        observed = frame[name].dropna().to_numpy()
        interior = (
            np.unique(np.quantile(observed, np.linspace(0.1, 0.9, 9))) if len(observed) else []
        )
        edges = np.r_[-np.inf, interior, np.inf]
        counts = np.histogram(observed, bins=edges)[0]
        output["numeric"][name] = {"interior": list(interior), "counts": counts.tolist()}
    for name in CATEGORICAL:
        output["categorical"][name] = frame[name].fillna("missing").value_counts().to_dict()
    output["prediction_counts"] = np.histogram(probabilities, bins=np.linspace(0, 1, 11))[
        0
    ].tolist()
    return output


def psi(reference, current) -> float:
    a, b = np.asarray(reference, dtype=float) + 0.5, np.asarray(current, dtype=float) + 0.5
    a, b = a / a.sum(), b / b.sum()
    return float(np.sum((b - a) * np.log(b / a)))


def drift(
    reference: dict, frame: pd.DataFrame, probabilities, alert_psi=0.2, alert_missing_delta=0.05
) -> dict:
    if len(frame) < 30:
        raise ValueError("Drift assessment requires at least 30 observations")
    scores = {}
    for name, baseline in reference["numeric"].items():
        edges = np.r_[-np.inf, baseline["interior"], np.inf]
        counts = np.histogram(frame[name].dropna(), bins=edges)[0]
        scores[name] = psi(baseline["counts"], counts)
    for name, baseline in reference["categorical"].items():
        current = frame[name].fillna("missing").value_counts().to_dict()
        keys = sorted(set(baseline) | set(current))
        scores[name] = psi([baseline.get(k, 0) for k in keys], [current.get(k, 0) for k in keys])
    scores["prediction"] = psi(
        reference["prediction_counts"], np.histogram(probabilities, bins=np.linspace(0, 1, 11))[0]
    )
    missing = {c: float(frame[c].isna().mean() - reference["missing"][c]) for c in frame}
    return {
        "psi": scores,
        "missing_rate_change": missing,
        "investigate": any(v >= alert_psi for v in scores.values())
        or any(abs(v) >= alert_missing_delta for v in missing.values()),
        "thresholds": {"psi": alert_psi, "missing_delta": alert_missing_delta},
        "action": "Investigate; never automatically retrain or promote",
    }


def delayed_performance(y, probabilities, groups=None, minimum_group_size=100) -> dict:
    """Caller must securely join matured labels; groups are separate audit data."""
    y, p = np.asarray(y), np.asarray(probabilities)
    if len(y) != len(p) or not np.isin(y, [0, 1]).all():
        raise ValueError("Aligned binary mature labels required")
    output = {"overall": metrics(y, p), "groups": {}}
    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != len(y):
            raise ValueError("Audit groups must align with scores")
        for group in np.unique(groups):
            mask = groups == group
            if mask.sum() < minimum_group_size or np.unique(y[mask]).size < 2:
                output["groups"][str(group)] = {
                    "suppressed": True,
                    "reason": "insufficient support",
                }
            else:
                output["groups"][str(group)] = {
                    **metrics(y[mask], p[mask]),
                    "ci95": uncertainty(y[mask], p[mask], 100, 42),
                    "calibration_gap": float(p[mask].mean() - y[mask].mean()),
                }
    return output
