"""Serving wrapper: a complete fitted preprocessing + calibrated model artifact."""

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from loan_risk.artifacts import load
from loan_risk.contracts import FEATURES
from loan_risk.data import validate_features


def segment(probability: float, low: float, high: float) -> str:
    if not 0 <= probability <= 1 or not 0 < low < high < 1:
        raise ValueError("Invalid probability or risk band")
    return "LOW" if probability < low else "MODERATE" if probability < high else "HIGH"


class Predictor:
    def __init__(self, directory, checksum: str, allow_candidate: bool = False):
        self.model, self.metadata = load(directory, checksum, allow_candidate)
        reference = pd.DataFrame([self.metadata["reference"]])[FEATURES]
        self.predict(reference)  # Readiness requires successful contract-compatible inference.

    def predict(self, records: pd.DataFrame) -> list[dict]:
        frame = validate_features(records)
        with threadpool_limits(limits=1):
            probabilities = self.model.predict_proba(frame)[:, 1]
        if not np.isfinite(probabilities).all():
            raise ValueError("Model returned nonfinite probabilities")
        bands = self.metadata["risk_bands"]
        return [
            {
                "model_version": self.metadata["model_version"],
                "default_probability": float(p),
                "risk_segment": segment(p, bands["low"], bands["high"]),
                "decision_support_only": True,
            }
            for p in probabilities
        ]
