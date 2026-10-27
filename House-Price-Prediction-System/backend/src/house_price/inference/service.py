from pathlib import Path
import logging
import joblib, numpy as np, pandas as pd
from house_price.data.validation import FEATURES

log = logging.getLogger(__name__)

class ModelService:
    def __init__(self, artifact: Path):
        self.ready = False
        self.metadata = {}
        self.pipeline = None
        self.load_error = None
        try:
            if not artifact.exists():
                raise FileNotFoundError(f"Model artifact not found: {artifact}")
            payload = joblib.load(artifact)
            metadata = payload.get("metadata", {})
            if metadata.get("features") != FEATURES:
                raise ValueError("Incompatible model feature contract")
            pipeline = payload["pipeline"]
            if not hasattr(pipeline, "predict"):
                raise TypeError("Model artifact does not contain a predictor")
            self.pipeline, self.metadata, self.ready = pipeline, metadata, True
        except Exception as exc:
            self.load_error = str(exc)
            log.warning("Model artifact unavailable: %s", exc)
    def predict(self, values: dict) -> float:
        if not self.ready: raise RuntimeError("Model is not ready")
        value = float(self.pipeline.predict(pd.DataFrame([values], columns=FEATURES))[0])
        if not np.isfinite(value): raise ValueError("Model returned a non-finite prediction")
        return value
