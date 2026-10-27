from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[4]
DATA_PATH = Path(os.getenv("HOUSE_DATA_PATH", ROOT / "datasets/raw/ames_subset.csv"))
ARTIFACT_PATH = Path(os.getenv("MODEL_ARTIFACT_PATH", ROOT / "artifacts/model.joblib"))
CURRENCY = os.getenv("TARGET_CURRENCY", "USD")
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
CORS_ORIGINS = tuple(
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
)
