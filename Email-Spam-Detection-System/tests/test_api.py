from pathlib import Path

from fastapi.testclient import TestClient

from spam_detection.api import create_app
from spam_detection.data import load_dataset
from spam_detection.model import save_bundle, train_bundle
from tests.test_pipeline import config


def client(tmp_path: Path) -> TestClient:
    frame, profile = load_dataset(
        Path(__file__).parents[1] / "data" / "fixtures" / "toy_emails.csv"
    )
    artifact = save_bundle(
        train_bundle(frame.text.tolist(), frame.label.tolist(), profile, config()),
        tmp_path / "model.joblib",
    )
    return TestClient(create_app(str(artifact)))


def test_api_predict_and_validation(tmp_path: Path) -> None:
    with client(tmp_path) as test_client:
        response = test_client.post(
            "/api/v1/predict", json={"subject": "Free prize", "body": "Claim your cash now"}
        )
        assert response.status_code == 200
        assert response.json()["label"] in {"SPAM", "HAM"}
        invalid = test_client.post("/api/v1/predict", json={"subject": "", "body": ""})
        assert invalid.status_code == 422


def test_readiness_fails_without_trusted_model(tmp_path: Path) -> None:
    with TestClient(create_app(str(tmp_path / "missing.joblib"))) as test_client:
        assert test_client.get("/health/ready").status_code == 503
