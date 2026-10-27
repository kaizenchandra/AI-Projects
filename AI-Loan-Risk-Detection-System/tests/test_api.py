import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from loan_risk.api import create_app
from loan_risk.config import Settings

AUTH = {"Authorization": "Bearer test-only-key"}


@pytest.fixture
def settings(trained):
    output, result = trained
    return Settings(
        model_dir=output / "candidate",
        trusted_manifest_sha256=result["manifest_sha256"],
        allow_candidate=True,
        api_key=SecretStr("test-only-key"),
    )


def test_valid_batch_and_auth(settings, request_record):
    with TestClient(create_app(settings)) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 200
        assert client.post("/api/v1/risk/predict", json=request_record).status_code == 401
        first = client.post("/api/v1/risk/predict", json=request_record, headers=AUTH)
        assert first.status_code == 200
        result = first.json()
        assert result["decision_support_only"] is True
        assert result["request_id"] == first.headers["X-Request-ID"]
        assert first.headers["Cache-Control"] == "no-store"
        batch = client.post(
            "/api/v1/risk/predict/batch", json={"records": [request_record] * 2}, headers=AUTH
        )
        assert batch.status_code == 200
        assert all(r["default_probability"] == result["default_probability"] for r in batch.json())
        assert client.get("/api/v1/model/info", headers=AUTH).json()["synthetic"]
        assert client.get("/metrics").status_code == 401
        assert "loan_risk_requests_total" in client.get("/metrics", headers=AUTH).text


@pytest.mark.parametrize("mutation", ["missing", "extra", "string", "extreme", "nonfinite"])
def test_invalid(settings, request_record, mutation):
    if mutation == "missing":
        request_record.pop("annual_income")
    elif mutation == "extra":
        request_record["ssn"] = "SENSITIVE-DO-NOT-ECHO"
    elif mutation == "string":
        request_record["annual_income"] = "SENSITIVE-DO-NOT-ECHO"
    elif mutation == "extreme":
        request_record["loan_amount"] = 1e20
    else:
        request_record["annual_income"] = float("nan")
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/api/v1/risk/predict",
            content=json.dumps(request_record),
            headers={**AUTH, "Content-Type": "application/json"},
        )
        assert response.status_code == 422
        assert "SENSITIVE" not in response.text
        assert "annual_income" not in response.text


def test_limits_and_failures(settings, request_record, tmp_path):
    small = settings.model_copy(update={"max_batch_size": 1, "max_body_bytes": 1024})
    with TestClient(create_app(small)) as client:
        assert (
            client.post(
                "/api/v1/risk/predict/batch", json={"records": [request_record] * 2}, headers=AUTH
            ).status_code
            == 413
        )
        assert (
            client.post("/api/v1/risk/predict", content="x" * 1025, headers=AUTH).status_code == 413
        )
        assert (
            client.post("/api/v1/risk/predict", content="{broken", headers=AUTH).status_code == 422
        )
    missing = settings.model_copy(update={"model_dir": tmp_path / "missing"})
    with TestClient(create_app(missing)) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503
        assert (
            client.post("/api/v1/risk/predict", json=request_record, headers=AUTH).status_code
            == 503
        )
    target = tmp_path / "corrupt"
    shutil.copytree(settings.model_dir, target)
    (target / "model.joblib").write_bytes(b"bad")
    with TestClient(create_app(settings.model_copy(update={"model_dir": target}))) as client:
        assert client.get("/health/ready").status_code == 503
    with TestClient(create_app(settings.model_copy(update={"api_key": SecretStr("")}))) as client:
        assert client.get("/health/ready").status_code == 503


def test_timeout_retains_capacity(settings, request_record):
    config = settings.model_copy(update={"request_timeout_seconds": 0.05, "concurrency": 1})
    app = create_app(config)
    with TestClient(app) as client:
        original = app.state.predictor.predict

        def slow(frame):
            time.sleep(0.2)
            return original(frame)

        app.state.predictor.predict = slow
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(
                client.post, "/api/v1/risk/predict", json=request_record, headers=AUTH
            )
            assert pending.result().status_code == 504
            assert (
                client.post("/api/v1/risk/predict", json=request_record, headers=AUTH).status_code
                == 429
            )
        time.sleep(0.3)
        app.state.predictor.predict = original
        assert (
            client.post("/api/v1/risk/predict", json=request_record, headers=AUTH).status_code
            == 200
        )


def test_null_unseen_large_batch_and_prediction_failure(settings, request_record, caplog):
    app = create_app(settings)
    with TestClient(app) as client:
        request_record["annual_income"] = None
        request_record["repayment_history"] = "unseen"
        response = client.post("/api/v1/risk/predict", json=request_record, headers=AUTH)
        assert response.status_code == 200
        response = client.post(
            "/api/v1/risk/predict/batch", json={"records": [request_record] * 101}, headers=AUTH
        )
        assert response.status_code == 422

        def fail(frame):
            raise RuntimeError("SENSITIVE-BORROWER")

        app.state.predictor.predict = fail
        response = client.post("/api/v1/risk/predict", json=request_record, headers=AUTH)
        assert response.status_code == 503
        assert "SENSITIVE" not in response.text
        assert "SENSITIVE" not in caplog.text
