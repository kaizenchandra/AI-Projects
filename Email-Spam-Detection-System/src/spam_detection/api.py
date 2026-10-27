"""FastAPI inference service."""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, make_asgi_app
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .data import combine_email
from .model import ModelBundle, explain, load_bundle

REQUESTS = Counter("spam_detection_requests_total", "Prediction requests", ["outcome", "label"])
LATENCY = Histogram("spam_detection_inference_seconds", "Prediction latency")


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject: str = Field(default="", max_length=100_000)
    body: str = Field(default="", max_length=200_000)

    @field_validator("subject", "body")
    @classmethod
    def valid_text(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("NUL characters are not supported")
        return value


class PredictResponse(BaseModel):
    label: str
    spam_score: float
    score_semantics: str
    threshold: float
    model_version: str
    request_id: str
    explanation: dict[str, Any] | None = None


def create_app(model_path: str | None = None) -> FastAPI:
    state: dict[str, ModelBundle | None] = {"bundle": None}
    configured_path: str = (
        model_path
        if model_path is not None
        else os.getenv("SPAM_MODEL_PATH", "artifacts/model.joblib")
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            state["bundle"] = load_bundle(configured_path)
        except Exception:
            state["bundle"] = None
        yield
        state["bundle"] = None

    app = FastAPI(title="Email Spam Detection API", version="1.0.0", lifespan=lifespan)
    app.mount("/metrics", make_asgi_app())

    @app.middleware("http")
    async def correlation(request: Request, call_next: Any) -> JSONResponse:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        try:
            response = await call_next(request)
        except Exception:
            response = JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": "internal_error",
                        "message": "internal server error",
                        "request_id": request_id,
                    }
                },
            )
        response.headers["X-Request-ID"] = request_id
        return response

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready() -> JSONResponse:
        if state["bundle"] is None:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": "trusted model unavailable"},
            )
        return JSONResponse(content={"status": "ready"})

    @app.get("/api/v1/model/info")
    def model_info() -> JSONResponse:
        if state["bundle"] is None:
            return JSONResponse(status_code=503, content={"error": "model unavailable"})
        metadata = dict(state["bundle"].metadata)
        metadata.pop("test_data", None)
        return JSONResponse(content=metadata)

    @app.post("/api/v1/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest, request: Request) -> PredictResponse | JSONResponse:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        bundle = state["bundle"]
        if bundle is None:
            return JSONResponse(
                status_code=503,
                content={
                    "error": {
                        "code": "model_unavailable",
                        "message": "model is not ready",
                        "request_id": request_id,
                    }
                },
            )
        text = combine_email(payload.subject, payload.body)
        if not payload.subject.strip() and not payload.body.strip():
            return JSONResponse(
                status_code=422,
                content={
                    "error": {
                        "code": "empty_message",
                        "message": "subject or body is required",
                        "request_id": request_id,
                    }
                },
            )
        started = time.perf_counter()
        classifier = bundle.pipeline.named_steps["classifier"]
        spam_index = list(classifier.classes_).index("spam")
        score = float(bundle.pipeline.predict_proba([text])[0, spam_index])
        threshold = float(bundle.metadata["decision_threshold"])
        label = "SPAM" if score >= threshold else "HAM"
        LATENCY.observe(time.perf_counter() - started)
        REQUESTS.labels("success", label).inc()
        return PredictResponse(
            label=label,
            spam_score=score,
            score_semantics="model score; calibration not established",
            threshold=threshold,
            model_version=bundle.metadata.get("model_version", "unknown"),
            request_id=request_id,
            explanation=explain(bundle, text),
        )

    return app


app = create_app()
