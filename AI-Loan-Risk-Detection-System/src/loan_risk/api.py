"""Protected API with bounded bodies, batches and actual in-flight model work."""

import asyncio
import hmac
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from starlette.exceptions import HTTPException as StarletteHTTPException

from loan_risk.config import Settings
from loan_risk.contracts import FEATURES, BatchRequest, Borrower, Prediction
from loan_risk.inference import Predictor

logger = logging.getLogger("loan_risk.audit")


class BodyLimit:
    """Bound actual bytes (including chunked bodies) before JSON parsing."""

    def __init__(self, app, maximum: int):
        self.app, self.maximum = app, maximum

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        chunks, size = [], 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body = message.get("body", b"")
            size += len(body)
            if size > self.maximum:
                response = JSONResponse(
                    {"error": "request_too_large", "request_id": scope["state"]["request_id"]},
                    status_code=413,
                )
                return await response(scope, receive, send)
            chunks.append(body)
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": b"".join(chunks), "more_body": False}

        await self.app(scope, replay, send)


def create_app(settings: Settings | None = None) -> FastAPI:
    config = settings or Settings()
    registry = CollectorRegistry()
    requests = Counter(
        "loan_risk_requests_total", "Requests", ["route", "status"], registry=registry
    )
    latency = Histogram("loan_risk_request_seconds", "Request duration", registry=registry)
    missing = Counter(
        "loan_risk_missing_features_total", "Missing features", ["feature"], registry=registry
    )
    scores = Histogram(
        "loan_risk_probability",
        "Predicted PD",
        buckets=[0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1],
        registry=registry,
    )
    available = Gauge("loan_risk_model_available", "Model readiness", registry=registry)
    failures = Counter("loan_risk_model_load_failures_total", "Startup failures", registry=registry)

    @asynccontextmanager
    async def lifespan(app):
        app.state.predictor = None
        app.state.executor = ThreadPoolExecutor(max_workers=config.concurrency)
        app.state.inflight = 0
        try:
            if not config.api_key.get_secret_value():
                raise ValueError("API key required")
            app.state.predictor = Predictor(
                config.model_dir, config.trusted_manifest_sha256, config.allow_candidate
            )
            available.set(1)
        except Exception:
            failures.inc()
            available.set(0)
            logger.error(json.dumps({"event": "model_load_failed"}))
        yield
        app.state.executor.shutdown(wait=True, cancel_futures=True)
        available.set(0)

    app = FastAPI(
        title="Loan Risk Decision Support",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )
    app.add_middleware(BodyLimit, maximum=config.max_body_bytes)
    routes = {
        "/api/v1/risk/predict",
        "/api/v1/risk/predict/batch",
        "/api/v1/model/info",
        "/health/live",
        "/health/ready",
        "/metrics",
    }

    @app.middleware("http")
    async def audit(request: Request, call_next):
        request_id = str(
            uuid.uuid4()
        )  # Server-generated to avoid log injection/high-cardinality input.
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await asyncio.wait_for(call_next(request), config.request_timeout_seconds)
        except TimeoutError:
            response = JSONResponse(
                {"error": "request_timeout", "request_id": request_id}, status_code=504
            )
        except Exception:
            response = JSONResponse(
                {"error": "internal_error", "request_id": request_id}, status_code=500
            )
        elapsed = time.perf_counter() - started
        route = request.url.path if request.url.path in routes else "other"
        requests.labels(route, str(response.status_code)).inc()
        latency.observe(elapsed)
        model = getattr(request.app.state, "predictor", None)
        logger.info(
            json.dumps(
                {
                    "event": "request",
                    "request_id": request_id,
                    "route": route,
                    "status": response.status_code,
                    "duration_ms": round(elapsed * 1000, 2),
                    "model_version": model.metadata["model_version"] if model else None,
                }
            )
        )
        response.headers["X-Request-ID"] = request_id
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid(request, exc):
        return JSONResponse(
            {"error": "invalid_request", "request_id": request.state.request_id}, status_code=422
        )

    @app.exception_handler(StarletteHTTPException)
    async def rejected(request, exc):
        return JSONResponse(
            {"error": str(exc.detail), "request_id": request.state.request_id},
            status_code=exc.status_code,
        )

    def authorize(request: Request):
        key = config.api_key.get_secret_value()
        supplied = request.headers.get("authorization", "")
        if not key or not hmac.compare_digest(supplied.encode(), f"Bearer {key}".encode()):
            raise HTTPException(401, "unauthorized")

    async def score(records: list[Borrower], request: Request):
        state = request.app.state
        if state.predictor is None:
            raise HTTPException(503, "model_unavailable")
        if state.inflight >= config.concurrency:
            raise HTTPException(429, "inference_capacity_exceeded")
        frame = pd.DataFrame([record.model_dump() for record in records])[FEATURES]
        for feature in FEATURES:
            missing.labels(feature).inc(int(frame[feature].isna().sum()))
        state.inflight += 1
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(state.executor, state.predictor.predict, frame)

        def release(completed):
            state.inflight -= 1
            if not completed.cancelled():
                completed.exception()  # Consume failures even after the HTTP caller timed out.

        future.add_done_callback(release)
        try:
            results = await asyncio.wait_for(asyncio.shield(future), config.request_timeout_seconds)
        except TimeoutError as exc:
            raise HTTPException(504, "inference_timeout") from exc
        except Exception as exc:
            raise HTTPException(503, "inference_failed") from exc
        for result in results:
            scores.observe(result["default_probability"])
            result["request_id"] = request.state.request_id
        return results

    @app.post("/api/v1/risk/predict", response_model=Prediction, dependencies=[Depends(authorize)])
    async def predict(body: Borrower, request: Request):
        return (await score([body], request))[0]

    @app.post(
        "/api/v1/risk/predict/batch",
        response_model=list[Prediction],
        dependencies=[Depends(authorize)],
    )
    async def batch(body: BatchRequest, request: Request):
        if len(body.records) > config.max_batch_size:
            raise HTTPException(413, "batch_too_large")
        return await score(body.records, request)

    @app.get("/health/live")
    async def live():
        return {"status": "live"}

    @app.get("/health/ready")
    async def ready(request: Request):
        if request.app.state.predictor is None:
            raise HTTPException(503, "model_unavailable")
        return {"status": "ready"}

    @app.get("/api/v1/model/info", dependencies=[Depends(authorize)])
    async def info(request: Request):
        if request.app.state.predictor is None:
            raise HTTPException(503, "model_unavailable")
        metadata = request.app.state.predictor.metadata
        return {
            key: metadata[key]
            for key in [
                "model_version",
                "model_family",
                "synthetic",
                "feature_schema_version",
                "prediction_definition",
                "risk_bands",
            ]
        }

    @app.get("/metrics", dependencies=[Depends(authorize)])
    async def prometheus():
        return Response(generate_latest(registry), media_type="text/plain; version=0.0.4")

    return app
