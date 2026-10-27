import csv
import hashlib
import io
import json
import math
import logging
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Depends, File, Form, HTTPException, UploadFile, Request, Query
import polars as pl
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from . import service as s
from .ingestion import ingest, MAX_BYTES
from .models import AnalysisConfig, DatasetMetadata, JobStatus, AnalysisResult
from .llm import explain
from .security import BodyGuard
from .reports import html_report, REPORT_SECTIONS
from .analytics import filtered, number

logger = logging.getLogger("eda")
logging.basicConfig(level=logging.INFO)
TOKENS = [t for t in os.getenv("EDA_API_TOKENS", "").split(",") if t]
DEV = os.getenv("EDA_DEV_MODE", "false") == "true"
RATE = {}


@asynccontextmanager
async def lifespan(app):
    if not TOKENS and not DEV:
        raise RuntimeError("Set EDA_API_TOKENS or explicitly enable EDA_DEV_MODE for loopback development.")
    if any(len(t) < 24 for t in TOKENS):
        raise RuntimeError("API tokens require at least 24 characters.")
    if os.getenv("EDA_EXTERNAL_AI") == "true":
        if not all(os.getenv(k) for k in ("EDA_LLM_URL", "EDA_LLM_MODEL", "EDA_LLM_KEY")) or not os.environ[
            "EDA_LLM_URL"
        ].startswith("https://"):
            raise RuntimeError("External AI requires a configured HTTPS endpoint, model and server key.")
    timeout = float(os.getenv("EDA_JOB_TIMEOUT", "120"))
    if not 1 <= timeout <= 3600:
        raise RuntimeError("EDA_JOB_TIMEOUT must be between 1 and 3600 seconds.")
    s.initialize()
    yield
    for job in list(s.ACTIVE):
        s.cancel(job)
    s.POOL.shutdown(wait=True, cancel_futures=True)


app = FastAPI(title="Prism EDA", version="1.0.0", lifespan=lifespan)
app.add_middleware(BodyGuard)


@app.middleware("http")
async def instrumentation(request, call_next):
    start = time.monotonic()
    request_id = str(uuid.uuid4())
    try:
        if int(request.headers.get("content-length", "0") or 0) > MAX_BYTES + 1024 * 1024:
            return Response("Upload exceeds limit.", status_code=413)
    except ValueError:
        return Response("Invalid content length.", status_code=400)
    response = await call_next(request)
    response.headers.update(
        {
            "X-Request-ID": request_id,
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "Cache-Control": "no-store",
        }
    )
    logger.info(
        json.dumps(
            {
                "request_id": request_id,
                "method": request.method,
                "status": response.status_code,
                "duration_ms": round((time.monotonic() - start) * 1000),
            }
        )
    )
    return response


def auth(request: Request):
    token = request.headers.get("authorization", "").removeprefix("Bearer ")
    if DEV and not TOKENS:
        if request.client and request.client.host not in ("127.0.0.1", "::1", "testclient"):
            raise HTTPException(403, "Development mode requires loopback.")
        return s.owner("local-development")
    if not any(secrets.compare_digest(token, t) for t in TOKENS):
        raise HTTPException(401, "A valid access token is required.")
    owner = s.owner(token)
    now = time.monotonic()
    recent = [t for t in RATE.get(owner, []) if now - t < 60]
    if len(recent) >= 120:
        raise HTTPException(429, "Request rate exceeded. Retry in one minute.")
    RATE[owner] = [*recent, now]
    return owner


def dataset(identifier, owner):
    rows = s.records("SELECT * FROM datasets WHERE id=? AND owner=?", (identifier, owner))
    if not rows:
        raise HTTPException(404, "Dataset not found.")
    return rows[0]


def job(identifier, owner):
    rows = s.records("SELECT * FROM jobs WHERE id=? AND owner=?", (identifier, owner))
    if not rows:
        raise HTTPException(404, "Analysis not found.")
    return rows[0]


def result(identifier, owner):
    j = job(identifier, owner)
    if j["state"] != "completed":
        raise HTTPException(409, "Analysis is not complete.")
    return json.loads(s.path(identifier, "json").read_text())


@app.get("/api/v1/health")
def health():
    return {"status": "ok"}


@app.get("/api/v1/ready")
def ready():
    s.records("SELECT 1")
    return {"status": "ready"}


@app.get("/api/v1/datasets", response_model=list[DatasetMetadata])
def datasets(owner=Depends(auth)):
    return s.records(
        "SELECT id,name,fingerprint,created,rows,cols,sheets FROM datasets WHERE owner=? ORDER BY created DESC",
        (owner,),
    )


@app.post("/api/v1/datasets", status_code=201, response_model=DatasetMetadata)
def upload(
    file: UploadFile = File(...),
    delimiter: str = Form(","),
    encoding: str = Form("utf-8"),
    header: bool = Form(True),
    sheet: str = Form(""),
    owner=Depends(auth),
):
    data = file.file.read(MAX_BYTES + 1)
    try:
        df, sheets = ingest(data, file.filename or "", delimiter, encoding, header, sheet or None)
    except Exception as e:
        detail = (
            str(e)
            if isinstance(e, ValueError) and not isinstance(e, UnicodeError)
            else "Unable to parse dataset. Check content, encoding and file format."
        )
        raise HTTPException(422, detail[:250]) from None
    identifier = str(uuid.uuid4())
    fingerprint = hashlib.sha256(data + json.dumps([delimiter, encoding, header, sheet]).encode()).hexdigest()
    df.write_parquet(s.path(identifier, "parquet"))
    s.execute(
        "INSERT INTO datasets VALUES(?,?,?,?,?,?,?,?)",
        (
            identifier,
            owner,
            Path(file.filename or "dataset").name[:150],
            fingerprint,
            time.time(),
            df.height,
            df.width,
            json.dumps(sheets),
        ),
    )
    return dataset(identifier, owner)


@app.delete("/api/v1/datasets/{identifier}", status_code=204)
def delete(identifier: str, owner=Depends(auth)):
    dataset(identifier, owner)
    jobs = s.records("SELECT id,state FROM jobs WHERE dataset=? AND owner=?", (identifier, owner))
    if any(j["state"] in ("queued", "running") or j["id"] in s.ACTIVE for j in jobs):
        raise HTTPException(409, "Cancel or finish active analyses before deleting.")
    for j in jobs:
        for suffix in ("json", "tmp", "error"):
            s.path(j["id"], suffix).unlink(missing_ok=True)
    s.path(identifier, "parquet").unlink(missing_ok=True)
    s.execute("DELETE FROM jobs WHERE dataset=?", (identifier,))
    s.execute("DELETE FROM datasets WHERE id=?", (identifier,))


@app.post("/api/v1/datasets/{identifier}/analyses", status_code=202)
def analyze(identifier: str, config: AnalysisConfig, owner=Depends(auth)):
    d = dataset(identifier, owner)
    try:
        return {"id": s.submit(identifier, owner, config.model_dump(), d["fingerprint"])}
    except ValueError as e:
        raise HTTPException(429, str(e)) from None


@app.get("/api/v1/jobs/{identifier}", response_model=JobStatus)
def status(identifier: str, owner=Depends(auth)):
    return job(identifier, owner)


@app.delete("/api/v1/jobs/{identifier}")
def cancel(identifier: str, owner=Depends(auth)):
    job(identifier, owner)
    s.cancel(identifier)
    return {"status": "cancellation_requested"}


@app.get("/api/v1/analyses/{identifier}", response_model=AnalysisResult)
def analysis(identifier: str, owner=Depends(auth)):
    return result(identifier, owner)


@app.get("/api/v1/analyses/{identifier}/rows")
def preview_rows(
    identifier: str, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200), owner=Depends(auth)
):
    j = job(identifier, owner)
    if j["state"] != "completed":
        raise HTTPException(409, "Analysis is not complete.")
    dataset(j["dataset"], owner)
    df = filtered(
        pl.read_parquet(s.path(j["dataset"], "parquet")), AnalysisConfig.model_validate_json(j["config"])
    )
    return {
        "offset": offset,
        "total": df.height,
        "rows": df.slice(offset, limit).cast({c: pl.String for c in df.columns}).to_dicts(),
    }


@app.post("/api/v1/analyses/{identifier}/explanations")
async def explanations(identifier: str, owner=Depends(auth)):
    return await explain(result(identifier, owner))


@app.get("/api/v1/analyses/{identifier}/report")
def report(identifier: str, format: str = "html", column: str = "", sections: str = "", owner=Depends(auth)):
    r = result(identifier, owner)
    if format == "json":
        content = json.dumps(r, indent=2)
        mime = "application/json"
    elif format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(
            [
                "column",
                "type",
                "non_null_count",
                "null_count",
                "distinct_count",
                "mean",
                "median",
                "sample_std",
            ]
        )
        for p in r["profiles"]:
            values = [
                p["name"],
                p["semantic_type"],
                p["non_null_count"],
                p["null_count"],
                p["distinct_count"],
                *[p["statistics"].get(k) for k in ("mean", "median", "sample_std")],
            ]
            writer.writerow(
                [
                    "'" + v
                    if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@", "\t", "\r"))
                    else v
                    for v in values
                ]
            )
        content = output.getvalue()
        mime = "text/csv"
    elif format == "outliers":
        profile = next((p for p in r["profiles"] if p["name"] == column), None)
        if not profile or not profile["outliers"] or not profile["outliers"]["eligible"]:
            raise HTTPException(422, "Select a column with an eligible outlier analysis.")
        j = job(identifier, owner)
        df = filtered(
            pl.read_parquet(s.path(j["dataset"], "parquet")), AnalysisConfig.model_validate(r["config"])
        )
        output = io.StringIO()
        writer = csv.writer(output)

        def safe(value):
            return (
                "'" + value
                if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r"))
                else value
            )

        writer.writerow(
            [
                safe(v)
                for v in [
                    "analysis_run_id",
                    "filtered_row_index",
                    "method",
                    "lower_fence",
                    "upper_fence",
                    *df.columns,
                ]
            ]
        )
        o = profile["outliers"]
        for i, row in enumerate(df.iter_rows(named=True)):
            value = number(row[column])
            if math.isfinite(value) and (value < o["lower"] or value > o["upper"]):
                writer.writerow(
                    [safe(v) for v in [identifier, i, o["method"], o["lower"], o["upper"], *row.values()]]
                )
        content = output.getvalue()
        mime = "text/csv"
        format = "csv"
    elif format == "html":
        chosen = set(sections.split(",")) if sections else None
        if chosen == {"none"}:
            chosen = set()
        if chosen and not chosen.issubset(REPORT_SECTIONS):
            raise HTTPException(422, "Unknown report section.")
        content = html_report(r, chosen)
        mime = "text/html"
    else:
        raise HTTPException(422, "Choose html, json, csv or outliers with a column.")
    return Response(
        content,
        media_type=mime,
        headers={
            "Content-Disposition": f'attachment; filename="analysis-{identifier}.{format}"',
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; sandbox",
        },
    )


frontend = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if frontend.exists():
    app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
