"""Persistent local metadata and isolated, bounded analytical jobs."""

import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sqlite3
import threading
import time
import uuid
import logging
from concurrent.futures import ThreadPoolExecutor
import polars as pl
from .analytics import analyze
from .models import AnalysisConfig

ROOT = Path(os.getenv("EDA_STORAGE", "./data")).resolve()
ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
DB = ROOT / "metadata.sqlite3"
POOL = ThreadPoolExecutor(max_workers=2)
SLOTS = threading.BoundedSemaphore(6)
ACTIVE = {}
LOCK = threading.Lock()
logger = logging.getLogger("eda.jobs")


def connect():
    c = sqlite3.connect(DB, timeout=15)
    c.row_factory = sqlite3.Row
    return c


def initialize():
    with connect() as c:
        c.executescript("""PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS datasets(id TEXT PRIMARY KEY,owner TEXT NOT NULL,name TEXT NOT NULL,fingerprint TEXT NOT NULL,created REAL NOT NULL,rows INTEGER,cols INTEGER,sheets TEXT);
        CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,dataset TEXT NOT NULL,owner TEXT NOT NULL,state TEXT NOT NULL,config TEXT NOT NULL,created REAL NOT NULL,error TEXT,duration REAL);
        PRAGMA user_version=1;""")
        c.execute(
            "UPDATE jobs SET state='failed',error='Server restarted; resubmit analysis.' WHERE state IN ('queued','running')"
        )
    retention = int(os.getenv("EDA_RETENTION_DAYS", "30"))
    if retention < 1:
        raise RuntimeError("EDA_RETENTION_DAYS must be positive.")
    for d in records("SELECT id FROM datasets WHERE created < ?", (time.time() - retention * 86400,)):
        for j in records("SELECT id FROM jobs WHERE dataset=?", (d["id"],)):
            for suffix in ("json", "tmp", "error"):
                path(j["id"], suffix).unlink(missing_ok=True)
        path(d["id"], "parquet").unlink(missing_ok=True)
        execute("DELETE FROM jobs WHERE dataset=?", (d["id"],))
        execute("DELETE FROM datasets WHERE id=?", (d["id"],))


def records(sql, args=()):
    with connect() as c:
        return [dict(r) for r in c.execute(sql, args).fetchall()]


def execute(sql, args=()):
    with connect() as c:
        c.execute(sql, args)


def path(identifier, suffix):
    # IDs never contain client filenames.
    uuid.UUID(identifier)
    return ROOT / f"{identifier}.{suffix}"


def child(dataset, config, run_id, fingerprint):
    try:
        result = analyze(
            pl.read_parquet(path(dataset, "parquet")),
            AnalysisConfig.model_validate(config),
            run_id,
            fingerprint,
        )
        tmp = path(run_id, "tmp")
        tmp.write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
        tmp.replace(path(run_id, "json"))
    except Exception:
        # Do not persist values or parser exception payloads.
        path(run_id, "error").write_text("Analysis failed. Check selected types, filters and numeric ranges.")


def run(job, dataset, config, fingerprint):
    started = time.monotonic()
    process = None
    try:
        with LOCK:
            state = records("SELECT state FROM jobs WHERE id=?", (job,))
            if not state or state[0]["state"] == "cancelled":
                return
            execute("UPDATE jobs SET state='running' WHERE id=?", (job,))
            process = mp.get_context("spawn").Process(target=child, args=(dataset, config, job, fingerprint))
            ACTIVE[job] = process
            process.start()
            logger.info(json.dumps({"analysis_run_id": job, "state": "running", "retry_attempt": 0}))
        process.join(float(os.getenv("EDA_JOB_TIMEOUT", "120")))
        if process.is_alive():
            process.terminate()
            process.join()
            error = "Analysis time limit exceeded."
        else:
            error = (
                "Analysis failed; check types and numeric ranges."
                if process.exitcode != 0 or not path(job, "json").exists()
                else None
            )
        with LOCK:
            execute(
                "UPDATE jobs SET state=?,error=?,duration=? WHERE id=? AND state='running'",
                ("failed" if error else "completed", error, time.monotonic() - started, job),
            )
            logger.info(
                json.dumps(
                    {
                        "analysis_run_id": job,
                        "state": "failed" if error else "completed",
                        "duration": time.monotonic() - started,
                        "failure_category": error,
                    }
                )
            )
    except Exception:
        execute(
            "UPDATE jobs SET state='failed',error='Worker startup or storage failure.' WHERE id=? AND state IN ('queued','running')",
            (job,),
        )
    finally:
        with LOCK:
            ACTIVE.pop(job, None)
            state = records("SELECT state FROM jobs WHERE id=?", (job,))
            if not state or state[0]["state"] != "completed":
                for suffix in ("json", "tmp", "error"):
                    path(job, suffix).unlink(missing_ok=True)
        SLOTS.release()


def submit(dataset, owner, config, fingerprint):
    if not SLOTS.acquire(blocking=False):
        raise ValueError("Analysis queue is full. Retry later.")
    job = str(uuid.uuid4())
    execute(
        "INSERT INTO jobs(id,dataset,owner,state,config,created) VALUES(?,?,?,?,?,?)",
        (job, dataset, owner, "queued", json.dumps(config), time.time()),
    )
    POOL.submit(run, job, dataset, config, fingerprint)
    return job


def cancel(job):
    with LOCK:
        execute("UPDATE jobs SET state='cancelled' WHERE id=? AND state IN ('queued','running')", (job,))
        p = ACTIVE.get(job)
        if p and p.is_alive():
            p.terminate()


def owner(token):
    return hashlib.sha256(token.encode()).hexdigest()
