# Prism — AI-powered exploratory data analysis

A working single-node EDA dashboard: upload CSV/XLSX/Parquet, inspect schema and quality, compute descriptive statistics, explore distributions and associations, investigate candidate anomalies, filter and rerun, inspect evidence-backed findings, and export a consistent report.

**Deployment boundary:** a bounded, single-node modular monolith for trusted small-team deployments. This is not a certified multi-tenant SaaS platform. Read [limitations](docs/architecture.md#limitations) before exposing it to untrusted workloads.

## Quick start

Prerequisites: Python 3.13 (security-patched dependencies require a modern runtime), Node 24, npm. Run these commands from this project directory:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements.txt
cd frontend
npm ci
npm run build
cd ..
EDA_DEV_MODE=true .venv/bin/python -m uvicorn eda.api:app --app-dir backend --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
```

Open http://127.0.0.1:8000. Expand **Upload dataset**, select [samples/operations.csv](samples/operations.csv), then press **Analyze dataset**. No AI credentials are required. Development mode is explicit and accepts loopback clients only. It uses a shared local owner; do not use it behind a reverse proxy or for production.

For frontend development, run `npm run dev` in `frontend` with the API running on port 8000. Vite proxies `/api` to the API.

## Docker

```sh
cp .env.example .env
# Edit .env: replace EDA_API_TOKENS with a randomly generated secret (24+ characters).
docker compose up --build
```

Open http://127.0.0.1:8000, expand **Connection & access**, paste the token and connect. Comma-separated tokens represent separate owners. Store secrets outside version control. Use a TLS reverse proxy for any network deployment; keep the service bound to loopback/private networks. Do not use more than one API process or replica with this local job manager.

The container uses a non-root account, read-only root filesystem, persistent data volume, bounded memory/CPU/PIDs, dropped capabilities, and a health check. Docker was not assumed available during implementation; see verification notes.

## Tests and checks

```sh
.venv/bin/python -m pytest backend/tests -q
.venv/bin/python -m ruff check backend
.venv/bin/python -m ruff format --check backend
cd frontend
npm run build
npm test
npm audit
npm exec playwright install chromium
# Start the API in explicit local development mode with built frontend first.
npm exec playwright test
cd ..
.venv/bin/python backend/benchmark.py 10000
```

The API publishes its schema and interactive docs at `/docs` and `/openapi.json`. See [architecture, security and operations](docs/architecture.md), [statistical methodology](docs/methodology.md), and [verification](docs/verification.md).

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `EDA_DEV_MODE` | `false` | Explicit loopback-only unauthenticated local mode |
| `EDA_API_TOKENS` | unset | Comma-separated server secrets, each at least 24 characters |
| `EDA_STORAGE` | `./data` | Private directory for SQLite, Parquet and run artifacts |
| `EDA_MAX_BYTES` | `20971520` | Upload bytes; maximum configurable 100 MiB |
| `EDA_MAX_ROWS` | `100000` | Rows; maximum configurable 1 million |
| `EDA_MAX_COLUMNS` | `100` | Columns; maximum configurable 500 |
| `EDA_JOB_TIMEOUT` | `120` | Maximum analysis process duration in seconds |
| `EDA_RETENTION_DAYS` | `30` | Startup cleanup of datasets and associated analyses older than this |
| `EDA_EXTERNAL_AI` | `false` | Explicit opt-in to external aggregate evidence processing |
| `EDA_LLM_URL` | unset | HTTPS OpenAI-compatible chat completions URL |
| `EDA_LLM_MODEL` | unset | Provider model identifier |
| `EDA_LLM_KEY` | unset | Server-only provider credential |
| `POLARS_MAX_THREADS` | library default | Set 2 in constrained deployments |
| `OPENBLAS_NUM_THREADS` | library default | Set 1 to prevent worker oversubscription |

The browser currently advertises and enforces the default 20 MiB limit. Adjust it alongside the server if raising upload limits. Increasing limits requires new memory and latency measurements.

## Export

HTML is a standalone escaped report with tables, numerical distribution graphics, findings, provenance and methods; use browser print-to-PDF if needed. JSON contains the complete bounded analysis artifact. CSV contains column statistics and defends spreadsheet formula prefixes. The Outlier Explorer can export all flagged original records for the selected column. HTML report sections are configurable. Plotly toolbars export chart PNGs. All refer to the completed analysis run, even when next-run controls have changed.
