# Verification record

Executed locally on macOS ARM64 on 2026-10-25. Final backend environment: project-local Python 3.13.17, dependencies pinned in `backend/requirements.txt`. Frontend: Node 24.22.1 and npm 11.19.1, lockfile included in source.

## Executed checks

| Check | Result |
|---|---|
| Backend pytest | 28 passed |
| Ruff lint | Passed |
| Ruff formatting | Applied and checked during final verification |
| TypeScript strict check | Passed as part of production build |
| Vite production build | Passed; large Plotly bundle warning remains |
| React component tests | 3 passed |
| Chromium end-to-end workflow | 1 comprehensive scenario passed against the real API |
| npm audit | 0 known vulnerabilities |
| pip-audit of pinned backend requirements | No known vulnerabilities |
| Editor diagnostics | No errors reported |
| Visual inspection | Inspected captured desktop dashboard; corrected sidebar overflow and missingness-axis bounds |

The browser scenario uploads actual CSV data, runs analysis, visits schema/preview and statistics, selects a column, renders and switches charts, filters and reruns, verifies evidence-backed findings, downloads JSON, and rejects malformed input. API tests additionally verify cross-owner access denial, report formats, failed-job recovery, cancellation, deletion, paginated records, full-record outlier CSV export, and configurable HTML report sections. Numerical tests compare known results and SciPy outputs; ingestion tests cover CSV, multi-sheet XLSX and Parquet. Security tests verify pre-parser authorization, chunked body limits, report escaping and queue capacity. Mocked provider tests verify aggregate-only requests and timeout fallback.

One upstream deprecation warning remains: Starlette's test client recommends `httpx2` over `httpx`. Tests pass; the application uses `httpx` for the optional provider adapter. This warning is not suppressed.

## Measurements

Fixed seed 42, four columns (two numerical, one categorical, one temporal), one isolated benchmark invocation per size. The measurement excludes interpreter/import startup and HTTP/UI overhead. RSS includes interpreter and libraries; it is process peak RSS, not incremental allocation.

| Rows | Engine elapsed | Process peak RSS | JSON artifact bytes |
|---:|---:|---:|---:|
| 10,000 | 0.068 s | 142.12 MiB | 64,782 |
| 100,000 | 0.656 s | 203.31 MiB | 77,863 |

These are local observations, not latency SLOs or evidence for maximum-width datasets, concurrent workloads or adversarial files. Reproduce with `python backend/benchmark.py 10000` / `100000` in the configured environment.

## Acceptance status

**Verified:** local launch; actual CSV/XLSX/Parquet ingestion; schema inference; data preview; full-data profiles; numerical correctness fixtures; histogram/KDE browser rendering and selection; Pearson/Spearman and Cramér’s V computation; three outlier algorithms; numeric and date filtering; deterministic evidence-linked insights; JSON/CSV/HTML report retrieval; ownership isolation; safe malformed-input errors; failed-run recovery; cancellation; real frontend/API integration; build and tests.

**Implemented but not fully verified:** every chart type across every supported datatype; upload cancellation on unreliable networks; long-running timeout and host-crash recovery under load; responsive tablet behavior and full assistive-technology accessibility; startup retention on a long-lived deployment; live external LLM providers; Docker image/Compose runtime; backup/restore procedures; safe behavior at every configurable upper limit.

**Not implemented:** PostgreSQL/S3 deployment, federated identity/RBAC and account-preserving token rotation, horizontally distributed workers and rate limits, per-owner storage quotas, temporal cadence/seasonality inference, arbitrary chart-specific settings, categorical-versus-numeric comparison views, native PDF generation, full observability metrics integration and load/chaos certification.

The complete specification's production definition of done is **not** claimed. This delivery is a verified end-to-end single-node application with the above remaining requirements explicitly tracked.

## Environment limitations

Docker was unavailable, so no container build/deployment claim is made. Git status/diff commands were blocked by missing macOS command-line developer tools (`xcrun`); discovery established the target directory initially contained only `prompt.md`. No existing application source was replaced. A workspace task was added in the parent `.vscode/tasks.json` to launch this application.
