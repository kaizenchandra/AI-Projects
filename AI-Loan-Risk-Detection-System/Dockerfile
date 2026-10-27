# Patch tag is explicit; deployment must additionally pin an approved registry digest.
FROM python:3.11.14-slim-bookworm AS builder
WORKDIR /build
COPY requirements.lock pyproject.toml ./
COPY src ./src
RUN python -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir --require-hashes -r requirements.lock && /opt/venv/bin/pip install --no-deps --no-cache-dir .

FROM python:3.11.14-slim-bookworm
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/matplotlib
RUN groupadd --gid 10001 risk && useradd --uid 10001 --gid risk --no-create-home risk
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=2)"]
CMD ["uvicorn", "loan_risk.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--limit-concurrency", "32", "--timeout-keep-alive", "5", "--no-access-log", "--log-config", "/opt/venv/lib/python3.11/site-packages/loan_risk/logging.json"]
