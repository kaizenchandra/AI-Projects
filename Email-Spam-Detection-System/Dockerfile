FROM python:3.12-slim AS builder
WORKDIR /build
COPY pyproject.toml ./
COPY src ./src
RUN python -m pip install --no-cache-dir --upgrade pip && python -m pip wheel --no-cache-dir --wheel-dir /wheels .

FROM python:3.12-slim
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels
COPY src ./src
COPY artifacts ./artifacts
USER appuser
ENV PYTHONPATH=/app/src SPAM_MODEL_PATH=/app/artifacts/model.joblib
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live')"
CMD ["uvicorn", "spam_detection.api:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
