FROM python:3.12-slim
WORKDIR /app
RUN addgroup --system app && adduser --system --ingroup app app && apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY backend/pyproject.toml backend/pyproject.toml
COPY backend/src backend/src
RUN pip install --no-cache-dir ./backend && mkdir -p artifacts && chown -R app:app /app
COPY scripts scripts
# Build a deterministic offline artifact into the image. Production deployments
# can replace it with a promoted artifact at build time or via a mounted volume.
RUN python scripts/generate_fixture.py && python -m house_price.training.train
RUN chown -R app:app /app/artifacts /app/reports /app/datasets
USER app
EXPOSE 8000
CMD ["uvicorn", "house_price.api.main:app", "--app-dir", "backend/src", "--host", "0.0.0.0", "--port", "8000"]
