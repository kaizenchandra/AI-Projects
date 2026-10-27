FROM node:24-alpine AS frontend
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 EDA_STORAGE=/app/data POLARS_MAX_THREADS=2 OPENBLAS_NUM_THREADS=1
WORKDIR /app
COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && useradd --uid 10001 --create-home eda && mkdir /app/data && chown eda:eda /app/data
COPY backend/ /app/backend/
COPY --from=frontend /app/frontend/dist /app/frontend/dist
USER eda
WORKDIR /app/backend
EXPOSE 8000
CMD ["uvicorn","eda.api:app","--host","0.0.0.0","--port","8000","--workers","1","--no-access-log"]
