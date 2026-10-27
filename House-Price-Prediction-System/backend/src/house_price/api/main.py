import json, logging, time, uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from house_price.core.config import ARTIFACT_PATH, CORS_ORIGINS
from house_price.inference.service import ModelService

logging.basicConfig(level=logging.INFO, format='%(message)s'); log = logging.getLogger("house-price")
app = FastAPI(title="House Price Prediction API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=list(CORS_ORIGINS), allow_methods=["GET", "POST"], allow_headers=["*"])
service = ModelService(ARTIFACT_PATH)

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    OverallQual: int = Field(ge=1, le=10, examples=[7]); GrLivArea: float = Field(gt=0, le=100000, examples=[1710])
    GarageCars: float = Field(ge=0, le=10, examples=[2]); TotalBsmtSF: float = Field(ge=0, le=100000, examples=[856])
    YearBuilt: int = Field(ge=1800, le=2100, examples=[2003]); Neighborhood: str = Field(min_length=1, max_length=64, examples=["CollgCr"])

class PredictionValue(BaseModel):
    price: float
    currency: str
    unit: str

class ModelSummary(BaseModel):
    name: str
    version: str

class PredictionResponse(BaseModel):
    prediction: PredictionValue
    model: ModelSummary
    request_id: str

@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = str(uuid.uuid4()); request.state.request_id = request_id; start = time.perf_counter(); response = await call_next(request)
    response.headers["X-Request-ID"] = request_id; log.info(json.dumps({"request_id": request_id, "path": request.url.path, "status": response.status_code, "latency_ms": round((time.perf_counter()-start)*1000, 2), "model_version": service.metadata.get("model_version")})); return response

@app.get("/api/v1/health/live")
def live(): return {"status": "live"}
@app.get("/api/v1/health/ready")
def ready():
    if not service.ready: raise HTTPException(503, "Model artifact is unavailable or incompatible")
    return {"status": "ready", "model_version": service.metadata["model_version"]}
@app.get("/api/v1/models/current")
def current():
    if not service.ready: raise HTTPException(503, "Model is not ready")
    return {k: service.metadata[k] for k in ("model_version", "algorithm", "features", "currency", "unit", "metrics")}
@app.post("/api/v1/predictions", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request):
    if not service.ready: raise HTTPException(503, "Model is not ready")
    value = service.predict(payload.model_dump())
    return {"prediction": {"price": value, "currency": service.metadata["currency"], "unit": service.metadata["unit"]}, "model": {"name": service.metadata["algorithm"], "version": service.metadata["model_version"]}, "request_id": request.state.request_id}
