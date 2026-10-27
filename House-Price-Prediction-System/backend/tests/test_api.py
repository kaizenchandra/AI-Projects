from fastapi.testclient import TestClient
from house_price.api.main import app

def test_live_contract(): assert TestClient(app).get("/api/v1/health/live").json() == {"status":"live"}
def test_validation_rejects_extra_field():
    payload={"OverallQual":7,"GrLivArea":1700,"GarageCars":2,"TotalBsmtSF":800,"YearBuilt":2000,"Neighborhood":"x","secret":1}
    assert TestClient(app).post("/api/v1/predictions", json=payload).status_code == 422
