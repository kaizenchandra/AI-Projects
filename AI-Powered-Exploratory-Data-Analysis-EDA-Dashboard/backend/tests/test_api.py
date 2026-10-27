import os
import tempfile
import time

os.environ["EDA_STORAGE"] = tempfile.mkdtemp(prefix="prism-tests-")
os.environ["EDA_API_TOKENS"] = "test-owner-one-token-123456789,test-owner-two-token-123456789"
from fastapi.testclient import TestClient
from eda.api import app

A = {"Authorization": "Bearer test-owner-one-token-123456789"}
B = {"Authorization": "Bearer test-owner-two-token-123456789"}


def test_workflow_and_isolation():
    with TestClient(app) as c:
        assert c.get("/api/v1/datasets").status_code == 401
        assert c.get("/api/v1/ready").status_code == 200
        bad = c.post("/api/v1/datasets", headers=A, files={"file": ("x.csv", b"a,a\n1,2")})
        assert bad.status_code == 422
        d = c.post(
            "/api/v1/datasets",
            headers=A,
            files={"file": ("../../test.csv", b"x,y,category\n1,2,a\n2,4,a\n3,6,b\n4,8,b\n100,200,b\n")},
        ).json()
        assert d["name"] == "test.csv"
        assert c.get("/api/v1/datasets", headers=B).json() == []
        assert c.post("/api/v1/datasets/" + d["id"] + "/analyses", headers=B, json={}).status_code == 404
        j = c.post("/api/v1/datasets/" + d["id"] + "/analyses", headers=A, json={}).json()["id"]
        for _ in range(100):
            state = c.get("/api/v1/jobs/" + j, headers=A).json()
            if state["state"] in ("completed", "failed"):
                break
            time.sleep(0.1)
        assert state["state"] == "completed", state
        assert c.get("/api/v1/analyses/" + j, headers=B).status_code == 404
        result = c.get("/api/v1/analyses/" + j, headers=A).json()
        assert result["overview"]["rows"] == 5
        assert result["insights"]
        outlier_csv = c.get("/api/v1/analyses/" + j + "/report?format=outliers&column=x", headers=A)
        assert outlier_csv.status_code == 200 and "100,200,b" in outlier_csv.text
        assert (
            c.get("/api/v1/analyses/" + j + "/report?format=outliers&column=x", headers=B).status_code == 404
        )
        page = c.get("/api/v1/analyses/" + j + "/rows?offset=2&limit=2", headers=A).json()
        assert page["total"] == 5 and page["rows"][0]["x"] == "3" and len(page["rows"]) == 2
        assert c.get("/api/v1/analyses/" + j + "/rows", headers=B).status_code == 404
        for format in ("html", "csv", "json"):
            r = c.get("/api/v1/analyses/" + j + "/report?format=" + format, headers=A)
            assert r.status_code == 200 and len(r.content) > 100
        assert c.post("/api/v1/analyses/" + j + "/explanations", headers=A).json()["status"] == "disabled"
        # A bad configuration fails in isolation; the next job can still finish.
        failed = c.post(
            "/api/v1/datasets/" + d["id"] + "/analyses", headers=A, json={"overrides": {"missing": "numeric"}}
        ).json()["id"]
        for _ in range(100):
            state = c.get("/api/v1/jobs/" + failed, headers=A).json()
            if state["state"] == "failed":
                break
            time.sleep(0.1)
        assert state["state"] == "failed"
        second = c.post(
            "/api/v1/datasets/" + d["id"] + "/analyses",
            headers=A,
            json={"filters": [{"column": "x", "op": "lt", "value": "10"}]},
        ).json()["id"]
        for _ in range(100):
            state = c.get("/api/v1/jobs/" + second, headers=A).json()
            if state["state"] == "completed":
                break
            time.sleep(0.1)
        assert c.get("/api/v1/analyses/" + second, headers=A).json()["overview"]["rows"] == 4
        cancelled = c.post("/api/v1/datasets/" + d["id"] + "/analyses", headers=A, json={}).json()["id"]
        assert c.delete("/api/v1/jobs/" + cancelled, headers=A).status_code == 200
        assert c.get("/api/v1/jobs/" + cancelled, headers=A).json()["state"] == "cancelled"
        assert c.get("/api/v1/analyses/" + cancelled, headers=A).status_code == 409
        from eda import service

        for _ in range(100):
            if cancelled not in service.ACTIVE:
                break
            time.sleep(0.05)
        assert c.delete("/api/v1/datasets/" + d["id"], headers=B).status_code == 404
        assert c.delete("/api/v1/datasets/" + d["id"], headers=A).status_code == 204
        assert c.get("/api/v1/analyses/" + j, headers=A).status_code == 404
