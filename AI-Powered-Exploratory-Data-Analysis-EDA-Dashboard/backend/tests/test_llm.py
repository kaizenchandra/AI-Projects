import asyncio
import httpx
from eda.llm import explain


def test_provider_failure_falls_back(monkeypatch):
    monkeypatch.setenv("EDA_EXTERNAL_AI", "true")
    monkeypatch.setenv("EDA_LLM_URL", "https://provider.invalid/chat/completions")
    monkeypatch.setenv("EDA_LLM_KEY", "test")

    async def fail(*args, **kwargs):
        raise httpx.ReadTimeout("timeout")

    monkeypatch.setattr(httpx.AsyncClient, "post", fail)
    assert asyncio.run(explain({"insights": []}))["status"] == "fallback"


def test_provider_receives_no_column_names_or_rows(monkeypatch):
    monkeypatch.setenv("EDA_EXTERNAL_AI", "true")
    monkeypatch.setenv("EDA_LLM_URL", "https://provider.invalid/chat/completions")
    monkeypatch.setenv("EDA_LLM_KEY", "test")

    async def respond(*args, **kwargs):
        payload = str(kwargs["json"])
        assert "secret_email_column" not in payload
        return httpx.Response(
            200,
            request=httpx.Request("POST", "https://provider.invalid"),
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"explanations":[{"insight_id":"r-1","explanation":"Investigate missingness before choosing a treatment."}]}'
                        }
                    }
                ]
            },
        )

    monkeypatch.setattr(httpx.AsyncClient, "post", respond)
    r = asyncio.run(
        explain(
            {
                "insights": [
                    {
                        "id": "r-1",
                        "type": "data_quality",
                        "affected_columns": ["secret_email_column"],
                        "evidence": {"metric": "missing_percentage", "value": 25},
                    }
                ]
            }
        )
    )
    assert r["status"] == "generated_unverified_interpretation"
