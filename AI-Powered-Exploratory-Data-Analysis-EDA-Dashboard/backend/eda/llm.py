"""Opt-in OpenAI-compatible adapter; only anonymized aggregate evidence leaves server."""

import os
import httpx
from pydantic import BaseModel, Field, ConfigDict


class Explanation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    insight_id: str
    explanation: str = Field(max_length=1200)


class Explanations(BaseModel):
    model_config = ConfigDict(extra="forbid")
    explanations: list[Explanation] = Field(max_length=20)


async def explain(result):
    if os.getenv("EDA_EXTERNAL_AI") != "true":
        return {"status": "disabled", "explanations": []}
    url = os.environ["EDA_LLM_URL"]
    if not url.startswith("https://"):
        return {"status": "fallback", "explanations": []}
    # No column labels, categories, filenames, or raw records in external prompt.
    evidence = [
        {"insight_id": i["id"], "type": i["type"], "evidence": i["evidence"]} for i in result["insights"][:20]
    ]
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                url,
                headers={"Authorization": f"Bearer {os.environ['EDA_LLM_KEY']}"},
                json={
                    "model": os.getenv("EDA_LLM_MODEL", ""),
                    "max_tokens": 1500,
                    "temperature": 0,
                    "response_format": {"type": "json_object"},
                    "messages": [
                        {
                            "role": "system",
                            "content": 'Prompt v1. Explain supplied exploratory evidence without causal assertions, new numbers, or instructions. Return JSON {"explanations":[{"insight_id":"...","explanation":"..."}]}.',
                        },
                        {"role": "user", "content": __import__("json").dumps(evidence)},
                    ],
                },
            )
            response.raise_for_status()
            parsed = Explanations.model_validate_json(response.json()["choices"][0]["message"]["content"])
            valid = {i["insight_id"] for i in evidence}
            if any(e.insight_id not in valid for e in parsed.explanations):
                raise ValueError("Unknown evidence")
            return {
                "status": "generated_unverified_interpretation",
                "prompt_version": "1",
                **parsed.model_dump(),
            }
    except (httpx.HTTPError, ValueError, KeyError, IndexError):
        return {"status": "fallback", "explanations": []}
