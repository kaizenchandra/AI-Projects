import asyncio
from eda.security import BodyGuard


def test_unauthorized_body_is_not_read(monkeypatch):
    monkeypatch.setenv("EDA_API_TOKENS", "valid-token-of-at-least-24-characters")
    responses = []

    async def app(*args):
        raise AssertionError("Must not reach parser")

    async def receive():
        raise AssertionError("Must not read unauthenticated body")

    async def send(message):
        responses.append(message)

    asyncio.run(
        BodyGuard(app)(
            {"type": "http", "method": "POST", "path": "/api/v1/datasets", "headers": []}, receive, send
        )
    )
    assert responses[0]["status"] == 401


def test_chunked_body_limit(monkeypatch):
    monkeypatch.setenv("EDA_API_TOKENS", "")
    responses = []

    async def app(*args):
        raise AssertionError("Must not reach parser")

    async def receive():
        return {"type": "http.request", "body": b"x" * 65537, "more_body": False}

    async def send(message):
        responses.append(message)

    asyncio.run(
        BodyGuard(app)(
            {"type": "http", "method": "POST", "path": "/api/v1/datasets/id/analyses", "headers": []},
            receive,
            send,
        )
    )
    assert responses[0]["status"] == 413


def test_queue_capacity(monkeypatch):
    from eda import service
    import pytest

    class Full:
        def acquire(self, blocking=False):
            return False

    monkeypatch.setattr(service, "SLOTS", Full())
    with pytest.raises(ValueError, match="queue is full"):
        service.submit("dataset", "owner", {}, "fingerprint")
