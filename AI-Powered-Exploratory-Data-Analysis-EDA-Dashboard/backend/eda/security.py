"""Reject oversized and unauthenticated bodies before multipart parsing."""

import asyncio
import json
import os
import secrets
from .ingestion import MAX_BYTES


class BodyGuard:
    def __init__(self, app):
        self.app = app
        self.slots = asyncio.Semaphore(2)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in ("POST", "PUT", "PATCH"):
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        tokens = [t for t in os.getenv("EDA_API_TOKENS", "").split(",") if t]
        supplied = headers.get(b"authorization", b"").decode("latin-1").removeprefix("Bearer ")

        async def reject(code, detail):
            await send(
                {
                    "type": "http.response.start",
                    "status": code,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": json.dumps({"detail": detail}).encode()})

        if tokens and not any(secrets.compare_digest(supplied, t) for t in tokens):
            return await reject(401, "A valid access token is required.")
        if self.slots.locked():
            return await reject(429, "Upload capacity reached. Retry later.")
        limit = MAX_BYTES + 1024 * 1024 if scope["path"] == "/api/v1/datasets" else 64 * 1024
        async with self.slots:
            chunks = []
            size = 0
            try:
                while True:
                    event = await asyncio.wait_for(receive(), timeout=30)
                    if event["type"] == "http.disconnect":
                        return
                    chunk = event.get("body", b"")
                    size += len(chunk)
                    if size > limit:
                        return await reject(413, "Request body exceeds configured limit.")
                    chunks.append(chunk)
                    if not event.get("more_body", False):
                        break
            except asyncio.TimeoutError:
                return await reject(408, "Upload timed out.")
            body = b"".join(chunks)
            delivered = False

            async def bounded_receive():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()

            await self.app(scope, bounded_receive, send)
