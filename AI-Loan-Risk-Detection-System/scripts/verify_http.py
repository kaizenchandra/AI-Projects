"""Exercise a real loopback server with an ephemeral development-only credential."""

import json
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

root = Path(__file__).resolve().parents[1]
candidate = root / "artifacts/verified/candidate"
# This is an explicitly local synthetic smoke test, not production trust establishment.
manifest_hash = (candidate / "manifest.sha256").read_text().strip()
key = secrets.token_urlsafe(32)
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
environment = {
    **os.environ,
    "LOAN_RISK_MODEL_DIR": str(candidate),
    "LOAN_RISK_TRUSTED_MANIFEST_SHA256": manifest_hash,
    "LOAN_RISK_ALLOW_CANDIDATE": "true",
    "LOAN_RISK_API_KEY": key,
    "MPLCONFIGDIR": str(root / ".tools/matplotlib"),
}
server = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "uvicorn",
        "loan_risk.api:create_app",
        "--factory",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--no-access-log",
        "--log-level",
        "warning",
    ],
    cwd=root,
    env=environment,
)
base = f"http://127.0.0.1:{port}"
try:
    for _ in range(100):
        try:
            with urllib.request.urlopen(base + "/health/ready", timeout=1) as response:
                assert response.status == 200
            break
        except (urllib.error.URLError, TimeoutError):
            if server.poll() is not None:
                raise RuntimeError("Server exited before readiness") from None
            time.sleep(0.1)
    else:
        raise RuntimeError("Server readiness timed out")
    record = json.loads((root / "configs/request.json").read_text())
    results = {}
    for endpoint, payload in [
        ("predict", record),
        ("predict/batch", {"records": [record, record]}),
    ]:
        request = urllib.request.Request(
            base + "/api/v1/risk/" + endpoint,
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            results[endpoint] = json.load(response)
    assert (
        results["predict"]["default_probability"]
        == results["predict/batch"][0]["default_probability"]
    )
    print(
        json.dumps(
            {
                "real_http_smoke": "passed",
                "single_and_batch_match": True,
                "probability": results["predict"]["default_probability"],
            }
        )
    )
finally:
    server.terminate()
    server.wait(timeout=10)
