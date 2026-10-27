# Operations

Set `HOUSE_DATA_PATH`, `MODEL_ARTIFACT_PATH`, `TARGET_CURRENCY`, and `RANDOM_STATE` through the environment. Readiness is 503 until a manifest-compatible artifact exists. The manifest records a SHA-256 dataset fingerprint, feature contract, algorithm, metrics, timestamp, and Python version. Promote a new artifact atomically, retain the prior file, and roll back by restoring the previous known-good artifact and restarting workers.

Logs are JSON lines with request ID, endpoint, status, latency, and model version; request payloads are not logged. Liveness does not depend on the model; readiness does. Add gateway authentication/TLS/rate limiting in production. Monitor request failures/latency, input ranges, feature and prediction distributions, and model-load errors. Accuracy drift requires verified sale-price labels; without outcomes, only input/prediction drift can be measured.
