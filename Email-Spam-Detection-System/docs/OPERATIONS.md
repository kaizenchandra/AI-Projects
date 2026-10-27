# Operations checklist

- Keep model artifacts in a trusted, access-controlled store; verify the SHA-256 sidecar before startup.
- Deploy with a non-root user, read-only filesystem, resource limits, TLS termination, authentication, and rate limiting.
- Alert on readiness failures, error rate, rejection rate, latency, CPU/memory saturation, and score-distribution changes.
- Never infer precision/recall from unlabeled production predictions.
- When labeled feedback is available, evaluate false positives first, then recalibrate or retrain with a documented dataset fingerprint.
- Promote a complete artifact atomically and retain the previous artifact for rollback.
- Quarantine rather than delete mail until a representative operating point is demonstrated.
