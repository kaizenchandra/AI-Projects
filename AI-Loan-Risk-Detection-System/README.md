# AI Loan Risk Prediction System

A working Python service and reproducible ML pipeline for **origination-time, 12-month
first-90+DPD probability of default**. Random Forest is the primary model; regularized
logistic regression and histogram gradient boosting are measured challengers. Outputs
support human review and never approve or reject loans.

**All bundled/generated evaluation data is synthetic. This is not a validated lending
model or a claim of legal compliance.** See [measured verification results](docs/verification.md)
and the [model card](docs/model-card.md). The generator, complete training workflow and
real HTTP single/batch inference have executable checks.

## Start locally

Python 3.11 is the verified runtime; 3.12–3.13 are declared CI targets, not locally certified.
Use an installed Python 3.11 and run from this project directory:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements-dev.lock
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/loan-risk generate --rows 6000 --seed 42 --output data/synthetic.csv
.venv/bin/loan-risk train --data data/synthetic.csv --config configs/training.json --output artifacts/verified
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
.venv/bin/mypy src/loan_risk
```

Generation/training refuse to overwrite existing outputs. Choose a new dataset/run path
for repeat experiments. This checkout already contains the gitignored demonstration run.
The project-local `.tools/uv` and Python installation were provisioned for this machine
because its system Python links were broken; they are not a deployment requirement.
`make install`, `make train`, `make test`, `make lint` wrap the standard commands.

## Serve and predict

For the **local synthetic demonstration only**, copy the hash from the trusted local
training run and create a disposable API key. Production digests must be delivered through
an independent approved channel; never blindly trust a model download's hash sidecar.

```sh
export LOAN_RISK_MODEL_DIR=artifacts/verified/candidate
export LOAN_RISK_TRUSTED_MANIFEST_SHA256="$(cat artifacts/verified/candidate/manifest.sha256)"
export LOAN_RISK_ALLOW_CANDIDATE=true
export LOAN_RISK_API_KEY="$(.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))')"
.venv/bin/uvicorn loan_risk.api:create_app --factory --host 127.0.0.1 --port 8000 \
  --workers 1 --limit-concurrency 32 --timeout-keep-alive 5 --no-access-log \
  --log-config src/loan_risk/logging.json
```

In another terminal with the same authorized environment:

```sh
curl -sS http://127.0.0.1:8000/health/ready
curl -sS http://127.0.0.1:8000/api/v1/risk/predict \
  -H "Authorization: Bearer $LOAN_RISK_API_KEY" -H 'Content-Type: application/json' \
  --data-binary @configs/request.json
```

Response: server-generated request ID, model version, calibrated default probability,
LOW/MODERATE/HIGH display segment and `decision_support_only: true`. Provisional bands
are configuration, not validated business policy. No review threshold is applied by API.

| Endpoint | Contract |
|---|---|
| POST /api/v1/risk/predict | One feature object; authenticated |
| POST /api/v1/risk/predict/batch | `{"records": [feature_object, ...]}`; max 100, ordered results |
| GET /api/v1/model/info | Version, schema, target, synthetic flag and provisional bands; authenticated |
| GET /health/live | Process liveness |
| GET /health/ready | 200 only after trusted model load and inference check; otherwise 503 |
| GET /metrics | Prometheus aggregates; authenticated |

Errors are safe `{error, request_id}` objects: 401 unauthorized, 413 size limit, 422 invalid
schema, 429 worker capacity, 503 missing/failed model, 504 timeout. All features must be
present; null requests imputation. Extra fields, PII columns, nonfinite values, strings in
numeric JSON fields and out-of-range values fail. See [data contract](docs/data-contract.md).
For an automatic real HTTP smoke test with a disposable credential and server cleanup:

```sh
.venv/bin/python scripts/verify_http.py
```

## Offline scoring, explanations and monitoring

Create a CSV with exactly the eleven feature columns from the contract, then:

```sh
.venv/bin/loan-risk predict --model artifacts/verified/candidate \
  --trusted-hash "$LOAN_RISK_TRUSTED_MANIFEST_SHA256" --allow-candidate \
  --input data/features.csv --output artifacts/predictions.json
.venv/bin/loan-risk explain --model artifacts/verified/candidate \
  --trusted-hash "$LOAN_RISK_TRUSTED_MANIFEST_SHA256" --allow-candidate \
  --input data/one-record.csv --output artifacts/explanation.json
.venv/bin/loan-risk monitor --model artifacts/verified/candidate \
  --trusted-hash "$LOAN_RISK_TRUSTED_MANIFEST_SHA256" --allow-candidate \
  --input data/features.csv --output artifacts/drift.json
.venv/bin/loan-risk evaluate-delayed --model artifacts/verified/candidate \
  --trusted-hash "$LOAN_RISK_TRUSTED_MANIFEST_SHA256" --allow-candidate \
  --input data/synthetic.csv --output artifacts/delayed-performance.json
```

`explain` accepts one record. Its reference replacement sensitivities are **not SHAP,
causal effects or adverse-action reasons**. Delayed evaluation requires the full mature
labeled data contract; use a properly retained cohort for historical scoring evaluation.
Monitoring defaults only trigger investigation. Controlled retraining follows the same
workflow and a fresh independent promotion review.

## Artifacts and layout

Training creates immutable `candidate/model.joblib`, manifest/digest, `comparison.json`,
`curves.json`, `eda.json`, EDA narrative and PNG charts. Metadata includes seed/config,
source commit when available, dataset digest/version, schema/feature ordering, dependency
versions, model size and validation/test results. Only trusted joblib bytes may be loaded.

- [src/loan_risk](src/loan_risk): contracts, data, features, model workflow, evaluation,
  explanations, artifact lifecycle, inference, monitoring, API and CLI.
- [tests](tests): unit, ML, integration and API tests using generated temporary artifacts.
- [configs](configs): explicit training/business costs, request, service, approval and alerts.
- [requirements.lock](requirements.lock) / [requirements-dev.lock](requirements-dev.lock):
  transitive version/hash locks, with platform/Python markers.
- [architecture](docs/architecture.md): boundaries, temporal CV, calibration and model comparisons.
- [operations](docs/operations.md): security, monitoring, promotion, rollback, privacy and fairness.
- [data](data/README.md): replacing synthetic data with licensed, authorized data.

There is no notebook-only business logic. No database, MLflow or additional service is
required for the current scope. Bagging and boosting provide ensemble comparisons;
voting/stacking is deferred until validation demonstrates enough gain to justify complexity.

## Containers and CI

```sh
docker build -t loan-risk:1.0.0 .
docker run --rm --read-only --cap-drop ALL --security-opt no-new-privileges \
  --memory 1g --cpus 2 --pids-limit 128 --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  -p 127.0.0.1:8000:8000 \
  --mount "type=bind,src=$PWD/artifacts/verified/candidate,dst=/models,readonly" \
  -e LOAN_RISK_MODEL_DIR=/models -e LOAN_RISK_ALLOW_CANDIDATE=true \
  -e LOAN_RISK_API_KEY -e LOAN_RISK_TRUSTED_MANIFEST_SHA256 loan-risk:1.0.0
```

No data/models/secrets are copied into the non-root runtime image. Pin the base image to
an approved registry digest before deployment; the checked-in patch tag is mutable.
Docker was not available in the implementation environment, so the image build is a CI
verification requirement. The workflow is intended for this directory as repository root;
when embedding in a monorepo, place it in that repository's root `.github/workflows` and
set the working directory/path filters accordingly. CI checks Python 3.11–3.13, lint/types,
tests, dependency audit, image build, Trivy and authenticated inference inside a restricted
non-root container; it never automatically promotes or deploys.

```sh
.venv/bin/pip-audit --strict -r requirements.lock
```

Update locks through a reviewed dependency change using `uv pip compile pyproject.toml
--universal --generate-hashes --output-file requirements.lock` and the corresponding
`--extra dev` command. Revalidate model compatibility and retrain/repackage as necessary.
See [verification](docs/verification.md) for actual checks and known limitations.
