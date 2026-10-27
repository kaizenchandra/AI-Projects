# Operations, security and controlled lifecycle

## Service boundary

TLS termination, network policy, client identity, rate limiting, connection/header/body
read deadlines and audit retention belong at the authenticated gateway. The service uses
constant-time bearer API-key comparison for a single trusted scoring role; scoring, model
info and metrics require the key. There is no anonymous mode. Keys come from a secret
manager through environment injection, never an image, repository or command history.
Rotate keys with rolling deployments; this reference implementation has no per-user RBAC,
OIDC issuer validation or tenant isolation. Place it behind the enterprise gateway before
using financial data. Probes are intentionally anonymous and reveal only availability.

Application limits default to 128 KiB actual body bytes, 100 rows/batch, 2 concurrent
model calls and 10-second processing timeouts. Input is buffered only to the byte limit.
Workers are bounded and capacity is held until actual model execution completes, even if
the caller times out. Python threads cannot safely kill a hung native operation: use
process-level health supervision and gateway deadlines. Run one Uvicorn worker per
container, limit concurrent connections to 32, cap CPU/memory/PIDs and scale replicas.
No CORS, debugger, public docs or file-serving endpoints are enabled. Error responses do
not echo input values, stack traces, paths or secrets. Request IDs are server-generated.
JSON audit events contain route, status, duration and model version, not borrower values,
predictions, auth tokens or identifiers. Disable generic access logs to avoid query leakage.

## Model trust and promotion

`joblib` can execute code. Artifact checksums alone do **not** make an arbitrary model safe.
Only the controlled training/release pipeline may create packages. Store models in a
private versioned encrypted bucket; separate training, approval and deployment roles.
The independently trusted manifest SHA-256 must arrive through deployment configuration.
Never take it from an untrusted request or automatically trust a downloaded sidecar hash.
The colocated sidecar is a local convenience, not the trust root. The loader checks the
manifest digest, schema/feature order, exact ML dependency versions, Python minor version,
model payload hash and class order before serving. Bytes are verified before deserialization.
Readiness fails closed for absent keys, missing/corrupt/incompatible/unapproved models.

`loan-risk promote --model SOURCE --trusted-hash APPROVED_SOURCE_DIGEST --approval APPROVAL.json
--output NEW_VERSION` creates a separate approved package. Required approvals cover
independent validation, evaluation acceptance, security, fairness and legal review with
reviewer/ticket/time metadata and `candidate_manifest_sha256` equal to the independently
trusted source digest. Reviewer and ticket must be nonblank strings, approval time must
be timezone-aware and not in the future, and every review gate must be an actual boolean
true. Unknown fields and approvals for another candidate are rejected before model loading.
These are attestations, not automated proof; enforce approver
identity and separation of duties in CI/IAM. Synthetic models cannot be production-promoted.
`LOAN_RISK_ALLOW_CANDIDATE=true` is for isolated synthetic demonstrations only.

Publish an approved version, configure its separately delivered digest, run readiness and
canary checks, then route traffic. Rollback means restoring the previous immutable model
path **and** its trusted digest, restarting and checking readiness. Never mutate a running
package or hot-swap files. CI does not automatically deploy or approve models.

## Monitoring and delayed outcomes

Protected `/metrics` exports request counts/status/route, latency histogram, model
availability/load failures, predicted-probability histogram and missing-value counts by
allowlisted feature. Status 422 measures schema violations, 429 capacity and 5xx failures.
Do not add borrower IDs, raw values or arbitrary paths as labels. Scrape with a dedicated
secret and network restriction. [Alert rules](../configs/alerts.yml) contain provisional
availability, error and p95 latency thresholds; tune with a production workload.

Offline `monitor` compares authorized feature windows with the training reference using
PSI (smoothed bin proportions), missingness deltas and prediction-distribution PSI.
Defaults are PSI 0.2 and absolute missingness change 0.05, configurable in the function;
these are investigation heuristics, not statistically calibrated tests. Minimum 30 rows
is only a technical guard. Set business-appropriate minimums and monitor seasonality.
The baseline includes training population predictions, which can differ from deployment;
review it against an approved reference window. Models do not automatically retrain.

`evaluate-delayed` validates a mature labeled cohort and reports discrimination/calibration
loss. For fairness auditing, call `delayed_performance` with separately authorized, securely
joined audit-group labels; raw sensitive columns are never accepted in training/API data.
It suppresses groups below 100 or with one outcome class, reports per-group metrics,
calibration gaps and bootstrap intervals. It does not prove fairness, handle intersectional
multiple testing, select legally appropriate groups, or join scores to identity records.
For real operations retain protected scored-cohort snapshots/model versions separately:
rescoring later with another model does not reconstruct historical performance.

Investigate changes in base rate, missingness, score distribution, Brier/log loss, observed
minus predicted default rate and subgroup outcomes after labels mature. Retrain only with
an approved new dataset version and frozen split protocol; rerun validation/security and
obtain fresh approvals. Keep previous models for reproducible audit and rollback.

## Privacy, fairness and legal review

Use only permitted origination information and verified provenance. Pseudonymous IDs
remain sensitive; keep raw datasets outside images and Git, encrypt at rest/in transit,
restrict analyst access and audit exports. Training aggregates and model artifacts may
still leak information; apply access and small-cell disclosure controls. Define retention
periods with counsel, propagate deletion to snapshots/backups/derived artifacts where
required, and record legal holds. Log retention should be shorter than regulated model
audit retention when permitted. No universal retention period is assumed.

Removing protected traits is insufficient: employment/history and income can be proxies.
Representative data, subgroup calibration/error analysis with uncertainty, disparate-impact
review, independent validation, stress scenarios and human escalation are deployment gates.
Review jurisdiction-specific credit reporting, fair lending/equal opportunity, adverse-action,
privacy, automated-decision and model-governance obligations (for example ECOA/FCRA in the
US, GDPR/EU AI Act where applicable, or relevant RBI/Indian privacy rules). This project
makes no claim of compliance. Human reviewers must have authority to challenge scores;
customer reason codes, approvals and rejections are outside this software's validated scope.

## Supply chain

Install hash-locked dependencies and run the vulnerability scan. Docker uses a patch-level
Python base tag and non-root UID 10001. The tag remains mutable: resolve and approve an
immutable registry digest before release and regularly rebuild for OS patches. Run Trivy
in CI and pin GitHub Actions to organization-reviewed commit SHAs before deployment.
A successful package audit does not audit first-party code, images or configuration.
Use a read-only root filesystem, read-only model mount, tmpfs, dropped capabilities and
no-new-privileges. A dedicated smaller inference dependency split is a future image-size
optimization; the current image includes scientific plotting dependencies but no datasets.
