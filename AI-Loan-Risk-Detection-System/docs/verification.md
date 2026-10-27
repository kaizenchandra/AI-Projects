# Verification and measured results

Executed on 2026-10-25 using Python 3.11.11 on macOS ARM64.

## Data and evaluation scope

**Synthetic data only. These results do not establish real-world lending accuracy or fairness.**

6,000 generated loans, seed 42. Train: 2,992; validation: 869; final out-of-time test: 1,266. 873 loans fall in maturity embargo periods. Test contains 290 defaults (22.91%). All loans in this generated sample have sufficient follow-up. Validation/tests exercise exclusion of immature records.

## Final out-of-time comparison

| Model | Calibration | ROC-AUC | Average precision | Brier | Log loss | Precision | Recall | F1 | Balanced accuracy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gradient_boosting | sigmoid | 0.6821 | 0.4094 | 0.1618 | 0.4988 | 0.4343 | 0.4448 | 0.4395 | 0.6363 |
| logistic | isotonic | 0.7024 | 0.4426 | 0.1576 | 0.4899 | 0.4318 | 0.4586 | 0.4448 | 0.6397 |
| random_forest | isotonic | 0.6824 | 0.4000 | 0.1631 | 0.5010 | 0.4386 | 0.4310 | 0.4348 | 0.6336 |

Classification metrics use each model’s validation-selected review threshold, not 0.5. Logistic regression performed best in this synthetic run. RF remains the requested primary; no test-driven family promotion occurred.

| Model | ROC-AUC 95% IID bootstrap interval | Brier 95% interval | Validation threshold | Test review fraction | Compressed bytes | Batch ms/row |
|---|---|---|---:|---:|---:|---:|
| gradient_boosting | 0.6440–0.7104 | 0.1524–0.1730 | 0.287095 | 0.2346 | 211388 | 0.0165 |
| logistic | 0.6650–0.7352 | 0.1489–0.1704 | 0.325363 | 0.2433 | 13909 | 0.0121 |
| random_forest | 0.6507–0.7108 | 0.1535–0.1751 | 0.306160 | 0.2251 | 426394 | 0.0223 |

100 bootstrap resamples; intervals exclude model-selection and temporal-dependence uncertainty. Latency is one local batch measurement, not a concurrent SLA benchmark.

## Calibration and review assumptions

| Model | Raw validation Brier | Sigmoid validation Brier | Isotonic validation Brier |
|---|---:|---:|---:|
| gradient_boosting | 0.157983 | 0.162493 | 0.162764 |
| logistic | 0.153457 | 0.155086 | 0.153533 |
| random_forest | 0.160345 | 0.163578 | 0.163455 |

**Calibration did not improve validation Brier over raw estimators in this run.** An explicitly calibrated estimator is packaged to meet the probability-scoring contract; calibration is not assumed to improve every cohort. Reassess this choice with representative data.

Cost assumptions: false-negative default 10,000 units; false-positive alert 200; manual review 50; validation review cap 25%. Thresholds minimize this stylized cost under capacity. FN-cost sensitivity at 0.5×, 1× and 2× is in the complete report. Risk bands 0.10/0.25 are provisional display settings, not validated policy.

## Explainability and robustness

Leading RF validation permutation features (mean increase in Brier loss):
- annual_income: 0.013307 (shuffle SD 0.001580).
- monthly_debt_obligations: 0.006593 (shuffle SD 0.001377).
- credit_utilization: 0.006129 (shuffle SD 0.001860).
- debt_to_income: 0.005607 (shuffle SD 0.000848).
- loan_amount: 0.004444 (shuffle SD 0.000785).

Income, obligations, DTI and engineered ratios are correlated. Neither ranking nor local feature replacement establishes causality or legally sufficient reason codes. Transformed-feature impurity and original-feature permutation have different feature spaces.

| Model | Validation Brier: 10% randomly missing | Validation Brier: unseen repayment category |
|---|---:|---:|
| gradient_boosting | 0.164248 | 0.163175 |
| logistic | 0.159243 | 0.154623 |
| random_forest | 0.166061 | 0.163729 |

These two controlled scenarios are not a complete stress or adversarial assessment.

## Executed checks

- `python -m pytest -q --junitxml=docs/test-results.xml`: **45 passed**, 0 failures, 0 errors, 0 skipped; final run 6.179 seconds.
- `ruff check src tests scripts`: passed.
- `ruff format --check src tests scripts`: 20 files formatted.
- `mypy src/loan_risk`: passed, 14 source files checked.
- `python -m compileall -q src scripts`: passed.
- `python -m pip check`: no broken requirements.
- Full CLI generation and three-model training: passed; immutable candidate and all reports created.
- Full training repeated in tests: same prediction arrays and final metrics with fixed seeds.
- CLI predict (40 rows), explain (one row), monitor and evaluate-delayed: passed.
- Real loopback Uvicorn HTTP test: passed; readiness and authenticated single/batch predictions agree. Sample request PD: 0.1012874076184146. Server stopped afterward.
- `uv build --wheel --out-dir dist`: installable wheel built successfully; logging configuration is included.
- `pip-audit --format json --output docs/dependency-audit.json`: no known vulnerabilities in audited installed dependencies after fixes. Local first-party package is not on PyPI and is explicitly unaudited by this tool.
- Initial dependency audit found eight advisories in Starlette/pytest. Pins were upgraded to Starlette 1.3.1, FastAPI 0.162.0 and pytest 9.0.3 before final verification.
- Initial SciPy 1.15.2 wheel failed to load on this machine. Pinned SciPy 1.16.3 passed numerical training/tests.

## Not executed / remaining limitations

- Docker build, image scan and container runtime: Docker is not installed. Dockerfile and CI build/Trivy steps are supplied, not locally verified.
- GitHub Actions and Python 3.12/3.13 matrix: not run here. The workflow assumes this project is repository root; a monorepo must relocate/adapt it.
- Git source commit: unavailable because the machine’s Xcode command-line tools are missing. No commit was fabricated.
- Production load tests, external penetration test, real-data fairness/credit validation and legal review: not performed.
- Deprecation warnings remain in matplotlib/pyparsing, Starlette/httpx/anyio and sklearn/SciPy; tests are not silently warning-filtered.
- No SHAP implementation; supported local sensitivity is non-additive and has explicit limitations.
- API-key authentication is a single service role; enterprise identity/RBAC/TLS/rate-limits need an external gateway. Native worker threads cannot forcibly interrupt hung model code.
- Hash checks require a trusted external digest; joblib cannot safely load arbitrary models. Approval records rely on release-system access control, not a cryptographic approver identity.
- Synthetic evaluation, IID intervals and heuristic drift thresholds cannot establish deployment validity.

## Evidence locations

- [Complete comparison](../artifacts/verified/comparison.json)
- [Curves](../artifacts/verified/curves.json) and [evaluation figure](../artifacts/verified/evaluation.png)
- [EDA](../artifacts/verified/EDA.md), [aggregate EDA JSON](../artifacts/verified/eda.json)
- [Permutation importance](../artifacts/verified/permutation_importance.png), [impurity importance](../artifacts/verified/impurity_importance.png)
- [Population and term relationships](../artifacts/verified/population_and_terms.png)
- [Manifest](../artifacts/verified/candidate/manifest.json)
- [JUnit test results](test-results.xml), [dependency audit](dependency-audit.json)

Generated data/models/reports under artifacts are intentionally gitignored but available in this checkout. Rerun the documented commands to reproduce them; do not commit real borrower data.

## Definition-of-done evidence

| Requirement | Implementation / evidence |
|---|---|
| Data contract, validation, maturity and leakage | data-contract.md; contracts.py/data.py; data tests |
| Feature processing and serving parity | features.py; pipeline and API consistency tests |
| Three models, tuning, calibration and comparison | training.py; comparison.json; repeated-training test |
| Costs and validation-only thresholds | evaluation.py; threshold tests; report sensitivity |
| Global/local explanation | explainability.py; importance plots; explanation tests and CLI |
| Reproducible versioned artifacts | artifacts.py; manifest; hash/schema/dependency corruption tests |
| Real-time and batch API | api.py; real HTTP smoke and API tests |
| Input/failure/concurrency handling | strict contracts; body limits; timeout/capacity tests |
| CI and Docker configuration | workflow and Dockerfile supplied; execution limitation above |
| Monitoring/retraining/promotion | monitoring.py; CLI; operations.md; promotion gates |
| Security/privacy/fairness/model card | operations.md and model-card.md; no real-world validation claim |

Functional engineering completion does not constitute lending deployment approval.

## Follow-up hardening

The follow-up regression run reproduced acceptance of infinite review costs, boolean
numeric training values, and malformed or unbound governance attestations before fixing
them. Fifteen regression cases were added; the complete suite now has 45 passing tests.
Approval validation rejects blank or non-string reviewer/ticket fields, invalid/naive/future
timestamps, non-boolean or incomplete reviews, unexpected fields, and candidate-digest
mismatches. Correctly formed approvals still cannot promote synthetic models.

CI now has a separate Python 3.11 container job that trains a synthetic fixture, builds
and scans the image, starts it non-root with a read-only filesystem/model mount and
resource limits, checks readiness, performs authenticated inference, and removes its
container. This job has **not** been executed here because Docker is unavailable.
The prior synthetic model measurements remain unchanged; these changes harden contracts
and release checks rather than change the estimation procedure.
