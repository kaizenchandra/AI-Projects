Act as a **Principal AI/ML Engineer, Machine Learning Architect, NLP Specialist, Senior Backend Engineer, MLOps Engineer, Security Engineer, and QA Automation Engineer**.

Your task is to **design, implement, test, document, and deliver a complete, production-oriented AI-powered Email Spam Detection System** using:

- Bayes' Theorem
- Conditional Probability
- Naive Bayes Classification
- Bag-of-Words (BoW)
- Natural Language Processing (NLP)
- Supervised Machine Learning

The system must classify emails as **SPAM** or **HAM (legitimate email)** and provide an interpretable spam score using a trained probabilistic classification model.

**This is an implementation task, not merely a tutorial, conceptual explanation, architecture proposal, or pseudocode exercise.**

Implement the actual solution within the available repository and execution environment.

Prioritize correctness, maintainability, reproducibility, security, reliability, model quality, and operational simplicity.

---

## 2. Execution Instructions

Follow these rules throughout the implementation:

1. Inspect the existing repository before making changes. Identify its architecture, dependencies, configuration, conventions, tests, and applicable instructions.
2. Preserve existing architecture and unrelated functionality. Do not unnecessarily restructure the project.
3. If starting from an empty repository, create a maintainable, modular Python project.
4. Make reasonable implementation decisions independently. Ask questions only when missing information materially affects correctness, security, or compatibility.
5. Implement functional code rather than placeholder functions, TODO-based solutions, or disconnected snippets.
6. Avoid unnecessary dependencies, microservices, distributed infrastructure, or abstraction layers.
7. Validate assumptions and consult official documentation when library or framework behavior is version-dependent.
8. Execute applicable tests, training workflows, API smoke tests, and code-quality checks where tooling is available.
9. Never fabricate training results, dataset statistics, model performance, benchmarks, or verification outcomes.
10. Clearly disclose environmental limitations, unavailable datasets, unexecuted checks, and remaining risks.

**Definition of success:** A developer can obtain the repository, provide a valid labeled dataset, install dependencies, train and evaluate the classifier, launch the inference API, submit email classification requests, execute tests, and follow documented deployment procedures.

---

## 3. Technology Stack

For a greenfield implementation, prefer the following stack:

| Layer | Preferred Technology |
|---|---|
| Programming language | Python 3.12+ |
| ML framework | scikit-learn |
| Data processing | pandas, NumPy |
| Text representation | CountVectorizer |
| Classification algorithm | Multinomial Naive Bayes |
| REST API | FastAPI |
| Schema validation | Pydantic |
| API server | Uvicorn |
| Model persistence | joblib for trusted local artifacts |
| Unit/integration testing | pytest |
| Static analysis | Ruff and mypy |
| Containerization | Docker |
| CI/CD | GitHub Actions, unless the repository uses another CI system |
| Experiment tracking | Lightweight local manifest by default; MLflow only if justified |
| Metrics | Prometheus-compatible instrumentation when operationally appropriate |
| Deployment | Container-based, Kubernetes-compatible |

Use compatible, explicitly pinned dependency versions and a reproducible dependency lock or constraints strategy.

Python should own the ML training and inference implementation for a greenfield project. If an existing repository is based on **Java 17/21 and Spring Boot**, integrate into its existing architecture without introducing a Python microservice unless the ML requirements justify that boundary.

Do not introduce Kafka, Redis, PostgreSQL, Kubernetes, or cloud services merely for architectural completeness.

---

## 4. Mathematical Foundations and Explainability

Implement and clearly explain the mathematical foundations underlying the classifier.

### 4.1 Bayes' Theorem

Explain and demonstrate:

\[
P(C\mid X)=\frac{P(X\mid C)P(C)}{P(X)}
\]

Where:

- \(C\): Email classification, SPAM or HAM.
- \(X\): Observed email features.
- \(P(C)\): Prior probability of the class.
- \(P(X\mid C)\): Likelihood of the observed features given the class.
- \(P(C\mid X)\): Posterior probability of the class.
- \(P(X)\): Evidence or normalization factor.

Include numerical examples showing how prior probabilities and word likelihoods influence classification.

### 4.2 Conditional Probability

Explain:

\[
P(A\mid B)=\frac{P(A\cap B)}{P(B)}
\]

Demonstrate conditional word probabilities in spam and legitimate emails.

For example:

- Probability of observing "free" given SPAM.
- Probability of observing "meeting" given HAM.
- Effect of repeatedly occurring words.
- Effect of class imbalance on priors.

### 4.3 Multinomial Naive Bayes

Implement and explain the multinomial model:

\[
\log P(C\mid X)\propto
\log P(C)+\sum_{i=1}^{|V|}x_i\log P(w_i\mid C)
\]

Use additive smoothing:

\[
P(w_i\mid C)=
\frac{N_{i,C}+\alpha}
{\sum_{j=1}^{|V|}N_{j,C}+\alpha|V|}
\]

Where:

- \(V\): Training vocabulary.
- \(x_i\): Frequency of word \(w_i\) in the email.
- \(N_{i,C}\): Training-token count of word \(w_i\) in class \(C\).
- \(\alpha\): Smoothing parameter.

Explain:

- Conditional independence assumption.
- Maximum a posteriori classification.
- Logarithmic probability computation and numerical stability.
- Class priors and class imbalance.
- Laplace/additive smoothing.
- Unknown and out-of-vocabulary tokens.
- Why Multinomial Naive Bayes works well with word-count vectors.
- Limitations of the independence assumption.
- Why high model confidence is not necessarily calibrated probability.

Use a trusted scikit-learn implementation for production inference.

Additionally, build a **small Naive Bayes implementation from scratch** as an isolated educational/reference component. Include automated tests comparing its mathematical behavior against scikit-learn on deterministic datasets.

Do not duplicate the production inference implementation merely for educational purposes.

---

## 5. Dataset Acquisition and Validation

Support configurable, labeled email datasets.

Allow a standard CSV schema:

```text
text,label
"Congratulations you won a prize",spam
"Please review tomorrow's meeting notes",ham
```

Optionally support datasets with separate `subject` and `body` fields and raw `.eml` input through a dedicated ingestion adapter.

Requirements:

- Use legitimate, appropriately licensed public datasets such as SpamAssassin or another suitable spam corpus.
- Document dataset origin, licenses, versions, and acquisition instructions.
- Do not assume a dataset is already available.
- Validate required columns and supported labels.
- Normalize labels to a consistent binary representation.
- Reject missing, invalid, or malformed records with actionable error reporting.
- Detect empty emails, exact duplicates, and near-duplicates.
- Analyze class imbalance and class distribution.
- Avoid training on personally identifiable information unless appropriately authorized.
- Minimize and protect sensitive email content.
- Prevent train/validation/test leakage caused by duplicate messages, threads, repeated templates, or corpus-source artifacts.

If no real dataset is available, provide small deterministic fixtures for software verification and reproducible instructions to obtain a real dataset.

**Never report toy fixture performance as representative spam detection accuracy.**

---

## 6. NLP Preprocessing and Feature Engineering

Implement a reusable preprocessing component.

Support:

1. Unicode normalization.
2. HTML-to-text extraction using a safe parser.
3. Subject/body handling.
4. Configurable lowercase conversion.
5. Tokenization.
6. Whitespace normalization.
7. Optional punctuation handling.
8. Optional stop-word filtering.
9. Optional stemming or lemmatization when empirically justified.
10. URL and email-address normalization when useful.
11. MIME and character-encoding validation for raw email ingestion.
12. Configurable message size and token-count limits.

Avoid aggressive preprocessing that destroys useful spam signals, including suspicious URLs, punctuation patterns, numbers, currency symbols, or unusual character sequences.

Ensure that preprocessing used during training and inference is identical.

### Bag-of-Words implementation

Use `CountVectorizer` within a scikit-learn `Pipeline`.

Support configurable:

- `ngram_range`
- `min_df`
- `max_df`
- `max_features`
- `binary` mode
- tokenization rules
- vocabulary filtering

Explain the conversion:

**Email text → tokens → vocabulary → word-count vector → classifier**

Default to raw token counts for Multinomial Naive Bayes.

Treat TF-IDF, character n-grams, and alternative vectorization strategies as optional experiments rather than silently replacing the requested Bag-of-Words foundation.

Fit vocabulary and preprocessing transformations only on appropriate training data.

Never allow validation/test examples to influence learned vocabulary or model parameters.

---

## 7. Model Training Pipeline

Implement a reproducible offline training workflow.

Required stages:

1. Load dataset.
2. Validate records and labels.
3. Perform dataset profiling.
4. Detect duplicates and leakage risks.
5. Split training, validation, and test data.
6. Fit the preprocessing/vectorization pipeline.
7. Train Multinomial Naive Bayes.
8. Tune eligible hyperparameters.
9. Select a decision threshold.
10. Evaluate against the untouched test set.
11. Save the model and metadata.
12. Generate a machine-readable evaluation report.

Use deterministic random seeds.

Use stratified splitting where suitable, or group-aware/time-aware splitting where relationships between emails make random splitting misleading.

Keep all hyperparameter and threshold decisions independent of the final test data.

Support configurable parameters including:

```yaml
model:
  algorithm: multinomial_naive_bayes
  alpha: 1.0
  fit_prior: true

vectorizer:
  ngram_range: [1, 2]
  min_df: 1
  max_df: 1.0
  max_features: 50000

training:
  random_seed: 42

classification:
  decision_threshold: 0.5
```

Treat these as initial configuration values, not proven optimal production parameters.

If fitting and evaluating multiple candidate models, use a defined selection protocol and retain only validated candidates.

---

## 8. Model Evaluation and Threshold Selection

Implement comprehensive classification evaluation.

Include:

| Metric | Purpose |
|---|---|
| Accuracy | Overall classification correctness |
| Precision (SPAM) | Reliability of spam predictions |
| Recall (SPAM) | Percentage of actual spam detected |
| F1-score | Precision/recall balance |
| Confusion matrix | TP, TN, FP, FN |
| False Positive Rate | Legitimate emails incorrectly classified as spam |
| False Negative Rate | Spam emails missed |
| PR-AUC | Performance across precision-recall trade-offs |
| ROC-AUC | Ranking discrimination |
| Brier score | Probability quality |
| Calibration curve | Reliability of predicted probabilities |

Give special attention to **false positives** because blocking legitimate emails can have significant operational consequences.

Implement configurable threshold selection using validation data, preferably optimizing spam recall subject to an explicitly configured false-positive-rate budget.

Do not claim that a stringent false-positive-rate target is satisfied without enough representative HAM test examples to support that conclusion.

Generate:

- Confusion matrix visualization.
- Precision-recall curve.
- ROC curve.
- Threshold sensitivity analysis.
- Calibration assessment.
- Classification report.
- Error analysis for false positives and false negatives.

Assess performance on representative email distributions and discuss dataset shift.

Do not equate `predict_proba` outputs with well-calibrated probabilities without supporting evidence.

If calibration is necessary, fit a supported calibration method without data leakage and re-evaluate it on untouched data.

Report sample counts and uncertainty where meaningful.

---

## 9. Model Explainability

Provide interpretable explanations for classifications.

Implement a method for identifying words that contribute to SPAM or HAM classification using the fitted model's token likelihoods and their log-likelihood contributions.

For a classification, optionally return:

- Strongest spam-indicating tokens.
- Strongest ham-indicating tokens.
- Relative contribution to classification.
- Model version.

Distinguish statistical associations from causal explanations.

Do not expose the complete training vocabulary or sensitive token information through public APIs without an explicit business requirement.

Ensure diagnostic explanations cannot leak private email contents to logs or monitoring systems.

---

## 10. REST API Implementation

Build a versioned REST inference API using FastAPI.

### Required endpoints

**POST `/api/v1/predict`**

Request:

```json
{
  "subject": "Congratulations!",
  "body": "You have won a free prize. Claim it now."
}
```

Illustrative response structure:

```json
{
  "label": "SPAM",
  "spam_score": 0.97,
  "threshold": 0.8,
  "model_version": "1.0.0",
  "request_id": "example-request-id"
}
```

The sample values are illustrative, not actual model outputs.

Only label a score as a calibrated probability when its calibration has been evaluated.

The prediction response must reflect the actual trained model, threshold, and version.

Additional endpoints:

- `GET /health/live`
- `GET /health/ready`
- `GET /api/v1/model/info`

Implement:

- Typed request/response models.
- Input validation.
- Maximum payload sizes.
- Unicode and malformed-text handling.
- Consistent exception handling.
- Standardized error responses.
- Appropriate HTTP status codes.
- Request correlation IDs.
- Configurable inference timeouts where relevant.
- Authentication and rate limiting when exposed outside a trusted network.
- OpenAPI documentation.

Load the model once during application startup rather than deserializing it on every request.

Readiness must fail if the required model artifact cannot be loaded or validated.

Prediction must remain side-effect-free and deterministic for a fixed model and input.

Define behavior for empty messages, oversized payloads, missing fields, corrupted model artifacts, and unsupported input formats.

Do not accept arbitrary client-supplied file paths or executable serialized objects.

---

## 11. Production Architecture

Use a modular architecture with clear responsibilities.

Suggested modules:

```text
email-spam-detection/
├── src/
│   └── spam_detection/
│       ├── api/
│       ├── application/
│       ├── domain/
│       ├── infrastructure/
│       ├── data/
│       ├── features/
│       ├── training/
│       ├── evaluation/
│       └── inference/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── configs/
├── scripts/
├── docs/
├── artifacts/
├── Dockerfile
├── compose.yaml
├── pyproject.toml
├── README.md
└── .github/
    └── workflows/
```

Adjust this structure to actual implementation complexity. Do not create layers, empty directories, or indirection without a clear reason.

Separate:

- Email/domain representation.
- Dataset loading.
- Preprocessing.
- Feature engineering.
- Model training.
- Evaluation.
- Model persistence.
- Inference.
- HTTP transport.
- Configuration and infrastructure.

Apply SOLID, dependency inversion, and Clean/Hexagonal Architecture pragmatically.

Keep training outside the inference request path.

---

## 12. Model Artifact Management

Save a complete, reproducible inference pipeline and its associated metadata.

Metadata should include:

- Model identifier and version.
- Training timestamp.
- Dataset version and fingerprint.
- Preprocessing/vectorizer configuration.
- Hyperparameters.
- Decision threshold.
- Label mapping.
- Dependency compatibility information.
- Evaluation metrics.
- Artifact checksum.

Use a consistent model artifact naming/versioning convention.

Never deserialize untrusted pickle/joblib artifacts. Treat model artifact integrity and provenance as a security boundary.

Implement safe model loading with compatibility checks and actionable failure diagnostics.

Support controlled promotion and rollback of model versions.

Do not hot-swap a partially written model artifact into a running service.

---

## 13. Security and Privacy

Apply secure-by-default engineering practices.

Implement or document:

- Strict input validation.
- Resource limits for email parsing and inference.
- Safe HTML/MIME handling.
- Protection against malformed or decompression-amplified messages.
- No execution of email attachments or embedded scripts.
- Trusted model artifact loading.
- Dependency and container vulnerability scanning.
- Non-root container execution.
- Externalized secrets.
- Authentication and authorization where required.
- Request throttling for publicly accessible deployments.
- TLS termination at the appropriate deployment boundary.
- No raw email contents in routine logs.
- No sensitive training data committed into source control.
- Secure storage, access control, and retention policies.
- Safe error responses that do not expose implementation details.

Document threats such as spam obfuscation, adversarial wording, training-data poisoning, and model extraction.

Do not claim that spam classification alone provides phishing, malware, or attachment security.

---

## 14. Observability and Operations

Add production-relevant observability.

Metrics should include:

- Request count.
- Error count.
- P50/P95/P99 inference latency when sufficient data exists.
- Request rejection rate.
- Prediction distribution.
- Model load status.
- Active model version.
- CPU and memory utilization through the deployment environment.

Where labeled feedback is legitimately available, support monitoring for:

- False positives.
- False negatives.
- Precision and recall.
- Data drift.
- Model performance degradation.

Do not claim ground-truth accuracy from unlabeled production predictions.

Use structured JSON logging where appropriate, with request IDs and model versions.

Avoid logging message text, sensitive features, credentials, or high-cardinality identifiers into metrics.

Document operational alerts and appropriate responses.

---

## 15. Automated Testing and Quality

Implement meaningful tests covering the mathematical model and the full inference workflow.

### Unit tests

- Bayes' theorem calculations.
- Conditional probability.
- Smoothing.
- Log-likelihood computation.
- Class prior handling.
- Unknown vocabulary.
- Tokenization and preprocessing.
- Vectorizer output.
- Label normalization.
- Threshold boundary behavior.
- Model serialization compatibility.

### Integration tests

- End-to-end scikit-learn training pipeline.
- Save/load prediction parity.
- API inference with actual fixture-trained model.
- Invalid request handling.
- Missing/corrupt artifact handling.
- Consistency across repeated predictions.
- Training/inference preprocessing parity.

### End-to-end tests

Exercise:

**Load data → validate → preprocess → train → evaluate → save model → start API → predict**

Also verify:

- Docker image build.
- Container startup.
- Health endpoints.
- Successful and rejected requests.
- API contract/schema compatibility.

Use deterministic fixtures.

Run linting, type checking, and relevant security checks.

Prefer strong behavioral coverage over arbitrary coverage percentages. Add regression tests for substantive bugs and edge cases.

Clearly distinguish tests executed from tests merely provided.

---

## 16. Deployment and CI/CD

Create a secure, reproducible container deployment.

Requirements:

- Minimal production container image.
- Multi-stage Docker build where beneficial.
- Non-root runtime user.
- Dependency pinning.
- Health checks.
- Graceful termination.
- Configurable deployment settings.
- No unnecessary privileged access.
- Externalized model artifacts/configuration where appropriate.

Implement a CI pipeline that performs:

1. Dependency installation.
2. Linting.
3. Type checking.
4. Unit tests.
5. Integration tests.
6. Container build.
7. Relevant security/dependency scanning.

Keep large-scale model retraining separate from normal pull-request checks.

Add Kubernetes manifests or Helm charts only when requested or justified by the target deployment platform.

Document resource requirements, model startup behavior, rollout strategy, and rollback steps.

Do not claim production readiness based solely on a successful local container build.

---

## 17. Documentation and Learning Material

Create a complete `README.md` and supporting documentation.

Include:

- Project overview.
- Business problem.
- System architecture.
- Mathematical foundations.
- Bayes' theorem worked example.
- Conditional probability example.
- Bag-of-Words example.
- Naive Bayes training example.
- Dataset schema and provenance.
- Environment setup.
- Training commands.
- Evaluation commands.
- Model versioning.
- API usage.
- Testing.
- Docker execution.
- Deployment.
- Configuration.
- Security considerations.
- Model limitations.
- Operational monitoring.
- Troubleshooting.

Include a Mermaid architecture diagram and an inference sequence diagram.

Explain the following pipeline:

**Email → Validation → Text Preprocessing → Bag-of-Words → Multinomial Naive Bayes → Decision Threshold → SPAM/HAM Response**

Provide practical explanations connecting the mathematics to the actual code.

---

## 18. Required Deliverables

Produce the following:

1. Complete implementation in the repository.
2. Reproducible data ingestion and validation pipeline.
3. Naive Bayes reference implementation with mathematical tests.
4. Production inference pipeline using scikit-learn.
5. Training and evaluation commands.
6. Model persistence and metadata handling.
7. FastAPI prediction service.
8. Environment-driven configuration.
9. Automated unit, integration, and end-to-end tests.
10. Model performance reporting and visualizations.
11. Docker configuration.
12. CI workflow.
13. Architecture and operational documentation.
14. Example API requests and responses.
15. Security and deployment guidance.

Do not include large public datasets, private emails, secrets, or trained-model binaries in source control by default.

---

## 19. Acceptance Criteria

Treat implementation as complete only when the following behavior is verified, or explicitly identify what remains unverified.

- A documented command trains a model from a valid labeled dataset.
- Training produces a versioned inference artifact and metadata.
- Predictions are consistent before and after artifact reload.
- The training and serving pipelines share identical preprocessing.
- Data leakage controls are implemented and tested where applicable.
- A held-out evaluation report is generated.
- The report includes false-positive and false-negative behavior.
- Threshold selection does not use the final test set.
- API predictions derive from the saved model, not hardcoded classification rules.
- Invalid inputs return defined, safe error responses.
- Readiness fails when the model cannot be loaded.
- Relevant automated tests pass.
- The container builds and starts if the runtime supports Docker.
- Dependency and configuration requirements are documented.
- No training data or message contents are unintentionally exposed in logs.
- Limitations and residual production risks are documented.

Any numerical performance target must be explicitly configured and validated against representative data. Do not invent pass/fail results for unprovided targets.

For email-blocking use cases, default to a conservative operational recommendation: use review/quarantine or human feedback rather than irrevocably discarding legitimate messages based on an unvalidated classifier.

---

## 20. Implementation Workflow

Execute the work in this order:

**Phase 1 — Repository Assessment**

Inspect the existing project, identify dependencies and architecture, and determine the smallest viable implementation approach.

**Phase 2 — Mathematical and Data Foundations**

Implement the educational Naive Bayes calculations, dataset validation, and preprocessing tests.

**Phase 3 — Machine Learning Pipeline**

Implement feature extraction, training, model selection, leakage-safe evaluation, thresholding, and artifact persistence.

**Phase 4 — Inference Application**

Implement model loading, domain/application logic, prediction API, validation, and error handling.

**Phase 5 — Quality and Security**

Implement tests, secure configuration, model artifact verification, observability, and code-quality checks.

**Phase 6 — Packaging and Deployment**

Implement Docker, CI, startup verification, and deployment configuration appropriate to the repository.

**Phase 7 — Documentation and Final Verification**

Run available checks, verify the end-to-end workflow, and document how the system is operated.

Do not stop after planning when repository editing and execution are available. Continue through implementation and verification within the authorized environment.

---

## 21. Final Response Format

At the end, provide a concise engineering delivery report with these sections:

### A. Implementation Summary

Describe what was implemented and key architectural choices.

### B. Files Created or Modified

Identify major changes and their responsibilities.

### C. Architecture and Data Flow

Explain training and inference, including mathematical foundations.

### D. Commands to Run

Provide exact installation, training, evaluation, testing, and API startup commands.

### E. Model Evaluation

Show real metrics only when produced from an actual evaluation run. Include dataset size, split strategy, false positives, false negatives, and threshold.

### F. Verification Results

Separate:

- Passed checks.
- Failed checks.
- Checks not executed.

### G. Security and Operational Considerations

Explain relevant controls, configuration, deployment, and monitoring.

### H. Remaining Risks and Improvements

List known gaps without overstating completeness or production readiness.

**Final instruction: Build a complete, mathematically correct, testable, maintainable, secure, and production-oriented Email Spam Detection System. Favor a reliable, focused solution over unnecessary enterprise complexity.**
