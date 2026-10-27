Act as a **Principal AI/ML Engineer, Machine Learning Architect, MLOps Engineer, and Senior Backend Engineer** with extensive experience designing, implementing, deploying, and operating enterprise-grade machine learning systems in regulated financial environments.

**Your task is to DESIGN, IMPLEMENT, TEST, DOCUMENT, and VERIFY a complete end-to-end AI Loan Risk Prediction System**, using:

- **Random Forests** as the primary predictive model.
- **Ensemble Learning** to compare and improve predictive performance.
- **Feature Importance and Explainable AI (XAI)** to understand and interpret predictions.
- Production-grade ML engineering, backend architecture, MLOps, security, observability, and operational practices.

This is an **implementation task, not merely a system-design exercise**.

Produce a functioning, maintainable, reproducible, and testable project with real source code, configurations, tests, documentation, and deployment artifacts.

Do not stop after generating an architecture diagram, notebook, tutorial, or code snippets.

---

## 2. Primary Business Objective

Build a system that estimates the probability that a borrower will default on a loan using information legitimately available at the time of loan origination.

The system must support:

1. Loan default probability prediction.
2. Configurable risk segmentation.
3. Model performance assessment.
4. Global and local model explainability.
5. Human review and credit-risk decision support.
6. Batch and real-time inference.
7. Model versioning and reproducibility.
8. Model monitoring and controlled retraining.

### Default prediction definition

Unless the provided dataset or business requirements establish a different definition:

- **Target:** Binary loan-default outcome.
- **Default event:** First occurrence of 90 or more days past due.
- **Prediction horizon:** 12 months from origination.
- **Prediction timestamp:** Loan origination.
- **Output:** Calibrated probability of default between 0 and 1.

Exclude loans without sufficient observable outcome history from supervised evaluation unless censoring is explicitly modeled.

Do not use any feature unavailable at the defined prediction timestamp.

Treat predictions as decision-support information. Do not automatically approve or reject loans without separately validated decision policies, human oversight, and applicable compliance reviews.

---

## 3. Engineering Principles and Execution Rules

Follow these requirements throughout implementation:

- Apply SOLID, DRY, KISS, and separation of concerns pragmatically.
- Use clean, modular architecture without unnecessary abstractions.
- Prefer simple, maintainable solutions over speculative enterprise complexity.
- Follow existing repository instructions, dependency versions, architecture, and conventions.
- Preserve unrelated files and existing behavior.
- Avoid unnecessary frameworks, dependencies, and microservices.
- Use explicit configuration rather than hidden constants.
- Implement deterministic and reproducible ML workflows.
- Define clear interfaces and data contracts.
- Handle invalid inputs, edge cases, failures, and unexpected conditions.
- Avoid data leakage, train-serving skew, and evaluation contamination.
- Treat data privacy, security, explainability, and auditability as fundamental requirements.
- Include operational documentation and observable system behavior.
- Never fabricate test results, model metrics, benchmark results, or production-readiness claims.

Inspect the existing repository before changing code. If no suitable project exists, scaffold one.

Make routine implementation decisions independently. Ask questions only when unresolved ambiguity materially affects correctness, safety, compliance, or compatibility.

---

## 4. Preferred Technology Stack

Use the following defaults when creating a new project:

| Layer | Technology |
|---|---|
| Programming language | Python 3.11+ |
| ML framework | scikit-learn |
| Data processing | pandas, NumPy |
| Statistical analysis | SciPy |
| Primary algorithm | RandomForestClassifier |
| Baseline model | LogisticRegression |
| Challenger model | HistGradientBoostingClassifier |
| Explainability | Permutation importance, optional SHAP |
| Model serialization | joblib with trusted artifact loading |
| API framework | FastAPI |
| API validation | Pydantic |
| Configuration | Pydantic Settings |
| Testing | pytest |
| Data/model validation | Custom validation and scikit-learn utilities |
| Experiment tracking | MLflow, when justified |
| Containerization | Docker |
| Automation | GitHub Actions |
| Logging | Structured JSON logs |
| Monitoring | OpenTelemetry / Prometheus-compatible metrics when appropriate |

Use pinned, mutually compatible dependency versions and document the supported Python environment.

Do not introduce a separate Java or Spring Boot microservice unless integration with an existing enterprise backend makes that boundary necessary.

---

## 5. Dataset Discovery, Validation, and Preparation

Implement a complete data-processing workflow.

### Dataset strategy

If an authorized dataset is supplied, inspect and use it.

If no dataset is available:

- Create a deterministic synthetic dataset generator with realistic structural relationships between borrower characteristics, loan terms, and repayment outcomes.
- Clearly identify all generated data as synthetic.
- Never represent synthetic evaluation results as evidence of real-world predictive accuracy.
- Document how a properly licensed real dataset can replace synthetic data.

### Example permitted input features

- Annual income.
- Loan amount.
- Loan term.
- Interest rate, only if available at the chosen decision point.
- Employment length.
- Debt-to-income ratio.
- Credit utilization.
- Credit history length.
- Existing debt obligations.
- Delinquency history available before origination.
- Number of active credit accounts.
- Prior repayment behavior available before origination.

Do not include personal identifiers, protected characteristics, post-origination outcomes, future payment data, or other target-leaking fields as predictive features.

If sensitive attributes are legitimately available for fairness auditing, store and process them separately under appropriate access controls.

### Required preprocessing

Implement:

1. Dataset schema validation.
2. Type validation and coercion where safe.
3. Missing-value analysis.
4. Appropriate numerical and categorical imputation.
5. Categorical encoding with unseen-category support.
6. Outlier and invalid-range detection.
7. Duplicate record handling.
8. Feature distributions and summary statistics.
9. Target-label validation.
10. Class-imbalance analysis.
11. Explicit feature inclusion/exclusion policies.
12. Training-serving feature consistency.

Use scikit-learn `Pipeline` and `ColumnTransformer` wherever appropriate.

Fit preprocessing only on training data.

Provide focused unit tests for preprocessing, validation, and leakage-prevention logic.

---

## 6. Exploratory Data Analysis

Create a reproducible EDA workflow that examines:

- Dataset dimensions and schema.
- Missing-value distributions.
- Loan-default class distribution.
- Numerical feature distributions.
- Categorical feature distributions.
- Correlation and multicollinearity.
- Outliers and suspicious values.
- Income versus default outcomes.
- Debt-to-income ratio versus default outcomes.
- Credit utilization versus default outcomes.
- Loan amount and loan term relationships.
- Time-based population changes when timestamps exist.

Produce useful charts and a concise EDA report.

Distinguish correlation from causation. Do not interpret observed associations as evidence of causal effects.

---

## 7. Feature Engineering

Implement meaningful, testable feature transformations.

Candidate engineered features include:

- Loan-to-income ratio.
- Debt-to-income ratio.
- Credit utilization bands.
- Credit history length.
- Income-to-obligation ratio.
- Loan amount relative to documented repayment capacity.
- Credit account utilization patterns.
- Prior delinquency counts or rates, where available before origination.

For every engineered feature:

- Document its definition.
- Identify source columns.
- Validate numerical constraints.
- Prevent division-by-zero errors.
- Handle missing or extreme values.
- Check availability at prediction time.
- Prevent target leakage.
- Verify the same transformation is applied in training and inference.

Avoid feature engineering that encodes prohibited information or introduces unjustified proxies for protected characteristics.

---

## 8. Train, Validation, and Test Strategy

Implement defensible data partitioning.

### Preferred validation strategy

For datasets containing meaningful origination timestamps:

- Use chronological train, validation, and test periods.
- Keep the final out-of-time test set untouched until final evaluation.
- Account for prediction-horizon maturity and label availability.
- Prevent repeated borrower records from leaking across evaluation boundaries where relevant.
- Document gaps, embargoes, or other safeguards required by the data.

For datasets without meaningful timestamps:

- Use stratified train/validation/test partitioning.
- Group by borrower or another relevant entity where necessary to avoid leakage.
- Explain why this is a weaker approximation of production performance.

Never fit preprocessing, feature selection, oversampling, calibration, or hyperparameter selection using held-out test data.

Implement split-integrity tests.

---

## 9. Machine Learning Model Development

Build and evaluate at least three models.

### Model A: Logistic Regression

Use as an interpretable baseline.

Include:

- Appropriate preprocessing.
- Regularization.
- Class-weight options.
- Cross-validation within the training partition.
- Probability calibration assessment.

### Model B: Random Forest — Primary Model

Use `RandomForestClassifier` as the primary candidate.

Tune appropriate parameters, including:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`
- `max_features`
- `class_weight`
- `max_samples`, where applicable

Use bounded `RandomizedSearchCV` or an equivalent reproducible search strategy.

Do not perform unnecessarily large searches or consume unbounded compute.

### Model C: Gradient Boosting — Challenger

Implement a gradient boosting challenger, preferably scikit-learn's histogram-based gradient boosting unless an existing project justifies another library.

Compare:

- Predictive discrimination.
- Probability calibration.
- Inference latency.
- Model size.
- Robustness.
- Explainability.
- Operational complexity.

### Ensemble learning

Explain and implement relevant concepts:

- Bagging through Random Forest.
- Boosting through the challenger model.
- Voting or stacking only if validated improvement justifies the additional complexity.

Do not assume an ensemble is automatically superior.

Use a common, defensible evaluation methodology.

---

## 10. Class Imbalance Handling

Evaluate appropriate strategies:

- Class weighting.
- Threshold optimization.
- Stratified or group-aware training folds.
- Resampling within training folds only, if justified.

Do not apply SMOTE or any other resampling technique before splitting the dataset.

Avoid optimizing solely for accuracy when defaults are rare.

Explain trade-offs between false positives and false negatives.

---

## 11. Model Performance Evaluation

Implement reusable evaluation utilities and produce a structured comparison report.

### Required metrics

- ROC-AUC.
- PR-AUC / Average Precision.
- Precision.
- Recall.
- F1-score.
- Balanced accuracy.
- Confusion matrix.
- Brier score.
- Log loss.
- Calibration curves.
- Precision-recall curves.
- ROC curves.

Evaluate probability quality separately from classification-threshold performance.

Include uncertainty estimates or confidence intervals where statistically and computationally appropriate.

Use held-out data for final reporting and clearly distinguish validation metrics from test metrics.

### Cost-sensitive evaluation

Implement a configurable cost model for:

- False-negative defaults.
- False-positive risk alerts.
- Manual review costs.
- Review capacity.

Determine candidate decision thresholds using validation data.

Report the business assumptions and sensitivity to changes in those assumptions.

Do not select thresholds based on the final test set.

---

## 12. Feature Importance and Explainable AI

Implement both global and local interpretability.

### Global feature importance

Support:

1. Random Forest impurity-based feature importance.
2. Permutation feature importance on held-out data.
3. Feature-importance visualizations.
4. Comparison of importance rankings.
5. Analysis of correlated-feature effects.
6. Stability checks when practical.

Clearly explain that impurity-based importance may be biased toward high-cardinality or continuous predictors, and that correlated features can distort permutation rankings.

### Local prediction explanations

Where appropriate, integrate SHAP for model-specific explanations.

Support:

- Local explanation of individual predictions.
- Global explanation summaries.
- Top contributing features.
- Validation of expected feature ordering.
- Explanation availability and failure handling.

Do not claim that feature attribution establishes causation.

Do not equate SHAP values or permutation importance directly with legally sufficient adverse-action reasons. Treat customer-facing explanation and reason-code generation as a separately validated compliance capability.

Provide human-readable model documentation and explicit limitations.

---

## 13. Probability Calibration and Risk Segmentation

Evaluate whether predicted probabilities correspond to observed default frequencies.

Compare relevant calibration methods, such as:

- Sigmoid calibration.
- Isotonic calibration.

Fit calibration using appropriate training or validation procedures without contaminating the final test set.

Define configurable risk bands, for example:

- Low risk.
- Moderate risk.
- High risk.

Do not hardcode arbitrary probability bands as validated business policy.

Keep model scoring distinct from downstream decision thresholds and human-review policies.

---

## 14. Production Inference API

Build a FastAPI application exposing versioned REST endpoints.

### Required endpoints

`POST /api/v1/risk/predict`

Accept validated borrower and loan attributes.

Return a response such as:

```json
{
  "request_id": "example-request-id",
  "model_version": "1.0.0",
  "default_probability": 0.174,
  "risk_segment": "MODERATE",
  "decision_support_only": true
}
```

The values above are illustrative, not actual model predictions.

Also implement:

- `POST /api/v1/risk/predict/batch`
- `GET /health/live`
- `GET /health/ready`
- `GET /api/v1/model/info`

### API engineering requirements

- Versioned request/response contracts.
- Strict input validation.
- Predictable error responses.
- Request-size and batch-size limits.
- Bounded request processing.
- Concurrency safeguards.
- Model loading at application startup.
- Explicit readiness failure when model artifacts are missing or invalid.
- Consistent inference across requests.
- Request correlation identifiers.
- Structured logging without sensitive borrower attributes.
- Authentication and authorization suitable for protected financial APIs.
- Configurable network and service timeouts.
- Secure configuration and secret management.
- Model version identifiers in responses.
- Tests for malformed inputs and operational failure paths.

Keep the API independent of model training.

Do not train models during ordinary inference requests.

Never deserialize untrusted model artifacts.

---

## 15. Software Architecture

Use a modular structure similar to:

```text
loan-risk-prediction/
├── configs/
├── data/
│   ├── synthetic/
│   └── README.md
├── notebooks/
├── src/
│   └── loan_risk/
│       ├── api/
│       ├── contracts/
│       ├── data/
│       ├── features/
│       ├── training/
│       ├── evaluation/
│       ├── explainability/
│       ├── inference/
│       ├── monitoring/
│       └── core/
├── scripts/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── api/
├── docs/
├── .github/
│   └── workflows/
├── pyproject.toml
├── Dockerfile
├── .dockerignore
├── .gitignore
├── Makefile
└── README.md
```

Adapt the directory structure if the repository already has appropriate conventions.

Ensure notebooks are exploratory interfaces rather than the only place where business logic and model functionality exist.

Do not create empty modules merely to satisfy a directory diagram.

---

## 16. MLOps and Model Lifecycle

Implement a reproducible workflow covering:

**Data → Validation → Feature Engineering → Training → Evaluation → Calibration → Model Selection → Packaging → Deployment → Monitoring → Retraining**

### Reproducibility

Record:

- Dataset identifier/version.
- Data schema version.
- Feature schema version.
- Training configuration.
- Random seeds.
- Dependency versions.
- Model hyperparameters.
- Evaluation metrics.
- Artifact checksums.
- Model version.
- Training timestamps.
- Source commit when available.

### Model registry and promotion

Provide a simple, explicit model artifact lifecycle.

Where justified, integrate MLflow tracking and model registry.

Keep candidate evaluation separate from deployment approval.

Require evaluation and governance checks before model promotion.

### Monitoring

Design or implement proportionate monitoring for:

- Request volume and failure rate.
- Inference latency.
- Input schema violations.
- Missing-feature rates.
- Input feature drift.
- Prediction distribution drift.
- Data quality changes.
- Model performance when delayed labels become available.
- Calibration deterioration.
- Segment-level performance.
- Model availability and load failures.

Use configurable alerts and document limitations.

Drift detection should trigger investigation, not unrestricted automatic retraining or deployment.

---

## 17. Security, Privacy, Fairness, and Compliance

Treat this system as potentially high-impact financial decision-support software.

Implement and document:

- Data minimization.
- No sensitive PII in logs.
- No secrets committed to source control.
- Secure model artifact storage.
- Restricted training-data access.
- API authentication and authorization.
- Dependency and container security checks.
- Input validation and bounded resource consumption.
- Audit metadata for model predictions and versions.
- Retention/deletion guidance.
- Fairness and model-risk evaluation.
- Human review and escalation procedures.
- Explainability limitations.
- Traceability of model selection and changes.

Fairness evaluation should consider applicable protected groups when legally and operationally appropriate audit data is available.

Measure subgroup performance and calibration where sample sizes permit, and explicitly report statistical uncertainty.

Do not treat simply removing protected attributes as proof of fairness.

Identify jurisdiction-specific lending, privacy, adverse-action, and model-governance obligations as review items rather than claiming compliance automatically.

---

## 18. Testing and Quality Engineering

Implement meaningful tests at multiple layers.

### Unit tests

Cover:

- Data validation.
- Missing values.
- Invalid ranges.
- Feature transformations.
- Unseen categories.
- Safe ratio calculations.
- Feature ordering.
- Probability bounds.
- Risk segmentation.
- Configuration parsing.

### ML tests

Cover:

- Training reproducibility.
- Split disjointness.
- Leakage prevention.
- Feature and target separation.
- Pipeline serialization/deserialization.
- Training/inference preprocessing parity.
- Prediction consistency.
- Calibration and threshold logic.
- Model compatibility checks.

### Integration tests

Cover:

- Complete synthetic training pipeline.
- Model artifact loading.
- API inference using generated artifacts.
- Batch inference.
- Health/readiness behavior.
- Missing or corrupted model artifacts.
- Model/schema incompatibility.

### API tests

Cover:

- Valid requests.
- Invalid requests.
- Missing fields.
- Extra fields.
- Invalid data types.
- Nonfinite numbers.
- Extreme values.
- Oversized batches.
- Unauthorized requests where authentication is enabled.
- Deterministic prediction responses.
- Error handling without sensitive data leakage.

Use pytest fixtures and realistic test boundaries.

Avoid brittle tests that expect exact model scores unless the test explicitly fixes all necessary artifacts and dependencies.

---

## 19. CI/CD, Containers, and Developer Experience

Provide:

- A reproducible local setup.
- Dependency management.
- Formatting and linting.
- Static type checks where practical.
- Automated tests.
- Container build.
- Dependency vulnerability checks.
- CI pipeline.
- Configuration examples without real secrets.
- Clear development commands.

Use a small, secure runtime image where feasible.

Run the application as a non-root user.

Exclude training datasets, caches, credentials, and unnecessary artifacts from the production container.

Pin container base images appropriately and document image update practices.

Do not package real borrower information in container images.

---

## 20. Documentation and Architecture Deliverables

Create a clear README and supporting documentation describing:

1. Business problem and prediction target.
2. Assumptions and limitations.
3. Technology choices.
4. Project architecture.
5. Data contract.
6. Data preparation.
7. Feature engineering.
8. ML algorithms.
9. Model selection and evaluation.
10. Explainability methodology.
11. API contracts and examples.
12. Local development.
13. Model training and inference commands.
14. Docker usage.
15. Testing.
16. Monitoring and drift detection.
17. Security and fairness considerations.
18. Model promotion and rollback.
19. Known risks and gaps.
20. Future improvements.

Use Mermaid diagrams where they materially improve architecture or workflow documentation.

Provide a model card summarizing intended use, inappropriate use, evaluation evidence, limitations, and governance requirements.

---

## 21. Implementation Workflow

Execute the implementation in the following order.

### Phase 1 — Repository Assessment and Design

- Inspect existing files and instructions.
- Identify current dependencies and conventions.
- Define the prediction contract and assumptions.
- Define the minimum viable architecture.
- Identify data availability and restrictions.

### Phase 2 — Data Pipeline

- Implement dataset generation or loading.
- Add validation.
- Implement data splitting.
- Build preprocessing and feature engineering.
- Add data-pipeline tests.

### Phase 3 — Model Development

- Implement Logistic Regression baseline.
- Implement Random Forest primary model.
- Implement Gradient Boosting challenger.
- Add bounded hyperparameter tuning.
- Compare models using appropriate validation data.

### Phase 4 — Evaluation and Explainability

- Implement performance reports.
- Evaluate probability calibration.
- Select configurable thresholds.
- Implement feature importance.
- Add optional SHAP support.
- Document model limitations.

### Phase 5 — Production API

- Package the selected model and preprocessing.
- Implement FastAPI endpoints.
- Add validation, security, and operational safeguards.
- Add API and integration tests.

### Phase 6 — MLOps and Deployment

- Add model metadata/versioning.
- Implement monitoring hooks.
- Add Docker and CI.
- Document promotion, rollback, and retraining procedures.

### Phase 7 — Verification and Hardening

- Execute automated tests.
- Run static quality checks.
- Run training smoke tests.
- Test API startup and inference.
- Test invalid-input and missing-model behavior.
- Build the container where the environment permits.
- Review security and privacy risks.
- Document unresolved gaps.

Complete actual implementation tasks within the authorized environment rather than describing how someone else should implement them.

If the environment prevents a verification step, clearly state what was not run and why.

---

## 22. Definition of Done

The project is functionally complete only when:

- [ ] A documented data contract exists.
- [ ] Dataset validation and preprocessing work.
- [ ] Leakage-prevention safeguards are implemented.
- [ ] A reproducible training pipeline works.
- [ ] The Logistic Regression baseline works.
- [ ] The Random Forest primary model works.
- [ ] The Gradient Boosting challenger works.
- [ ] Models are compared using appropriate metrics.
- [ ] Probability calibration is evaluated.
- [ ] Threshold selection uses documented business assumptions.
- [ ] Global feature importance works.
- [ ] Local explainability is supported or limitations are clearly documented.
- [ ] Model artifacts include version metadata.
- [ ] Real-time inference works through the API.
- [ ] Batch inference works.
- [ ] Input validation and failure handling work.
- [ ] Automated tests cover critical paths.
- [ ] CI and Docker configurations are provided.
- [ ] Security and privacy safeguards are documented.
- [ ] Monitoring and retraining guidance exists.
- [ ] A model card and README are available.
- [ ] Verification results and limitations are reported accurately.

Do not mark an item complete without implementation evidence.

A project passing these checks is not automatically validated for real-world lending use. Production deployment additionally requires representative real-world data, independent validation, security assessment, governance approval, and applicable legal/compliance review.

---

## 23. Final Response Requirements

When implementation is complete, provide a concise engineering summary in this order:

### A. Implementation Summary
What was implemented and the important design decisions.

### B. Architecture
Main components, boundaries, and data flow.

### C. Repository Changes
Important added or modified files.

### D. Machine Learning Results
Actual model comparison metrics, calibration results, and explainability findings, clearly identifying synthetic or real-data evaluations.

### E. Verification Results
Commands executed, tests passed/failed, and checks not performed.

### F. Running the Project
Exact installation, training, testing, inference, and container commands.

### G. Risks and Limitations
Remaining security, model-risk, performance, fairness, and production-deployment concerns.

### H. Recommended Next Steps
Only high-value follow-up work, prioritized by actual risk and business value.

## Final Instruction

**Build the complete AI Loan Risk Prediction System now.**

Prioritize correct data handling, leakage-free evaluation, calibrated predictions, explainability, security, maintainability, and observable runtime behavior over unnecessary architectural complexity.

Write production-quality code, implement meaningful failure handling, verify what you build, and make all remaining assumptions and limitations explicit.

Do not replace working implementation with pseudocode, generic recommendations, or unsupported claims of production readiness.
