Act as a **Principal Machine Learning Engineer, AI Solution Architect, Senior Python Engineer, MLOps Engineer, and Technical Lead** with extensive experience building production-grade machine learning applications.

Your task is to **design, implement, test, document, and containerize a complete end-to-end AI-powered House Price Prediction System** using classical supervised machine learning.

The implementation must demonstrate the mathematical foundations and practical applications of:

- Simple and Multiple Linear Regression
- Ordinary Least Squares (OLS)
- Gradient Descent (implemented from scratch)
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Ridge Regression (L2 Regularization)
- Lasso Regression (L1 Regularization)
- Bias–Variance Trade-off
- Feature Engineering and Scaling
- Hyperparameter Optimization
- Model Evaluation, Selection, and Explainability

Build a genuinely functional application, not merely an architectural proposal, notebook demonstration, or collection of disconnected code snippets.

**Primary objective:** Deliver a maintainable, reproducible, secure, observable, and production-oriented ML system that demonstrates the complete journey from raw housing data to deployed price predictions.

---

## 2. Execution Instructions

Follow these rules throughout implementation:

1. Inspect the existing repository before making changes.
2. Follow all applicable repository instructions, dependencies, architecture, conventions, and existing tests.
3. If the repository is empty, scaffold the application using the architecture specified below.
4. Implement working code, configuration, tests, documentation, and deployment assets.
5. Make reasonable architectural decisions independently; ask questions only when missing information materially affects correctness or operational safety.
6. Prefer simple, maintainable solutions over unnecessary abstractions, frameworks, infrastructure, or microservices.
7. Use stable, compatible dependency versions and verify version-dependent behavior against official documentation when necessary.
8. Separate data ingestion, preprocessing, feature engineering, model training, evaluation, inference, and API concerns.
9. Do not hardcode training data, predictions, model evaluation results, credentials, or environment-specific paths.
10. Do not use mock implementations in production execution paths.
11. Do not add LLMs, deep learning, vector databases, or distributed infrastructure when classical ML adequately solves the problem.
12. Complete implementation and verification rather than stopping after planning or scaffolding.
13. Report exactly which checks were executed, their results, and anything that could not be verified.

Use pragmatic SOLID, DRY, KISS, separation of concerns, and Clean/Hexagonal Architecture principles without introducing needless layers.

---

## 3. Technology Stack

Use the following defaults unless an existing repository requires otherwise.

### Machine Learning and Data Engineering

- Python 3.12+
- NumPy
- Pandas
- scikit-learn
- SciPy where justified
- Matplotlib and Seaborn for offline analysis
- Joblib or another justified model serialization approach
- Jupyter for optional educational experiments

### Backend and Inference

- FastAPI
- Pydantic v2
- Uvicorn
- OpenAPI 3.x
- Structured JSON logging
- Configuration through environment variables
- Dependency injection where useful

### Frontend

- React
- TypeScript
- Vite
- A lightweight, accessible, responsive UI
- A minimal charting library only where beneficial

### Engineering and Operations

- Docker
- Docker Compose for local execution
- GitHub Actions for CI
- pytest
- Ruff
- mypy
- Frontend linting, type checking, and testing
- Environment-based configuration
- Reproducible dependency locking

Use PostgreSQL, Redis, MLflow, Kubernetes, or cloud-managed infrastructure only if a concrete requirement justifies them. Document how the design could integrate these later without implementing unnecessary infrastructure.

---

## 4. Functional Requirements

### FR-01: Dataset Acquisition

Implement a reproducible housing dataset ingestion pipeline.

Prefer the public Ames Housing dataset if no dataset is supplied.

Requirements:

- Support CSV-based datasets through configurable paths.
- Provide a reproducible dataset acquisition command for the selected public dataset.
- Document dataset provenance, source, version, license, features, and target units.
- Validate required columns, schema, target type, and record validity.
- Detect missing values, duplicates, inconsistent data types, and suspicious values.
- Generate data quality summaries.
- Separate raw and processed data.
- Avoid modifying original source data.
- Do not commit large datasets unnecessarily.
- Support an offline synthetic dataset strictly for development and tests, clearly identified as synthetic.

Do not assume that all housing datasets represent the same geography, currency, or target units.

### FR-02: Exploratory Data Analysis

Implement a reusable EDA workflow covering:

- Dataset dimensions and schema
- Descriptive statistics
- Missing-value distributions
- Numerical and categorical features
- Price distributions
- Correlation analysis
- Feature-target relationships
- Potentially influential outliers
- Multicollinearity
- Data leakage risks
- Distribution skewness

Produce appropriate artifacts such as a correlation heatmap, price distribution, feature relationship charts, and a concise data quality report.

Explain which findings influence preprocessing or model design.

EDA must not leak test-set information into model selection or fitted preprocessing.

### FR-03: Data Preprocessing and Feature Engineering

Implement leakage-safe preprocessing using fitted scikit-learn pipelines and column transformers.

Include:

- Numerical missing-value imputation
- Categorical missing-value handling
- Categorical encoding with unseen-category support
- StandardScaler for scale-sensitive models
- Configurable outlier handling, when justified
- Relevant feature engineering
- Data type and domain validation
- Training-only fitting of preprocessing operations
- Consistent inference-time transformations

Avoid blanket removal of records or arbitrary outlier filtering without evidence.

The identical fitted preprocessing pipeline must be reused during inference.

### FR-04: Dataset Splitting

Create reproducible training, validation, and test partitions, using an appropriate default such as 70/15/15.

Requirements:

- Fixed configurable random seed
- No train/test contamination
- No duplicate or related-property leakage across splits
- Cross-validation on development/training data
- Dedicated holdout evaluation
- Grouped or temporal splitting where dataset characteristics require it

Never optimize hyperparameters using the final test set.

Document the split strategy and its limitations.

---

## 5. Machine Learning Algorithms

### ML-01: Simple Linear Regression

Implement a basic linear regression model illustrating:

- Independent and dependent variables
- Slope and intercept
- Predictions
- Residuals
- Least-squares fitting
- MSE minimization

Explain the model mathematically:

ŷ = β₀ + β₁x

Provide a simple visualization of fitted regression lines and residual errors.

### ML-02: Multiple Linear Regression

Build a multivariate regression model using scikit-learn's LinearRegression.

Requirements:

- Multiple numerical and categorical input features
- Fitted preprocessing pipeline
- Coefficient and intercept inspection
- Evaluation against a baseline predictor
- Interpretation of coefficient limitations under correlated and encoded features

Avoid manually inverting potentially singular matrices.

### ML-03: Gradient Descent From Scratch

Implement a custom linear regression estimator using NumPy, without relying on scikit-learn's optimization implementation.

Support:

- Vectorized forward predictions
- MSE loss calculation
- Analytical gradients
- Batch Gradient Descent
- Configurable learning rate
- Configurable maximum iterations
- Convergence tolerance
- Early stopping based on a documented criterion
- Intercept handling
- Deterministic initialization
- Numerical stability checks
- Training loss history

Use:

ŷ = Xw + b

MSE = (1/n) Σ(yᵢ − ŷᵢ)²

∂MSE/∂w = (2/n)Xᵀ(Xw + b − y)

∂MSE/∂b = (2/n)Σ(ŷᵢ − yᵢ)

Update the parameters iteratively using the gradients and configured learning rate.

Requirements:

- Do not penalize the intercept.
- Normalize numerical features appropriately.
- Avoid target leakage.
- Reject non-finite inputs and unstable optimization states.
- Test the analytical gradients against numerical finite differences.
- Compare converged results against scikit-learn LinearRegression within documented numerical tolerances.
- Plot convergence and loss versus training iteration.
- Demonstrate the effect of learning-rate changes.

Structure the custom estimator so it can be evaluated consistently alongside other models.

### ML-04: Ridge Regression

Implement Ridge Regression using L2 regularization.

Explain and demonstrate:

- Regularization strength alpha
- Coefficient shrinkage
- Impact of feature scaling
- Multicollinearity handling
- Bias–variance trade-off
- Underfitting caused by excessive regularization

Use cross-validation to tune alpha.

Document the exact objective and scaling conventions used by the selected library.

### ML-05: Lasso Regression

Implement Lasso Regression using L1 regularization.

Demonstrate:

- Sparse coefficients
- Feature selection behavior
- Regularization-strength tuning
- Sensitivity to correlated predictors
- Convergence behavior
- Comparison against Ridge and OLS

Ensure that scaling and convergence parameters are appropriate.

### ML-06: Baseline Model

Include a simple, valid baseline such as median-price prediction using a dummy regressor.

All trained models must be evaluated relative to the baseline.

---

## 6. Model Training, Evaluation, and Selection

Build an executable training workflow.

Example commands:

```bash
python -m house_price.train
python -m house_price.evaluate
```

These commands are illustrative; choose consistent package names and verify the final commands work.

### Evaluation metrics

Calculate and compare:

- MSE
- MAE
- RMSE
- R²
- Cross-validation results
- Training versus validation error
- Training duration
- Inference latency measurements, where practical

Explain:

- Why MSE penalizes large prediction errors more strongly.
- Why MAE is useful for interpreting errors in price units.
- When RMSE may better communicate the impact of large errors.
- How regularization influences bias and variance.
- Why training performance alone does not establish generalization.

### Hyperparameter optimization

Use GridSearchCV or RandomizedSearchCV where appropriate.

Requirements:

- Fit preprocessing within cross-validation folds.
- Use suitable, deterministic CV splitting.
- Define an explicit primary model-selection metric, preferably validation MAE.
- Use secondary metrics to resolve meaningful trade-offs.
- Compare baseline, OLS, custom Gradient Descent, Ridge, and Lasso.
- Select the final candidate using development data only.
- Refit the selected configuration on appropriate development data.
- Evaluate once on the untouched test partition.
- Do not repeatedly tune based on test results.

Do not assume Ridge or Lasso will outperform ordinary Linear Regression. Select using measured evidence.

### Model diagnostic artifacts

Generate:

- Model comparison table
- Actual versus predicted price plot
- Residual distribution
- Residuals versus predictions
- Gradient Descent loss curve
- Regularization strength versus CV performance
- Selected-model coefficient summary
- Error breakdown across meaningful property-price segments where sample sizes permit

Clearly distinguish diagnostics from production inference outputs.

---

## 7. Model Serialization and Lifecycle

Package the complete fitted inference pipeline into a versioned artifact.

Persist metadata including:

- Model version
- Algorithm and hyperparameters
- Feature schema
- Target unit/currency
- Dataset version or fingerprint
- Training timestamp
- Evaluation metrics
- Preprocessing version
- Application and relevant library versions
- Artifact checksum

Requirements:

- Use only trusted model artifacts.
- Never deserialize arbitrary user-provided artifacts.
- Validate artifact integrity and compatibility at startup.
- Fail readiness checks when no valid model is available.
- Load the selected model once per application worker where appropriate.
- Avoid training models during normal prediction requests.
- Ensure a reproducible artifact build process.
- Define a documented rollback mechanism for replacing an incompatible or degraded model.

A full model registry is optional; a simple versioned artifact store and manifest are sufficient initially.

---

## 8. Backend REST API

Implement a versioned FastAPI inference service.

### Required endpoints

**GET /api/v1/health/live**

Returns basic service liveness without requiring a successful model prediction.

**GET /api/v1/health/ready**

Returns readiness based on model loading and required dependencies.

**GET /api/v1/models/current**

Returns safe public metadata such as model version, algorithm, feature schema information, and measurement units.

**POST /api/v1/predictions**

Accepts a validated housing-property request and returns a predicted price.

Example response shape (illustrative, not a fabricated prediction):

```json
{
  "prediction": {
    "price": 0.0,
    "currency": "DATASET_CURRENCY",
    "unit": "currency_per_property"
  },
  "model": {
    "name": "selected-regression-model",
    "version": "1.0.0"
  },
  "request_id": "generated-request-id"
}
```

The actual response must contain the model's computed result. Currency and unit values must come from verified dataset metadata and configuration.

### API engineering requirements

- Typed Pydantic request and response models
- Strict data validation
- Domain-specific constraints for property features
- Rejection of non-finite numerical values
- Reasonable payload limits
- Explicit handling of unsupported features
- No silent feature defaulting when inputs are required
- Stable error response structure
- Correct HTTP status codes
- Request correlation IDs
- Structured logging
- No sensitive request payloads in logs
- Centralized exception handling
- OpenAPI documentation with examples
- Configurable CORS allowlist
- Authentication at a documented application or gateway boundary
- Configurable timeouts and resource limits
- No external dependencies in the synchronous prediction path unless necessary

Avoid exposing debugging tracebacks, internal model paths, or sensitive configuration to API consumers.

Prediction requests are read-only and should not create unnecessary database state.

---

## 9. Frontend Web Application

Implement a minimal but polished House Price Prediction dashboard.

### Features

**Property Details Form**

Collect only the supported property features.

Use the actual trained model's feature contract rather than inventing unrelated fields.

Provide:

- Appropriate field types
- Client-side validation
- Clear descriptions and units
- Required/optional field indicators
- Accessible form labels
- Submission progress indicators
- Field-specific validation errors

**Prediction Results**

Display:

- Predicted house price
- Correct currency and units
- Model version
- Short explanation of what the prediction represents
- A clear statement that predictions are estimates, not guaranteed appraisals

Do not display fabricated confidence scores or prediction intervals.

If uncertainty intervals are implemented, derive and validate them using a defensible statistical method and document their limitations.

**Model Insights**

Display selected, verified model-evaluation information, with useful visualizations where appropriate.

Do not present global coefficients as universally valid causal explanations or individualized feature contributions.

**UX Requirements**

- Responsive layout
- Accessible keyboard navigation
- Graceful API error handling
- Loading and empty states
- No hardcoded model predictions
- Environment-configurable API base URL
- No secret tokens embedded in client bundles

---

## 10. Architecture and Project Structure

Design a modular repository approximately as follows, adapting it to existing conventions:

```text
house-price-prediction/
├── backend/
│   ├── src/house_price/
│   │   ├── api/
│   │   ├── core/
│   │   ├── data/
│   │   ├── features/
│   │   ├── models/
│   │   ├── training/
│   │   ├── evaluation/
│   │   └── inference/
│   ├── tests/
│   │   ├── unit/
│   │   ├── integration/
│   │   └── contracts/
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── features/
│   │   ├── api/
│   │   └── types/
│   └── package.json
├── datasets/
├── artifacts/
├── notebooks/
├── reports/
├── scripts/
├── docs/
├── .github/workflows/
├── compose.yaml
├── .env.example
└── README.md
```

This structure is guidance, not a mandate to create unnecessary empty packages.

### Architectural boundaries

**Data layer:** Dataset acquisition, validation, and transformations.

**ML layer:** Algorithms, training, evaluation, and model selection.

**Inference layer:** Model loading, feature contract enforcement, and predictions.

**API layer:** Transport, validation, authentication integration, and error mapping.

**Frontend layer:** Presentation, user input, and API communication.

Keep training workflows independent of the runtime HTTP service.

Do not introduce distributed services or messaging systems unless justified by an actual requirement.

---

## 11. Testing and Verification

Implement meaningful automated tests.

### Unit Tests

Cover:

- Data schema validation
- Missing-value handling
- Feature preprocessing
- Numerical feature scaling
- Unseen categorical values
- Gradient calculations
- Gradient Descent convergence
- Learning-rate edge cases
- MSE and MAE correctness
- Ridge and Lasso pipeline configuration
- Model metadata validation
- Prediction request validation

### ML Tests

Verify:

- Reproducible results for fixed inputs and seeds
- Leakage-safe preprocessing
- Stable predictions for known fixtures
- Convergence against a trusted linear regression reference
- Correct training/test separation
- Correct target-unit handling
- No NaN or infinite predictions
- Valid model artifact loading
- Baseline comparison reporting

### Integration Tests

Cover:

- End-to-end training on a small fixture
- Artifact creation and loading
- Successful inference requests
- Invalid feature values
- Missing required fields
- Incompatible model artifacts
- Model-loading failures
- Health/readiness behavior
- Stable API response contracts

### Frontend Tests

Cover:

- Form validation
- Valid prediction submission
- Successful result display
- Network failures
- Server-side validation errors
- Relevant accessibility behavior

### End-to-End Smoke Test

Verify the full flow:

Dataset → Validation → Preprocessing → Training → Evaluation → Model Selection → Serialization → API Startup → Frontend Prediction.

Tests must be deterministic, self-contained where possible, and executable without production credentials.

If any acceptance test requires downloading external data, provide a separate offline fixture-based path.

---

## 12. Security and Reliability

Apply relevant production security controls.

### Security

- Validate all external inputs.
- Do not place secrets in source code.
- Use environment configuration for credentials.
- Do not log sensitive property-owner information.
- Use trusted and integrity-checked artifacts.
- Pin or lock compatible dependencies.
- Run containers as non-root.
- Limit filesystem permissions.
- Avoid unnecessary container capabilities.
- Exclude secret files, raw private datasets, and large artifacts from source control.
- Document authentication, TLS termination, and rate limiting at appropriate boundaries.
- Use dependency vulnerability checks in CI where feasible.

### Reliability

- Fail clearly when the trained model is missing or corrupt.
- Handle unexpected input categories.
- Guard against non-finite outputs.
- Do not mutate shared inference model state during requests.
- Keep request behavior deterministic for the same model and inputs.
- Avoid unnecessary retry loops for deterministic prediction failures.
- Expose meaningful startup/readiness failures.
- Document operational limits and recovery steps.

---

## 13. Observability and Operations

Include:

- Structured application logs
- Request identifiers
- Model version in relevant inference logs or metrics
- Prediction request count
- Prediction failure count
- Request latency metrics
- Model loading failures
- Liveness and readiness health checks

Use a lightweight metrics approach if appropriate.

Explain how production monitoring could detect:

- Input schema changes
- Data drift
- Feature distribution changes
- Prediction distribution shifts
- Out-of-range property inputs
- Serving latency regressions
- Model quality deterioration once verified ground-truth sale prices become available

Do not claim model accuracy drift can be measured without actual outcome labels.

Provide a documented retraining and rollback workflow. Automated retraining orchestration is optional unless justified.

---

## 14. Docker and Deployment

Provide production-oriented container definitions.

Requirements:

- Reproducible builds
- Multi-stage builds where useful
- Small runtime images
- Non-root execution
- Health checks
- Environment-driven configuration
- Minimal runtime dependencies
- Proper startup and shutdown behavior
- Separate build-time training from inference serving
- Documented model artifact provisioning

Create Docker Compose configuration for local execution.

Provide commands for:

1. Installing dependencies
2. Acquiring the dataset
3. Running EDA
4. Training models
5. Comparing models
6. Creating the selected model artifact
7. Starting the API
8. Starting the frontend
9. Running automated tests
10. Running the complete application with Docker Compose

Validate all documented commands against the actual repository.

---

## 15. CI/CD

Implement GitHub Actions workflows covering:

- Python linting and formatting checks
- Python type checking
- Backend unit and integration tests
- ML fixture tests
- Frontend linting and type checking
- Frontend automated tests
- Dependency vulnerability scanning where configured
- Container build verification

Separate model-training pipelines from application build pipelines when appropriate.

Do not require long-running production-sized model training for every source change.

Use fast, deterministic fixture-based checks in routine CI.

Document how release builds associate a particular application version with an immutable trained model artifact.

---

## 16. Documentation and Learning Materials

Create developer-friendly documentation covering:

### README.md

- Project overview
- System architecture
- Technology stack
- Prerequisites
- Quick start
- Dataset setup
- Local training
- Running the API and frontend
- Docker execution
- Testing commands
- Example prediction request and response
- Troubleshooting

### docs/architecture.md

Include a Mermaid architecture diagram showing:

User → Frontend → Prediction API → Validation → Preprocessing Pipeline → Trained Regression Model → Prediction Response.

Also describe the separate offline training workflow.

### docs/ml-concepts.md

Explain with formulas and actual project examples:

- Linear Regression
- OLS
- Gradient Descent
- Cost/Loss Functions
- MSE versus MAE
- Ridge Regression
- Lasso Regression
- Regularization
- Overfitting versus Underfitting
- Bias–Variance Trade-off
- Model Selection
- Cross-Validation
- Feature Scaling

Explain why and when each algorithm is useful, not just how to call its library API.

### docs/model-evaluation.md

Describe:

- Dataset and splitting methodology
- Experimental setup
- Model configurations
- Evaluation results
- Selected model
- Model limitations
- Relevant error distributions
- Applicability boundaries

### docs/operations.md

Document:

- Deployment configuration
- Health checks
- Model versioning
- Monitoring
- Retraining
- Rollback
- Incident troubleshooting

---

## 17. Performance and Quality Requirements

The solution should:

- Support stateless, concurrent inference safely.
- Avoid retraining or loading artifacts on every HTTP request.
- Minimize inference-time transformations unrelated to the selected pipeline.
- Avoid unnecessary external network requests during inference.
- Produce numerically valid predictions for supported inputs.
- Use clearly documented input and output units.
- Provide measured inference latency rather than invented performance claims.
- Avoid arbitrary accuracy thresholds disconnected from dataset characteristics.
- Measure and report MAE, MSE, RMSE, and R².
- Include meaningful tests for correctness and reliability.
- Provide reproducible commands for training and evaluation.

Any model-quality acceptance threshold must be explicitly justified or configurable.

---

## 18. Implementation Strategy

Execute the task in the following order.

**Phase 1 — Repository and Architecture**

Inspect the repository, identify applicable constraints, establish dependency versions, and create a concise implementation plan.

**Phase 2 — Data Engineering**

Implement ingestion, validation, data quality analysis, preprocessing, and reproducible splitting.

**Phase 3 — Core ML Algorithms**

Implement OLS, custom Gradient Descent, Ridge, Lasso, and baseline regression.

**Phase 4 — Training and Evaluation**

Implement pipelines, cross-validation, hyperparameter tuning, diagnostics, model selection, and artifact versioning.

**Phase 5 — Backend**

Implement inference services, schema contracts, prediction endpoints, health checks, logging, and exception handling.

**Phase 6 — Frontend**

Implement property input forms, API integration, prediction visualization, and error handling.

**Phase 7 — Testing**

Add unit, integration, contract, frontend, and complete workflow verification.

**Phase 8 — Deployment and Documentation**

Implement Docker, CI, configuration samples, operational guidance, and developer documentation.

Complete the relevant code and verification for each phase before considering it finished. Keep the solution functional throughout implementation.

Do not stop after generating a folder structure or explaining what should be built.

---

## 19. Definition of Done

The task is complete only when:

- [ ] The repository has a functional, documented project structure.
- [ ] The dataset can be acquired or supplied reproducibly.
- [ ] Dataset validation and preprocessing work.
- [ ] Data leakage protections are implemented.
- [ ] Simple and Multiple Linear Regression are demonstrated.
- [ ] Gradient Descent is implemented from scratch and tested.
- [ ] MSE and MAE are correctly implemented or calculated and verified.
- [ ] Ridge and Lasso are trained and tuned.
- [ ] A baseline regression model is included.
- [ ] All candidate models are evaluated consistently.
- [ ] A best model is selected using development data.
- [ ] The final untouched test evaluation is reported.
- [ ] The complete fitted pipeline can be serialized and loaded.
- [ ] The REST API serves real model predictions.
- [ ] The frontend submits property details and displays predictions.
- [ ] Validation and failure paths are tested.
- [ ] Docker-based local execution is functional.
- [ ] CI workflows are implemented and, where possible, validated.
- [ ] Architecture and ML concepts are documented.
- [ ] Relevant code quality and test checks have actually been run.
- [ ] Remaining limitations and operational risks are disclosed.

A generated file or written plan does not count as a successfully implemented feature without supporting verification.

---

## 20. Final Response Requirements

After implementing the solution, provide a concise engineering delivery report containing:

**1. Architecture Summary**

Explain the implemented components and primary design decisions.

**2. Implementation Summary**

List the actual features and modules implemented.

**3. ML Evaluation**

Provide a table of real, measured model results:

| Model | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| Baseline | Measured | Measured | Measured | Measured |
| Linear Regression | Measured | Measured | Measured | Measured |
| Custom Gradient Descent | Measured | Measured | Measured | Measured |
| Ridge Regression | Measured | Measured | Measured | Measured |
| Lasso Regression | Measured | Measured | Measured | Measured |

Identify the selected model and justify the selection.

Never fabricate evaluation values. If training or evaluation could not be executed, clearly identify the missing results.

**4. API Examples**

Show at least one actual representative valid request shape and response contract.

**5. Running Instructions**

Provide verified commands to run training, inference, frontend, tests, and Docker.

**6. Verification**

Distinguish:

- Checks executed and passed
- Checks executed and failed
- Checks not executed and why

**7. Remaining Risks**

Report relevant blockers, limitations, model assumptions, security issues, and deployment dependencies.

---

## Final Instruction

**Implement the complete AI House Price Prediction System end to end.**

Prioritize correct ML mathematics, data integrity, reproducibility, maintainable engineering, observable inference, secure defaults, and verified execution.

Make architecture and implementation decisions independently within the repository's constraints.

Deliver working code, tests, trained model artifacts when training is possible, usable API endpoints, a functional frontend, Docker configuration, and documentation.

Do not substitute pseudocode, high-level explanations, synthetic evaluation claims, or incomplete scaffolding for the requested implementation.

The final result should be suitable as a well-engineered machine learning reference application and a credible foundation for further production hardening.