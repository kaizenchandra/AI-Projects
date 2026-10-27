
Act as a **Principal Software Architect, Senior Full-Stack Engineer, Data Scientist, ML Engineer, Data Engineer, Security Engineer, and SRE**.

Your task is to **design, implement, test, and document a complete, production-grade AI-Powered Exploratory Data Analysis (EDA) Dashboard**.

The application must enable users to upload datasets, perform automated data profiling, calculate descriptive statistics, explore distributions and relationships, detect missing values and outliers, visualize meaningful patterns, and generate accurate, explainable AI-powered insights.

Build an actual functioning application, not merely a proof of concept, UI mockup, architecture proposal, or collection of disconnected examples.

Follow these engineering principles:

- Clean Architecture and separation of concerns.
- SOLID, DRY, KISS, and pragmatic domain-driven design.
- Secure-by-default implementation.
- Statistical correctness and reproducibility.
- Maintainable, modular, testable code.
- Performance and scalability appropriate to workload.
- Accessibility and responsive design.
- Observability and operational reliability.
- Minimal justified dependencies and abstractions.
- Incremental implementation with verification.

**Primary directive:** Deliver working, integrated functionality from dataset ingestion through analytics, visualizations, AI insights, and report generation.

---

## 2. Repository Discovery and Technology Decisions

Before implementation:

1. Inspect the repository structure, existing source code, configuration, and documentation.
2. Identify the application architecture, package managers, framework versions, testing conventions, and deployment approach.
3. Read and follow applicable repository instructions.
4. Identify reusable modules, UI components, utilities, and services.
5. Preserve existing functionality, dependencies, configuration patterns, and unrelated changes.
6. Identify missing components necessary for complete implementation.
7. Produce a concise implementation plan and immediately execute it.

### Technology preferences

**Existing repository:** Retain its established technology stack and architectural conventions unless a change is technically necessary.

**Greenfield implementation:** Prefer a maintainable architecture using:

| Layer | Preferred technology |
|---|---|
| Frontend | React, TypeScript |
| UI | Tailwind CSS and accessible component primitives |
| Visualization | Apache ECharts or Plotly |
| API | Python FastAPI, or Java 21 / Spring Boot where the repository favors Java |
| Analytical computation | Polars, NumPy, SciPy, and scikit-learn where justified |
| Database | PostgreSQL |
| File storage | Local development storage; S3-compatible production storage |
| Background processing | A bounded job queue and worker when asynchronous workloads require it |
| Cache | Redis only when profiling or operational evidence justifies it |
| AI integration | Provider-agnostic interface supporting configurable LLM providers |
| Testing | Pytest or JUnit, Vitest, React Testing Library, Playwright |
| Deployment | Docker and Docker Compose; Kubernetes manifests if relevant |

Prefer a **modular monolith** for a greenfield implementation unless independently scalable services are warranted by concrete requirements. Do not introduce Kafka, microservices, a separate API gateway, or other infrastructure without justification.

Use compatible, maintained dependency versions. Verify uncertain version-specific APIs against official documentation.

---

## 3. Product Objectives

Build a self-service analytical platform that transforms raw datasets into understandable, actionable, evidence-backed insights.

The primary user workflow must support:

**Upload Dataset → Validate → Infer Schema → Profile Data → Calculate Statistics → Explore Distributions → Analyze Correlations → Detect Outliers → Generate AI Insights → Export Report**

### Functional outcomes

Users must be able to:

1. Import supported datasets.
2. Preview and understand the data schema.
3. Inspect data quality and completeness.
4. Examine individual column statistics.
5. Explore numerical and categorical distributions.
6. Investigate relationships between variables.
7. Detect and investigate potential outliers.
8. Filter datasets and regenerate relevant analysis.
9. Receive explainable AI-assisted findings.
10. Export analysis results and visualizations.

All dashboard results must come from actual processed data. Never substitute hardcoded metrics, fabricated findings, or mock visualizations in production flows.

---

## 4. Dataset Ingestion and Management

### 4.1 Supported formats

Implement ingestion for:

- CSV
- XLSX
- Parquet

Design extensible format handlers without overengineering.

Support configurable file size, row count, and column count limits.

### 4.2 Upload workflow

Implement:

- Drag-and-drop upload.
- File selection.
- Upload progress and cancellation where supported.
- Format and size validation.
- Clear error reporting.
- Dataset preview.
- Sheet selection for multi-sheet Excel workbooks.
- Configurable CSV delimiter, encoding, and header handling.
- Dataset metadata and unique identifiers.
- Upload history and dataset deletion.
- Dataset processing status.

### 4.3 Data validation

Handle:

- Empty files.
- Malformed CSV rows.
- Invalid encodings.
- Mixed column types.
- Missing headers.
- Duplicate column names.
- Null-only columns.
- Invalid date values.
- Unsupported formats.
- Corrupt workbooks.
- Excessively large inputs.
- Nonfinite numerical values.
- Duplicate records.

Use content validation rather than trusting filename extensions.

Prevent path traversal, spreadsheet-related injection risks, decompression bombs where applicable, and unauthorized file access.

### 4.4 Schema inference

Detect and expose:

- Integer and floating-point types.
- Boolean values.
- Categorical and text columns.
- Date and timestamp values.
- High-cardinality identifiers.
- Numeric-looking strings.
- Potential identifiers and sensitive fields.

Separate **physical data type**, **inferred semantic type**, and **user-selected override**.

Preserve original values unless the user explicitly approves a transformation.

---

## 5. Automated Data Profiling Engine

Implement a reusable profiling engine with deterministic outputs and structured result models.

### 5.1 Dataset-level profiling

Calculate:

- Total records.
- Total columns.
- Column type breakdown.
- Missing cell percentage.
- Duplicate row count and percentage.
- Memory usage estimates.
- Dataset completeness.
- Constant columns.
- High-cardinality columns.
- Invalid or inconsistent values.
- Potential identifiers.
- Basic data-quality warnings.

### 5.2 Column-level profiling

For every column, compute applicable metrics:

| Category | Metrics |
|---|---|
| General | Name, type, count, non-null count |
| Missingness | Null count, percentage |
| Uniqueness | Distinct count, uniqueness ratio |
| Numerical | Min, max, mean, median |
| Dispersion | Standard deviation, variance, range |
| Quantiles | Q1, Q3, percentiles, IQR |
| Shape | Skewness, kurtosis |
| Categorical | Mode, frequency, top values |
| Text | Length statistics, blank counts |
| Temporal | Earliest, latest, range, temporal frequency |
| Quality | Invalid values, type inconsistencies |
| Outliers | Count, percentage, applicable thresholds |

Use mathematically appropriate calculations for each semantic type.

Distinguish unavailable metrics from zero-valued metrics.

### 5.3 Profiling correctness

Account for:

- All-null columns.
- Single-value columns.
- Zero variance.
- Extremely skewed data.
- Small samples.
- Boolean values.
- Large integers.
- Floating-point precision.
- Infinity and NaN.
- Timezone-aware dates.
- Approximate versus exact distinct counts.

Return structured profiling results that can be consumed by the frontend, AI layer, and reporting system.

---

## 6. Descriptive Statistical Analysis

Provide comprehensive descriptive statistics for numerical, categorical, and temporal data.

### 6.1 Numerical statistics

Implement:

- Arithmetic mean.
- Median.
- Mode when meaningful.
- Minimum and maximum.
- Range.
- Sample and population variance, clearly distinguished.
- Sample and population standard deviation.
- Quartiles and percentiles.
- Interquartile range.
- Median absolute deviation.
- Coefficient of variation where valid.
- Skewness.
- Kurtosis.
- Missing and nonfinite value statistics.

Use established numerical libraries rather than writing unnecessary custom statistical implementations.

### 6.2 Categorical statistics

Implement:

- Category count.
- Frequency distribution.
- Relative frequency.
- Most frequent category.
- Category imbalance.
- High-cardinality detection.
- Rare-category identification.

### 6.3 Temporal statistics

Where applicable, calculate:

- Date range.
- Temporal coverage.
- Records by time period.
- Missing time intervals where a valid cadence exists.
- Time-based trends.
- Seasonal patterns as exploratory observations, not unsupported forecasts.

### 6.4 Statistical integrity

Specify and test:

- Null exclusion rules.
- Sample versus population definitions.
- Quantile interpolation conventions.
- Minimum sample requirements.
- Type applicability.
- Approximation and sampling methods.

Do not show misleading calculations for unsupported data types or insufficient sample sizes.

---

## 7. Distribution Analysis and Visualization

Create an interactive visualization layer using real computed analytical data.

### Required visualizations

| Visualization | Purpose |
|---|---|
| Histogram | Numerical frequency distribution |
| KDE plot | Smoothed numerical density |
| Box plot | Quartiles and potential outliers |
| Violin plot | Distribution shape where meaningful |
| Bar chart | Category frequency |
| Donut/Pie chart | Small-cardinality proportions where appropriate |
| Scatter plot | Numerical relationships |
| Correlation heatmap | Pairwise correlation |
| Time-series line chart | Temporal trends |
| Missingness chart | Missing-value comparison |
| Missingness matrix | Missing-value patterns |
| Outlier plot | Potential anomaly visualization |

### Visualization requirements

- Interactive tooltips.
- Zoom and pan where useful.
- Responsive resizing.
- Accessible chart labels and alternative summaries.
- Column selection.
- Visualization-specific settings.
- Data filtering.
- Date filtering.
- Category filtering.
- Exportable chart images.
- Consistent legends and units.
- Explicit sample sizes.
- Loading, empty, and error states.

### Distribution processing

Implement robust binning strategies such as Freedman–Diaconis or appropriate alternatives.

Handle constant data, extreme values, sparse distributions, and high-cardinality categories.

Apply category grouping or top-K selection where necessary and clearly label omitted or aggregated categories.

Avoid silently truncating the data represented by charts.

---

## 8. Correlation and Relationship Analysis

Build a statistically correct correlation engine.

### 8.1 Numerical correlations

Support:

- Pearson correlation.
- Spearman rank correlation.

Evaluate Kendall correlation if justified by the dataset size and analytical requirements.

### 8.2 Categorical associations

Support an appropriate categorical association measure such as Cramér's V.

Where useful, provide categorical-versus-numerical comparison visualizations.

Avoid applying Pearson correlation to arbitrary category encodings.

### 8.3 Correlation visualizations

Implement:

- Interactive correlation heatmap.
- Pairwise scatter plots.
- Selectable X and Y variables.
- Optional trend lines.
- Correlation method selection.
- Pairwise observation counts.
- Correlation ranking by absolute strength.

### 8.4 Edge cases and interpretation

Handle:

- Missing observations.
- Constant variables.
- Insufficient valid pairs.
- High-dimensional datasets.
- Nearly identical columns.
- Duplicate or redundant features.

Clearly disclose pairwise versus listwise deletion and any sampling.

**Never present correlation as proof of causation.**

If significance measures are implemented, correctly state their assumptions and avoid overinterpreting large numbers of simultaneous comparisons.

---

## 9. Outlier Detection and Investigation

Implement multiple transparent outlier detection methods.

### Required algorithms

**IQR method**

Detect candidate outliers using configurable lower and upper fences based on quartiles.

**Z-score method**

Support configurable thresholds and validate that standard deviation is meaningful.

**Modified Z-score / MAD**

Support robust detection when median absolute deviation is valid.

### Optional advanced algorithms

Add Isolation Forest only if it provides clear product value and its computational costs are controlled.

### Outlier experience

Users must be able to:

- Choose an algorithm.
- Configure relevant thresholds.
- Select analytical columns.
- View candidate outlier counts.
- Inspect affected records with appropriate authorization.
- See why a particular value was flagged.
- Compare outliers with the underlying distribution.
- Export outlier analysis.

### Safeguards

- Label outliers as potential anomalies rather than automatic errors.
- Do not delete or modify rows during analysis.
- Handle constant, small, sparse, and highly skewed distributions.
- Do not apply numerical algorithms blindly to identifiers or categorical codes.
- Explicitly distinguish univariate from multivariate detection.
- Show algorithm settings and sample coverage.

---

## 10. AI-Powered Insights Engine

Build an AI-assisted analysis layer that explains the dataset using calculated evidence.

### 10.1 Insight generation

Generate insights about:

- Data quality problems.
- High missingness.
- Duplicate records.
- Skewed distributions.
- Potentially unusual values.
- Strong associations.
- Redundant features.
- Categorical imbalance.
- Temporal irregularities.
- Potential data-preparation considerations.

### 10.2 Deterministic analytical foundation

The statistical engine must be the source of truth.

AI must consume structured, verified metrics rather than invent statistics.

Create a deterministic, rules-based insight engine that works without an LLM API key.

Optional LLM generation should add interpretation, explanations, and user-friendly summaries—not replace statistical computation.

### 10.3 Structured insight model

Every insight should contain fields equivalent to:

```json
{
  "id": "insight-id",
  "type": "data_quality",
  "severity": "warning",
  "title": "High missingness detected",
  "description": "Column X has substantial missing values.",
  "affected_columns": ["column_x"],
  "evidence": {
    "metric": "missing_percentage",
    "value": 32.4,
    "unit": "percent"
  },
  "recommendation": "Investigate the missingness mechanism before selecting a treatment.",
  "method": "deterministic_rule",
  "limitations": [],
  "analysis_run_id": "analysis-run-id"
}
```

The example illustrates a response contract, not a hardcoded production insight.

### 10.4 AI reliability requirements

- Ground insights in calculated metrics.
- Identify evidence and relevant columns.
- Report uncertainty and limitations.
- Avoid fabricated numerical findings.
- Avoid causal assertions unsupported by analysis.
- Never execute instructions embedded in uploaded data.
- Treat dataset text as untrusted input.
- Validate structured LLM outputs.
- Bound token usage and execution time.
- Handle provider timeouts, rate limits, and unavailable services.
- Provide a deterministic fallback.
- Version AI prompt templates and insight rules.

### 10.5 Data privacy

Do not send raw datasets, sensitive rows, or potential PII to external AI providers by default.

Require explicit configuration and appropriate authorization for external processing.

Minimize, aggregate, or redact data sent to LLMs.

Never expose secrets or provider credentials to the browser.

### 10.6 AI Insights Dashboard

Provide:

- Executive analysis summary.
- Key findings.
- Data-quality warnings.
- Statistical observations.
- Suggested next analytical steps.
- Evidence-linked insights.
- Insight filtering and categorization.
- Clear separation between deterministic findings and LLM-generated explanations.

---

## 11. Dashboard and UX Requirements

Build a polished analytical workspace designed for real users.

### Application sections

**Dataset Workspace**

- Dataset selection and management.
- Upload interface.
- Dataset overview.
- Data preview.
- Schema and column explorer.

**Overview Dashboard**

- Row and column counts.
- Dataset health indicators.
- Missingness summary.
- Duplicate summary.
- Data-type summary.
- Key insights.

**Data Profiling**

- Column-level profiling cards.
- Searchable and sortable profiles.
- Missing values and cardinality.
- Data-quality warnings.

**Statistics Explorer**

- Numerical statistics.
- Categorical summaries.
- Temporal summaries.
- Selected-column comparisons.

**Distribution Explorer**

- Interactive charts.
- Configurable binning.
- Column selection.
- Distribution summaries.

**Correlation Explorer**

- Correlation heatmap.
- Method selection.
- Pairwise comparisons.
- Relationship exploration.

**Outlier Explorer**

- Algorithm selection.
- Configurable thresholds.
- Candidate outlier visualization.
- Investigation table.

**AI Insights**

- Insight summary.
- Severity or importance categorization.
- Supporting statistical evidence.
- Suggested analytical follow-ups.

**Reports**

- Analysis summary.
- Configurable report sections.
- Export options.
- Analysis configuration and provenance.

### UX quality

Implement:

- Responsive desktop and tablet layouts.
- Consistent typography and spacing.
- Accessible navigation.
- Keyboard-accessible controls.
- Skeleton and loading states.
- Clear error and retry states.
- Empty dataset guidance.
- Notifications for completed processing.
- Pagination or virtualization for large tables.
- Clearly labeled statistical units and methods.

Avoid decorative charts that do not support meaningful investigation.

---

## 12. Backend Architecture and APIs

Use clear boundaries between ingestion, storage, profiling, statistical computation, visualization preparation, insights, and reporting.

Recommended logical modules:

```text
dataset/
  ingestion
  validation
  metadata
  storage

profiling/
  schema
  quality
  column_profiles

analytics/
  descriptive_statistics
  distributions
  correlations
  outliers
  temporal_analysis

insights/
  rules
  evidence
  llm_adapter

visualization/
  chart_data

reports/
  generation
  export

jobs/
  execution
  status
  cancellation

security/
  authentication
  authorization

shared/
  configuration
  errors
  observability
```

Adapt these boundaries to the existing repository rather than mechanically creating every directory or service.

### API capabilities

Provide versioned or convention-compatible APIs for:

- Dataset upload.
- Dataset listing and metadata.
- Schema and preview retrieval.
- Profiling results.
- Descriptive statistics.
- Distribution chart data.
- Correlation analysis.
- Outlier analysis.
- AI insights.
- Analysis job status and cancellation.
- Report generation and retrieval.
- Dataset deletion.

Use typed request and response contracts with clear validation errors.

### Data consistency

Introduce a dataset fingerprint and an analysis-run identity.

An analysis run should preserve:

- Dataset identity or version.
- Analytical parameters.
- Filter configuration.
- Algorithm version.
- Sample details.
- Execution timestamp.
- Completion status.

Analysis artifacts must not silently mix results from different dataset versions or filter configurations.

---

## 13. Performance, Scalability, and Reliability

### Large dataset handling

- Prefer columnar processing where appropriate.
- Avoid loading entire large datasets into browser memory.
- Use chunked, lazy, streaming, or bounded execution where supported.
- Implement configurable memory and execution limits.
- Prevent uncontrolled high-dimensional pairwise computation.
- Use sensible pagination and chart aggregation.

### Sampling

For expensive workloads, implement configurable sampling only when justified.

Disclose:

- Sampling method.
- Sample size.
- Source dataset size.
- Approximation status.
- Relevant limitations.

Never represent sampled results as exact full-dataset findings.

### Asynchronous execution

Use asynchronous jobs when profiling or exporting may exceed normal HTTP request timeouts.

Support:

- Queued, running, completed, failed, and cancelled states.
- Idempotent job submission where appropriate.
- Bounded retries for transient failures.
- Cancellation and timeout handling.
- Controlled resource concurrency.
- Recovery from worker failures.
- Prevention of duplicate or inconsistent results.

### Caching

Cache expensive analysis results when beneficial.

Cache identity must account for dataset version, filters, algorithm settings, and relevant computation versions.

Ensure cache invalidation prevents stale or cross-user results.

### Performance validation

Define configurable workload limits and benchmark representative datasets.

Record actual latency, peak memory use, and processing time.

Do not claim a performance target has been achieved without measurement.

---

## 14. Security and Privacy

Implement appropriate protections for a data-upload and analytics system.

Requirements:

- Authentication and authorization consistent with the deployment model.
- Dataset ownership and tenant isolation where applicable.
- Server-side input validation.
- Secure file storage.
- Configurable upload restrictions.
- Secure secrets management.
- HTTPS-compatible deployment.
- Restrictive CORS settings.
- Safe error responses.
- Prevention of sensitive data leakage in logs.
- Appropriate API abuse controls.
- Report and download authorization.
- Dataset deletion and associated artifact cleanup.
- Configurable data-retention policies.
- Dependency vulnerability checks.

Apply reasonable protection against denial-of-service through oversized files, excessive columns, expensive correlation matrices, and unbounded analysis requests.

For a local development environment, document simplified authentication or infrastructure behavior. Do not silently carry development shortcuts into production configuration.

---

## 15. Observability and Operations

Implement operational diagnostics appropriate to the chosen architecture.

### Structured logging

Log:

- Request and correlation identifiers.
- Analysis-run identifiers.
- Dataset-processing transitions.
- Execution duration.
- Failure category.
- Worker retry attempts.
- Relevant operational events.

Do not log full dataset rows, credentials, or sensitive payloads.

### Metrics

Where suitable, track:

- Upload success and failure rates.
- Dataset processing duration.
- Analysis latency.
- Queue depth.
- Job failures.
- Memory usage.
- LLM provider latency and failure rates.
- Cache hit rates.

### Health and configuration

Provide:

- Health and readiness endpoints where applicable.
- Environment-based configuration.
- Configuration validation.
- Graceful shutdown.
- Database migration support.
- Actionable operational errors.

---

## 16. Testing Strategy

Build meaningful automated coverage rather than superficial happy-path tests.

### Backend tests

Verify:

- File format validation.
- Data type inference.
- Missingness calculations.
- Duplicate identification.
- Numerical statistical correctness.
- Quantile and dispersion calculations.
- Distribution aggregation.
- Pearson and Spearman correlations.
- Categorical association calculations.
- Outlier algorithms.
- Empty and constant columns.
- NaN and infinity handling.
- Analysis parameter validation.
- Job state transitions.
- Cache correctness where implemented.
- Authorization boundaries.
- Insight-to-evidence consistency.

Compare important numerical calculations against trusted library results and carefully chosen analytical fixtures.

### Frontend tests

Verify:

- Upload flow.
- Dataset selection.
- Dashboard rendering.
- Column selection.
- Filtering.
- Chart interactions.
- Statistical value formatting.
- Loading and failure states.
- Error recovery.
- Accessibility-critical interactions.

### Integration tests

Test:

- Upload to schema inference.
- Upload to profiling.
- Profiling to chart rendering.
- Correlation analysis.
- Outlier detection.
- AI insight generation.
- Report generation.
- Job lifecycle.

### End-to-end scenarios

At minimum:

1. Upload a valid dataset and complete analysis.
2. Explore column profiles and distributions.
3. Change filters and verify recalculated results.
4. Generate and inspect evidence-backed insights.
5. Export and retrieve a report.
6. Reject an invalid dataset with a useful error.
7. Recover from a processing failure.
8. Enforce dataset access boundaries.

### Test fixtures

Use small deterministic fixtures covering:

- Normal distributions.
- Skewed distributions.
- Correlated and uncorrelated variables.
- Known outliers.
- Missing data.
- Duplicate rows.
- High-cardinality categories.
- Constant columns.
- Mixed data types.
- Temporal data.

Avoid depending on external LLM services for deterministic automated tests.

---

## 17. Reporting and Export

Support structured export of analytical results.

Required capabilities:

- CSV export for relevant tabular outputs.
- JSON export of structured analysis results.
- Downloadable visualizations.
- Human-readable analytical summary report.

Prefer PDF as the human-readable report format if supported naturally by the application's architecture.

Report sections should include:

- Dataset overview.
- Data-quality summary.
- Column profiling.
- Statistical summaries.
- Distribution analysis.
- Correlations.
- Outlier findings.
- AI insights.
- Analysis methods, parameters, and limitations.

Ensure reports use consistent values from the same analysis run.

Apply export authorization, safe filename generation, and spreadsheet formula-injection defenses where relevant.

---

## 18. Documentation and Developer Experience

Provide:

- Project README.
- Architecture overview.
- Local setup instructions.
- Environment variable reference.
- Data storage configuration.
- Database migration instructions.
- Development commands.
- Test commands.
- API documentation.
- Supported dataset formats and limits.
- Statistical methodology notes.
- AI configuration and privacy behavior.
- Deployment instructions.
- Known limitations.

A developer should be able to clone the repository, configure documented prerequisites, launch the application, upload a sample dataset, and observe a complete analysis workflow.

Provide Docker-based local orchestration if appropriate to the repository.

Never require undocumented credentials or private infrastructure for the default local demonstration.

---

## 19. Implementation Workflow

Execute the work in the following order.

### Phase 1 — Discovery and architecture

- Inspect repository and applicable instructions.
- Identify existing capabilities and gaps.
- Select minimal justified technologies.
- Define domain models, component boundaries, and API contracts.
- Establish an implementation and testing plan.

### Phase 2 — Data foundation

- Implement dataset ingestion.
- Add validation and secure storage.
- Implement metadata and schema inference.
- Build dataset preview and management APIs.

### Phase 3 — Statistical engine

- Implement dataset and column profiling.
- Add descriptive statistics.
- Implement distributions.
- Add correlation calculations.
- Implement outlier detection.
- Validate numerical correctness.

### Phase 4 — Dashboard

- Build the application shell.
- Implement dataset workspace.
- Add overview and profiling views.
- Implement statistical explorer.
- Build distribution, correlation, and outlier visualizations.

### Phase 5 — AI insights

- Implement deterministic insight rules.
- Add structured evidence references.
- Implement optional LLM integration.
- Add graceful fallback and privacy controls.
- Build insight presentation.

### Phase 6 — Reports and operational hardening

- Implement report generation and export.
- Add authentication and authorization as required.
- Improve execution limits, failure handling, and logging.
- Add deployment configuration.

### Phase 7 — Verification

- Run relevant automated tests.
- Run linting and static type checks.
- Run security and dependency checks where tooling is available.
- Build the application.
- Verify integrated frontend/backend workflows.
- Benchmark representative workloads.
- Resolve discovered defects.

Follow dependencies between phases, but favor working vertical slices over disconnected infrastructure.

**Do not stop after the architecture plan or project scaffolding. Continue into implementation and verification.**

---

## 20. Definition of Done

The task is complete only when the following acceptance criteria are met:

- [ ] The application launches using documented instructions.
- [ ] Real datasets can be uploaded and validated.
- [ ] Dataset schema and previews display correctly.
- [ ] Profiling results are derived from actual data.
- [ ] Descriptive statistics pass correctness tests.
- [ ] Distribution visualizations render and respond to user selections.
- [ ] Correlation analysis uses statistically valid methods.
- [ ] Outlier detection supports documented algorithms and thresholds.
- [ ] Dataset filtering updates relevant analysis results.
- [ ] Deterministic AI insights work without external credentials.
- [ ] Optional LLM integration is configurable, secure, and failure-tolerant.
- [ ] Insights reference genuine analytical evidence.
- [ ] Report exports contain consistent analysis results.
- [ ] Errors, empty datasets, and important edge cases are handled.
- [ ] Data access is protected according to the deployment model.
- [ ] Backend and frontend integrate through real APIs.
- [ ] Relevant unit, integration, and end-to-end tests pass.
- [ ] Builds, linting, and type checks pass where configured.
- [ ] Documentation describes setup, operation, and limitations.
- [ ] No required functionality relies on fabricated data or unfinished placeholders.

Do not declare completion solely because the application compiles.

---

## 21. Execution Rules for the AI Coding Agent

1. **Implement instead of merely describing.** Write and integrate the actual application code.
2. **Inspect before modifying.** Follow existing repository patterns and preserve unrelated functionality.
3. **Make sound engineering choices independently.** Ask a focused question only when an unresolved issue materially affects correctness, compatibility, security, or architecture.
4. **Prefer focused changes.** Avoid unnecessary abstractions, dependencies, and microservices.
5. **Prioritize statistical correctness.** Validate formulas, null handling, assumptions, and numerical edge cases.
6. **Use real data.** Never fabricate production metrics or insights.
7. **Keep AI grounded.** Generated findings must be traceable to computed evidence.
8. **Design for failure.** Handle validation, timeouts, retries, concurrency, cancellations, and partial failures.
9. **Protect user data.** Treat datasets as untrusted and potentially sensitive.
10. **Verify incrementally.** Run appropriate tests after meaningful changes.
11. **Do not mask failures.** Report genuine test failures and unresolved issues.
12. **Do not overengineer.** Implement only what is necessary to satisfy the functional and quality requirements.
13. **Maintain compatibility.** Preserve established API, database, and frontend contracts unless deliberate migration is required.
14. **Document trade-offs.** Explain meaningful architectural choices and limitations.
15. **Finish the end-to-end workflow.** Prioritize a complete usable system over isolated sophisticated components.

If time, environment, or available tooling prevents full verification, complete as much working functionality as possible and explicitly identify what remains unverified. Do not claim production readiness without supporting evidence.

---

## 22. Final Response Format

After implementation, provide a concise engineering report containing:

### A. Implementation Summary
What was actually implemented, including working capabilities.

### B. Architecture
Technology choices, significant boundaries, and their rationale.

### C. Files Changed
Important added or modified modules with their responsibilities.

### D. Verification
Tests, builds, static checks, and end-to-end scenarios actually executed, including results.

### E. Running the Application
Exact documented commands and necessary configuration.

### F. Performance and Security
Controls implemented, benchmark findings if measured, and any unresolved concerns.

### G. Limitations and Remaining Work
Incomplete requirements, operational risks, and checks that could not be performed.

### H. Acceptance Criteria
Clearly distinguish **verified**, **implemented but unverified**, and **not implemented** requirements.

## Final Instruction

**Start by inspecting the repository. Build a complete, maintainable, secure, statistically correct, AI-powered Exploratory Data Analysis Dashboard with working ingestion, profiling, statistical analysis, distribution visualizations, correlation exploration, outlier detection, evidence-backed insights, and report generation. Validate the integrated application and report only what was genuinely implemented and verified.**