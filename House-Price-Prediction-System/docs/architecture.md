# Architecture

```mermaid
flowchart LR
 U[User] --> F[React frontend]
 F --> A[FastAPI /api/v1/predictions]
 A --> V[Pydantic validation]
 V --> P[Fitted ColumnTransformer]
 P --> M[Versioned regression artifact]
 M --> R[Prediction response]
 R --> F
 D[CSV / OpenML] --> T[Offline train + CV + holdout evaluation]
 T --> S[Joblib artifact + manifest]
 S --> M
```

The runtime loads one trusted artifact at startup. Training is an offline process and never runs in a request. Numeric imputation/scaling and categorical imputation/one-hot encoding are fitted inside the estimator pipeline, including within CV folds. The fixture makes CI self-contained; the acquisition script supports a public Ames CSV.
