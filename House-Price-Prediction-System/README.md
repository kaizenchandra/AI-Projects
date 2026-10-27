# House Price Prediction System

A runnable reference application for classical supervised regression. It trains and serves a house-price model using a compact Ames Housing feature contract:

`OverallQual`, `GrLivArea`, `GarageCars`, `TotalBsmtSF`, `YearBuilt`, and `Neighborhood`.

The repository separates data acquisition, validation, leakage-safe preprocessing, model training, evaluation, artifact loading, FastAPI inference, and the React frontend.

## Prerequisites

- Python 3.9.6+ (Python 3.12 is also supported)
- Node.js 22+ and npm
- Docker Desktop, only required for the Compose workflow

On macOS, if `python3` fails with `xcrun: invalid active developer path`, install the Command Line Tools and reopen the terminal:

```bash
xcode-select --install
```

## Local setup and training

Run these commands from the parent `AI-Projects` directory shown in the project environment:

```bash
cd House-Price-Prediction-System
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e './backend[dev]'
python scripts/generate_fixture.py
python -m house_price.training.train
python scripts/eda.py
```

The fixture is deterministic synthetic data for offline development and tests only. It is not a real-estate valuation dataset. The training command produces:

- `artifacts/model.joblib` and its JSON manifest
- `reports/model_comparison.json`
- `reports/actual_vs_predicted.png`
- `reports/residuals_vs_predictions.png`
- `reports/gradient_descent_loss.png`

To use the public Ames source instead, run this from the project root before training:

```bash
python scripts/acquire_ames.py
python -m house_price.training.train
```

The acquisition command requires network access. Verify the downloaded dataset’s provenance, geography, currency, and target units before using it for any real decision.

## Run the local application

Start the API from the project root:

```bash
source .venv/bin/activate
python -m uvicorn house_price.api.main:app --app-dir backend/src --reload
```

In a second terminal, from the project root, start the frontend:

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The frontend uses `VITE_API_URL` when supplied; otherwise it calls `http://localhost:8000`.

## API smoke test

With the API running:

```bash
curl http://localhost:8000/api/v1/health/live
curl http://localhost:8000/api/v1/health/ready
curl -X POST http://localhost:8000/api/v1/predictions \
  -H 'content-type: application/json' \
  -d '{"OverallQual":7,"GrLivArea":1700,"GarageCars":2,"TotalBsmtSF":800,"YearBuilt":2000,"Neighborhood":"CollgCr"}'
```

The prediction response contains the computed price, configured currency and unit, selected model, model version, and a request correlation ID. The API also exposes `/docs` for interactive OpenAPI documentation.

## Tests and checks

From the project root with the virtual environment active:

```bash
pytest -q backend/tests
ruff check backend/src backend/tests scripts
mypy backend/src --ignore-missing-imports
```

For a frontend production build:

```bash
cd frontend
npm install
npm run build
```

## Docker Compose

Run Compose from the parent `AI-Projects` directory, not from `frontend`:

```bash
cd House-Price-Prediction-System
docker compose up --build
```

The API image creates the deterministic offline fixture and model during its build, so the Compose stack does not need a dataset download. The API is available at <http://localhost:8000> and the frontend at <http://localhost:5173>. Stop the stack with `Ctrl-C` or `docker compose down`.

## Configuration

Copy the values in [.env.example](.env.example) into your shell or deployment environment as needed:

- `HOUSE_DATA_PATH`: training CSV path
- `MODEL_ARTIFACT_PATH`: serialized model path
- `TARGET_CURRENCY`: output currency label
- `RANDOM_STATE`: reproducibility seed
- `CORS_ORIGINS`: comma-separated browser origins

Authentication, TLS termination, rate limiting, and gateway-level request limits belong at the deployment boundary. Model artifacts are trusted server-side files and are never accepted from prediction requests.

## Project documentation

- [Architecture](docs/architecture.md)
- [ML concepts](docs/ml-concepts.md)
- [Model evaluation](docs/model-evaluation.md)
- [Operations](docs/operations.md)
