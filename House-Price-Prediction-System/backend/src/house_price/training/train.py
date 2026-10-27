import argparse, hashlib, json, platform, time
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from house_price.core.config import DATA_PATH, ARTIFACT_PATH, CURRENCY, RANDOM_STATE
from house_price.data.validation import FEATURES, TARGET, validate_dataset
from house_price.features.pipeline import build_preprocessor
from house_price.evaluation.diagnostics import save_diagnostics
from house_price.models.gradient_descent import GradientDescentRegressor
from house_price.training.metrics import regression_metrics

def load_data(path=DATA_PATH):
    frame = pd.read_csv(path); validate_dataset(frame); return frame

def make_pipeline(model): return Pipeline([("preprocess", build_preprocessor()), ("model", model)])

def train(path=DATA_PATH, artifact_path=ARTIFACT_PATH):
    frame = load_data(path); X, y = frame[FEATURES], frame[TARGET]
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=.15, random_state=RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.1765, random_state=RANDOM_STATE)
    candidates = {"Baseline": make_pipeline(DummyRegressor(strategy="median")),
                  "Linear Regression": make_pipeline(LinearRegression()),
                  "Custom Gradient Descent": make_pipeline(GradientDescentRegressor()),
                  "Ridge Regression": GridSearchCV(make_pipeline(Ridge()), {"model__alpha": [0.1, 1, 10, 100]}, scoring="neg_mean_absolute_error", cv=KFold(4, shuffle=True, random_state=RANDOM_STATE)),
                  "Lasso Regression": GridSearchCV(make_pipeline(Lasso(max_iter=10000)), {"model__alpha": [0.001, 0.01, 0.1, 1]}, scoring="neg_mean_absolute_error", cv=KFold(4, shuffle=True, random_state=RANDOM_STATE))}
    results = {}
    test_predictions = {}
    gradient_loss = []
    for name, estimator in candidates.items():
        start = time.perf_counter(); estimator.fit(X_train, y_train); pred = estimator.predict(X_test)
        test_predictions[name] = pred
        if name == "Custom Gradient Descent":
            gradient_loss = estimator.named_steps["model"].loss_history_
        results[name] = {**regression_metrics(y_test, pred), "training_seconds": time.perf_counter()-start}
    # Selection is based on validation MAE only; final metrics above remain a single holdout readout.
    selection = {}
    for name, estimator in candidates.items(): selection[name] = regression_metrics(y_val, estimator.predict(X_val))["mae"]
    selected_name = min(selection, key=selection.get); selected = candidates[selected_name]; selected.fit(X_dev, y_dev)
    metadata = {"model_version": "1.0.0", "algorithm": selected_name, "features": FEATURES, "target": TARGET,
                "currency": CURRENCY, "unit": "currency_per_property", "dataset_fingerprint": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                "trained_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "metrics": results,
                "selection_validation_mae": selection[selected_name], "python": platform.python_version()}
    artifact_path.parent.mkdir(parents=True, exist_ok=True); joblib.dump({"pipeline": selected, "metadata": metadata}, artifact_path)
    artifact_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2)); Path("reports").mkdir(exist_ok=True)
    Path("reports/model_comparison.json").write_text(json.dumps({"results": results, "validation_mae": selection, "selected": selected_name}, indent=2))
    save_diagnostics(y_test, test_predictions[selected_name], selected_name, gradient_loss, Path("reports"))
    print(json.dumps({"selected": selected_name, "results": results}, indent=2)); return metadata

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--data", type=Path, default=DATA_PATH); parser.add_argument("--artifact", type=Path, default=ARTIFACT_PATH)
    args = parser.parse_args(); train(args.data, args.artifact)
