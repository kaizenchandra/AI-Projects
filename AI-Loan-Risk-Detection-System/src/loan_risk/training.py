"""Bounded training orchestration; test outcomes never select an estimator."""

import hashlib
import io
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from threadpoolctl import threadpool_limits

from loan_risk.artifacts import package
from loan_risk.config import TrainConfig
from loan_risk.contracts import FEATURES, NUMERIC, TARGET
from loan_risk.data import temporal_folds, temporal_split, validate_dataset
from loan_risk.evaluation import (
    choose_threshold,
    curves,
    metrics,
    robustness,
    sensitivity,
    uncertainty,
)
from loan_risk.explainability import global_importance
from loan_risk.features import pipeline
from loan_risk.monitoring import profile
from loan_risk.reporting import eda, evaluation_plots, write_json


def train(frame, output: Path, config: TrainConfig) -> dict:
    """Do not overwrite a run. RF remains primary; challengers require separate promotion."""
    if output.exists():
        raise ValueError("Run output already exists; use a new version directory")
    data, validation = validate_dataset(frame)
    splits = temporal_split(data, config)
    tr, va, te = [splits[name] for name in ["train", "validation", "test"]]
    folds = temporal_folds(tr, config.folds)
    output.mkdir(parents=True)
    eda(tr, output)
    candidates = {
        "logistic": (
            LogisticRegression(max_iter=1500, random_state=config.seed),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(random_state=config.seed, n_jobs=1),
            {
                "model__n_estimators": [config.trees, config.trees + 20],
                "model__max_depth": [5, 10, None],
                "model__min_samples_split": [2, 10],
                "model__min_samples_leaf": [5, 15, 30],
                "model__max_features": ["sqrt", 0.8],
                "model__class_weight": [None, "balanced"],
                "model__max_samples": [0.8, 1.0],
            },
        ),
        "gradient_boosting": (
            HistGradientBoostingClassifier(
                random_state=config.seed, max_iter=100, early_stopping=False
            ),
            {
                "model__learning_rate": [0.05, 0.1],
                "model__max_leaf_nodes": [7, 15],
                "model__l2_regularization": [1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
    }
    report: dict = {
        "synthetic": config.synthetic,
        "dataset_id": config.dataset_id,
        "interpretation": "Synthetic performance is not evidence of real-world accuracy"
        if config.synthetic
        else "Requires independent representative-data validation",
        "validation": validation,
        "split_counts": {k: len(v) for k, v in splits.items()},
        "embargo_excluded": len(data) - sum(map(len, splits.values())),
        "models": {},
        "selection": "Random Forest primary; calibration selected on validation Brier",
    }
    fitted = {}
    selected_curves = {}
    with threadpool_limits(limits=1):
        for name, (estimator, parameters) in candidates.items():
            search = RandomizedSearchCV(
                pipeline(estimator),
                parameters,
                n_iter=config.search_iterations,
                scoring="neg_brier_score",
                cv=folds,
                random_state=config.seed,
                n_jobs=1,
                error_score="raise",
                refit=True,
            )
            search.fit(tr[FEATURES], tr[TARGET])
            raw = search.best_estimator_
            variants = {"uncalibrated": raw}
            for method in ["sigmoid", "isotonic"]:
                calibrated = CalibratedClassifierCV(
                    raw, method=method, cv=folds, ensemble=True, n_jobs=1
                )
                calibrated.fit(tr[FEATURES], tr[TARGET])
                variants[method] = calibrated
            assessments = {
                method: metrics(va[TARGET], model.predict_proba(va[FEATURES])[:, 1])
                for method, model in variants.items()
            }
            # Deliver a calibrated RF even if raw validation Brier is slightly better.
            method = min(["sigmoid", "isotonic"], key=lambda m: assessments[m]["brier"])
            selected = variants[method]
            vp = selected.predict_proba(va[FEATURES])[:, 1]
            threshold = choose_threshold(va[TARGET], vp, config)
            fitted[name] = (selected, raw)
            report["models"][name] = {
                "hyperparameters": search.best_params_,
                "cv_neg_brier": float(search.best_score_),
                "calibration_comparison_validation": assessments,
                "selected_calibration": method,
                "validation_metrics": metrics(va[TARGET], vp, threshold["threshold"]),
                "review_policy_candidate": threshold,
                "cost_sensitivity": sensitivity(va[TARGET], vp, config),
                "validation_robustness": robustness(
                    selected, va[FEATURES], va[TARGET], config.seed
                ),
            }
        # All model/calibration/threshold choices are fixed before any test predictions.
        for name, (selected, _raw) in fitted.items():
            started = time.perf_counter()
            tp = selected.predict_proba(te[FEATURES])[:, 1]
            latency = (time.perf_counter() - started) * 1000 / len(te)
            test_threshold = float(report["models"][name]["review_policy_candidate"]["threshold"])
            buffer = io.BytesIO()
            joblib.dump(selected, buffer, compress=3)
            report["models"][name].update(
                {
                    "test_metrics": metrics(te[TARGET], tp, test_threshold),
                    "test_ci95_iid_bootstrap": uncertainty(
                        te[TARGET], tp, config.bootstrap_iterations, config.seed
                    ),
                    "test_review_rate": float((tp >= test_threshold).mean()),
                    "batch_ms_per_row": latency,
                    "compressed_bytes": buffer.tell(),
                }
            )
            selected_curves[name] = curves(te[TARGET], tp)
        selected, raw = fitted["random_forest"]
        report["importance"] = global_importance(
            raw, selected, va[FEATURES], va[TARGET], config.seed, output
        )
        reference: dict[str, Any] = {
            name: float(tr[name].median()) if tr[name].notna().any() else 0.0 for name in NUMERIC
        }
        reference["repayment_history"] = (
            tr.repayment_history.mode().iloc[0] if tr.repayment_history.notna().any() else "missing"
        )
        # Missing-safe references must also satisfy the input contract.
        reference["annual_income"] = max(1.0, reference["annual_income"])
        reference["loan_amount"] = max(100.0, reference["loan_amount"])
        reference["loan_term_months"] = max(6, int(reference["loan_term_months"]))
        for name in ["prior_delinquencies", "active_credit_accounts"]:
            reference[name] = int(reference[name])
        monitoring = profile(tr[FEATURES], selected.predict_proba(tr[FEATURES])[:, 1])
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unavailable"
    write_json(output / "comparison.json", report)
    write_json(output / "curves.json", selected_curves)
    evaluation_plots(selected_curves, output)
    metadata = {
        "model_version": config.model_version,
        "model_family": "random_forest",
        "synthetic": config.synthetic,
        "dataset_id": config.dataset_id,
        "dataset_sha256": hashlib.sha256(data.to_csv(index=False).encode()).hexdigest(),
        "training_config": config.model_dump(),
        "source_commit": commit,
        "trained_at": datetime.now(UTC).isoformat(),
        "reference": reference,
        "monitoring_reference": monitoring,
        "evaluation": report["models"]["random_forest"],
        "report_sha256": hashlib.sha256((output / "comparison.json").read_bytes()).hexdigest(),
        "risk_bands": {
            "low": config.low_risk_cutoff,
            "high": config.high_risk_cutoff,
            "validated_policy": False,
        },
        "prediction_definition": "First 90+ DPD event within 12 calendar months of origination",
    }
    checksum = package(selected, metadata, output / "candidate")
    return {"report": report, "manifest_sha256": checksum, "output": str(output)}
