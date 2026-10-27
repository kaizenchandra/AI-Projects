from pathlib import Path

from spam_detection.data import load_dataset
from spam_detection.model import load_bundle, save_bundle, train_bundle


def config() -> dict:
    return {
        "model": {"alpha": 1.0, "fit_prior": True},
        "vectorizer": {"ngram_range": [1, 2], "min_df": 1, "max_df": 1.0, "max_features": 50000},
        "training": {"random_seed": 42, "test_size": 0.2, "validation_size": 0.2},
        "classification": {"false_positive_rate_budget": 0.5},
    }


def test_train_save_load_prediction_parity(tmp_path: Path) -> None:
    frame, profile = load_dataset(
        Path(__file__).parents[1] / "data" / "fixtures" / "toy_emails.csv"
    )
    bundle = train_bundle(frame.text.tolist(), frame.label.tolist(), profile, config())
    artifact = save_bundle(bundle, tmp_path / "model.joblib")
    loaded = load_bundle(artifact)
    texts = frame.text.iloc[:3].tolist()
    assert (bundle.pipeline.predict(texts) == loaded.pipeline.predict(texts)).all()
