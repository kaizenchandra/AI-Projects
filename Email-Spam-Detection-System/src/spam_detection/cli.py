"""Command-line training and evaluation workflow."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from .data import load_dataset
from .evaluation import evaluate, save_report
from .model import load_bundle, save_bundle, train_bundle


def load_config(path: str | None) -> dict[str, Any]:
    default = Path(__file__).parents[2] / "configs" / "default.yaml"
    config_path = Path(path) if path else default
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Email spam detection training and evaluation")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--data", required=True)
    train.add_argument("--config")
    train.add_argument("--artifact", default="artifacts/model.joblib")
    train.add_argument("--report", default="artifacts/evaluation.json")
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--data", required=True)
    evaluate_parser.add_argument("--artifact", default="artifacts/model.joblib")
    evaluate_parser.add_argument("--report", default="artifacts/evaluation.json")
    args = parser.parse_args()
    if args.command == "train":
        frame, profile = load_dataset(args.data)
        bundle = train_bundle(
            frame["text"].tolist(), frame["label"].tolist(), profile, load_config(args.config)
        )
        save_bundle(bundle, args.artifact)
        test = bundle.metadata.pop("test_data")
        save_report(evaluate(bundle, test["texts"], test["labels"]), args.report)
        print(f"saved {args.artifact} and {args.report}; model={bundle.metadata['model_id']}")
    else:
        frame, _ = load_dataset(args.data)
        bundle = load_bundle(args.artifact)
        save_report(evaluate(bundle, frame["text"].tolist(), frame["label"].tolist()), args.report)
        print(f"saved {args.report}")


if __name__ == "__main__":
    main()
