"""Explicit offline commands; serving never triggers training."""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from loan_risk.artifacts import promote
from loan_risk.config import TrainConfig
from loan_risk.contracts import FEATURES, TARGET
from loan_risk.data import synthetic_data, validate_dataset, validate_features
from loan_risk.explainability import local_explanation
from loan_risk.inference import Predictor
from loan_risk.monitoring import delayed_performance, drift
from loan_risk.reporting import write_json
from loan_risk.training import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--rows", type=int, default=6000)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--output", type=Path, required=True)
    training = commands.add_parser("train")
    training.add_argument("--data", type=Path, required=True)
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    for name in ["predict", "explain", "monitor", "evaluate-delayed"]:
        command = commands.add_parser(name)
        command.add_argument("--model", type=Path, required=True)
        command.add_argument("--trusted-hash", required=True)
        command.add_argument("--allow-candidate", action="store_true")
        command.add_argument("--input", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    promotion = commands.add_parser("promote")
    promotion.add_argument("--model", type=Path, required=True)
    promotion.add_argument("--output", type=Path, required=True)
    promotion.add_argument("--trusted-hash", required=True)
    promotion.add_argument("--approval", type=Path, required=True)
    args = parser.parse_args()
    result: Any
    if args.command == "generate":
        if args.output.exists():
            parser.error("Output exists")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        synthetic_data(args.rows, args.seed).to_csv(args.output, index=False)
    elif args.command == "train":
        config = TrainConfig.model_validate_json(args.config.read_text())
        result = train(pd.read_csv(args.data), args.output, config)
        print(json.dumps({k: v for k, v in result.items() if k != "report"}))
    elif args.command == "promote":
        print(
            promote(
                args.model, args.output, args.trusted_hash, json.loads(args.approval.read_text())
            )
        )
    else:
        predictor = Predictor(args.model, args.trusted_hash, args.allow_candidate)
        if args.command == "evaluate-delayed":
            mature, _ = validate_dataset(pd.read_csv(args.input))
            predictions = predictor.predict(mature[FEATURES])
            result = delayed_performance(
                mature[TARGET], [p["default_probability"] for p in predictions]
            )
        else:
            frame = validate_features(pd.read_csv(args.input))
            if args.command == "explain":
                result = local_explanation(predictor.model, frame, predictor.metadata["reference"])
            else:
                result = predictor.predict(frame)
                if args.command == "monitor":
                    result = drift(
                        predictor.metadata["monitoring_reference"],
                        frame,
                        [p["default_probability"] for p in result],
                    )
        write_json(args.output, result)


if __name__ == "__main__":
    main()
