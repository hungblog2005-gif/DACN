"""Evaluate a saved Random Forest on the held-out PhishVN test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

if __package__:
    from . import train_random_forest as trainer
    from . import training_config
else:
    import train_random_forest as trainer
    import training_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = trainer.DEFAULT_DATA_DIR
DEFAULT_MODEL_IN = trainer.DEFAULT_MODEL_OUT
DEFAULT_METRICS_OUT = PROJECT_ROOT / "reports" / "random_forest_test_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Random Forest on url_test.csv."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-in", type=Path, default=DEFAULT_MODEL_IN)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_OUT)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the decision threshold stored with the model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_in.is_file():
        raise FileNotFoundError(f"Trained model not found: {args.model_in}")

    artifact = joblib.load(args.model_in)
    model = artifact["model"]
    threshold = (
        artifact.get("decision_threshold", training_config.DECISION_THRESHOLD)
        if args.threshold is None
        else args.threshold
    )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Decision threshold must be between 0 and 1")

    test_features, test_labels = trainer.load_splits(
        args.data_dir, split_names=("test",)
    )["test"]
    result = trainer.evaluate(model, test_features, test_labels, threshold)
    report = {
        "data_dir": str(args.data_dir.resolve()),
        "model_path": str(args.model_in.resolve()),
        "split": "test",
        "decision_threshold": threshold,
        "metrics": result,
    }

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    trainer.print_metrics_table({"test": result}, heading="Held-out test results")
    print(f"Test metrics saved to: {args.metrics_out}")


if __name__ == "__main__":
    main()