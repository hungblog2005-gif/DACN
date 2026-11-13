"""Train a Random Forest on the PhishVN official training split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

if __package__:
    from . import training_config
else:
    import training_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "PhishVN_v3.1.0_open" / "data" / "splits"
DEFAULT_MODEL_OUT = PROJECT_ROOT / "models" / "random_forest.joblib"
DEFAULT_METRICS_OUT = PROJECT_ROOT / "reports" / "random_forest_training_metrics.json"
DEFAULT_IMPORTANCE_OUT = PROJECT_ROOT / "reports" / "random_forest_feature_importance.csv"

FEATURE_COLUMNS = [
    "url_len",
    "num_dots",
    "num_subdomains",
    "has_ip",
    "has_at",
    "suspicious_tld",
]
LABEL_TO_INT = {"benign": 0, "phishing": 1}
SPLIT_FILES = {
    "train": "url_train.csv",
    "val": "url_val.csv",
    "test": "url_test.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Random Forest and report train/validation metrics each round."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the official PhishVN split CSV files",
    )
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_OUT)
    parser.add_argument("--importance-out", type=Path, default=DEFAULT_IMPORTANCE_OUT)
    parser.add_argument("--n-estimators", type=int, default=training_config.N_ESTIMATORS)
    parser.add_argument("--max-depth", type=int, default=training_config.MAX_DEPTH)
    parser.add_argument("--random-state", type=int, default=training_config.RANDOM_STATE)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=training_config.PROGRESS_EVERY,
        help="Trees added per training round",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=training_config.DECISION_THRESHOLD,
        help="Probability threshold used for classification metrics",
    )
    return parser.parse_args()


def load_splits(
    data_dir: Path, split_names: tuple[str, ...] | None = None
) -> dict[str, tuple[pd.DataFrame, np.ndarray]]:
    names = tuple(split_names) if split_names is not None else tuple(SPLIT_FILES)
    unknown_splits = sorted(set(names) - set(SPLIT_FILES))
    if unknown_splits:
        raise ValueError(f"Unknown split names: {unknown_splits}")

    required = set(FEATURE_COLUMNS) | {"label"}
    splits: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}

    for split_name in names:
        data_path = data_dir / SPLIT_FILES[split_name]
        if not data_path.is_file():
            raise FileNotFoundError(f"Dataset split not found: {data_path}")

        frame = pd.read_csv(data_path, usecols=lambda column: column in required)
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"{data_path.name} is missing required columns: {sorted(missing)}"
            )

        frame["label"] = frame["label"].astype("string").str.strip().str.lower()
        unknown_labels = sorted(set(frame["label"].dropna()) - set(LABEL_TO_INT))
        if unknown_labels:
            raise ValueError(f"Unexpected labels in {data_path.name}: {unknown_labels}")

        features = frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        labels = frame["label"].map(LABEL_TO_INT)
        valid_labels = labels.notna()
        features = features.loc[valid_labels].reset_index(drop=True)
        y = labels.loc[valid_labels].astype("int8").to_numpy()
        if len(y) == 0:
            raise ValueError(f"Split '{split_name}' has no rows with valid labels")
        splits[split_name] = (features, y)

    if "train" in splits and len(np.unique(splits["train"][1])) < 2:
        raise ValueError("Training split must contain both benign and phishing labels")
    return splits


def evaluate(
    model: Pipeline,
    features: pd.DataFrame,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    probabilities = model.predict_proba(features)
    classifier = model.named_steps["classifier"]
    positive_index = int(np.flatnonzero(classifier.classes_ == 1)[0])
    positive_probabilities = probabilities[:, positive_index]
    predictions = (positive_probabilities >= threshold).astype("int8")
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "rows": int(len(labels)),
        "class_counts": {
            "benign": int(np.sum(labels == 0)),
            "phishing": int(np.sum(labels == 1)),
        },
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_phishing": float(
            precision_score(labels, predictions, pos_label=1, zero_division=0)
        ),
        "recall_phishing": float(
            recall_score(labels, predictions, pos_label=1, zero_division=0)
        ),
        "f1_phishing": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(labels, positive_probabilities))
            if len(np.unique(labels)) == 2
            else None
        ),
        "confusion_matrix_labels_benign_phishing": matrix.tolist(),
    }


def print_metrics_table(
    splits: dict[str, dict[str, Any]], heading: str = "Metrics by data split"
) -> None:
    headers = [
        "Split",
        "Rows",
        "Benign",
        "Phishing",
        "LogLoss",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
    ]
    rows = []
    for split_name, result in splits.items():
        counts = result["class_counts"]
        rows.append(
            [
                split_name,
                str(result["rows"]),
                str(counts["benign"]),
                str(counts["phishing"]),
                f"{result['log_loss']:.4f}",
                f"{result['accuracy']:.4f}",
                f"{result['precision_phishing']:.4f}",
                f"{result['recall_phishing']:.4f}",
                f"{result['f1_phishing']:.4f}",
                "N/A" if result["roc_auc"] is None else f"{result['roc_auc']:.4f}",
            ]
        )

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    separator = "+-" + "-+-".join("-" * width for width in widths) + "-+"

    def format_row(row: list[str]) -> str:
        return "| " + " | ".join(
            value.ljust(width) for value, width in zip(row, widths)
        ) + " |"

    print(f"\n{heading}:")
    print(separator)
    print(format_row(headers))
    print(separator)
    for row in rows:
        print(format_row(row))
    print(separator)


def main() -> None:
    args = parse_args()
    if args.n_estimators < 1:
        raise ValueError("--n-estimators must be at least 1")
    if args.progress_every < 1:
        raise ValueError("--progress-every must be at least 1")
    if args.max_depth is not None and args.max_depth < 1:
        raise ValueError("--max-depth must be at least 1")
    if not 0.0 <= args.decision_threshold <= 1.0:
        raise ValueError("--decision-threshold must be between 0 and 1")

    splits = load_splits(args.data_dir, split_names=("train", "val"))
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=args.n_estimators,
                    max_depth=args.max_depth,
                    min_samples_split=training_config.MIN_SAMPLES_SPLIT,
                    min_samples_leaf=training_config.MIN_SAMPLES_LEAF,
                    max_features=training_config.MAX_FEATURES,
                    criterion=training_config.CRITERION,
                    class_weight=training_config.CLASS_WEIGHT,
                    random_state=args.random_state,
                    n_jobs=-1,
                    warm_start=True,
                ),
            ),
        ]
    )

    train_x, train_y = splits["train"]
    val_x, val_y = splits["val"]
    batch_size = min(args.progress_every, args.n_estimators)
    total_rounds = (args.n_estimators + batch_size - 1) // batch_size
    history: list[dict[str, Any]] = []
    training_started = time.perf_counter()

    for round_index, first_tree in enumerate(
        range(0, args.n_estimators, batch_size), start=1
    ):
        last_tree = min(first_tree + batch_size, args.n_estimators)
        model.set_params(classifier__n_estimators=last_tree)
        print(
            f"Training round {round_index}/{total_rounds}: "
            f"trees {first_tree + 1}-{last_tree}/{args.n_estimators}...",
            flush=True,
        )
        round_started = time.perf_counter()
        model.fit(train_x, train_y)

        round_metrics = {
            "train": evaluate(model, train_x, train_y, args.decision_threshold),
            "val": evaluate(model, val_x, val_y, args.decision_threshold),
        }
        history.append(
            {"round": round_index, "tree_count": last_tree, "metrics": round_metrics}
        )
        print(
            f"Completed round {round_index}/{total_rounds} "
            f"in {time.perf_counter() - round_started:.1f} seconds."
        )
        print_metrics_table(
            round_metrics, heading=f"Metrics after training round {round_index}"
        )

    metrics = {
        "data_dir": str(args.data_dir.resolve()),
        "target": "label (benign=0, phishing=1)",
        "features": FEATURE_COLUMNS,
        "model_parameters": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_split": training_config.MIN_SAMPLES_SPLIT,
            "min_samples_leaf": training_config.MIN_SAMPLES_LEAF,
            "max_features": training_config.MAX_FEATURES,
            "criterion": training_config.CRITERION,
            "class_weight": training_config.CLASS_WEIGHT,
            "decision_threshold": args.decision_threshold,
            "random_state": args.random_state,
        },
        "training_history": history,
        "final_metrics": history[-1]["metrics"],
    }

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.importance_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": FEATURE_COLUMNS,
            "label_to_int": LABEL_TO_INT,
            "decision_threshold": args.decision_threshold,
        },
        args.model_out,
    )
    args.metrics_out.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    forest = model.named_steps["classifier"]
    importance = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "importance": forest.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(args.importance_out, index=False, encoding="utf-8")

    print(f"Training finished in {time.perf_counter() - training_started:.1f} seconds.")
    print(f"Model saved to: {args.model_out}")
    print(f"Training metrics saved to: {args.metrics_out}")
    print(f"Feature importances saved to: {args.importance_out}")


if __name__ == "__main__":
    main()