"""Evaluate a trained PhoBERT URL classifier on the official test split.

Run from the repository root:
    python src/phoBERT/test_phobert.py

If the model was trained in Colab, download its model folder and place it at
models/phobert, or pass the folder path with --model-dir.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except ImportError as exc:
    raise SystemExit(
        "Missing evaluation dependency. Install them with: "
        "pip install torch transformers sentencepiece scikit-learn pandas"
    ) from exc

if __package__:
    from . import training_config
else:
    import training_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "phoBERT_output" / "model"
DEFAULT_DATA_DIR = (
    PROJECT_ROOT / "data" / "PhishVN_v3.1.0_open" / "data" / "splits"
)
DEFAULT_METRICS_OUT = PROJECT_ROOT / "reports" / "phobert_test_metrics.json"
DEFAULT_PREDICTIONS_OUT = PROJECT_ROOT / "reports" / "phobert_test_predictions.csv"

LABEL_TO_ID = {"benign": 0, "phishing": 1, "0": 0, "1": 1}
ID_TO_LABEL = {0: "benign", 1: "phishing"}


class URLDataset(Dataset):
    """Tokenize URL examples for batched inference."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {
            key: values[index]
            for key, values in self.encodings.items()
        }
        item["labels"] = self.labels[index]
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved PhoBERT classifier on url_test.csv."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_OUT)
    parser.add_argument(
        "--predictions-out", type=Path, default=DEFAULT_PREDICTIONS_OUT
    )
    parser.add_argument(
        "--max-length", type=int, default=training_config.MAX_LENGTH
    )
    parser.add_argument(
        "--batch-size", type=int, default=training_config.EVAL_BATCH_SIZE
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Phishing probability threshold from 0 to 1.",
    )
    return parser.parse_args()


def load_test_data(data_dir: Path) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    data_path = data_dir / "url_test.csv"
    if not data_path.is_file():
        raise FileNotFoundError(f"Test split not found: {data_path}")

    frame = pd.read_csv(data_path, usecols=["url", "label"])
    frame["url"] = frame["url"].fillna("").astype("string").str.strip()
    raw_labels = frame["label"].astype("string").str.strip().str.lower()

    unknown_labels = sorted(set(raw_labels.dropna()) - set(LABEL_TO_ID))
    if unknown_labels:
        raise ValueError(
            f"Unexpected labels in {data_path.name}: {unknown_labels}"
        )

    frame["label_id"] = raw_labels.map(LABEL_TO_ID)
    frame = frame.loc[
        frame["url"].ne("") & frame["label_id"].notna()
    ].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No usable test rows found in {data_path}")
    labels = frame["label_id"].astype("int64").to_numpy()
    if len(np.unique(labels)) < 2:
        raise ValueError("The test split must contain both benign and phishing rows.")

    return frame, frame["url"].astype(str).tolist(), labels


def collate_batch(features: list[dict[str, Any]], tokenizer: Any) -> dict[str, torch.Tensor]:
    labels = torch.tensor([feature["labels"] for feature in features], dtype=torch.long)
    token_features = [
        {key: value for key, value in feature.items() if key != "labels"}
        for feature in features
    ]
    batch = tokenizer.pad(
        token_features,
        padding=True,
        return_tensors="pt",
    )
    batch["labels"] = labels
    return batch


def main() -> None:
    args = parse_args()
    if not args.model_dir.is_dir():
        raise FileNotFoundError(
            f"Trained model directory not found: {args.model_dir}\n"
            "Copy the model folder downloaded from Colab to this path or use --model-dir."
        )
    if not 2 <= args.max_length <= 256:
        raise ValueError("--max-length must be between 2 and 256 for PhoBERT.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1.")

    test_frame, texts, labels = load_test_data(args.data_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    dataset = URLDataset(texts, labels.tolist(), tokenizer, args.max_length)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda features: collate_batch(features, tokenizer),
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            model_inputs = {
                key: value.to(device)
                for key, value in batch.items()
                if key != "labels"
            }
            logits = model(**model_inputs).logits
            phishing_probabilities = torch.softmax(logits, dim=-1)[:, 1]
            all_probabilities.append(phishing_probabilities.cpu().numpy())

            if batch_index % 100 == 0:
                print(f"Evaluated {batch_index * args.batch_size:,} rows...", flush=True)

    phishing_probabilities = np.concatenate(all_probabilities)
    predictions = (phishing_probabilities >= args.threshold).astype("int64")

    metrics = {
        "model_dir": str(args.model_dir.resolve()),
        "data_file": str((args.data_dir / "url_test.csv").resolve()),
        "rows": int(len(labels)),
        "class_counts": {
            "benign": int(np.sum(labels == 0)),
            "phishing": int(np.sum(labels == 1)),
        },
        "threshold": args.threshold,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_phishing": float(
            precision_score(labels, predictions, pos_label=1, zero_division=0)
        ),
        "recall_phishing": float(
            recall_score(labels, predictions, pos_label=1, zero_division=0)
        ),
        "f1_phishing": float(
            f1_score(labels, predictions, pos_label=1, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(labels, phishing_probabilities)),
        "confusion_matrix_labels_benign_phishing": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=["benign", "phishing"],
            output_dict=True,
            zero_division=0,
        ),
    }

    test_frame["label_true"] = [ID_TO_LABEL[int(label)] for label in labels]
    test_frame["label_pred"] = [
        ID_TO_LABEL[int(label)] for label in predictions
    ]
    test_frame["prob_phishing"] = phishing_probabilities
    test_frame["correct"] = predictions == labels

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False, default=float) + "\n",
        encoding="utf-8",
    )
    test_frame.to_csv(args.predictions_out, index=False, encoding="utf-8-sig")

    print(f"Device: {device}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision (phishing): {metrics['precision_phishing']:.4f}")
    print(f"Recall (phishing): {metrics['recall_phishing']:.4f}")
    print(f"F1 (phishing): {metrics['f1_phishing']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(
        "Confusion matrix [benign, phishing]: "
        f"{metrics['confusion_matrix_labels_benign_phishing']}"
    )
    print(f"Metrics saved to: {args.metrics_out}")
    print(f"Predictions saved to: {args.predictions_out}")


if __name__ == "__main__":
    main()


