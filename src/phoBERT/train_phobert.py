"""Fine-tune PhoBERT to classify URLs as benign or phishing.

Install the additional dependencies first:
    pip install torch transformers sentencepiece accelerate

Run from the repository root:
    python src/phoBERT/train_phobert.py
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import pandas as pd
    import torch
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )
except ImportError as exc:
    raise SystemExit(
        "Missing training dependency. Install them with: "
        "pip install torch transformers sentencepiece accelerate scikit-learn pandas"
    ) from exc


if __package__:
    from . import training_config
else:
    import training_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = (
    PROJECT_ROOT / "data" / "PhishVN_v3.1.0_open" / "data" / "splits"
)
DEFAULT_MODEL_OUT = PROJECT_ROOT / "models" / "phobert"
DEFAULT_METRICS_OUT = PROJECT_ROOT / "reports" / "phobert_training_metrics.json"
DEFAULT_TRAINING_OUT = PROJECT_ROOT / "reports" / "phobert_checkpoints"
MODEL_NAME = training_config.MODEL_NAME
LABEL_TO_ID = {"benign": 0, "phishing": 1, "0": 0, "1": 1}
SPLIT_FILES = {
    "train": "url_train.csv",
    "validation": "url_val.csv",
    "test": "url_test.csv",
}


class URLDataset(torch.utils.data.Dataset):
    """Tokenized URL examples for Hugging Face Trainer."""

    def __init__(self, encodings: dict[str, list[list[int]]], labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            name: torch.tensor(values[index], dtype=torch.long)
            for name, values in self.encodings.items()
        }
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune PhoBERT to classify phishing URLs."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_OUT)
    parser.add_argument("--training-out", type=Path, default=DEFAULT_TRAINING_OUT)
    parser.add_argument("--max-length", type=int, default=training_config.MAX_LENGTH)
    parser.add_argument("--epochs", type=float, default=training_config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=training_config.BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=training_config.EVAL_BATCH_SIZE)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=training_config.GRADIENT_ACCUMULATION_STEPS)
    parser.add_argument("--learning-rate", type=float, default=training_config.LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=training_config.WEIGHT_DECAY)
    parser.add_argument("--logging-steps", type=int, default=training_config.LOGGING_STEPS)
    parser.add_argument("--seed", type=int, default=training_config.SEED)
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=training_config.FP16,
        help="Use half precision on a compatible CUDA GPU to reduce memory use.",
    )
    return parser.parse_args()


def load_split(data_dir: Path, split_name: str) -> tuple[list[str], list[int]]:
    path = data_dir / SPLIT_FILES[split_name]
    if not path.is_file():
        raise FileNotFoundError(f"Dataset split not found: {path}")

    frame = pd.read_csv(path, usecols=["url", "label"])
    frame["url"] = frame["url"].fillna("").astype("string").str.strip()
    raw_labels = frame["label"].astype("string").str.strip().str.lower()
    unknown_labels = sorted(set(raw_labels.dropna()) - set(LABEL_TO_ID))
    if unknown_labels:
        raise ValueError(f"Unexpected labels in {path.name}: {unknown_labels}")

    frame["label_id"] = raw_labels.map(LABEL_TO_ID)
    frame = frame.loc[frame["url"].ne("") & frame["label_id"].notna()].copy()
    if frame.empty:
        raise ValueError(f"Split '{split_name}' has no usable URL rows: {path}")

    labels = frame["label_id"].astype("int64").tolist()
    if len(set(labels)) < 2:
        raise ValueError(f"Split '{split_name}' must contain both classes: {path}")

    return frame["url"].astype(str).tolist(), labels


def tokenize_split(
    tokenizer: Any, texts: list[str], labels: list[int], max_length: int
) -> URLDataset:
    encodings = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    return URLDataset(encodings, labels)


def compute_metrics(prediction: Any) -> dict[str, float]:
    logits = prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]

    logits = np.asarray(logits)
    labels = np.asarray(prediction.label_ids)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predicted_labels = probabilities.argmax(axis=1)

    return {
        "accuracy": float(accuracy_score(labels, predicted_labels)),
        "precision": float(
            precision_score(labels, predicted_labels, pos_label=1, zero_division=0)
        ),
        "recall": float(
            recall_score(labels, predicted_labels, pos_label=1, zero_division=0)
        ),
        "f1": float(f1_score(labels, predicted_labels, pos_label=1, zero_division=0)),
        "roc_auc": (
            float(roc_auc_score(labels, probabilities[:, 1]))
            if len(np.unique(labels)) == 2
            else 0.0
        ),
    }


def main() -> None:
    args = parse_args()
    if not 2 <= args.max_length <= 256:
        raise ValueError("--max-length must be between 2 and 256 for PhoBERT.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be greater than 0.")
    if args.batch_size < 1 or args.eval_batch_size < 1:
        raise ValueError("Batch sizes must be at least 1.")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be at least 1.")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Learning rate must be positive and weight decay non-negative.")
    if args.logging_steps < 1:
        raise ValueError("--logging-steps must be at least 1.")
    if args.fp16 and not torch.cuda.is_available():
        raise ValueError("--fp16 requires a CUDA-capable GPU.")

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "benign", 1: "phishing"},
        label2id={"benign": 0, "phishing": 1},
        classifier_dropout=training_config.CLASSIFIER_DROPOUT,
        hidden_dropout_prob=training_config.HIDDEN_DROPOUT_PROB,
        attention_probs_dropout_prob=training_config.ATTENTION_PROBS_DROPOUT_PROB,
    )

    datasets: dict[str, URLDataset] = {}
    row_counts: dict[str, int] = {}
    for split_name in SPLIT_FILES:
        texts, labels = load_split(args.data_dir, split_name)
        datasets[split_name] = tokenize_split(
            tokenizer, texts, labels, args.max_length
        )
        row_counts[split_name] = len(labels)
        print(f"Loaded {split_name}: {len(labels):,} URLs", flush=True)

    # Transformers renamed evaluation_strategy to eval_strategy in newer versions.
    eval_argument = (
        "eval_strategy"
        if "eval_strategy" in inspect.signature(TrainingArguments).parameters
        else "evaluation_strategy"
    )
    training_arguments = TrainingArguments(
        output_dir=str(args.training_out),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=training_config.WARMUP_RATIO,
        lr_scheduler_type=training_config.LR_SCHEDULER_TYPE,
        max_grad_norm=training_config.MAX_GRAD_NORM,
        adam_beta1=training_config.ADAM_BETA1,
        adam_beta2=training_config.ADAM_BETA2,
        adam_epsilon=training_config.ADAM_EPSILON,
        label_smoothing_factor=training_config.LABEL_SMOOTHING_FACTOR,
        gradient_checkpointing=training_config.GRADIENT_CHECKPOINTING,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        greater_is_better=True,
        save_total_limit=training_config.SAVE_TOTAL_LIMIT,
        fp16=args.fp16,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
        dataloader_num_workers=0,
        **{eval_argument: "epoch"},
    )
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    print(
        f"Training {args.model_name} on {row_counts['train']:,} URLs "
        f"for {args.epochs:g} epoch(s)...",
        flush=True,
    )
    train_result = trainer.train()
    validation_metrics = trainer.evaluate(
        eval_dataset=datasets["validation"], metric_key_prefix="validation"
    )
    test_metrics = trainer.evaluate(
        eval_dataset=datasets["test"], metric_key_prefix="test"
    )

    args.model_out.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.model_out))
    tokenizer.save_pretrained(str(args.model_out))

    results = {
        "model_name": args.model_name,
        "data_dir": str(args.data_dir.resolve()),
        "model_out": str(args.model_out.resolve()),
        "label_to_id": {"benign": 0, "phishing": 1},
        "max_length": args.max_length,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "weight_decay": args.weight_decay,
        "warmup_ratio": training_config.WARMUP_RATIO,
        "lr_scheduler_type": training_config.LR_SCHEDULER_TYPE,
        "classifier_dropout": training_config.CLASSIFIER_DROPOUT,
        "hidden_dropout_prob": training_config.HIDDEN_DROPOUT_PROB,
        "attention_probs_dropout_prob": training_config.ATTENTION_PROBS_DROPOUT_PROB,
        "gradient_checkpointing": training_config.GRADIENT_CHECKPOINTING,
        "fp16": args.fp16,
        "seed": args.seed,
        "rows": row_counts,
        "train": train_result.metrics,
        "validation": validation_metrics,
        "test": test_metrics,
    }
    args.metrics_out.write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=float) + "\n",
        encoding="utf-8",
    )

    print(f"Model and tokenizer saved to: {args.model_out}")
    print(f"Metrics saved to: {args.metrics_out}")
    print("Test metrics:")
    for name, value in test_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {name}: {value:.4f}")


if __name__ == "__main__":
    main()


