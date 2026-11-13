"""Integration test of training and held-out evaluation on real PhishVN data."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "src" / "randomForest"
DATA_DIR = PROJECT_ROOT / "data" / "PhishVN_v3.1.0_open" / "data" / "splits"
sys.path.insert(0, str(SOURCE_DIR))

import evaluate_random_forest as evaluator  # noqa: E402
import train_random_forest as trainer  # noqa: E402


REQUIRED_SPLITS = [DATA_DIR / filename for filename in trainer.SPLIT_FILES.values()]


@unittest.skipUnless(
    all(path.is_file() for path in REQUIRED_SPLITS),
    "The real PhishVN split CSV files are required for this integration test.",
)
class RealDatasetTrainingTests(unittest.TestCase):
    def test_training_and_later_test_evaluation(self) -> None:
        training_splits = trainer.load_splits(DATA_DIR, split_names=("train", "val"))
        expected_rows = {
            name: len(labels) for name, (_, labels) in training_splits.items()
        }

        for split_name, (_, labels) in training_splits.items():
            with self.subTest(split=split_name):
                self.assertGreater(len(labels), 0)
                self.assertEqual(set(labels.tolist()), {0, 1})

        with tempfile.TemporaryDirectory(prefix="random_forest_test_") as temp_dir:
            output_dir = Path(temp_dir)
            model_path = output_dir / "random_forest.joblib"
            training_metrics_path = output_dir / "training_metrics.json"
            importance_path = output_dir / "feature_importance.csv"
            test_metrics_path = output_dir / "test_metrics.json"

            training_arguments = [
                "train_random_forest.py",
                "--data-dir",
                str(DATA_DIR),
                "--model-out",
                str(model_path),
                "--metrics-out",
                str(training_metrics_path),
                "--importance-out",
                str(importance_path),
                "--n-estimators",
                "10",
                "--max-depth",
                "12",
                "--progress-every",
                "5",
                "--decision-threshold",
                "0.5",
            ]
            real_random_forest = trainer.RandomForestClassifier

            def single_worker_forest(**kwargs):
                kwargs["n_jobs"] = 1
                return real_random_forest(**kwargs)

            with patch.object(
                trainer,
                "RandomForestClassifier",
                side_effect=single_worker_forest,
            ), patch.object(sys, "argv", training_arguments):
                trainer.main()

            self.assertTrue(model_path.is_file())
            self.assertTrue(training_metrics_path.is_file())
            self.assertTrue(importance_path.is_file())

            with training_metrics_path.open(encoding="utf-8") as metrics_file:
                training_metrics = json.load(metrics_file)
            self.assertEqual(
                set(training_metrics["final_metrics"]), {"train", "val"}
            )
            self.assertEqual(len(training_metrics["training_history"]), 2)
            self.assertEqual(
                [item["tree_count"] for item in training_metrics["training_history"]],
                [5, 10],
            )
            for split_name, row_count in expected_rows.items():
                self.assertEqual(
                    training_metrics["final_metrics"][split_name]["rows"], row_count
                )
                self.assertIn(
                    "log_loss", training_metrics["final_metrics"][split_name]
                )

            artifact = joblib.load(model_path)
            self.assertEqual(artifact["model"].named_steps["classifier"].n_estimators, 10)

            test_arguments = [
                "evaluate_random_forest.py",
                "--data-dir",
                str(DATA_DIR),
                "--model-in",
                str(model_path),
                "--metrics-out",
                str(test_metrics_path),
            ]
            with patch.object(sys, "argv", test_arguments):
                evaluator.main()

            with test_metrics_path.open(encoding="utf-8") as metrics_file:
                test_report = json.load(metrics_file)
            self.assertEqual(test_report["split"], "test")
            self.assertEqual(test_report["metrics"]["class_counts"]["benign"] > 0, True)
            self.assertEqual(test_report["metrics"]["class_counts"]["phishing"] > 0, True)
            self.assertIn("log_loss", test_report["metrics"])


if __name__ == "__main__":
    unittest.main()