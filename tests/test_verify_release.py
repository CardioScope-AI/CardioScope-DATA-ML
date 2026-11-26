"""Tests for release integrity and cross-artifact consistency checks."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.verify_release import ROOT, validate_release


FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
CV_METRICS = {
    "accuracy": 0.8, "precision": 0.8, "recall": 0.8,
    "f1": 0.8, "roc_auc": 0.9,
}
TEST_METRICS = {
    **CV_METRICS, "specificity": 0.75,
}
CONFUSION = {
    "true_negative": 3, "false_positive": 1,
    "false_negative": 1, "true_positive": 5,
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class VerifyReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.root = Path(self.temp.name)
        self.release = self.root / "rf-cleveland-v1"
        self.release.mkdir()
        self.dataset = self.root / "processed.cleveland.data"
        dataset_bytes = b"fictional test dataset fixture\n"
        self.dataset.write_bytes(dataset_bytes)
        dataset_hash = sha256(dataset_bytes)

        self.files = {
            "pipeline.joblib": b"fictional joblib fixture, never loaded by this checker",
            "metadata.json": {
                "schema_version": 1,
                "dataset_sha256": dataset_hash,
                "split": {"test_size": 0.2, "random_state": 42, "stratify": True},
                "model": "RandomForestClassifier",
                "dataset": "Cleveland",
                "features": FEATURES,
                "target": {"name": "num"},
                "train_samples": 8,
                "test_samples": 10,
                "cv_folds": 5,
                "cv_metrics": CV_METRICS,
            },
            "test_evaluation.json": {
                "schema_version": 1,
                "dataset_sha256": dataset_hash,
                "dataset": "Cleveland",
                "test_samples": 10,
                "metrics": TEST_METRICS,
                "confusion_matrix": CONFUSION,
            },
            "verification.json": {
                "protocol": "Cleveland/80-20-stratified/random42/CV5",
                "dataset_sha256": dataset_hash,
                "train_samples": 8,
                "test_samples": 10,
                "cv_folds": 5,
                "cv_metrics": CV_METRICS,
                "test_metrics": TEST_METRICS,
                "confusion_matrix": CONFUSION,
            },
        }
        for name, value in self.files.items():
            data = value if isinstance(value, bytes) else (
                json.dumps(value, indent=2, ensure_ascii=False) + "\n"
            ).encode("utf-8")
            (self.release / name).write_bytes(data)

        manifest = {
            "schema_version": 1,
            "model_version": "rf-cleveland-v1",
            "artifact_sha256": sha256((self.release / "pipeline.joblib").read_bytes()),
            "metadata_sha256": sha256((self.release / "metadata.json").read_bytes()),
            "test_evaluation_sha256": sha256((self.release / "test_evaluation.json").read_bytes()),
            "limitations": ["Fixture pédagogique"],
        }
        (self.release / "release.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_accepts_matching_release_and_dataset(self):
        result = validate_release(self.release, self.dataset)
        self.assertEqual(result["model_version"], "rf-cleveland-v1")
        self.assertTrue(result["dataset_checked"])

    def test_rejects_missing_pipeline(self):
        (self.release / "pipeline.joblib").unlink()
        with self.assertRaisesRegex(ValueError, "pipeline.joblib"):
            validate_release(self.release, self.dataset)

    def test_rejects_artifact_hash_mismatch(self):
        (self.release / "pipeline.joblib").write_bytes(b"different bytes")
        with self.assertRaisesRegex(ValueError, "Empreinte SHA-256"):
            validate_release(self.release, self.dataset)

    def test_rejects_metadata_report_disagreement(self):
        report_path = self.release / "test_evaluation.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["test_samples"] = 9
        report_path.write_text(json.dumps(report), encoding="utf-8")
        manifest_path = self.release / "release.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["test_evaluation_sha256"] = sha256(report_path.read_bytes())
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Effectifs"):
            validate_release(self.release, self.dataset)

    def test_rejects_dataset_mismatch(self):
        self.dataset.write_bytes(b"different dataset fixture\n")
        with self.assertRaisesRegex(ValueError, "empreinte du jeu de données"):
            validate_release(self.release, self.dataset)


if __name__ == "__main__":
    unittest.main()
