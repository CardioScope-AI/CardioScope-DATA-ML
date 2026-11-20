"""Publish verified, immutable model snapshots without replacing the trained model."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path
import re
import shutil
import tempfile

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
FILES = ("pipeline.joblib", "metadata.json", "test_evaluation.json")
LIMITATIONS = [
    "Démonstrateur pédagogique ; ceci ne constitue pas un diagnostic médical.",
    "Utiliser uniquement des profils fictifs.",
    "Dataset historique Cleveland de 303 observations ; absence de validation externe démontrée.",
    "La classe 0 ou 1 ne prédit pas un risque à 5 ou 10 ans ; des erreurs sont possibles.",
    "Probabilités non calibrées et performances par sous-groupe non établies.",
    "Le modèle ne se réentraîne pas automatiquement à partir des nouvelles données ou des simulations enregistrées.",
]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def read_json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=unique)


def assert_scores(reported, actual):
    if set(reported) != set(actual):
        raise ValueError("Missing or unexpected metrics")
    for name, expected in actual.items():
        value = reported[name]
        if (isinstance(value, bool) or not isinstance(value, (float, int))
                or not np.isfinite(value) or not 0 <= value <= 1
                or not np.isclose(value, expected, atol=0.00005, rtol=0)):
            raise ValueError(f"Metric mismatch for {name}: reported={value}, recomputed={expected}")


def verify_snapshot(snapshot, dataset):
    """Check saved predictions and recompute CV on fresh clones, never refit the saved object."""
    metadata = read_json(snapshot["metadata.json"])
    report = read_json(snapshot["test_evaluation.json"])
    packages = {"numpy_version": "numpy", "scipy_version": "scipy",
                "sklearn_version": "scikit-learn", "joblib_version": "joblib"}
    for field, package in packages.items():
        if metadata[field] != version(package):
            raise ValueError(f"Use the pinned training environment: {package}")
    if (metadata["model"] != "RandomForestClassifier" or metadata["dataset"] != "Cleveland"
            or report["dataset"] != "Cleveland" or metadata["features"] != FEATURES
            or metadata["target"]["name"] != "num" or metadata["cv_folds"] != 5):
        raise ValueError("Unsupported metadata schema or training protocol")
    frame = pd.read_csv(BytesIO(dataset), header=None, names=FEATURES + ["num"], na_values="?")
    if frame.shape != (303, 14) or not frame["num"].isin([0, 1, 2, 3, 4]).all():
        raise ValueError("Expected the original Cleveland dataset and valid target")
    X, y = frame.drop(columns="num"), (frame["num"] > 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    if (metadata["train_samples"] != len(X_train) or metadata["test_samples"] != len(X_test)
            or report["test_samples"] != len(X_test)):
        raise ValueError("Inconsistent split counts")
    pipeline = joblib.load(BytesIO(snapshot["pipeline.joblib"]))
    classifier = pipeline.named_steps["classifier"]
    if (not isinstance(classifier, RandomForestClassifier)
            or list(pipeline.feature_names_in_) != FEATURES
            or not np.array_equal(classifier.classes_, [0, 1])
            or classifier.n_estimators != metadata["n_estimators"]
            or classifier.random_state != metadata["random_state"]):
        raise ValueError("Pipeline does not match metadata")
    prediction = pipeline.predict(X_test)
    probability = pipeline.predict_proba(X_test)[:, list(classifier.classes_).index(1)]
    tn, fp, fn, tp = confusion_matrix(y_test, prediction, labels=[0, 1]).ravel()
    matrix = dict(zip(("true_negative", "false_positive", "false_negative", "true_positive"),
                      map(int, (tn, fp, fn, tp))))
    if (report["confusion_matrix"] != matrix
            or any(type(n) is not int for n in report["confusion_matrix"].values())):
        raise ValueError("Confusion matrix does not match saved model")
    metrics = dict(accuracy=round(float(accuracy_score(y_test, prediction)), 4),
                   precision=round(float(precision_score(y_test, prediction, zero_division=0)), 4),
                   recall=round(float(recall_score(y_test, prediction, zero_division=0)), 4),
                   specificity=round(float(tn / (tn + fp)), 4),
                   f1=round(float(f1_score(y_test, prediction, zero_division=0)), 4),
                   roc_auc=round(float(roc_auc_score(y_test, probability)), 4))
    assert_scores(report["metrics"], metrics)
    cv = cross_validate(clone(pipeline), X_train, y_train,
                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                        scoring=["accuracy", "precision", "recall", "f1", "roc_auc"], error_score="raise")
    cv_metrics = {name: round(float(cv["test_" + name].mean()), 4)
                  for name in ("accuracy", "precision", "recall", "f1", "roc_auc")}
    assert_scores(metadata["cv_metrics"], cv_metrics)
    return {"dataset_sha256": sha256(dataset), "train_samples": len(X_train),
            "test_samples": len(X_test), "cv_folds": 5,
            "cv_metrics": cv_metrics, "test_metrics": metrics, "confusion_matrix": matrix}


def export_release(source, dataset_path, output, model_version):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", model_version):
        raise ValueError("Version must contain 1-100 ASCII letters, digits, dots, underscores or hyphens")
    destination = output / model_version
    if destination.exists():
        raise FileExistsError("Release already exists; choose a new version, never overwrite")
    snapshot = {name: (source / name).read_bytes() for name in FILES}
    dataset = dataset_path.read_bytes()
    verification = verify_snapshot(snapshot, dataset)
    manifest = dict(schema_version=1, model_version=model_version,
                    artifact_sha256=sha256(snapshot["pipeline.joblib"]),
                    metadata_sha256=sha256(snapshot["metadata.json"]),
                    test_evaluation_sha256=sha256(snapshot["test_evaluation.json"]), limitations=LIMITATIONS)
    output.mkdir(parents=True, exist_ok=True)
    # Stage on the same filesystem. Publish only a complete, validated directory.
    staging = Path(tempfile.mkdtemp(prefix=".release-", dir=output))
    try:
        for name, data in snapshot.items():
            (staging / name).write_bytes(data)
        (staging / "release.json").write_bytes(json_bytes(manifest))
        (staging / "verification.json").write_bytes(json_bytes({
            "verified_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "protocol": "Cleveland/80-20-stratified/random42/CV5", **verification,
        }))
        lock = output / ("." + model_version + ".lock")
        # Exclusive publication lock coordinates concurrent exporters on Windows/Linux.
        lock.open("x").close()
        try:
            if destination.exists():
                raise FileExistsError("Release already exists; never overwrite")
            staging.rename(destination)
        finally:
            lock.unlink()
    finally:
        if staging.exists():
            if staging.resolve().parent != output.resolve() or not staging.name.startswith(".release-"):
                raise ValueError("Refusing cleanup outside release staging directory")
            shutil.rmtree(staging)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/raw/processed.cleveland.data")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/releases")
    args = parser.parse_args()
    try:
        destination = export_release(args.source, args.dataset, args.output, args.version)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        parser.exit(1, f"Release not published: {exc}\n")
    print(f"Verified release published: {destination}")


if __name__ == "__main__":
    main()
