"""Recompute scores before publishing a trusted XGBoost artifact."""
from importlib.metadata import version
from io import BytesIO
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, make_scorer, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from xgboost import XGBClassifier

if __package__:
    from .export_release import assert_scores, read_json, sha256
    from .train_xgboost import COLUMNS, PARAMETERS, specificity
else:
    from export_release import assert_scores, read_json, sha256
    from train_xgboost import COLUMNS, PARAMETERS, specificity


def verify_snapshot(snapshot, dataset):
    metadata = read_json(snapshot["metadata.json"])
    report = read_json(snapshot["test_evaluation.json"])
    dataset_hash, artifact_hash = sha256(dataset), sha256(snapshot["pipeline.joblib"])
    for record in (metadata, report):
        if (record.get("schema_version") != 1 or record.get("model") != "XGBClassifier"
                or record.get("dataset_sha256") != dataset_hash
                or record.get("artifact_sha256") != artifact_hash):
            raise ValueError("XGBoost schema or provenance mismatch")
    if (metadata.get("dataset") != "Cleveland" or report.get("dataset") != "Cleveland"
            or metadata.get("features") != COLUMNS[:-1]
            or metadata.get("target", {}).get("name") != "num"
            or metadata.get("split") != {"test_size": .2, "random_state": 42, "stratify": True}
            or metadata.get("cv_folds") != 5 or metadata.get("cv_random_state") != 42
            or report.get("threshold") != .5):
        raise ValueError("XGBoost protocol mismatch")
    versions = {"numpy": "numpy", "scipy": "scipy", "sklearn": "scikit-learn", "joblib": "joblib", "xgboost": "xgboost-cpu"}
    if metadata.get("python_version") != sys.version.split()[0]:
        raise ValueError("Python version mismatch")
    for key, distribution in versions.items():
        if metadata.get(key + "_version") != version(distribution):
            raise ValueError(f"Version mismatch: {key}")
    pipeline = joblib.load(BytesIO(snapshot["pipeline.joblib"]))
    classifier = pipeline.named_steps.get("classifier")
    if (not isinstance(classifier, XGBClassifier)
            or list(pipeline.feature_names_in_) != COLUMNS[:-1]
            or not np.array_equal(classifier.classes_, [0, 1])):
        raise ValueError("XGBoost object mismatch")
    if metadata.get("parameters") != PARAMETERS or any(classifier.get_params().get(k) != v for k, v in PARAMETERS.items()):
        raise ValueError("XGBoost parameters mismatch")
    df = pd.read_csv(BytesIO(dataset), header=None, names=COLUMNS, na_values="?")
    X, y = df.drop(columns="num"), (df.num > 0).astype(int)
    if len(df) != 303:
        raise ValueError("Expected Cleveland 303")
    xt, xv, yt, yv = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    if metadata.get("train_samples") != len(xt) or metadata.get("test_samples") != len(xv) or report.get("test_samples") != len(xv):
        raise ValueError("XGBoost sample count mismatch")
    prediction = pipeline.predict(xv)
    probability = pipeline.predict_proba(xv)[:, list(classifier.classes_).index(1)]
    names = ["true_negative", "false_positive", "false_negative", "true_positive"]
    matrix = dict(zip(names, map(int, confusion_matrix(yv, prediction, labels=[0, 1]).ravel())))
    if report.get("confusion_matrix") != matrix:
        raise ValueError("XGBoost confusion matrix mismatch")
    metrics = dict(accuracy=accuracy_score(yv, prediction), precision=precision_score(yv, prediction, zero_division=0),
                   recall=recall_score(yv, prediction, zero_division=0), specificity=specificity(yv, prediction),
                   f1=f1_score(yv, prediction, zero_division=0), roc_auc=roc_auc_score(yv, probability))
    assert_scores(report["metrics"], metrics)
    scoring = {"accuracy": "accuracy", "precision": make_scorer(precision_score, zero_division=0),
               "recall": make_scorer(recall_score, zero_division=0), "specificity": make_scorer(specificity),
               "f1": make_scorer(f1_score, zero_division=0), "roc_auc": "roc_auc"}
    cv = cross_validate(clone(pipeline), xt, yt, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring=scoring, error_score="raise")
    cv_metrics = {name: float(cv["test_" + name].mean()) for name in scoring}
    assert_scores(metadata["cv_metrics"], cv_metrics)
    for name in scoring:
        details = metadata["cv_details"][name]
        if (not np.allclose(details["fold_scores"], cv["test_" + name], rtol=0, atol=1e-8)
                or not np.isclose(details["std"], cv["test_" + name].std(), rtol=0, atol=1e-8)):
            raise ValueError("XGBoost fold scores mismatch")
    return {"dataset_sha256": dataset_hash, "train_samples": len(xt), "test_samples": len(xv),
            "cv_folds": 5, "cv_metrics": metadata["cv_metrics"],
            "test_metrics": report["metrics"], "confusion_matrix": matrix}
