"""Entraîner et évaluer une référence XGBoost séparée du Random Forest."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, make_scorer,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from xgboost import XGBClassifier

if __package__:
    from .pipeline import build_pipeline
    from .train import BASE_DIR, DATA_PATH, COLUMNS, TEST_SIZE, SPLIT_RANDOM_STATE, CV_FOLDS, CV_RANDOM_STATE
else:
    from pipeline import build_pipeline
    from train import BASE_DIR, DATA_PATH, COLUMNS, TEST_SIZE, SPLIT_RANDOM_STATE, CV_FOLDS, CV_RANDOM_STATE


PARAMETERS = dict(
    n_estimators=100, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=1,
    reg_lambda=1.0, objective="binary:logistic", eval_metric="logloss",
    tree_method="hist", device="cpu", n_jobs=1, random_state=42,
)


def build_xgboost_pipeline():
    """Imputation et encodage appris dans chaque groupe, sans standardisation."""
    return build_pipeline(XGBClassifier(**PARAMETERS))


def specificity(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if tn + fp else 0.0


def run(output_dir):
    output_dir = Path(output_dir).resolve()
    # Refuser un dossier existant avant tout entraînement : aucun export écrasé.
    if output_dir.exists():
        raise FileExistsError(f"Choisir un nouveau dossier de sortie : {output_dir}")
    df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS, na_values="?")
    X, y = df.drop(columns="num"), (df["num"] > 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SPLIT_RANDOM_STATE, stratify=y,
    )
    pipeline = build_xgboost_pipeline()
    scoring = {
        "accuracy": "accuracy", "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "specificity": make_scorer(specificity),
        "f1": make_scorer(f1_score, zero_division=0), "roc_auc": "roc_auc",
    }
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=CV_RANDOM_STATE)
    scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, error_score="raise")
    pipeline.fit(X_train, y_train)
    prediction = pipeline.predict(X_test)
    positive_index = list(pipeline.named_steps["classifier"].classes_).index(1)
    probability = pipeline.predict_proba(X_test)[:, positive_index]
    tn, fp, fn, tp = confusion_matrix(y_test, prediction, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": accuracy_score(y_test, prediction),
        "precision": precision_score(y_test, prediction, zero_division=0),
        "recall": recall_score(y_test, prediction, zero_division=0),
        "specificity": specificity(y_test, prediction),
        "f1": f1_score(y_test, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probability),
    }
    dataset_sha256 = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    metadata = {
        "schema_version": 1, "model": "XGBClassifier", "parameters": PARAMETERS,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "Cleveland", "dataset_sha256": dataset_sha256,
        "features": list(X.columns), "target": {"name": "num", "definition": "0 si num == 0 ; 1 si num > 0"},
        "split": {"test_size": TEST_SIZE, "random_state": SPLIT_RANDOM_STATE, "stratify": True},
        "train_samples": len(X_train), "test_samples": len(X_test),
        "cv_folds": CV_FOLDS, "cv_random_state": CV_RANDOM_STATE,
        "cv_metrics": {name: round(float(scores[f"test_{name}"].mean()), 4) for name in scoring},
        "cv_details": {name: {"fold_scores": scores[f"test_{name}"].tolist(),
                              "std": float(scores[f"test_{name}"].std())} for name in scoring},
        "python_version": sys.version.split()[0], "numpy_version": np.__version__,
        "scipy_version": scipy.__version__, "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__, "xgboost_version": xgboost.__version__,
        "selection": "Paramètres fixes, sans recherche ni réglage sur le test réservé.",
    }
    # Calculs terminés avant de créer le dossier exclusif de sortie.
    output_dir.mkdir(parents=True, exist_ok=False)
    model_path = output_dir / "pipeline.joblib"
    joblib.dump(pipeline, model_path)
    metadata["artifact_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    report = {
        "schema_version": 1, "model": "XGBClassifier", "dataset_sha256": dataset_sha256,
        "artifact_sha256": metadata["artifact_sha256"], "test_samples": len(y_test),
        "positive_class": "1 = présence de pathologie", "threshold": 0.5,
        "confusion_matrix": dict(zip(
            ["true_negative", "false_positive", "false_negative", "true_positive"],
            map(int, [tn, fp, fn, tp]),
        )),
        "metrics": {name: round(float(value), 4) for name, value in metrics.items()},
    }
    for name, content in [("metadata.json", metadata), ("test_evaluation.json", report)]:
        (output_dir / name).write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"cv": metadata["cv_metrics"], "test": report["metrics"],
                      "confusion_matrix": report["confusion_matrix"]}, indent=2))
    print(f"Exports XGBoost : {output_dir}")
    return pipeline, metadata, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "artifacts" / "xgboost")
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
