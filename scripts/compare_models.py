"""Comparer Random Forest et régression logistique en validation croisée."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

if __package__:
    from .pipeline import build_logistic_regression_pipeline, build_pipeline
else:
    from pipeline import build_logistic_regression_pipeline, build_pipeline


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "processed.cleveland.data"
DEFAULT_OUTPUT = BASE_DIR / "artifacts" / "model_comparison.json"
TEST_SIZE = 0.20
SPLIT_RANDOM_STATE = 42
CV_FOLDS = 5
CV_RANDOM_STATE = 42

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def compare_models(dataset_path=DATA_PATH, output_path=DEFAULT_OUTPUT):
    """Comparer les pipelines sur les mêmes plis du seul jeu d'entraînement.

    Le jeu de test réservé est isolé conformément au protocole du projet, mais
    n'est ni utilisé pour choisir le modèle ni inclus dans cette comparaison.
    """
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Données introuvables : {dataset_path}")

    raw_bytes = dataset_path.read_bytes()
    df = pd.read_csv(dataset_path, header=None, names=COLUMNS, na_values="?")
    X = df.drop(columns=["num"])
    y = (df["num"] > 0).astype(int)
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SPLIT_RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=CV_RANDOM_STATE,
    )
    # Materialize once so both models see the exact same examples in each fold.
    folds = list(cv.split(X_train, y_train))
    models = {
        "RandomForestClassifier": (
            build_pipeline(),
            {"n_estimators": 100, "random_state": 42},
        ),
        "LogisticRegression": (
            build_logistic_regression_pipeline(),
            {"C": 1.0, "max_iter": 1000, "random_state": 42},
        ),
    }

    results = {}
    for model_name, (pipeline, parameters) in models.items():
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=folds,
            scoring=SCORING,
            error_score="raise",
        )
        results[model_name] = {
            "parameters": parameters,
            "metrics": {
                metric: {
                    "mean": round(float(scores[f"test_{metric}"].mean()), 4),
                    "std": round(float(scores[f"test_{metric}"].std()), 4),
                }
                for metric in SCORING
            },
        }

    report = {
        "schema_version": 1,
        "dataset": "Cleveland",
        "dataset_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "split": {
            "test_size": TEST_SIZE,
            "random_state": SPLIT_RANDOM_STATE,
            "stratify": True,
            "train_samples": len(X_train),
            "test_samples": len(X) - len(X_train),
        },
        "cross_validation": {
            "folds": CV_FOLDS,
            "random_state": CV_RANDOM_STATE,
            "stratified": True,
            "metrics": results,
        },
        "test_set_used_for_comparison": False,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    for model_name, model_result in results.items():
        summary = ", ".join(
            f"{name}={values['mean']:.4f} +/- {values['std']:.4f}"
            for name, values in model_result["metrics"].items()
        )
        print(f"{model_name}: {summary}")
    print(f"Rapport écrit dans {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Chemin du rapport JSON (défaut : {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()
    compare_models(output_path=args.output)


if __name__ == "__main__":
    main()
