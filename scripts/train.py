
"""Entraîner le modèle Cleveland et enregistrer le pipeline et ses métadonnées."""

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from pipeline import build_pipeline


# Chemins du projet et du dataset Cleveland
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "processed.cleveland.data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Paramètres du découpage et de la validation croisée
TEST_SIZE = 0.20
SPLIT_RANDOM_STATE = 42
CV_FOLDS = 5
CV_RANDOM_STATE = 42

# Noms des 14 colonnes du dataset
COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Données introuvables : {DATA_PATH}")

    # Calculer l'empreinte du fichier source pour identifier exactement le dataset.
    dataset_sha256 = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()

    # Charger les données ; "?" représente une valeur manquante.
    df = pd.read_csv(
        DATA_PATH,
        header=None,
        names=COLUMNS,
        na_values="?",
    )

    print("Dimensions du dataset :", df.shape)

    # Transformer la cible en binaire :
    # 0 = absence de maladie, 1 = présence de maladie.
    X = df.drop(columns=["num"])
    y = (df["num"] > 0).astype(int)

    # Séparer les données en entraînement et test.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SPLIT_RANDOM_STATE,
        stratify=y,
    )

    print("Observations d'entraînement :", len(X_train))
    print("Observations de test :", len(X_test))

    # Construire le pipeline du modèle.
    pipeline = build_pipeline()

    # Préparer la validation croisée stratifiée.
    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=CV_RANDOM_STATE,
    )

    # Calculer les métriques de validation croisée sur les données d'entraînement.
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        error_score="raise",
    )

    cv_metrics = {
        metric: round(float(scores[f"test_{metric}"].mean()), 4)
        for metric in ("accuracy", "precision", "recall", "f1", "roc_auc")
    }

    print("Métriques moyennes de validation croisée :")
    for metric, value in cv_metrics.items():
        print(f"{metric} : {value:.4f}")

    # Entraîner le pipeline final sur toutes les données d'entraînement.
    pipeline.fit(X_train, y_train)
    print("Entraînement final terminé.")

    # Enregistrer le modèle entraîné.
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / "pipeline.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Modèle enregistré dans {model_path.relative_to(BASE_DIR)}")

    # Enregistrer les métadonnées. Les clés existantes sont conservées
    # pour rester compatibles avec le backend.
    metadata = {
        "schema_version": 1,
        "dataset_sha256": dataset_sha256,
        "split": {
            "test_size": TEST_SIZE,
            "random_state": SPLIT_RANDOM_STATE,
            "stratify": True,
        },
        "model": "RandomForestClassifier",
        "n_estimators": 100,
        "random_state": 42,
        "dataset": "Cleveland",
        "features": list(X.columns),
        "target": {
            "name": "num",
            "definition": (
                "0 si num vaut 0 (absence de pathologie), "
                "1 si num est supérieur à 0 (présence de pathologie)"
            ),
        },
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "cv_folds": CV_FOLDS,
        "cv_metrics": cv_metrics,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "joblib_version": joblib.__version__,
    }

    metadata_path = ARTIFACTS_DIR / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        )
        file.write("\n")

    print(f"Métadonnées enregistrées dans {metadata_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()