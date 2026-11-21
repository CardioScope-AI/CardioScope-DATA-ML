"""Évaluer le modèle final sur le test Cleveland réservé."""

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "processed.cleveland.data"
MODEL_PATH = BASE_DIR / "artifacts" / "pipeline.joblib"
METADATA_PATH = BASE_DIR / "artifacts" / "metadata.json"
REPORT_PATH = BASE_DIR / "artifacts" / "test_evaluation.json"

SCHEMA_VERSION = 1

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

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}. Exécute d'abord scripts/train.py."
        )

    # Calculer l'empreinte du dataset utilisé pour l'évaluation.
    dataset_sha256 = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()

    df = pd.read_csv(
        DATA_PATH,
        header=None,
        names=COLUMNS,
        na_values="?",
    )

    X = df.drop(columns=["num"])
    y = (df["num"] > 0).astype(int)

    with METADATA_PATH.open(encoding="utf-8") as file:
        metadata = json.load(file)

    # Vérifier le schéma, le dataset et le protocole de séparation.
    if metadata["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            "Version de schéma des métadonnées non prise en charge."
        )

    if metadata["dataset_sha256"] != dataset_sha256:
        raise ValueError(
            "Le dataset ne correspond pas à celui des métadonnées."
        )

    split = metadata["split"]
    if (
        split["test_size"] != 0.2
        or split["random_state"] != 42
        or split["stratify"] is not True
    ):
        raise ValueError(
            "Le découpage du dataset ne correspond pas au protocole attendu."
        )

    # Reproduire la séparation utilisée par train.py.
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=split["test_size"],
        random_state=split["random_state"],
        stratify=y if split["stratify"] else None,
    )

    if list(X_test.columns) != metadata["features"]:
        raise ValueError(
            "Les variables du test ne correspondent pas au modèle."
        )

    if len(X_test) != metadata["test_samples"]:
        raise ValueError(
            "La taille du test ne correspond pas aux métadonnées."
        )

    # Charger le modèle déjà entraîné, sans refaire d'entraînement.
    pipeline = joblib.load(MODEL_PATH)
    y_pred = pipeline.predict(X_test)

    classes = list(pipeline.named_steps["classifier"].classes_)
    positive_index = classes.index(1)
    y_proba = pipeline.predict_proba(X_test)[:, positive_index]

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(
            float(precision_score(y_test, y_pred, zero_division=0)), 4
        ),
        "recall": round(
            float(recall_score(y_test, y_pred, zero_division=0)), 4
        ),
        "specificity": round(float(tn / (tn + fp)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
    }

    report = {
        "schema_version": SCHEMA_VERSION,
        "dataset_sha256": dataset_sha256,
        "dataset": "Cleveland",
        "test_samples": len(y_test),
        "positive_class": "1 = présence de pathologie",
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "metrics": metrics,
    }

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
            allow_nan=False,
        )
        file.write("\n")

    print(f"Test réservé : {len(y_test)} profils")
    print("Matrice de confusion :")
    print("Lignes = classe réelle 0/1 ; colonnes = classe prédite 0/1")
    print(f"[[{tn}, {fp}], [{fn}, {tp}]]")

    for name, value in metrics.items():
        print(f"{name} : {value:.4f}")

    print(
        "Rapport enregistré dans : "
        f"{REPORT_PATH.relative_to(BASE_DIR)}"
    )


if __name__ == "__main__":
    main()