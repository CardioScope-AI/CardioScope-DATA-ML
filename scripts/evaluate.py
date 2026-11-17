"""Évaluer le modèle final sur le test Cleveland réservé."""

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

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Données introuvables : {DATA_PATH}")
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}. Exécute d'abord scripts/train.py."
        )

    df = pd.read_csv(
        DATA_PATH,
        header=None,
        names=COLUMNS,
        na_values="?",
    )

    X = df.drop(columns=["num"])
    y = (df["num"] > 0).astype(int)

    # Reproduire exactement la séparation utilisée par train.py.
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    with METADATA_PATH.open(encoding="utf-8") as file:
        metadata = json.load(file)

    if list(X_test.columns) != metadata["features"]:
        raise ValueError("Les variables du test ne correspondent pas au modèle.")
    if len(X_test) != metadata["test_samples"]:
        raise ValueError("La taille du test ne correspond pas aux métadonnées.")

    # Charger le modèle déjà entraîné, sans refaire d'entraînement.
    pipeline = joblib.load(MODEL_PATH)
    y_pred = pipeline.predict(X_test)

    classes = list(pipeline.named_steps["classifier"].classes_)
    positive_index = classes.index(1)
    y_proba = pipeline.predict_proba(X_test)[:, positive_index]

    tn, fp, fn, tp = confusion_matrix(
        y_test, y_pred, labels=[0, 1]
    ).ravel()

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "specificity": round(tn / (tn + fp), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }

    report = {
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
        json.dump(report, file, indent=4, ensure_ascii=False)

    print(f"Test réservé : {len(y_test)} profils")
    print("Matrice de confusion :")
    print("Lignes = classe réelle 0/1 ; colonnes = classe prédite 0/1")
    print(f"[[{tn}, {fp}], [{fn}, {tp}]]")
    for name, value in metrics.items():
        print(f"{name} : {value:.4f}")
    print(f"Rapport enregistré dans : {REPORT_PATH.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()