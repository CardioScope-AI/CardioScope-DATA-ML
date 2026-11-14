
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from pipeline import build_pipeline

# Chemin vers le dataset Cleveland
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "processed.cleveland.data"

# Noms des 14 colonnes
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "num"
]

# Chargement des données
df = pd.read_csv(
    DATA_PATH,
    header=None,
    names=COLUMNS,
    na_values="?"
)

print("Dimensions du dataset :", df.shape)

# Transformer la cible en binaire
# 0 = absence de maladie, 1 = présence de maladie
X = df.drop(columns=["num"])
y = (df["num"] > 0).astype(int)

# Séparer les données : 80 % entraînement, 20 % test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Observations d'entraînement :", len(X_train))
print("Observations de test :", len(X_test))

# Créer le pipeline avec Random Forest
pipeline = build_pipeline()

# Préparer la validation croisée stratifiée à 5 plis
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("Pipeline Random Forest prêt.")
print("Validation croisée à 5 plis prête.")

# Évaluer le Random Forest avec la validation croisée
scores = cross_validate(
    pipeline,
    X_train,
    y_train,
    cv=cv,
    scoring={
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }
)

# Afficher les performances moyennes
for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    moyenne = scores["test_" + metric].mean()
    print(f"{metric} : {moyenne:.4f}")
# Entraîner le modèle final sur toutes les données d'entraînement
pipeline.fit(X_train, y_train)

print("Entraînement final terminé.")
# Enregistrer le modèle entraîné
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(pipeline, ARTIFACTS_DIR / "pipeline.joblib")

print("Modèle enregistré dans artifacts/pipeline.joblib")

# Enregistrer les métadonnées du modèle
metadata = {
    "model": "RandomForestClassifier",
    "n_estimators": 100,
    "random_state": 42,
    "dataset": "Cleveland",
    "features": list(X.columns),
    "target": {
        "name": "num",
        "definition": "0 si num vaut 0 (absence de pathologie), 1 si num est supérieur à 0 (présence de pathologie)"
    },
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "cv_folds": 5,
    "cv_metrics": {
        metric: round(float(scores["test_" + metric].mean()), 4)
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]
    },
    "python_version": sys.version.split()[0],
    "numpy_version": np.__version__,
    "scipy_version": scipy.__version__,
    "sklearn_version": sklearn.__version__,
    "joblib_version": joblib.__version__
}

with open(ARTIFACTS_DIR / "metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)

print("Métadonnées enregistrées.")
