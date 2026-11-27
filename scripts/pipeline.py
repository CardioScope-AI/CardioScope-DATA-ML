"""Construction du pipeline Cleveland, sans entraînement à l'import."""

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = ("age", "trestbps", "chol", "thalach", "oldpeak")
CATEGORICAL_FEATURES = (
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal",
)


def build_pipeline(estimator=None, *, scale_numeric=False):
    """Créer un pipeline neuf, à ajuster uniquement sur X_train et y_train.

    Les entrées sont un DataFrame avec les 13 colonnes UCI, codes originaux
    et valeurs manquantes représentées par NaN. La cible num en est exclue.
    Random Forest est le modèle retenu, avec des paramètres initiaux à évaluer.
    Un autre estimateur Scikit-learn peut être fourni. Activer scale_numeric
    pour les modèles sensibles à l'échelle, comme la régression logistique.
    Aucun fichier de modèle n'est écrit par cette fonction.
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
        ("categorical", categorical_pipeline, list(CATEGORICAL_FEATURES)),
    ], remainder="drop")
    classifier = (
        RandomForestClassifier(n_estimators=100, random_state=42)
        if estimator is None else clone(estimator)
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def build_logistic_regression_pipeline():
    """Créer un pipeline logistique neuf, prêt à être ajusté sur l'entraînement."""
    return build_pipeline(
        estimator=LogisticRegression(max_iter=1000, random_state=42),
        scale_numeric=True,
    )
