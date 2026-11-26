"""Validate the files and cross-file metadata of a published model release."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = ROOT / "artifacts" / "releases" / "rf-cleveland-v1"
DEFAULT_DATASET = ROOT / "data" / "raw" / "processed.cleveland.data"
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
ARTIFACT_HASHES = {
    "pipeline.joblib": "artifact_sha256",
    "metadata.json": "metadata_sha256",
    "test_evaluation.json": "test_evaluation_sha256",
}
CONFUSION_FIELDS = (
    "true_negative", "false_positive", "false_negative", "true_positive",
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    def unique_keys(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"Clé JSON dupliquée dans {path}: {key}")
            value[key] = item
        return value

    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file, object_pairs_hook=unique_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"JSON illisible ({path.name}): {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Le fichier {path.name} doit contenir un objet JSON.")
    return value


def require_int(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"Valeur entière invalide: {label}")
    return value


def validate_metrics(value, label, expected_keys=None):
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Métriques absentes ou invalides: {label}")
    if expected_keys is not None and set(value) != set(expected_keys):
        raise ValueError(f"Clés de métriques incohérentes: {label}")
    for name, score in value.items():
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError(f"Métrique invalide: {label}.{name}")


def same_metrics(left, right, label):
    if set(left) != set(right):
        raise ValueError(f"Métriques différentes entre fichiers: {label}")
    for name in left:
        if not math.isclose(left[name], right[name], rel_tol=0, abs_tol=0.00005):
            raise ValueError(f"Métrique incohérente: {label}.{name}")


def validate_release(release_dir, dataset_path=None):
    """Raise ValueError unless the release manifest, JSON and files agree.

    The joblib is checked as bytes and never deserialized by this verifier.
    This detects an incomplete or mismatched package; it does not authenticate
    the publisher or prove that a trusted joblib is safe to load.
    """
    release_dir = Path(release_dir)
    filenames = (*ARTIFACT_HASHES, "release.json", "verification.json")
    missing = [name for name in filenames if not (release_dir / name).is_file()]
    if missing:
        raise ValueError("Fichiers de release manquants: " + ", ".join(missing))

    manifest = read_json(release_dir / "release.json")
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("Version de schéma release.json non prise en charge.")
    model_version = manifest.get("model_version")
    if (not isinstance(model_version, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", model_version)):
        raise ValueError("model_version invalide dans release.json.")

    for filename, field in ARTIFACT_HASHES.items():
        expected = manifest.get(field)
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"Empreinte invalide dans release.json: {field}")
        actual = sha256((release_dir / filename).read_bytes())
        if actual != expected:
            raise ValueError(f"Empreinte SHA-256 incohérente: {filename}")

    metadata = read_json(release_dir / "metadata.json")
    report = read_json(release_dir / "test_evaluation.json")
    verification = read_json(release_dir / "verification.json")

    if type(metadata.get("schema_version")) is not int or metadata["schema_version"] != 1:
        raise ValueError("Version de schéma metadata.json non prise en charge.")
    if metadata.get("model") != "RandomForestClassifier" or metadata.get("dataset") != "Cleveland":
        raise ValueError("Le modèle ou le jeu de données de metadata.json est inattendu.")
    if metadata.get("features") != FEATURES or not isinstance(metadata.get("target"), dict) \
            or metadata["target"].get("name") != "num":
        raise ValueError("Variables ou cible incohérentes dans metadata.json.")
    if type(report.get("schema_version")) is not int or report["schema_version"] != 1:
        raise ValueError("Version de schéma test_evaluation.json non prise en charge.")
    if report.get("dataset") != "Cleveland":
        raise ValueError("Le jeu de données de test_evaluation.json est inattendu.")
    if verification.get("protocol") != "Cleveland/80-20-stratified/random42/CV5":
        raise ValueError("Protocole de vérification absent ou inattendu.")

    dataset_hashes = {
        metadata.get("dataset_sha256"), report.get("dataset_sha256"),
        verification.get("dataset_sha256"),
    }
    if len(dataset_hashes) != 1 or not isinstance(next(iter(dataset_hashes)), str) \
            or not re.fullmatch(r"[0-9a-f]{64}", next(iter(dataset_hashes))):
        raise ValueError("Empreintes du jeu de données incohérentes entre fichiers.")
    dataset_sha = next(iter(dataset_hashes))

    if dataset_path is not None:
        dataset_path = Path(dataset_path)
        if not dataset_path.is_file():
            raise ValueError(f"Jeu de données introuvable: {dataset_path}")
        if sha256(dataset_path.read_bytes()) != dataset_sha:
            raise ValueError("L’empreinte du jeu de données ne correspond pas à la release.")

    train_samples = require_int(metadata.get("train_samples"), "train_samples", 1)
    test_samples = require_int(metadata.get("test_samples"), "metadata.test_samples", 1)
    cv_folds = require_int(metadata.get("cv_folds"), "cv_folds", 2)
    if (require_int(report.get("test_samples"), "report.test_samples", 1) != test_samples
            or require_int(verification.get("train_samples"), "verification.train_samples", 1) != train_samples
            or require_int(verification.get("test_samples"), "verification.test_samples", 1) != test_samples
            or require_int(verification.get("cv_folds"), "verification.cv_folds", 2) != cv_folds
            or cv_folds > train_samples):
        raise ValueError("Effectifs ou nombre de plis incohérents entre fichiers.")

    cv_keys = ("accuracy", "precision", "recall", "f1", "roc_auc")
    test_keys = (*cv_keys, "specificity")
    cv_metrics = metadata.get("cv_metrics")
    test_metrics = report.get("metrics")
    verified_cv = verification.get("cv_metrics")
    verified_test = verification.get("test_metrics")
    validate_metrics(cv_metrics, "metadata.cv_metrics", cv_keys)
    validate_metrics(test_metrics, "test_evaluation.metrics", test_keys)
    validate_metrics(verified_cv, "verification.cv_metrics", cv_keys)
    validate_metrics(verified_test, "verification.test_metrics", test_keys)
    same_metrics(cv_metrics, verified_cv, "validation croisée")
    same_metrics(test_metrics, verified_test, "test")

    confusion = report.get("confusion_matrix")
    verified_confusion = verification.get("confusion_matrix")
    if not isinstance(confusion, dict) or set(confusion) != set(CONFUSION_FIELDS):
        raise ValueError("Matrice de confusion absente ou invalide.")
    if not isinstance(verified_confusion, dict) or set(verified_confusion) != set(CONFUSION_FIELDS):
        raise ValueError("Matrice de confusion de vérification absente ou invalide.")
    for field in CONFUSION_FIELDS:
        require_int(confusion[field], f"confusion_matrix.{field}")
        require_int(verified_confusion[field], f"verification.confusion_matrix.{field}")
    if confusion != verified_confusion or sum(confusion.values()) != test_samples:
        raise ValueError("Matrice de confusion incohérente avec la vérification ou l’effectif test.")

    limitations = manifest.get("limitations")
    if not isinstance(limitations, list) or not limitations or any(
            not isinstance(item, str) or not item.strip() for item in limitations):
        raise ValueError("Liste des limites absente ou invalide dans release.json.")

    return {
        "model_version": model_version,
        "artifact_sha256": manifest["artifact_sha256"],
        "dataset_sha256": dataset_sha,
        "train_samples": train_samples,
        "test_samples": test_samples,
        "dataset_checked": dataset_path is not None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--dataset", type=Path, default=None)
    args = parser.parse_args(argv)

    dataset = args.dataset
    if dataset is None and DEFAULT_DATASET.is_file():
        dataset = DEFAULT_DATASET
    try:
        result = validate_release(args.release, dataset)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Release incohérente : {exc}\n")

    dataset_status = "vérifié" if result["dataset_checked"] else "non vérifié (données absentes)"
    print(
        f"Release cohérente : {result['model_version']} ; "
        f"SHA-256 pipeline={result['artifact_sha256']} ; "
        f"dataset={dataset_status} ; "
        f"entraînement={result['train_samples']} ; test={result['test_samples']}"
    )


if __name__ == "__main__":
    main()
