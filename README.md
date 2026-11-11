# CardioScope · Data / ML

Dataset, préparation des données, entraînement et évaluation du futur modèle. Le notebook charge les données locales et prépare la séparation. Le pipeline est construit séparément dans `scripts/pipeline.py`. Aucun entraînement ni téléchargement ne se déclenche automatiquement.

## Source
[UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease), sous-ensemble Cleveland : fichier `processed.cleveland.data`, 303 observations, 13 variables explicatives et cible `num`. La collection contient quatre bases ; sélectionner explicitement Cleveland. Source officielle : https://archive.ics.uci.edu/static/public/45/heart+disease.zip ; citer UCI et respecter la licence affichée sur sa fiche.

Prévoir la gestion des valeurs `?`, la séparation apprentissage/test avant ajustement des transformations et la validation de l’encodage des classes avec l’équipe. L’entraînement et l’évaluation feront l’objet d’une tâche séparée.

## Organisation prévue
- `data/` : données téléchargées localement, ignorées par Git.
- `notebooks/` : exploration et expériences.
- `scripts/` : futurs scripts de préparation, entraînement et export.
- `artifacts/pipeline.joblib` : futur pipeline complet (prétraitement + estimateur), ignoré par Git. Conserver le dossier artifacts même sans modèle.

Exporter aussi les versions et le schéma des variables dans un futur fichier de métadonnées. Aligner les versions du backend et du ML : Python `3.13.7`, scikit-learn `1.6.1`, joblib `1.4.2`, NumPy `2.2.6`, SciPy `1.15.3`. Ne pas changer une seule des deux piles sans réexport et validation.

## Environnement Docker ML séparé
Depuis ce dépôt, mêmes commandes sous Windows et Linux :
```text
docker build -t cardioscope-ml:dev .
docker run --rm -it -p 127.0.0.1:8888:8888 --mount "type=bind,source=.,target=/workspace" cardioscope-ml:dev
```
Ouvrir l’URL Jupyter avec le jeton imprimé dans le terminal. Cette image n’est pas démarrée par Compose.

Une fois `scripts/train.py` développé lors de la tâche ML, la commande d’entraînement séparée sera :
```text
docker run --rm --mount "type=bind,source=.,target=/workspace" cardioscope-ml:dev python scripts/train.py
```
Cette commande n’est pas encore exécutable : le script reste à écrire. Elle devra produire `artifacts/pipeline.joblib`. Compose monte ce dossier en lecture seule dans le backend à `/models`; redémarrer le backend après export. Ne pas charger de fichier joblib non fiable.

## Audit des données et Définition de la cible (Carte 02)

- **Source** : UCI Machine Learning Repository - Heart Disease (Cleveland). Fichier brut `processed.cleveland.data` (récupéré le 11/11/2026).
- **Dimensions auditées** : 303 observations et 13 variables explicatives d'entrée (X).
- **Contrôle qualité** :
  - **Doublons** : 0 doublon strict.
  - **Valeurs manquantes** : 6 valeurs manquantes au total encodées par `?` dans la source, détectées sur `ca` (4 valeurs) et `thal` (2 valeurs). L'imputation sera gérée dans le pipeline ML pour éviter les fuites de données.
- **Définition de la cible binaire (y)** :
  - La cible d'origine `num` prend ses valeurs dans {0, 1, 2, 3, 4}.
  - **Classe 0 (Absence de pathologie)** : `num == 0` -> **164 patients (54.1 %)**.
  - **Classe 1 (Présence de pathologie)** : `num > 0` -> **139 patients (45.9 %)**.
  - La distribution est équilibrée.

## Séparation entraînement / test (Carte 04)

Exécuter `notebooks/01_eda.ipynb` dans l’ordre, après avoir placé le fichier source dans `data/raw/processed.cleveland.data`. Le chemin de lecture est relatif au dossier `notebooks/`.

- Séparation conjointe de `X` et `y` avec `train_test_split` : 80 % entraînement, 20 % test, `stratify=y`, `random_state=42`.
- Pour les 303 observations Cleveland : 242 observations d’entraînement et 61 de test.
- Le notebook affiche les effectifs et proportions des classes et vérifie l’absence de chevauchement ainsi que l’alignement entre entrées et cibles.
- Aucune transformation n’est ajustée avant la séparation. Les valeurs manquantes seront traitées par le pipeline ajusté sur l’entraînement ; le test est réservé à l’évaluation finale.

## Construction du pipeline (Carte 05)

`scripts/pipeline.py` expose `build_pipeline(estimator=None)` : chaque appel construit un pipeline neuf et non ajusté. Le notebook reste consacré à l’exploration et à la séparation entraînement/test.

- Numériques : `age`, `trestbps`, `chol`, `thalach`, `oldpeak` ; imputation médiane, sans standardisation (inutile pour les arbres de Random Forest).
- Catégorielles : `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal` ; imputation par le mode puis `OneHotEncoder(handle_unknown="ignore")`. Le traitement catégoriel de `ca` est un choix initial à comparer pendant la validation.
- Entrée : DataFrame avec les 13 noms de colonnes UCI et leurs codes originaux ; valeurs manquantes sous forme de NaN. `num` est exclu des entrées. La validation des champs API reste une responsabilité du backend.
- Estimateur retenu : `RandomForestClassifier(n_estimators=100, random_state=42)`, remplaçable via l'argument `estimator`. Les hyperparamètres sont initiaux ; leur sélection et l’évaluation relèvent de la tâche d’entraînement.
- Entraînement ultérieur : `pipeline.fit(X_train, y_train)` uniquement. Le prétraitement est ainsi ajusté dans le pipeline sans utiliser le test. Pour la sélection du modèle, passer le pipeline complet à la validation croisée sur l'entraînement.
- Aucun ajustement ni export à l'import du module ou lors de sa construction. Le script d'entraînement et l'export de `artifacts/pipeline.joblib` restent à réaliser.
