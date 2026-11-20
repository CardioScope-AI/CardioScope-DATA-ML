# Versionner les exports — DATA-ML #22

## But

Conserver un ensemble identifiable et complet : modèle, métadonnées, résultats du
test, limites et empreintes. Le backend #24 peut ainsi associer une fiche au modèle
effectivement chargé. Une empreinte n’est pas une signature : utiliser seulement
des fichiers joblib provenant d’une source de confiance.

## Commandes Docker

Construire l’image ML avec le Dockerfile du dépôt si elle n’existe pas :

```powershell
docker build -t cardioscope-ml:dev .
```

Depuis la racine DATA-ML, après les tâches d’entraînement et d’évaluation :

```powershell
docker run --rm --mount "type=bind,source=.,target=/workspace" -w /workspace cardioscope-ml:dev python -m scripts.export_release --version rf-cleveland-v1
```

La commande ne lance pas train.py et ne remplace pas pipeline.joblib. Elle vérifie
le modèle sauvegardé sur le test, puis recalcule les cinq plis sur des clones du
pipeline, uniquement avec les observations d’entraînement. Cela ajuste des modèles
temporaires pour vérifier les moyennes CV ; le modèle exporté reste inchangé.
La validation ne choisit aucun hyperparamètre à partir du test réservé.

Options : --source (artifacts), --dataset (fichier Cleveland), --output
(artifacts/releases), --version (obligatoire, 1 à 100 lettres/chiffres ASCII,
points, tirets et underscores ; premier caractère alphanumérique).

## Fichiers produits

```text
artifacts/releases/rf-cleveland-v1/
  pipeline.joblib
  metadata.json
  test_evaluation.json
  release.json
  verification.json
```

Les trois premiers fichiers sont des copies exactes des sources validées.
release.json est compatible avec BACKEND #24 : schema_version=1, model_version,
artifact_sha256, metadata_sha256, test_evaluation_sha256 et limitations.
verification.json conserve la date UTC, le hash du dataset, le protocole et les
scores recalculés. Les empreintes SHA-256 portent sur les octets exacts des fichiers.
Le fichier .gitattributes empêche la conversion automatique des fins de ligne dans
les releases : un clone Windows/Linux doit conserver les mêmes empreintes.

Le contrôle compare les effectifs, les variables, le modèle, ses paramètres et
classes, les versions des bibliothèques, la matrice et les métriques recalculées.
Un désaccord interrompt la publication au lieu de corriger silencieusement les JSON.
Cela ne remplace pas l’audit complet ML #21 : absence de fuite, provenance des
données, limites cliniques et revue de l’entraînement restent nécessaires.

### Correction des rapports, 20 novembre 2026

La première tentative a été refusée : le rapport indiquait 28 vrais négatifs et
5 faux positifs, contre 27 et 6 avec le joblib local. evaluate.py a été réexécuté
sans réentraînement : accuracy 0,8689, précision 0,8125, recall 0,9286,
spécificité 0,8182, F1 0,8667, ROC-AUC 0,9421.

Les anciennes moyennes CV différaient aussi. Elles ont été recalculées sur des
clones du pipeline sauvegardé avec cinq plis stratifiés, random_state=42, sur les
242 lignes d’entraînement : accuracy 0,8181, précision 0,8173, recall 0,7826,
F1 0,7948, ROC-AUC 0,8967. metadata.json et les tableaux documentaires ont été corrigés.
Le modèle sauvegardé n’a pas été remplacé. Ces vérifications ne reconstituent pas
la provenance historique de son entraînement. L’export vérifie les rapports
avant publication ; il ne se contente pas de calculer leurs empreintes.

## Publication immuable

Le script prépare un dossier temporaire sur le même système de fichiers, puis
publie le dossier complet par renommage. Un verrou exclusif empêche deux commandes
d’écraser la même version. Un nom déjà existant est refusé, même si les fichiers sont
identiques. Utiliser v2 pour un nouvel export ; ne jamais modifier v1 en place.

Après une interruption brutale, un fichier .<version>.lock ou un dossier .release-*
peut subsister. Vérifier qu’aucun export n’est actif avant tout nettoyage manuel.
Le script ne supprime jamais une release déjà publiée. Aucun pointeur latest ni
déploiement automatique n’est créé.

## Utilisation backend et Docker

Le backend doit recevoir :

```text
MODEL_PATH=/models/releases/rf-cleveland-v1/pipeline.joblib
```

Compose monte déjà artifacts sur /models en lecture seule. Sa configuration
actuelle fixe toutefois MODEL_PATH=/models/pipeline.joblib : un override explicite
ou une modification coordonnée de la configuration APP est nécessaire pour choisir
la release. Une simple variable dans .env ne remplace pas une valeur Compose fixe.
Redémarrer/recréer le backend pour charger les nouveaux fichiers. Si le code #24
n’est pas encore dans son image, reconstruire aussi cette image.

Une fois connecté, GET /model-info doit renvoyer 200 avec la version et le hash
du manifeste. Un fichier modifié ou manquant provoque un refus, pas des scores par
défaut. Revenir à une ancienne version consiste à repointer MODEL_PATH sur son
dossier complet et redémarrer ; aucun réentraînement n’est nécessaire.

## Git et distribution

Les JSON versionnés et les scripts sont suivis par Git. Les joblib restent ignorés,
y compris dans les sous-dossiers de releases. Un clone Git seul ne contient donc
pas le modèle : distribuer le dossier complet via le canal d’artefacts de l’équipe,
vérifier ses empreintes et conserver les versions utiles au retour arrière.

Le notebook n’est pas modifié par cette tâche. Les scripts train.py et evaluate.py
continuent à produire les exports de travail à la racine artifacts ; ils ne doivent
jamais écrire dans un dossier de release publié.

## Tests

```powershell
docker run --rm --mount "type=bind,source=.,target=/workspace" -w /workspace cardioscope-ml:dev python -m unittest discover -s tests -v
```

Les tests contrôlent les copies et empreintes, le refus d’écrasement, les versions
invalides, le verrouillage, les erreurs d’écriture et l’absence de publication
partielle. La commande d’export effectue séparément la vérification du vrai modèle.

Validation réalisée le 20 novembre 2026 : 5 tests de publication réussis sous
Docker/Linux, export réel rf-cleveland-v1 publié après contrôle du test et de la CV,
puis lecture réussie par load_snapshot du backend #24 (version rf-cleveland-v1,
accuracy de test 0,8689). L’image backend locale, qui possède les mêmes versions
NumPy/SciPy/scikit-learn/joblib/pandas, a servi d’environnement Python Docker de
vérification ; aucune nouvelle dépendance n’a été ajoutée. Les variables
OPENBLAS_NUM_THREADS=1 et OMP_NUM_THREADS=1 limitaient les threads pendant ces essais.
Cela valide la lecture entre repos, pas encore un appel HTTP authentifié au
conteneur de déploiement : sa sélection MODEL_PATH doit encore être actualisée.
