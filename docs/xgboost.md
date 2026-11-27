# Référence XGBoost

Cette expérience ajoute XGBoost à côté du Random Forest, sans remplacer le
modèle chargé par le backend. SVM et régression logistique restent des travaux
séparés. Les scripts et exports Random Forest conservent leurs commandes.

## Exécuter avec les versions de l’équipe

Depuis DATA-ML, avec Docker démarré et le dataset Cleveland présent dans
`data/raw/processed.cleveland.data` :

```text
docker build -t cardioscope-ml:dev .
docker run --rm --mount "type=bind,source=.,target=/workspace" cardioscope-ml:dev python -m scripts.train_xgboost
docker run --rm --mount "type=bind,source=.,target=/workspace" cardioscope-ml:dev python -m unittest discover -s tests
```

La dépendance `xgboost-cpu==3.1.2` fournit le module Python `xgboost`, sans
nécessiter de GPU. L’image installe aussi le runtime OpenMP `libgomp1`.
Les dépendances existantes restent à leurs versions actuelles.

## Protocole

1. Charger les 303 observations et transformer `num` en cible binaire.
2. Reprendre les constantes du script Random Forest : test de 20 %, seed 42,
   stratification, soit 242 observations d’entraînement et 61 de test.
3. Réutiliser le prétraitement partagé : médiane pour les nombres, modalité
   fréquente et one-hot pour les catégories. Aucune standardisation requise.
4. Faire cinq validations stratifiées, mélangées avec seed 42, uniquement sur
   l’entraînement. Le prétraitement est réappris dans chaque groupe.
5. Ajuster le pipeline final sur les 242 observations puis évaluer une fois sur
   les 61 observations réservées, avec seuil 0,5 pour la classification.

Paramètres initiaux : 100 arbres, profondeur 3, taux d’apprentissage 0,05,
sous-échantillonnage des lignes et colonnes à 0,8, régularisation L2 à 1.
Ce sont des paramètres de départ, pas des paramètres optimaux démontrés.
Pas de recherche d’hyperparamètres ni d’arrêt anticipé sur le test.

## Sorties et comparaison

Le dossier `artifacts/xgboost/` contient le pipeline complet `pipeline.joblib`
(ignoré par Git), `metadata.json` et `test_evaluation.json`. Les JSON enregistrent
les versions, le hash du dataset et du pipeline, le protocole, les moyennes CV,
les scores des cinq groupes et leurs écarts-types, les métriques du test et la
matrice de confusion. L’écart-type des groupes n’est pas un intervalle de confiance.

Un dossier de sortie existant est refusé. Pour une nouvelle expérience, choisir
un nouveau chemin, par exemple `--output-dir artifacts/xgboost-essai-2` ; ne pas
réutiliser le test pour choisir les paramètres. Les modèles joblib de dossiers
personnalisés doivent aussi rester hors de Git.

Comparer les scores CV aux autres modèles sur les mêmes groupes. Le test final
ne doit pas devenir un jeu de réglage par essais répétés. Ces scores ne prouvent
pas la validité clinique ; voir aussi les limites du modèle.

## Première exécution (27 novembre 2026)

Entraînement exécuté sous Linux dans un conteneur Python 3.13.7 avec les
dépendances scientifiques du backend et l’ajout de `xgboost-cpu==3.1.2`.
La reconstruction complète de l’image ML décrite plus haut reste à vérifier.
Les 16 tests unitaires ont également réussi dans l’environnement Windows.

| Mesure | Moyenne CV (242 profils) | Test réservé (61 profils) |
|---|---:|---:|
| Accuracy | 0,8017 | 0,9180 |
| Précision | 0,7869 | 0,8710 |
| Rappel | 0,7834 | 0,9643 |
| Spécificité | 0,8165 | 0,8788 |
| F1 | 0,7828 | 0,9153 |
| ROC-AUC | 0,8843 | 0,9524 |

Matrice du test : 29 vrais négatifs, 4 faux positifs, 1 faux négatif et
27 vrais positifs. Le score test élevé ne démontre pas une supériorité : le test
est petit et les scores CV sont moins élevés. Conserver ces paramètres fixes
pour cette référence et comparer les protocoles avant de conclure.

## Limite d’intégration actuelle

Les outils `export_release.py`, `verify_release.py` et le chargeur backend
valident actuellement le Random Forest. Ils ne sont pas étendus ici : ne pas
présenter cet export XGBoost comme une release déployable ni changer `MODEL_PATH`.
L’API, le frontend et la sélection entre modèles feront l’objet d’une intégration
commune après les travaux des autres membres.

Sources : [distribution CPU 3.1.2](https://pypi.org/project/xgboost-cpu/3.1.2/),
[API XGBoost](https://xgboost.readthedocs.io/en/release_3.1.0/python/python_api.html).
