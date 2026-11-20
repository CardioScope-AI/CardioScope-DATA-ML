# Limites du modèle CardioScope — DATA-ML #20

Document établi le 19 novembre 2026 à partir du code et des rapports présents
dans le dépôt, scores actualisés le 20 novembre 2026 après réévaluation du modèle
sauvegardé, sans le réentraîner. Usage prévu : démonstrateur pédagogique réservé à l’administrateur,
sur des profils fictifs. Le modèle n’est pas validé pour un usage clinique.

## 1. Ce que prédit réellement le modèle

Le pipeline utilise RandomForestClassifier avec 100 arbres et random_state=42.
Ses 13 entrées sont celles du sous-ensemble Cleveland de UCI Heart Disease.
La cible originale num est transformée en deux classes :

- 0 : num vaut 0, absence selon l’étiquette du dataset ;
- 1 : num vaut 1, 2, 3 ou 4, présence selon l’étiquette du dataset.

La cible correspond à une présence de maladie cardiaque documentée dans cette
base, dont la documentation décrit le statut angiographique coronarien. Elle ne
décrit pas la survenue d’un infarctus à 5 ou 10 ans. La binarisation ne permet pas
de distinguer les valeurs originales 1 à 4 ni de déterminer une gravité.

Le résultat 0 ou 1 est une classe prédite, pas un diagnostic individuel. Éviter
de présenter 0 comme « aucun risque » et 1 comme « maladie certaine ».

Source de cette définition : [fiche officielle UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease).

## 2. Données limitées et représentativité

Le projet emploie uniquement processed.cleveland.data : 303 observations et
13 variables. Il n’entraîne pas le modèle sur les quatre bases de la collection.
La fiche UCI indique un dépôt en 1988 : ces données historiques ne suffisent pas
à établir une performance sur la population actuelle ou sur le contexte marocain.
La date de dépôt ne doit pas être confondue avec la date de collecte de chaque ligne.

Le fichier local contient des âges de 29 à 77 ans et 206 lignes avec sex=1 contre
97 avec sex=0, selon les codes de la source. Ces effectifs ne démontrent aucune
équité du modèle. Les performances par sexe, tranche d’âge ou autre sous-groupe
ne sont pas fournies dans les rapports actuels. Une performance globale élevée
ne prouve donc pas une performance identique pour chaque groupe.

Le test est une séparation interne du même fichier, pas une validation dans un
autre établissement, un autre pays ou une cohorte prospective. La généralisation
hors de cette population, notamment chez les enfants ou en dehors des âges
observés, n’est pas démontrée. Un formulaire qui accepte une valeur ne prouve pas
que le modèle a été validé pour cette valeur.

## 3. Entrées, valeurs manquantes et encodages

Le fichier local contient 4 valeurs manquantes pour ca et 2 pour thal.
Le pipeline remplace les valeurs numériques manquantes par la médiane et les
catégorielles par le mode, appris sur l’entraînement. Cette imputation ne reconstitue
pas la vraie mesure et peut masquer une information manquante.

OneHotEncoder(handle_unknown="ignore") permet techniquement d’encoder une catégorie
inconnue sans erreur ; cela ne rend pas cette catégorie valide ou fiable. Le
backend doit vérifier les codes originaux, les unités et les 13 champs obligatoires.
Une absence dans le formulaire ne doit pas devenir silencieusement une valeur zéro.
Le pipeline accepte techniquement NaN, alors que le contrat API exige des champs
non nuls : ces deux interfaces ne doivent pas être confondues.

Certaines entrées proviennent d’examens (ECG, effort, ca, thal) ; elles ne peuvent
pas toutes être remplacées par une appréciation personnelle de l’utilisateur.
Les unités et codes doivent suivre la documentation UCI. Des erreurs de saisie,
d’unité ou de codage peuvent changer le résultat. Les associations apprises ne
prouvent pas qu’une variable cause une maladie ni qu’en changer la valeur constitue
une recommandation de traitement.

## 4. Ce que montrent les évaluations disponibles

Sources locales : [métadonnées](../artifacts/metadata.json),
[rapport de test](../artifacts/test_evaluation.json),
[entraînement](../scripts/train.py) et [évaluation](../scripts/evaluate.py).
Les valeurs de test ont été recalculées le 20 novembre 2026 : l’ancien rapport
indiquait 54/61 alors que le modèle sauvegardé donne 53/61. Aucun réglage du modèle
n’a été choisi à partir de cette vérification. Les moyennes CV ont aussi été
recalculées sur des clones du pipeline, uniquement sur l’entraînement, puis
corrigées dans metadata.json. L’export de release vérifie de nouveau les deux rapports.

La séparation stratifiée utilise 242 profils d’entraînement et 61 de test,
test_size=0.20 et random_state=42. La validation croisée à cinq plis emploie
uniquement l’entraînement ; le pipeline complet est ajusté à chaque pli.

| Mesure | Moyenne CV sur l’entraînement | Test réservé |
| --- | ---: | ---: |
| Accuracy | 81,81 % | 86,89 % |
| Précision classe 1 | 81,73 % | 81,25 % |
| Sensibilité / recall classe 1 | 78,26 % | 92,86 % |
| F1 classe 1 | 79,48 % | 86,67 % |
| ROC-AUC | 0,8967 | 0,9421 |
| Spécificité classe 0 | Non rapportée | 81,82 % |

La ROC-AUC est calculée sur les scores du modèle, pas sur une probabilité clinique
validée. Les écarts entre CV et test ne prouvent ni une amélioration sur une autre
population ni une garantie de performance individuelle.

Le rapport de test contient cette matrice :

| Classe réelle | Prédite 0 | Prédite 1 |
| --- | ---: | ---: |
| 0 | 27 vrais négatifs | 6 faux positifs |
| 1 | 2 faux négatifs | 26 vrais positifs |

Soit 53 prédictions correctes sur 61, mais 8 erreurs :

- Les 2 faux négatifs illustrent qu’un résultat 0 peut manquer une présence
  selon la cible ; il ne doit pas rassurer à tort.
- Les 6 faux positifs illustrent qu’un résultat 1 peut signaler à tort une présence
  selon la cible ; il ne suffit pas à conclure à une maladie.

Ces nombres décrivent ce test uniquement. Une seule erreur supplémentaire change
l’accuracy d’environ 1,64 point de pourcentage ; avec 28 positifs réels, une erreur
change la sensibilité d’environ 3,57 points. Le faible effectif rend les estimations
sensibles au choix de la séparation.

## 5. Ce qui n’a pas été démontré

- Validation externe ou prospective, robustesse dans un autre contexte clinique.
- Intervalles de confiance, dispersion des scores entre plis et variabilité entre
  plusieurs séparations : les rapports actuels donnent surtout des moyennes.
- Équité et performances détaillées par sous-groupe.
- Calibration des probabilités, seuil choisi pour un objectif clinique ou bénéfice clinique.
- Comparaison systématique avec des modèles de référence et recherche d’hyperparamètres.
- Résistance aux saisies erronées, combinaisons inhabituelles et changement de population.

Random Forest peut surapprendre malgré l’agrégation de plusieurs arbres. Les
paramètres initiaux ne constituent pas une preuve d’optimalité. Une importance
de variable éventuelle ne fournit pas, à elle seule, une explication fiable du
résultat d’un profil particulier.

Le test réservé ne doit pas servir à choisir les réglages. Si des décisions de
modélisation sont prises après observation de ses scores, une nouvelle évaluation
indépendante est nécessaire pour estimer honnêtement la généralisation.

## 6. Limites de traçabilité et d’exploitation

Les exports de travail à la racine artifacts ne contiennent pas de version commune.
La commande DATA-ML #22 produit désormais un dossier de release avec un manifeste
liant le modèle et ses rapports par SHA-256, après vérification des scores.
Cela ne certifie pas l’identité d’un autre joblib présent ou déployé : contrôler
le manifeste et la version effectivement chargée. L’audit complet relève de #21.

Le script evaluate.py reconstruit le test depuis le fichier courant. Modifier
ce fichier ou son ordre peut changer le test même avec random_state=42 : conserver
une version identifiable des données et des exports est nécessaire.

L’application n’apprend pas automatiquement avec les simulations. L’historique
contient des prédictions, pas de nouvelles vérités diagnostiques ; les réutiliser
comme cibles ferait apprendre au modèle ses propres réponses. Tout réentraînement
est une opération séparée avec des données et une évaluation appropriées.

## 7. Texte court pour l’interface et le PDF

> Démonstrateur pédagogique utilisant un modèle entraîné sur le dataset historique
> Cleveland. Le résultat est une classification estimée, peut être erroné et ne
> constitue pas un diagnostic médical ni une estimation du risque à 5 ou 10 ans.
> Utiliser uniquement des profils fictifs. Un résultat 0 n’exclut pas une maladie.

Ce texte peut être repris par FRONTEND #21 et par l’export PDF. La restriction
au compte administrateur ne transforme pas le modèle en outil clinique validé.

## 8. Actualisation du document

Mettre à jour les scores, la matrice et les limites après toute nouvelle release,
modification de cible ou de données. Garder distincts résultats observés, contrôles
réalisés et améliorations envisagées. Cette tâche est documentaire : elle ne modifie
ni le modèle, ni les données, ni les scripts d’entraînement.
