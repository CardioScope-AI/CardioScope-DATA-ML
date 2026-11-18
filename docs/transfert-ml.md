# Transfert des champs CardioScope — ML

## Entrée attendue par le pipeline

Les champs doivent être envoyés dans cet ordre. La cible `num` ne fait pas partie des données d’entrée.

| Ordre | Champ | Type proposé | Unité / codes |
|---:|---|---|---|
| 1 | `age` | Nombre | Années |
| 2 | `sex` | Catégorie | 0 = femme ; 1 = homme |
| 3 | `cp` | Catégorie | 1 = angine typique ; 2 = angine atypique ; 3 = douleur non angineuse ; 4 = asymptomatique |
| 4 | `trestbps` | Nombre | Pression artérielle au repos, mmHg |
| 5 | `chol` | Nombre | Cholestérol sérique, mg/dL |
| 6 | `fbs` | Catégorie | 0 = glycémie à jeun ≤ 120 mg/dL ; 1 = glycémie à jeun > 120 mg/dL |
| 7 | `restecg` | Catégorie | 0 = normal ; 1 = anomalie ST-T ; 2 = hypertrophie ventriculaire gauche probable ou certaine |
| 8 | `thalach` | Nombre | Fréquence cardiaque maximale atteinte |
| 9 | `exang` | Catégorie | 0 = non ; 1 = oui, angine induite par l’effort |
| 10 | `oldpeak` | Nombre décimal | Dépression du segment ST induite par l’effort par rapport au repos |
| 11 | `slope` | Catégorie | 1 = pente ascendante ; 2 = plate ; 3 = descendante |
| 12 | `ca` | Catégorie | Nombre de vaisseaux principaux observés : 0, 1, 2 ou 3 |
| 13 | `thal` | Catégorie | Codes UCI : 3 = normal ; 6 = défaut fixe ; 7 = défaut réversible |

## Cible du modèle

- `num` n’est pas envoyé au modèle comme champ d’entrée.
- Classe 0 : `num == 0`, absence de pathologie dans la définition du jeu.
- Classe 1 : `num > 0`, présence de pathologie dans la définition du jeu.

## À confirmer avec l’équipe backend

- Les 13 champs sont-ils obligatoires dans la requête ?
- Le JSON doit-il transmettre les catégories sous forme de nombres ou de chaînes ?
- Quelles bornes de validation appliquer aux nombres ?
- Comment représenter une valeur manquante ?
- Quelle sera la structure exacte de la requête et de la réponse de l’API ?

Référence : [UCI Heart Disease — Cleveland](https://archive.ics.uci.edu/dataset/45/heart+disease)
## Profils fictifs pour la démonstration

Les profils ci-dessous sont fictifs et servent uniquement à la démonstration.

### Profil de référence

`age=54, sex=1, cp=2, trestbps=130, chol=215, fbs=0, restecg=0, thalach=150, exang=0, oldpeak=1.0, slope=1, ca=0, thal=3`

Classe obtenue après vérification du rechargement : **0**

### Profil fictif 02

`age=45, sex=0, cp=1, trestbps=120, chol=180, fbs=0, restecg=0, thalach=170, exang=0, oldpeak=0.0, slope=1, ca=0, thal=3`

Classe prédite : à calculer

### Profil fictif 03

`age=65, sex=1, cp=4, trestbps=150, chol=270, fbs=1, restecg=2, thalach=120, exang=1, oldpeak=2.5, slope=2, ca=2, thal=7`

Profil fictif 02 : 0
Profil fictif 03 : 1
## Résultats du modèle Random Forest

### Validation croisée à 5 plis

| Mesure | Résultat |
|---|---:|
| Accuracy | 0.8140 |
| Précision | 0.8099 |
| Recall | 0.7830 |
| F1 | 0.7916 |
| ROC-AUC | 0.9013 |

### Évaluation finale sur le test réservé

Le test contient 61 profils. Les lignes de la matrice représentent les classes réelles et les colonnes les classes prédites.

| Mesure | Résultat |
|---|---:|
| Accuracy | 0.8852 |
| Précision | 0.8387 |
| Recall / sensibilité | 0.9286 |
| Spécificité | 0.8485 |
| F1 | 0.8814 |
| ROC-AUC | 0.9529 |

Matrice de confusion : `[[28, 5], [2, 26]]`

### Versions de l’environnement

- Python : 3.13.7
- scikit-learn : 1.6.1
- joblib : 1.4.2
- NumPy : 2.2.6
- SciPy : 1.15.3

### Interprétation de la cible

- Classe 0 : `num == 0`
- Classe 1 : `num > 0`