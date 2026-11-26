# Vérifier une release ML — DATA-ML #21

Le vérificateur contrôle qu’un dossier de release est complet et que le manifeste,
les métadonnées, le rapport de test et le relevé de vérification décrivent les
mêmes artefacts. Il compare les empreintes SHA-256 des fichiers exacts et les
effectifs, métriques, variables, matrice de confusion et empreinte du dataset.
Il ne désérialise jamais le fichier joblib.

## Utilisation

Depuis la racine de DATA-ML, avec la release récupérée au complet :

```powershell
python -m scripts.verify_release --release artifacts/releases/rf-cleveland-v1
```

Le dataset local `data/raw/processed.cleveland.data` est vérifié s’il est présent.
Pour imposer un chemin ou rendre ce contrôle obligatoire :

```powershell
python -m scripts.verify_release `
  --release artifacts/releases/rf-cleveland-v1 `
  --dataset data/raw/processed.cleveland.data
```

Sous Linux, remplacer les accents graves PowerShell par `\` en fin de ligne, ou
placer toute la commande sur une seule ligne. Pour une autre version, remplacer
le chemin par `artifacts/releases/<version>`.

Succès : le programme affiche la version, l’empreinte du modèle, l’état du contrôle
dataset et les effectifs. Un fichier manquant, une empreinte erronée ou une valeur
contradictoire termine avec un code de sortie non nul et indique la cause.

## Fichiers et sécurité

La release doit contenir `pipeline.joblib`, `metadata.json`,
`test_evaluation.json`, `release.json` et `verification.json`. Git ignore les
fichiers joblib ; un clone seul peut donc produire l’erreur « pipeline.joblib
manquant ». Récupérer le dossier complet depuis le canal de distribution des
artefacts avant de vérifier. Ne pas remplacer automatiquement le pipeline de la
release par `artifacts/pipeline.joblib` à la racine : son empreinte peut être celle
d’une autre version.

Les empreintes détectent les fichiers manquants ou modifiés par rapport au
manifeste, mais ne prouvent pas l’identité de son auteur. Le vérificateur traite
joblib comme des octets et ne garantit pas qu’un fichier soit sûr à charger. Ne
charger que des modèles issus d’une source de confiance.
