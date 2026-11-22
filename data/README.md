# Données

L'entrepôt complet n'est pas publié dans ce dépôt.

`data_warehouse.csv` (280 985 dossiers, 39 colonnes) fusionne trois jeux de données publics consacrés à l'hypertension, au diabète et aux maladies cardiaques. La colonne `source_dataset` indique l'origine de chaque dossier. Placez ce fichier dans ce dossier pour lancer `python -m ml.train` et `python -m ml.analyze`.

Fichiers versionnés :

| Fichier | Contenu | Utilisation |
| --- | --- | --- |
| `holdout_sample.csv` | 5 000 patients du jeu de test, avec la cible | Contrôle de qualité de la CI (`ml/quality_gate.py`) |
| `reference_sample.csv` | 5 000 patients du jeu d'entraînement | Population de référence pour la surveillance de la dérive (`ml/drift.py`) |

Les notebooks de `notebooks/` sont l'exploration initiale du projet d'équipe. Ils attendent le fichier `Data Warehouse.csv` dans leur propre dossier.
