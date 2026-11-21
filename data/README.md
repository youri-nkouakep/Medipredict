# Données

Les données ne sont pas publiées dans ce dépôt.

L'entrepôt `Data Warehouse.csv` (280 985 dossiers, 39 colonnes) fusionne trois jeux de données publics, consacrés respectivement à l'hypertension, au diabète et aux maladies cardiaques. La colonne `source_dataset` indique l'origine de chaque dossier.

Pour relancer les notebooks, placez `Data Warehouse.csv` dans le dossier `notebooks/` :

1. `01_preparation_random_forest.ipynb` nettoie, encode, découpe (80/20 stratifié) et standardise les données. Il produit `X_train.csv`, `X_test.csv`, `y_train.csv` et `y_test.csv`.
2. `02_comparaison_modeles.ipynb` entraîne et compare les modèles à partir de ces fichiers.
