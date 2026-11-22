# Fiche modèle : MédiPredict (XGBoost)

## Usage prévu
- **Rôle :** aider au **dépistage** du diabète, de l'hypertension et des maladies cardiaques. Le modèle estime la probabilité qu'au moins une de ces pathologies soit présente, et donne les trois facteurs qui pèsent le plus pour chaque patient.
- **Utilisateurs :** des professionnels de santé, dans un cadre de démonstration académique.
- **Hors usage :** le diagnostic, la décision thérapeutique, le tri automatique de patients, ou tout usage sans contrôle médical.

## Modèle
- **Pipeline scikit-learn :** standardisation des 13 variables numériques, puis XGBoost (400 arbres, profondeur 10, taux d'apprentissage 0,05).
- **Seuil de décision :** 0,521, choisi par l'indice de Youden sur des prédictions hors échantillon (validation croisée en 5 plis sur le jeu d'entraînement).
- **Fichier :** `backend/model/medipredict.joblib` (pipeline, seuil et liste des variables).

## Données
- **Volume :** 278 701 dossiers après suppression des doublons, issus de 3 jeux publics (hypertension, diabète, cœur).
- **Découpage :** 80/20 stratifié, soit 222 960 dossiers d'entraînement et 55 741 de test.
- **Variables :** 20 au total (biologie, mode de vie, profil social).
- **Cible :** au moins une des trois pathologies (62 % des dossiers).

## Performances (jeu de test)

| AUC-ROC [IC 95 %] | AUC-PR | Brier | ECE | Sensibilité | Spécificité |
| --- | --- | --- | --- | --- | --- |
| 0,940 [0,938–0,941] | 0,963 | 0,088 | 0,015 | 95,4 % | 76,1 % |

## Limites connues
1. **Le modèle apprend la provenance des données.**
   - Les 20 variables suffisent à retrouver la source d'un dossier dans 99,9 % des cas.
   - La source seule prédit déjà la cible avec une AUC de 0,83.
   - Sur une source jamais vue, l'AUC tombe entre 0,71 et 0,81.
   - Sur une nouvelle population, il faut donc s'attendre à une performance nettement inférieure à 0,94.
2. **HbA1c domine les explications.** Cette variable n'existe que dans le jeu « diabète » : dans les autres sources, elle a été complétée lors de la fusion. Son poids reflète en partie cet artefact.
3. **Les faux positifs augmentent avec l'âge.**
   - La spécificité est de 84 % avant 40 ans, mais de 59 % à partir de 60 ans.
   - Elle est de 79 % chez les femmes et de 72 % chez les hommes.
   - Les patients âgés sont plus souvent orientés à tort vers des examens.
4. **Aucune validation clinique.** Les données sont publiques et en partie synthétiques ou harmonisées : aucune validation prospective n'a été faite.

## Suivi en production
- Chaque prédiction est enregistrée (variables et explication) pour la **surveillance de la dérive** : `python -m ml.drift --db <url>` calcule le PSI de chaque variable par rapport à la population d'entraînement.
- La **CI** refuse un modèle dont l'AUC passe sous 0,93 sur un échantillon de contrôle fixe de 5 000 patients.
