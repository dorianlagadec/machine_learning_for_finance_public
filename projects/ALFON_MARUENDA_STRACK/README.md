# ML-203 — Group Project ML For Finance (QRT Challenge Data ENS #167)

## Membres du groupe

- **Thea Alfon**
- **Antoine Maruenda**
- **Sami Strack**

*Master 203 - Machine Learning For Finance Class - Dorian Lagadec*

---

## Description du problème

Ce projet s'inscrit dans le cadre du **QRT Challenge Data ENS #167**, une compétition de machine learning appliquée à la finance quantitative.

L'objectif est de prédire le **signe du rendement du lendemain** pour un ensemble de stratégies d'allocation d'actifs, formalisé comme un problème de **classification binaire**.

### Données

- **~278 allocations** suivies sur **~2 522 timestamps** d'entraînement
- **Features disponibles** :
  - Séries de rendements passés : `RET_1` à `RET_20`
  - Séries de volume signé : `SIGNED_VOLUME_1` à `SIGNED_VOLUME_20`
  - `MEDIAN_DAILY_TURNOVER` — turnover médian journalier
  - `GROUP` — groupe sectoriel (1 à 4)
  - `ALLOCATION`, `TS` — identifiants d'allocation et de timestamp
- La **variable cible** est binaire : 1 si le rendement du lendemain est positif, 0 sinon

### Métrique d'évaluation

La performance est mesurée par l'**accuracy** sur un jeu de test équilibré (50/50), avec le benchmark public LightGBM à **0.5079** comme référence.

---

## Structure du dépôt

```
.
├── code_qrt.py          # Code principal : feature engineering, modélisation, génération des prédictions
├── report_qrt.ipynb     # Notebook de rapport : analyse, résultats et visualisations
├── report_qrt.html      # Version HTML exportée du rapport
└── README.md
```

> Les fichiers de données (`X_train.csv`, `X_test.csv`, `y_train.csv`) ne sont pas inclus dans ce dépôt (données propriétaires QRT).

---

## Usage

Si vous voulez vous amuser à utiliser nos modèles ou tenter de les améliorer, rendez-vous dans la section code. 
Cette section vous permet de re-train tous les modeles et de générer les différents graphes et visualisations que vous retrouverez dans le report_qrt.html. Ces graphes sont générés automatiquement. 

Pour pouvoir retrain un modèle en particulier après avoir changer le code, vous pouvez utiliser :


python code_qrt.py --eda — EDA seul
python code_qrt.py --gbdt-v1 — baseline ensemble v1
python code_qrt.py --gbdt-v2b — modèle final v2b
python code_qrt.py --transformer --autoencoder — les deux NN
python code_qrt.py --stability — analyse KS / dense vs sparse
python code_qrt.py --all — tout d'un coup

## Environnement

Python 3.9.6 — dépendances principales : `lightgbm`, `xgboost`, `catboost`, `torch`, `scikit-learn`, `pandas`, `numpy`.
