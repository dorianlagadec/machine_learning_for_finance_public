# Asset Allocation Performance Prediction

This project focuses on predicting the performance of asset allocations in systematic trading. The central objective is to develop a predictive model that determines whether to trust a given allocation or bet against it, based on historical market data and diverse engineered features.

## Table of Contents
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage Workflow](#usage-workflow)
- [Modules Overview](#modules-overview)

## Project Overview

In systematic trading, asset allocation strategies generate target portfolios. However, these strategies can experience periods of underperformance. This project builds an overlay model to predict the future performance of such allocations. The output can be used to size the strategy up, scale it down, or even reverse the positions.

## Directory Structure

```text
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/               <- Original QRT data (X_train.csv, X_test.csv, y_train.csv).
│   ├── processed/         <- Preprocessed data ready for modeling (qrt_ready.csv).
│   └── sample.csv         <- 20% sample used for EDA and development.
│
├── models/                <- Core machine learning pipelines and helper classes.
│   ├── __init__.py
│   ├── modeling.py        <- ModelTrainer with Optuna hyperparameter search.
│   ├── preparing.py       <- Dataset module for cleaning, splitting, and scaling.
│   └── visualizing.py     <- EDAVisualizer and ModelVisualizer.
│
├── notebooks/             <- Jupyter notebooks for exploration and modeling.
│   └── eda.ipynb          <- EDA + feature engineering + full modeling pipeline.
│
├── scripts/
│   └── preprocess.py  <- Preprocessing script: raw data → qrt_ready.csv.
│
├── frontend/              <- FastAPI dashboard for interactive model training.
│   ├── main.py            <- API endpoints (upload CSV, run pipeline).
│   └── static/
│       └── index.html     <- Single-page UI (feature selection, results display).
│
├── submissions/           <- CSV files ready for leaderboard submission.
│
└── experiments/           <- YAML configs for model experiments and results tracking.
    └── results.csv
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd projet_ml
   ```

2. **Set up a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On MacOS/Linux
   # On Windows: venv\Scripts\activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage Workflow

### Option A — Notebook
Open `notebooks/eda.ipynb` for a full end-to-end pipeline: EDA → feature engineering → Optuna tuning → evaluation + visualizations.

### Option B — Dashboard
Launch the interactive FastAPI dashboard to train models on any CSV without writing code:
```bash
uvicorn frontend.main:app --reload --port 8001
```
Then open `http://localhost:8001` in your browser. Upload a CSV, select features and target, configure Optuna trials, and run.

### Option C — Script
Regenerate our preprocessed dataset from scratch:
```bash
python scripts/preprocess.py
```
This reads `data/raw/X_train.csv` + `data/raw/y_train.csv` and writes `data/processed/qrt_ready.csv`.

### Typical programmatic workflow
1. **Preprocessing:** Run `scripts/preprocess.py` to build `data/processed/qrt_ready.csv`.
2. **EDA:** Use `EDAVisualizer` (from `models.visualizing`) to inspect distributions, missing values, and correlations.
3. **Data Preparation:** Wrap your DataFrame in `Dataset` (from `models.preparing`) — handles encoding, imputation, scaling, and train/val/test splitting.
4. **Model Training:** Pass the prepared `Dataset` to `ModelTrainer` (from `models.modeling`) — runs Optuna over RF / XGBoost / LightGBM / Logistic Regression and returns the best model.
5. **Evaluation:** Inspect metrics and plots with `ModelVisualizer` (from `models.visualizing`) — confusion matrix, feature importances, residuals.

## Modules Overview

### Data Preparation (`models.preparing`)
The `Dataset` class wraps around a pandas DataFrame. It automatically tracks target variables and timestamps, encodes categorical labels, and provides built-in methods to extract lag/rolling metrics. It leverages Scikit-Learn transformers to safely handle `split_dataset` operations to prevent data leakage.

### Visualization (`models.visualizing`)
- **`EDAVisualizer`**: Generates comprehensive automated pair-plots, missing-value heatmaps, and target-association plots. 
- **`ModelVisualizer`**: Provides visual analysis of classification boundaries, feature importances, chronological timelines, regression residuals, and confusion matrices.

### Modeling (`models.modeling`)
`ModelTrainer` takes a fully prepared `Dataset` and uses **Optuna** to automatically search for the best model and hyperparameters across a configurable number of trials (default: 30), evaluated with stratified k-fold cross-validation. Supported model families: Random Forest, XGBoost, LightGBM, Logistic/Linear Regression. Evaluates metrics immediately (accuracy, ROC-AUC, classification report, RMSE, R²) and exposes the best model, best params, and test predictions.

### Dashboard (`frontend/`)
A FastAPI single-page application that exposes the full pipeline (EDA → prep → Optuna → evaluation) through a browser UI. Upload any CSV, select features, choose the target and task type (classification / regression), configure Optuna trials, and get results + plots without writing any code.