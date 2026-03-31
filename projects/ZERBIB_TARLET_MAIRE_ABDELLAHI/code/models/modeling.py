from .preparing import Dataset
import math
import numpy as np
import pandas as pd
import os
from pathlib import Path
import yaml
import optuna
from optuna.pruners import MedianPruner
from sklearn.model_selection import StratifiedKFold, KFold
import sklearn.base
import lightgbm as lgb
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    accuracy_score,
)
import xgboost as xgb
from sklearn.naive_bayes import MultinomialNB

####################################################################################################################################################
#                                                                   HELPERS                                                                        #
####################################################################################################################################################

_MODEL_ALIASES = {
    "logistic":    "lr",
    "lr":          "lr",
    "linear":      "linear",
    "ridge":       "ridge",
    "rf":          "rf",
    "randomforest":"rf",
    "xgb":         "xgb",
    "xgboost":     "xgb",
    "lgb":         "lgb",
    "lgbm":        "lgb",
    "lightgbm":    "lgb",
    "nb": "nb",
    "nlp": "nb",
}


def _load_yaml_config(config_path: str) -> dict:
    """Load and return a YAML experiment config, or {} if path is None."""
    if config_path is None:
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def _build_model_from_params(model_type: str, params: dict, is_classification: bool):
    """Instantiate a model given its type key and a flat params dict."""
    key = _MODEL_ALIASES.get(model_type.lower(), model_type.lower())
    p = dict(params)  # avoid mutating caller's dict
    if key == "rf":
        return (RandomForestClassifier(**p, random_state=42)
                if is_classification else RandomForestRegressor(**p, random_state=42))
    elif key == "xgb":
        if not is_classification:
            p.setdefault("objective", "reg:squarederror")
        return (xgb.XGBClassifier(**p, random_state=42)
                if is_classification else xgb.XGBRegressor(**p, random_state=42))
    elif key == "lgb":
        return (lgb.LGBMClassifier(**p, random_state=42, verbose=-1)
                if is_classification else lgb.LGBMRegressor(**p, random_state=42, verbose=-1))
    elif key == "ridge":
        return Ridge(**p)
    elif key == "lr":
        p.setdefault("solver", "liblinear" if p.get("penalty") == "l1" else "lbfgs")
        return LogisticRegression(**p, max_iter=1000, random_state=42)
    elif key == "linear":
        return LinearRegression(**p)
    elif key == "nb":
        if is_classification:
            return MultinomialNB(**p)
        else:
            raise ValueError("Naive Bayes cannot be used to predict continuous numbers.")
    else:
        raise ValueError(f"Unknown model type: '{model_type}'")


####################################################################################################################################################
#                                                                   MODEL TRAINER                                                                  #
####################################################################################################################################################

class ModelTrainer:

    def __init__(
        self,
        dataset: Dataset,
        is_classification: bool = False,
        n_trials: int = 50,
        shuffle: bool = False,
        config_path: str = None,
    ):
        self.dataset     = dataset
        self.shuffle     = shuffle
        self.n_trials    = n_trials
        self.config_path = config_path

        # Load YAML (empty dict if no config supplied)
        cfg = _load_yaml_config(config_path)

        # YAML can override is_classification
        self.is_classification = cfg.get("is_classification", is_classification)

        # Resolve model alias if present in YAML
        raw_model = cfg.get("model", "")
        self._yaml_model_type = _MODEL_ALIASES.get(raw_model.lower()) if raw_model else None

        self._yaml_params = cfg.get("params", None)        # fixed-params mode
        self._yaml_hints  = cfg.get("optuna_hints", None)  # Optuna-hints mode

        self.model = self._get_model()
        self._train()
        self._evaluate()
        self._log_results(self.model)  # always after fit

    ####################################################################################################################################################
    #                                                                   INTERNAL HELPERS                                                               #
    ####################################################################################################################################################

    def _get_model(self):
        if not hasattr(self.dataset, "splits") or self.dataset.splits is None:
            raise ValueError("Dataset splits not found. Please run prepare_dataset() first.")

        # ── Mode 1: fixed params from YAML (no Optuna) ────────────────
        if self._yaml_params is not None:
            if self._yaml_model_type is None:
                raise ValueError("YAML config with 'params' must also specify 'model'.")
            print(f"[YAML] Fixed-params mode — model: {self._yaml_model_type}, params: {self._yaml_params}")
            best_model = _build_model_from_params(
                self._yaml_model_type, self._yaml_params, self.is_classification
            )
            self.best_params = dict(self._yaml_params)
            self.study = None
            return best_model

        # ── Mode 2 / 3: Optuna (with optional hints) ──────────────────
        print(f"Starting model optimization with {self.n_trials} trials...")
        best_model, best_params, study = self.optimize_model(
            self.is_classification, self.n_trials, self.shuffle
        )
        self.best_params = best_params
        self.study = study
        print(
            f"Optimization complete\n"
            f"  Best model : {type(best_model).__name__}\n"
            f"  Best params: {best_params}"
        )
        return best_model

    def _train(self):
        def _np(arr):
            return arr.values if hasattr(arr, "values") else arr

        self.model.fit(_np(self.dataset.splits.X_train), _np(self.dataset.splits.y_train))

        if hasattr(self.model, 'feature_importances_'):
            print("Feature Importances (Top 20):")
            importances = self.model.feature_importances_
            top_indices = np.argsort(importances)[::-1][:20]
            feat_names  = self.dataset.splits.X_train.columns
            for i in top_indices:
                print(f"  {feat_names[i]}: {importances[i]:.6f}")
        elif hasattr(self.model, 'coef_'):
            print("Top 20 Coefficients (by abs value):")
            coef = np.array(self.model.coef_).flatten()
            top_indices = np.argsort(np.abs(coef))[::-1][:20]
            feat_names  = self.dataset.splits.X_train.columns
            for i in top_indices:
                print(f"  {feat_names[i]}: {coef[i]:.6f}")

        self.y_train_pred = self.model.predict(_np(self.dataset.splits.X_train))
        if len(self.dataset.splits.X_test) > 0:
            self.y_test_pred = self.model.predict(_np(self.dataset.splits.X_test))
        else:
            self.y_test_pred = np.array([])

    def _evaluate(self):
        has_test = len(self.dataset.splits.y_test) > 0
        if self.is_classification:
            print("Classification Report (Train):")
            print(classification_report(self.dataset.splits.y_train, self.y_train_pred))
            if has_test:
                print("Classification Report (Test):")
                print(classification_report(self.dataset.splits.y_test, self.y_test_pred))
        else:
            train_mse = mean_squared_error(self.dataset.splits.y_train, self.y_train_pred)
            print(f"Mean Squared Error (Train): {train_mse:.6f}")
            print(f"RMSE             (Train): {math.sqrt(train_mse):.6f}")
            print(f"R^2 Score        (Train): {r2_score(self.dataset.splits.y_train, self.y_train_pred):.6f}")
            if has_test:
                test_mse = mean_squared_error(self.dataset.splits.y_test, self.y_test_pred)
                print(f"Mean Squared Error (Test): {test_mse:.6f}")
                print(f"RMSE             (Test):  {math.sqrt(test_mse):.6f}")
                print(f"R^2 Score        (Test):  {r2_score(self.dataset.splits.y_test, self.y_test_pred):.6f}")

    def _log_results(self, best_model):
        """Append one row to experiments/results.csv."""
        project_root    = Path(__file__).resolve().parent.parent
        experiments_dir = project_root / "experiments"
        os.makedirs(experiments_dir, exist_ok=True)
        results_file = experiments_dir / "results.csv"

        def _np(arr):
            return arr.values if hasattr(arr, "values") else arr

        y_test = self.dataset.splits.y_test
        if len(y_test) == 0:
            accuracy, auc, rmse, r2 = np.nan, np.nan, np.nan, np.nan
        else:
          y_pred = best_model.predict(_np(self.dataset.splits.X_test))

        if len(y_test) > 0 and self.is_classification:
            accuracy = accuracy_score(y_test, y_pred)
            rmse, r2 = np.nan, np.nan
            try:
                if len(np.unique(y_test)) > 2:
                    proba = best_model.predict_proba(_np(self.dataset.splits.X_test))
                    auc   = roc_auc_score(y_test, proba, multi_class="ovr")
                else:
                    proba = best_model.predict_proba(_np(self.dataset.splits.X_test))[:, 1]
                    auc   = roc_auc_score(y_test, proba)
            except Exception:
                auc = np.nan
        elif len(y_test) > 0:
            accuracy, auc = np.nan, np.nan
            test_mse = mean_squared_error(y_test, y_pred)
            rmse     = math.sqrt(test_mse)
            r2       = r2_score(y_test, y_pred)

        config_name   = Path(self.config_path).stem if self.config_path else "no_config"
        experiment_id = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

        row = pd.DataFrame([{
            "experiment_id": experiment_id,
            "config":        config_name,
            "model":         type(best_model).__name__,
            "accuracy":      accuracy,
            "auc":           auc,
            "rmse":          rmse,
            "r2":            r2,
        }])

        if os.path.exists(results_file):
            try:
                existing = pd.read_csv(results_file, nrows=0)  # headers only
                if list(existing.columns) != list(row.columns):
                    row.to_csv(results_file, index=False)
                else:
                    row.to_csv(results_file, mode="a", header=False, index=False)
            except Exception:
                row.to_csv(results_file, index=False)
        else:
            row.to_csv(results_file, index=False)


    def evaluate_date_cv(
        self,
        n_splits: int = 8,
        random_state: int = 0,
        threshold: float = 0.0,
        verbose: bool = True,
    ) -> dict:
        """
        Date-based cross-validation (same approach as the benchmark).
        Folds are built on unique dates in the training set so that all rows
        from the same date are in the same fold — preventing look-ahead bias.

        Requires the Dataset to have been built with Dataset.from_split() and
        a timestamp_col so that train_dates is available.

        Parameters
        ----------
        n_splits      : number of CV folds (default 8, same as benchmark)
        random_state  : KFold random state
        threshold     : decision threshold for binary accuracy (default 0)
        verbose       : print per-fold accuracy

        Returns
        -------
        dict with keys: scores, mean, std
        """
        ds = self.dataset
        if not hasattr(ds, "train_dates") or ds.train_dates is None:
            raise ValueError(
                "Dataset has no train_dates. Build it with Dataset.from_split(timestamp_col=...).")

        unique_dates = ds.train_dates.unique()
        folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(unique_dates)

        raw_X  = ds._raw_X_train
        raw_y  = ds._raw_y_train
        feat   = ds.feature_names_in_
        scores = []

        for fold_i, (tr_date_ids, val_date_ids) in enumerate(folds):
            tr_dates  = unique_dates[tr_date_ids]
            val_dates = unique_dates[val_date_ids]

            mask_tr  = ds.train_dates.isin(tr_dates)
            mask_val = ds.train_dates.isin(val_dates)

            # Apply the already-fitted preprocessor to local folds
            X_loc_tr  = pd.DataFrame(
                ds._preprocessor.transform(raw_X.loc[mask_tr,  feat]),
                columns=ds.feature_names_out_, index=raw_X.loc[mask_tr].index,
            )
            X_loc_val = pd.DataFrame(
                ds._preprocessor.transform(raw_X.loc[mask_val, feat]),
                columns=ds.feature_names_out_, index=raw_X.loc[mask_val].index,
            )
            y_loc_tr  = raw_y.loc[mask_tr]
            y_loc_val = raw_y.loc[mask_val]

            # Clone + fit the base model on local train
            import sklearn.base
            local_model = sklearn.base.clone(self.model)
            local_model.fit(X_loc_tr.values, y_loc_tr.values)

            preds = local_model.predict(X_loc_val.values)
            acc   = accuracy_score(
                (y_loc_val > threshold).astype(int),
                (preds       > threshold).astype(int),
            )
            scores.append(acc)
            if verbose:
                print(f"  Fold {fold_i+1}/{n_splits} — Accuracy: {acc*100:.2f}%")

        mean_acc = float(np.mean(scores)) * 100
        std_acc  = float(np.std(scores))  * 100
        if verbose:
            lo, hi = mean_acc - std_acc, mean_acc + std_acc
            print(f"\nCV Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%  [{lo:.2f} ; {hi:.2f}]")
        return {"scores": scores, "mean": mean_acc, "std": std_acc}


    ####################################################################################################################################################
    #                                                                   OPTUNA OPTIMISATION                                                                  #
    ####################################################################################################################################################


    def optimize_model(self, is_classification: bool, n_trials: int = 50, shuffle: bool = False):

        direction  = "maximize" if is_classification else "minimize"

        X_train    = self.dataset.splits.X_train
        y_train    = self.dataset.splits.y_train
        X_train_np = X_train.values if hasattr(X_train, "values") else np.array(X_train)
        y_train_np = y_train.values if hasattr(y_train, "values") else np.array(y_train)

        # Adapt n_splits to the smallest class size to avoid StratifiedKFold errors
        if is_classification:
            _, counts = np.unique(y_train_np, return_counts=True)
            min_class = int(counts.min())
            n_splits = min(5, min_class)
        else:
            n_splits = min(5, len(y_train_np))
        n_splits = max(2, n_splits)

        cv_class   = StratifiedKFold if is_classification else KFold
        cv         = cv_class(n_splits=n_splits, shuffle=shuffle, random_state=42 if shuffle else None)

        if self._yaml_model_type:
            allowed_types = [self._yaml_model_type]
        elif is_classification:
            allowed_types = ["rf", "xgb", "lgb", "lr"]
        else:
            allowed_types = ["rf", "xgb", "linear"]

        hints = self._yaml_hints or {}

        def objective(trial):
            model_type = (
                allowed_types[0]
                if len(allowed_types) == 1
                else trial.suggest_categorical("model_type", allowed_types)
            )

            params = {}
            if model_type == "rf":
                params["n_estimators"]      = trial.suggest_int("n_estimators",      50, 300)
                params["max_depth"]         = trial.suggest_int("max_depth",           3,  20)
                params["min_samples_split"] = trial.suggest_int("min_samples_split",   2,  20)
                params["min_samples_leaf"]  = trial.suggest_int("min_samples_leaf",    1,  20)
                model = (RandomForestClassifier(**params, random_state=42)
                         if is_classification else RandomForestRegressor(**params, random_state=42))

            elif model_type == "xgb":
                params["n_estimators"]     = trial.suggest_int  ("n_estimators",    50,  500)
                params["learning_rate"]    = trial.suggest_float("learning_rate",  1e-3, 0.3, log=True)
                params["max_depth"]        = trial.suggest_int  ("max_depth",        3,   10)
                params["subsample"]        = trial.suggest_float("subsample",       0.5,  1.0) if shuffle else 1.0
                params["colsample_bytree"] = trial.suggest_float("colsample_bytree",0.5,  1.0)
                params["reg_alpha"]        = trial.suggest_float("reg_alpha",      1e-8, 10.0, log=True)
                params["reg_lambda"]       = trial.suggest_float("reg_lambda",     1e-8, 10.0, log=True)
                if not is_classification:
                    params["objective"] = "reg:squarederror"
                model = (xgb.XGBClassifier(**params, random_state=42, early_stopping_rounds=10)
                         if is_classification else xgb.XGBRegressor(**params, random_state=42, early_stopping_rounds=10))

            elif model_type == "lgb":
                params["n_estimators"]     = trial.suggest_int  ("n_estimators",    50,  500)
                params["learning_rate"]    = trial.suggest_float("learning_rate",  1e-3, 0.3, log=True)
                params["max_depth"]        = trial.suggest_int  ("max_depth",       -1,   10)
                params["num_leaves"]       = trial.suggest_int  ("num_leaves",       20,  150)
                params["subsample"]        = trial.suggest_float("subsample",        0.5,  1.0) if shuffle else 1.0
                params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5,  1.0)
                params["reg_alpha"]        = trial.suggest_float("reg_alpha",       1e-8, 10.0, log=True)
                params["reg_lambda"]       = trial.suggest_float("reg_lambda",      1e-8, 10.0, log=True)
                model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)

            elif model_type == "lr":
                penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
                params["C"]       = trial.suggest_float("C", 1e-3, 10.0, log=True)
                params["penalty"] = penalty
                params["solver"]  = "liblinear" if penalty == "l1" else "lbfgs"
                model = LogisticRegression(**params, max_iter=1000, random_state=42)

            elif model_type == "linear":
                params["fit_intercept"] = trial.suggest_categorical("fit_intercept", [True, False])
                model = LinearRegression(**params)

            elif model_type == "nb":
                params["alpha"] = trial.suggest_float("alpha", 1e-3, 10.0, log=True)
                model = MultinomialNB(**params)
                
            else:
                raise ValueError(f"Model type '{model_type}' not supported")

            scores = []
            for step, (train_idx, val_idx) in enumerate(cv.split(X_train_np, y_train_np)):
                X_tr, y_tr = X_train_np[train_idx], y_train_np[train_idx]
                X_va, y_va = X_train_np[val_idx],   y_train_np[val_idx]

                if model_type == "xgb":
                    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
                elif model_type == "lgb":
                    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                              callbacks=[lgb.early_stopping(10, verbose=False)])
                else:
                    model.fit(X_tr, y_tr)

                if is_classification:
                    try:
                        if len(np.unique(y_tr)) > 2:
                            proba = model.predict_proba(X_va)
                            score = roc_auc_score(y_va, proba, multi_class="ovr")
                        else:
                            proba = model.predict_proba(X_va)[:, 1]
                            score = roc_auc_score(y_va, proba)
                    except Exception:
                        score = accuracy_score(y_va, model.predict(X_va))
                else:
                    score = math.sqrt(mean_squared_error(y_va, model.predict(X_va)))

                scores.append(score)
                trial.report(np.mean(scores), step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return np.mean(scores)

        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)
        study  = optuna.create_study(direction=direction, pruner=pruner)

        # Seed the first trial with hint values (optuna_hints mode)
        if hints and self._yaml_model_type:
            seed = dict(hints)
            if len(allowed_types) > 1:
                seed["model_type"] = self._yaml_model_type
            study.enqueue_trial(seed)

        study.optimize(objective, n_trials=n_trials)

        best_params     = study.best_params.copy()
        best_model_type = best_params.pop("model_type", self._yaml_model_type or allowed_types[0])

        best_model = _build_model_from_params(best_model_type, best_params, is_classification)
        best_model.fit(X_train_np, y_train_np)

        return best_model, best_params, study