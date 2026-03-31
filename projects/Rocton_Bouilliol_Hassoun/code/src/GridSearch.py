from sklearn.model_selection import GridSearchCV
import pandas as pd
from DataManagement import DataHandler, DefaultDataHandler, ImprovedDataHandler
from ModelPipelines import DefaultPipeline, XGBoostSurvivalPipeline
from sksurv.util import Surv
from config import PARAMS_RSF, PARAMS_XGB
from sksurv.metrics import concordance_index_ipcw,concordance_index_censored
import logging
import os
import numpy as np
from LearningCurve import learning_curve_analysis, RiskScorePlotter

# logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ModelSelection:
    
    def __init__(self, model, param_grid, cv=5, n_jobs=-1):
        """
        model: sklearn estimator
        param_grid: dict of hyperparameters
        cv: number of folds
        scoring: scoring metric
        n_jobs: number of parallel jobs (-1 = all cores)
        """
        self.model = model
        self.param_grid = param_grid
        self.cv = cv
        self.n_jobs = n_jobs
        self.grid = None
        self.best_model = None

        logging.info("Initialized ModelSelection with model: %s, cv: %d, n_jobs: %d",
                     model.__class__.__name__, cv, n_jobs)


    def cindex_scorer(self, estimator, X, y):
        pred = estimator.predict(X)
        event, time = y[y.dtype.names[0]], y[y.dtype.names[1]]
        return concordance_index_censored(event, time, pred)[0]
        #return concordance_index_ipcw(y, y, pred, tau=7)[0]

    def fit(self, X, y):

        logging.info("Starting Grid Search with %d combinations", len(self.param_grid))

        self.grid = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            cv=self.cv,
            scoring=self.cindex_scorer,
            n_jobs=self.n_jobs,
            verbose=2
        )
        
        self.grid.fit(X, y)
        self.best_model = self.grid.best_estimator_
        
        return self
    

    def best_params(self):
        return self.grid.best_params_
    
    
    def best_score(self):
        return self.grid.best_score_
    

    def predict(self, X):
        return self.best_model.predict(X)


    def save_submission(self, X_test, out_path="submission_from_gridsearch.csv", ids=None):

        if self.best_model is None:
            raise RuntimeError("No fitted best_model available. Run fit() first.")

        raw_preds = self.predict(X_test)

        p_min = raw_preds.min()
        p_max = raw_preds.max()
        risk_scores = (raw_preds - p_min) / (p_max - p_min)

        if ids is None:
            try:
                ids = X_test.index
            except Exception:
                ids = range(len(risk_scores))

        sub_df = pd.DataFrame({
            "ID": ids, 
            "risk_score": risk_scores
        })
        
        sub_df.to_csv(out_path, index=False)
        logging.info(f"Submission saved : {out_path}")


if __name__ == "__main__":
    # Load data
    df = pd.read_csv("./data/raw/X_train/clinical_train.csv", index_col=0)
    maf_df = pd.read_csv("./data/raw/X_train/molecular_train.csv", index_col=0)
    target_df = pd.read_csv("./data/raw/target_train.csv", index_col=0)

    # Build and fit pipeline
    data_handler = ImprovedDataHandler(df, maf_df, target_df)
    prepared_data = data_handler.prepare()
    pipeline_builder = XGBoostSurvivalPipeline(prepared_data)
    pipeline = pipeline_builder.build_pipeline()
    # Save training columns to align test set
    train_cols = prepared_data[0].columns.tolist()

    y_surv = Surv.from_dataframe(
        event='OS_STATUS',   # 1 = event, 0 = censored
        time='OS_YEARS',
        data=prepared_data[1]
    )

    grid_search = ModelSelection(model=pipeline, param_grid=PARAMS_XGB, cv=5)
    # grid_search = ModelSelection(model=pipeline, param_grid=PARAMS_RSF, cv=5)
    grid_search.fit(prepared_data[0], y_surv)
    print("Best Parameters:", grid_search.best_params())
    print("Best Score:", grid_search.best_score())

    # --- Generate submission on test set (align columns with training) ---
    clin_test = pd.read_csv("./data/raw/X_test/clinical_test.csv", index_col=0)
    try:
        maf_test = pd.read_csv("./data/raw/X_test/molecular_test.csv", index_col=0)
    except FileNotFoundError:
        maf_test = None

    test_handler = ImprovedDataHandler(clin_test, maf_test, None)
    test_prepared = test_handler.prepare()
    X_test_prepared = test_prepared[0]

    # align test columns to training columns (adds missing columns as NaN, drops extras)
    X_test_aligned = X_test_prepared.reindex(columns=train_cols)

    out_file = os.path.join(os.getcwd(), "data", "submissions", "submission_from_gridsearch.csv")
    grid_search.save_submission(X_test_aligned, out_path=out_file)
    print(f"Saved submission to {out_file}")

    # --- Learning curve analysis ---
    learning_curve_analysis(pipeline, prepared_data[0], y_surv, PARAMS_XGB)
    # learning_curve_analysis(pipeline, prepared_data[0], y_surv, PARAMS_RSF)

    best_model=grid_search.best_model

    print("Loading Scores...")

    plotter = RiskScorePlotter(model=best_model, X=prepared_data[0], y=y_surv)
    print("Done.")
    plotter.plot_overall_distribution()
    plotter.plot_by_event_status()
    plotter.plot_kaplan_meier()

