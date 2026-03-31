import pandas as pd
from sklearn.model_selection import GridSearchCV
import sys
from pathlib import Path

# Add src to the module search path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

from DataManagement import DataHandler, DefaultDataHandler, ImprovedDataHandler # type: ignore
from ModelPipelines import DefaultPipeline, XGBoostSurvivalPipeline # type: ignore
from sksurv.util import Surv
from config import PARAMS_RSF, PARAMS_XGB # type: ignore
from sksurv.metrics import concordance_index_ipcw,concordance_index_censored
import logging
import os
import numpy as np
from LearningCurve import learning_curve_analysis, RiskScorePlotter # type: ignore
from GridSearch import ModelSelection # type: ignore

if __name__ == "__main__":

    # Load data
    data_dir = project_root / "data" / "raw"
    df = pd.read_csv(data_dir / "X_train" / "clinical_train.csv", index_col=0)
    maf_df = pd.read_csv(data_dir / "X_train" / "molecular_train.csv", index_col=0)
    target_df = pd.read_csv(data_dir / "target_train.csv", index_col=0)

    # Build and fit pipeline
    data_handler = ImprovedDataHandler(df, maf_df, target_df)
    prepared_data = data_handler.prepare()
    pipeline_builder = DefaultPipeline(prepared_data)
    pipeline = pipeline_builder.build_pipeline()

    train_cols = prepared_data[0].columns.tolist()

    # transform the target data
    y_surv = Surv.from_dataframe(
        event='OS_STATUS',   # 1 = event, 0 = censored
        time='OS_YEARS',
        data=prepared_data[1]
    )

    # perform the gridsearch
    grid_search = ModelSelection(model=pipeline, param_grid=PARAMS_RSF, cv=5)
    grid_search.fit(prepared_data[0], y_surv)
    print("Best Parameters:", grid_search.best_params())
    print("Best Score:", grid_search.best_score())

    # generate submission on test set
    clin_test = pd.read_csv(data_dir / "X_test" / "clinical_test.csv", index_col=0)
    try:
        maf_test = pd.read_csv(data_dir / "X_test" / "molecular_test.csv", index_col=0)
    except FileNotFoundError:
        maf_test = None

    test_handler = ImprovedDataHandler(clin_test, maf_test, None)
    test_prepared = test_handler.prepare()
    X_test_prepared = test_prepared[0]
    X_test_aligned = X_test_prepared.reindex(columns=train_cols)

    out_file = os.path.join(os.getcwd(), "submission_from_gridsearch.csv")
    grid_search.save_submission(X_test_aligned, out_path=out_file)
    print(f"Saved submission to {out_file}")

    # Learning curve analysis (check we don't overfit)
    learning_curve_analysis(pipeline, prepared_data[0], y_surv, PARAMS_RSF)

    # plots of the risk scores (distribution of prediction per event, kaplan-meyer curves...)
    best_model=grid_search.best_model
    print("Loading Scores...")
    plotter = RiskScorePlotter(model=best_model, X=prepared_data[0], y=y_surv)
    print("Done.")
    plotter.plot_overall_distribution()
    plotter.plot_by_event_status()
    plotter.plot_kaplan_meier()

