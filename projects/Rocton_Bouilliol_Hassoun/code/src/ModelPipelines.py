from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.compose import make_column_selector
from abc import ABC, abstractmethod
from sklearn.ensemble import  RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import logging
from DataManagement import DataHandler, DefaultDataHandler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, RegressorMixin



# logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureRatioCreator(BaseEstimator, TransformerMixin):
    """
    Create derived ratio features after imputation and preprocessing.
    """
    # ANC/WBC: fraction of neutrophils over the total count of white blood cells. functional immune capacity vs leukemia burden
    # Monocytes/WBC: proportion of monocytes, allowing to detect myelomonocytic disease patterns.
    # ANC/monocytes: Balance between two myeloid lineages, is it normal marrow recovery or AML with monocytic differenciation
    # PLT/WBC: in agressive leukemias, less platelets more WBC. marrow failure vs leukemic expansion
    # log transforms because blood values are highly skewed, so this stabilises variance and improve performance.
    # blast burden: total leukemic cell load in proxy
    # blast clone: product of tumor burden and max vaf (clonal dominance). estimates whether it's one dominant clone that expands or many subclones. dominant clone expansion.


    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self  # no fitting required

    def transform(self, X, y=None):
        X = X.copy()

        # --- Ratios ---
        if {"ANC", "WBC"}.issubset(X.columns):
            X["anc_ratio"] = X["ANC"] / X["WBC"]

        if {"MONOCYTES", "WBC"}.issubset(X.columns):
            X["mono_ratio"] = X["MONOCYTES"] / X["WBC"]

        if {"ANC", "MONOCYTES"}.issubset(X.columns):
            X["anc_mono_ratio"] = X["ANC"] / X["MONOCYTES"]

        if {"PLT", "WBC"}.issubset(X.columns):
            X["plt_wbc_ratio"] = X["PLT"] / X["WBC"]

        # --- Log transforms ---
        if "WBC" in X.columns:
            X["log_WBC"] = np.log1p(X["WBC"])

        if "BM_BLAST" in X.columns:
            X["log_BM_BLAST"] = np.log1p(X["BM_BLAST"])

        # --- Interactions ---
        if {"log_WBC", "BM_BLAST"}.issubset(X.columns):
            X["blast_burden"] = X["log_WBC"] * (X["BM_BLAST"] / 100)

        if {"BM_BLAST", "max_vaf"}.issubset(X.columns):
            X["blast_clone_interaction"] = X["BM_BLAST"] * X["max_vaf"]

        if {"cyto_complex", "nb_mutations"}.issubset(X.columns):
            X["complex_mutation"] = (X["cyto_complex"].fillna(0) > 0).astype(int) * X["nb_mutations"]

        return X

# Missing Values Transformer
class DropMissingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.2):
        self.threshold = threshold
        self.cols_to_keep_ = None

    def fit(self, X, y=None):
        missing_ratio = X.isna().sum() / len(X)
        self.cols_to_keep_ = missing_ratio[missing_ratio <= self.threshold].index.tolist()
        return self

    def transform(self, X):
        return X[self.cols_to_keep_]



# %%%%%%%%%%%%% Pipelines Builder %%%%%%%%%%%%%

# === Abstract Base Class for Pipelines ===
class ModelPipeline(ABC):

    def __init__(self, *args):
        pass

    @abstractmethod
    def build_pipeline(self) -> Pipeline:
        pass
#------------------------------

#-------------- Pipeline Factory ----------------
class PipelineFactory:

    @staticmethod
    def get_pipeline(model_type: str, prepared_data: tuple, **kwargs) -> ModelPipeline:
        """
        Factory method to return the appropriate pipeline based on model type.
        Supports 'xgboost' (or 'xgb') and 'rsf' (or 'randomsurvivalforest').
        """
        model_type = model_type.lower()
        if model_type in ['xgboost', 'xgb']:
            return XGBoostSurvivalPipeline(prepared_data, **kwargs)
        elif model_type in ['rsf', 'randomsurvivalforest', 'random_survival_forest']:
            return DefaultPipeline(prepared_data, **kwargs)
        else:
            raise ValueError(f"Unknown model_type for survival analysis: {model_type}")
#------------------------------

class DefaultPipeline(ModelPipeline):

    def __init__(
            self, 
            prep_:tuple[pd.DataFrame, pd.DataFrame, list, list, list], 
            *args
            ):
        super().__init__(*args)
        self.prepared_data = prep_

    def build_pipeline(self) -> Pipeline:

        # return prepared data set copies (non-mutating)
        X, y, float_cols, categorical_cols, binary_cols = self.prepared_data

        # Float preprocessing
        float_pipe = Pipeline(steps=[
            ("impute_num", SimpleImputer(strategy="median"))
        ])

        # Categorial Preprocessing
        multi_cat_pipe = Pipeline(steps=[
            ('encoder', OneHotEncoder(
                handle_unknown='ignore',
                sparse_output=False
            )),

            ("impute_cat", KNNImputer(
                n_neighbors=5,
                weights="distance"
            )),
        ])

        # binary categorical preprocessing
        binary_pipe = Pipeline(steps=[
            ("impute_cat", KNNImputer(
                n_neighbors=5,
                weights="distance"
            )),
        ])

        # columns transformer
        col_trans = ColumnTransformer(
            transformers=[
                ("num", float_pipe, float_cols),
                ("cat", multi_cat_pipe, categorical_cols),
                ("binary_cat", binary_pipe, binary_cols)
            ],
            remainder='passthrough'
        )
        col_trans.set_output(transform="pandas")

        # === MODEL ===

        model = RandomSurvivalForest(
            n_jobs=-1,
            random_state=42,
            bootstrap=True,
            oob_score=False
        )


        # return full Pipeline
        return Pipeline([
            ("drop_missing", DropMissingTransformer(threshold=0.2)),
            ('column_transformer', col_trans),
            ('feature_ratios', FeatureRatioCreator()),
            ('model', model)
        ])
    




class XGBSurvivalWrapper(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=42
    ):

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.random_state = random_state

        self.model = None


    def fit(self, X, y):

        event = y[y.dtype.names[0]].astype(bool)
        time = y[y.dtype.names[1]].astype(float)

        # For XGBoost survival:cox, right-censored instances must have negative time values
        y_xgb = np.where(event, time, -time)

        self.model_ = XGBRegressor(

            objective="survival:cox",

            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,

            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,

            reg_lambda=self.reg_lambda,
            reg_alpha=self.reg_alpha,

            tree_method="hist",
            n_jobs=-1,
            random_state=self.random_state
        )

        self.model_.fit(X, y_xgb)

        return self


    def predict(self, X):

        return self.model_.predict(X)


class XGBoostSurvivalPipeline(ModelPipeline):

    def __init__(
        self,
        prep_: tuple[pd.DataFrame, pd.DataFrame, list, list, list],
        *args
    ):
        super().__init__(*args)
        self.prepared_data = prep_

    def build_pipeline(self) -> Pipeline:

        X, y, float_cols, categorical_cols, binary_cols = self.prepared_data

        float_pipe = Pipeline([
            ("impute_num", SimpleImputer(strategy="median"))
        ])

        multi_cat_pipe = Pipeline([
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )),
            ("impute_cat", KNNImputer(
                n_neighbors=5,
                weights="distance"
            ))
        ])

        binary_pipe = Pipeline([
            ("impute_cat", KNNImputer(
                n_neighbors=5,
                weights="distance"
            ))
        ])

        col_trans = ColumnTransformer(
            transformers=[
                ("num", float_pipe, float_cols),
                ("cat", multi_cat_pipe, categorical_cols),
                ("binary_cat", binary_pipe, binary_cols)
            ],
            remainder="passthrough"
        )

        col_trans.set_output(transform="pandas")

        model = XGBSurvivalWrapper()

        return Pipeline([
            ("drop_missing", DropMissingTransformer(threshold=0.2)),
            ("column_transformer", col_trans),
            ("feature_ratios", FeatureRatioCreator()),
            ("model", model)
        ])




# %%%%%%%%%%%%% === MAIN TEST === %%%%%%%%%%%%%
if __name__ == "__main__":
    from pathlib import Path
    import sys
    
    # Path setup to ensure it runs independently
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data" / "raw"

    # Load data
    df = pd.read_csv(data_dir / "X_train" / "clinical_train.csv", index_col=0)
    maf_df = pd.read_csv(data_dir / "X_train" / "molecular_train.csv", index_col=0)
    target_df = pd.read_csv(data_dir / "target_train.csv", index_col=0)

    # Prepare data
    data_handler = DefaultDataHandler(df, maf_df, target_df)
    prepared_data = data_handler.prepare()

    y_surv = Surv.from_dataframe(
        event='OS_STATUS',   # 1 = event, 0 = censored
        time='OS_YEARS',
        data=prepared_data[1]
    )
    
    print("Testing XGBoost Pipeline...")
    xgb_pipeline_builder = PipelineFactory.get_pipeline("xgb", prepared_data)
    xgb_pipeline = xgb_pipeline_builder.build_pipeline()
    xgb_pipeline.fit(prepared_data[0], y_surv)
    predictions_xgb = xgb_pipeline.predict(prepared_data[0])
    print(f"XGBoost initial predictions sample: {predictions_xgb[:5]}")
    
    print("\nTesting Random Survival Forest Pipeline...")
    rsf_pipeline_builder = PipelineFactory.get_pipeline("rsf", prepared_data)
    rsf_pipeline = rsf_pipeline_builder.build_pipeline()
    rsf_pipeline.fit(prepared_data[0], y_surv)
    predictions_rsf = rsf_pipeline.predict(prepared_data[0])
    print(f"RSF initial predictions sample: {predictions_rsf[:5]}")
    
    print("\nBoth pipelines working correctly!")
