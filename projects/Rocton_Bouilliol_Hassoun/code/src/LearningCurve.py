import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
from ModelPipelines import ModelPipeline
from sklearn.model_selection import learning_curve,train_test_split, GridSearchCV
from sklearn.metrics import make_scorer
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from config import PARAMS_RSF, PARAMS_XGB
from lifelines import KaplanMeierFitter

# logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def cindex_score(y_true, risk_scores):
    return concordance_index_censored(
        y_true["OS_STATUS"],
        y_true["OS_YEARS"],
        risk_scores
    )[0]


def learning_curve_analysis(pipeline:ModelPipeline, X:np.array, y:np.array, param_grid:dict)->None:

    cindex_scorer = make_scorer(cindex_score, greater_is_better=True)

    # function that will print the score of train / validation samples
    # according to the fraction of data used inside the train sample
    # default param grid is the same as the one we used in the grid search

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=cindex_scorer,
        cv=5,
        n_jobs=-1,
        verbose=0
    )

    train_sizes = np.linspace(0.1, 0.9, 6)
    train_scores = []
    val_scores = []

    logging.info("Start looping ...")

    for frac in train_sizes:

        logging.info(f"Iteration : {frac}")

        X_sub, _, y_sub, _ = train_test_split(
            X,
            y,
            train_size=float(frac),
            random_state=42
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_sub,
            y_sub,
            test_size=0.3,
            random_state=42
        )

        # refit grid search
        grid.fit(X_train, y_train)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        # risk predictions
        train_risk = best_model.predict(X_train)
        val_risk = best_model.predict(X_val)

        # C-index train
        train_cindex = concordance_index_censored(
            y_train["OS_STATUS"],
            y_train["OS_YEARS"],
            train_risk
        )[0]

        # C-index validation
        val_cindex = concordance_index_censored(
            y_val["OS_STATUS"],
            y_val["OS_YEARS"],
            val_risk
        )[0]

        train_scores.append(train_cindex)
        val_scores.append(val_cindex)

    learning_df = pd.DataFrame({
        "fraction": np.tile(train_sizes, 2),
        "cindex": np.concatenate([train_scores, val_scores]),
        "set": ["Train"] * len(train_sizes) + ["Validation"] * len(train_sizes)
    })

    sns.set(style="whitegrid", font_scale=1.2)

    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=learning_df,
        x="fraction",
        y="cindex",
        hue="set",
        style="set",
        markers=True,
        dashes=False,
        linewidth=2,
        palette=sns.color_palette("pastel"), 
        marker="o",
        markersize=8
    )

    plt.title("Learning Curve", fontsize=16, weight="bold")
    plt.xlabel("Fraction of Training Data", fontsize=14)
    plt.ylabel("C-index", fontsize=14)
    plt.ylim(0, 1)
    plt.legend(title="", loc="lower right")
    plt.tight_layout()
    plt.show()


class RiskScorePlotter:
    def __init__(self, model=None, X=None, y=None):
        """
        model: fitted model
        X: features for predictions (required if model is provided)
        y: structured array or dataframe with 'OS_STATUS' (event indicator)
        """
        self.model = model
        self.X = X
        self.y = y

    def get_predictions(self):
        return self.model.predict(self.X)

    # Risk-Score graphs ---

    def plot_overall_distribution(self, bins=40, kde=True, grid=True):
        # shows how are the risk scores spread among the patient base
        risk_scores = self.get_predictions()
        plt.figure(figsize=(8,5))
        sns.histplot(risk_scores, bins=bins, kde=kde, color="skyblue")
        if grid:
            plt.grid(True, linestyle="--", alpha=0.5)
        plt.title("Predicted Risk Score Distribution")
        plt.xlabel("Risk Score")
        plt.ylabel("Count")
        plt.show()

    def plot_by_event_status(self, bins=30, kde=True, grid=True):
        # distribution of scores separated by the event occurence (model ability to separate low/high risk patients)
        if self.y is None:
            raise ValueError("y (event status) must be provided for this plot.")

        risk_scores = self.get_predictions()
        risk_df = pd.DataFrame({
            "risk_score": risk_scores,
            "OS_STATUS": self.y["OS_STATUS"].astype(int)
        })

        plt.figure(figsize=(8,5))
        sns.histplot(risk_df, x="risk_score", hue="OS_STATUS", bins=bins, kde=kde, palette="Set2", alpha=0.7)
        if grid:
            plt.grid(True, linestyle="--", alpha=0.5)
        plt.title("Predicted Risk Score Distribution by Event Status")
        plt.xlabel("Risk Score")
        plt.ylabel("Count")
        plt.show()
    
    # Kaplan-Meier Curve ---
    
    def plot_kaplan_meier(self, title="Kaplan-Meier Curves by Predicted Risk Group"):
        # survival over time by predicted risk group
        
        risk_scores = self.get_predictions()
        # Make a pandas Series for indexing, regardless of y type
        risk_series = pd.Series(risk_scores, index=getattr(self.y, 'index', None))

        # Determine median risk
        median_risk = risk_series.median()
        low_idx = risk_series <= median_risk
        high_idx = risk_series > median_risk

        # Extract survival data depending on type of self.y
        if isinstance(self.y, pd.DataFrame):
            low_risk = self.y.loc[low_idx, ["OS_YEARS", "OS_STATUS"]].copy()
            high_risk = self.y.loc[high_idx, ["OS_YEARS", "OS_STATUS"]].copy()
        else:  # assume structured array
            import numpy as np
            low_risk = pd.DataFrame({
                "OS_YEARS": np.array(self.y["OS_YEARS"])[low_idx.values],
                "OS_STATUS": np.array(self.y["OS_STATUS"])[low_idx.values]
            })
            high_risk = pd.DataFrame({
                "OS_YEARS": np.array(self.y["OS_YEARS"])[high_idx.values],
                "OS_STATUS": np.array(self.y["OS_STATUS"])[high_idx.values]
            })

        # Plot Kaplan-Meier curves
        kmf = KaplanMeierFitter()
        ax = None
        for group_name, df_group in [("Low Risk", low_risk), ("High Risk", high_risk)]:
            kmf.fit(
                durations=df_group["OS_YEARS"],
                event_observed=df_group["OS_STATUS"],
                label=group_name
            )
            ax = kmf.plot_survival_function(ax=ax)

        ax.set_xlabel("Time (years)")
        ax.set_ylabel("Survival probability")
        ax.set_title(title)
        ax.grid(True)
        plt.show()