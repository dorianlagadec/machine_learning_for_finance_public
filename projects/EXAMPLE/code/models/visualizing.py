import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression

import lightgbm as lgb

class EDAVisualizer:

    def __init__(self,
                 df: Optional[pd.DataFrame] = None,
                 target: Optional[str] = None,
                 timestamp_col: Optional[str] = None,
                 price_col: Optional[str] = None,
                 ) -> None:

        self.df = df.copy()
        self.target = target
        self.timestamp_col = timestamp_col
        self.price_col = price_col

        self.numeric_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

        if self.timestamp_col and self.timestamp_col in self.df.columns:
            self.df[self.timestamp_col] = pd.to_datetime(self.df[self.timestamp_col], errors='coerce')
            self.df = self.df.sort_values(self.timestamp_col)

        self.categorical_cols = [c for c in self.df.select_dtypes(include=['object', 'category', 'bool']).columns 
                         if self.df[c].nunique() < 50]

    def run_all(self, output_dir=None):
        n_plots = 0
        
        # Missing (bar + heatmap)
        if self.df.isnull().sum().sum() > 0:
            n_plots += 2
        else:
             pass 
        
        # Missing
        missing_series = self.df.isnull().mean() * 100
        if (missing_series > 0).any():
             n_plots += 1 # bar
        n_plots += 1 # heatmap

        # Distributions (hist + box per numeric col)
        n_plots += 2 * len(self.numeric_cols)

        # Correlations
        if len(self.numeric_cols) >= 2: n_plots += 1
        if len(self.categorical_cols) >= 2: n_plots += 1

        # Categorical (bar + potential pie)
        for col in self.categorical_cols:
            n_plots += 1 # bar
            if self.df[col].nunique() <= 10:
                n_plots += 1 # pie

        # Target Relationships
        if self.target and self.target in self.df.columns:
            if self.target in self.numeric_cols:
                # Scatter for numeric
                for col in self.numeric_cols:
                    if col != self.target: n_plots += 1
                # Box for categorical
                n_plots += len(self.categorical_cols)
            else:
                # Box for numeric
                n_plots += len(self.numeric_cols)
                # Stacked bar for categorical
                n_plots += len(self.categorical_cols)

        # Time Series
        if self.timestamp_col and self.timestamp_col in self.df.columns:
            if self.price_col and self.price_col in self.df.columns:
                n_plots += 4 # price, returns, returns hist, returns box

        # 2. Setup Grid
        ncols = 2
        nrows = math.ceil(n_plots / ncols)
        
        # Larger per-axis area improves readability for dense labels/annotations.
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.8 * nrows))
        axes_flat = axes.flatten()
        axes_iter = iter(axes_flat)

        # 3. Plotting
        self.plot_missing(axes_iter)
        self.plot_distributions(axes_iter)
        self.plot_correlations(axes_iter)
        self.plot_categorical(axes_iter)
        self.plot_categorical_associations(axes_iter)
        self.plot_target_relationships(axes_iter)
        self.plot_time_series(axes_iter)

        # Hide unused axes
        # We can't easily know index of axes_iter, so we iterate remaining
        for ax in axes_iter:
            ax.set_visible(False)

        plt.tight_layout()
        plt.show()
        print("EDA completed: Charts displayed inline.")

    def _get_ax(self, axes_iter):
        try:
            return next(axes_iter)
        except StopIteration:
            return None


    def plot_missing(self, axes_iter):
        missing = self.df.isnull().mean() * 100
        missing = missing[missing > 0].sort_values(ascending=False)

        if len(missing) > 0:
            ax = self._get_ax(axes_iter)
            if ax:
                missing.plot(kind='bar', ax=ax)
                ax.set_title("Missing Data Percentage")
                ax.set_ylabel("Percentage")
                ax.tick_params(axis='x', rotation=45)

        # Missing heatmap
        ax = self._get_ax(axes_iter)
        if ax:
            # Use 'binary' cmap where 0 (False/Present) is White, 1 (True/Missing) is Black
            ax.imshow(self.df.isnull(), aspect='auto', cmap='binary', interpolation='none')
            ax.set_title("Missing Data Pattern")
            ax.set_ylabel("Samples")
            
            # Set x-ticks to column names
            ax.set_xticks(range(len(self.df.columns)))
            ax.set_xticklabels(self.df.columns, rotation=90)
            
            # Removing y-ticks for cleaner look on heatmap if many rows
            ax.set_yticks([]) 
            # Add grid on x-axis to separate features clearly
            ax.grid(axis='x', color='lightgray', linestyle='-', linewidth=0.5, alpha=0.5) 


    def plot_distributions(self, axes_iter):
        for col in self.numeric_cols:
            
            # Histogram
            ax = self._get_ax(axes_iter)
            if ax:
                self.df[col].hist(bins=50, ax=ax)
                ax.set_title(f"{col} Distribution")

            # Boxplot
            ax = self._get_ax(axes_iter)
            if ax:
                self.df.boxplot(column=col, ax=ax)
                ax.set_title(f"{col} Boxplot")

    
    def plot_correlations(self, axes_iter):
        if len(self.numeric_cols) < 2:
            return

        corr = self.df[self.numeric_cols].corr()

        ax = self._get_ax(axes_iter)
        if ax:
            im = ax.imshow(corr, aspect='auto')
            fig = ax.get_figure()
            fig.colorbar(im, ax=ax)
            
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45)
            ax.set_yticklabels(corr.columns)
            ax.set_title("Correlation Matrix")

            # Add values
            for i in range(len(corr.columns)):
                for j in range(len(corr.columns)):
                    ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                             ha='center', va='center', fontsize=8)


    def plot_categorical(self, axes_iter):
        for col in self.categorical_cols:
            counts = self.df[col].value_counts(dropna=False)
            
            # If too many categories, take top 20
            if len(counts) > 20:
                counts = counts.head(20)

            # Bar plot
            ax = self._get_ax(axes_iter)
            if ax:
                counts.plot(kind='bar', ax=ax)
                ax.set_title(f"{col} Distribution (Top {len(counts)})")
                ax.set_ylabel("Count")
                ax.tick_params(axis='x', rotation=45)
                
                # Add percentage labels
                total = len(self.df)
                for p in ax.patches:
                    percentage = '{:.1f}%'.format(100 * p.get_height() / total)
                    x = p.get_x() + p.get_width() / 2
                    y = p.get_height()
                    ax.annotate(percentage, (x, y), ha='center', va='bottom')

            # Pie chart for low cardinality (< 10)
            if len(counts) <= 10:
                ax = self._get_ax(axes_iter)
                if ax:
                    counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, ax=ax)
                    ax.set_title(f"{col} proportions")
                    ax.set_ylabel("")


    def plot_categorical_associations(self, axes_iter):
        if len(self.categorical_cols) < 2:
            return

        # Helper to calculate Cramer's V (simplified)
        def cramers_v(x, y):
            confusion_matrix = pd.crosstab(x, y)
            try:
                from scipy.stats import chi2_contingency
                chi2, _, _, _ = chi2_contingency(confusion_matrix)
                n = confusion_matrix.sum().sum()
                return np.sqrt(chi2 / (n * (min(confusion_matrix.shape) - 1)))
            except:
                return 0.0

        cols = self.categorical_cols
        matrix = np.zeros((len(cols), len(cols)))
        
        for i in range(len(cols)):
            for j in range(len(cols)):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    matrix[i, j] = cramers_v(self.df[cols[i]], self.df[cols[j]])

        if np.all(matrix == 0):
            return

        ax = self._get_ax(axes_iter)
        if ax:
            im = ax.imshow(matrix, cmap='coolwarm', vmin=0, vmax=1)
            fig = ax.get_figure()
            fig.colorbar(im, ax=ax)
            
            ax.set_xticks(range(len(cols)))
            ax.set_yticks(range(len(cols)))
            ax.set_xticklabels(cols, rotation=45)
            ax.set_yticklabels(cols)
            ax.set_title("Categorical Associations (Cramer's V)")
            
            for i in range(len(cols)):
                for j in range(len(cols)):
                    ax.text(j, i, f"{matrix[i, j]:.2f}",
                             ha='center', va='center', fontsize=8)

    
    def plot_target_relationships(self, axes_iter):
        if not self.target or self.target not in self.df.columns:
            return

        # Numeric target
        if self.target in self.numeric_cols:
            for col in self.numeric_cols:
                if col == self.target:
                    continue
                ax = self._get_ax(axes_iter)
                if ax:
                    ax.scatter(self.df[col], self.df[self.target], alpha=0.5)
                    ax.set_xlabel(col)
                    ax.set_ylabel(self.target)
                    ax.set_title(f"{col} vs {self.target}")

            for col in self.categorical_cols:
                ax = self._get_ax(axes_iter)
                if ax:
                    self.df.boxplot(column=self.target, by=col, ax=ax)
                    ax.set_title(f"{self.target} by {col}")
                    ax.tick_params(axis='x', rotation=45)
                    # Remove auto-generated title from pandas boxplot if possible or just let it be
                    # pandas boxplot 'by' usually adds a suptitle to the figure, which might be annoying in subplots.
                    # We can try to suppress it.
                    ax.get_figure().suptitle("") 


        # Categorical target
        else:
            for col in self.numeric_cols:
                ax = self._get_ax(axes_iter)
                if ax:
                    self.df.boxplot(column=col, by=self.target, ax=ax)
                    ax.set_title(f"{col} by {self.target}")
                    ax.get_figure().suptitle("")

            for col in self.categorical_cols:
                ax = self._get_ax(axes_iter)
                if ax:
                    counts = pd.crosstab(self.df[col], self.df[self.target])
                    counts.plot(kind='bar', stacked=True, ax=ax)
                    ax.set_title(f"{col} vs {self.target}")
                    ax.tick_params(axis='x', rotation=45)

    
    def plot_time_series(self, axes_iter):
        if not self.timestamp_col or self.timestamp_col not in self.df.columns:
            return
        if not self.price_col or self.price_col not in self.df.columns:
            return

        df = self.df.dropna(subset=[self.price_col])
        returns = df[self.price_col].pct_change().dropna()

        # Price
        ax = self._get_ax(axes_iter)
        if ax:
            ax.plot(df[self.timestamp_col], df[self.price_col])
            ax.set_title("Price Series")

        # Returns
        ax = self._get_ax(axes_iter)
        if ax:
            ax.plot(df[self.timestamp_col].iloc[1:], returns)
            ax.set_title("Returns Series")

        # Returns Dist
        ax = self._get_ax(axes_iter)
        if ax:
            returns.hist(bins=50, ax=ax)
            ax.set_title("Returns Distribution")

        # Returns Box
        ax = self._get_ax(axes_iter)
        if ax:
            ax.boxplot(returns)
            ax.set_title("Returns Boxplot")


# =============================================================================
# Model Evaluation Visualizer
# =============================================================================

class ModelVisualizer:
    """
    Visualizes model predictions following the same conventions as EDAVisualizer.

    Usage:
        viz = ModelVisualizer(trainer)
        viz.run_all()
    """

    def __init__(self, trainer) -> None:
        import copy
        self.trainer = trainer
        self.is_classification = trainer.is_classification
        self.model = trainer.model
        self.y_train_pred = trainer.y_train_pred
        # Fall back to train data when no labelled test split exists
        # (covers both test_size=0 and Dataset.from_split where y_test is empty)
        if len(trainer.dataset.splits.y_test) > 0:
            self.splits = trainer.dataset.splits
            self.y_test_pred = trainer.y_test_pred
        else:
            self.splits = copy.copy(trainer.dataset.splits)
            self.splits.y_test = trainer.dataset.splits.y_train
            self.splits.X_test = trainer.dataset.splits.X_train
            self.y_test_pred = trainer.y_train_pred

    def _get_ax(self, axes_iter):
        try:
            return next(axes_iter)
        except StopIteration:
            return None

    def _count_plots(self) -> int:
        n = 0
        if self.is_classification:
            n += 1                                          # confusion matrix
            if (hasattr(self.model, 'feature_importances_') or hasattr(self.model, 'feature_log_prob_')
                    or hasattr(self.model, 'coef_')):
                n += 1                                     # feature importances / coefficients
        else:
            n += 4                                         # scatter + timeline + residuals + sign accuracy
            if hasattr(self.model, 'coef_'):
                n += 1                                     # coefficient bar chart
            elif hasattr(self.model, 'feature_importances_'):
                n += 1                                     # feature importances
        return n

    def run_all(self):
        n_plots = self._count_plots()
        ncols = 2
        nrows = math.ceil(n_plots / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.8 * nrows))
        if nrows == 1 and ncols == 1:
            axes_flat = [axes]
        else:
            axes_flat = np.array(axes).flatten()
        axes_iter = iter(axes_flat)

        if self.is_classification:
            self.plot_confusion_matrix(axes_iter)
            if hasattr(self.model, 'feature_importances_'):
                self.plot_feature_importances(axes_iter)
            elif hasattr(self.model, 'coef_'):
                self.plot_coefficients(axes_iter)
        else:
            self.plot_predictions_scatter(axes_iter)
            self.plot_predictions_timeline(axes_iter)
            self.plot_residuals(axes_iter)
            self.plot_sign_accuracy(axes_iter)
            if hasattr(self.model, 'coef_'):
                self.plot_coefficients(axes_iter)
            elif hasattr(self.model, 'feature_importances_'):
                self.plot_feature_importances(axes_iter)

        for ax in axes_iter:
            ax.set_visible(False)

        title = "Model Evaluation — Classification" if self.is_classification else "Model Evaluation — Regression"
        fig.suptitle(title, fontsize=14, y=1.01)
        plt.tight_layout()
        plt.show()

    def plot_predictions_scatter(self, axes_iter):
        ax = self._get_ax(axes_iter)
        if ax is None:
            return

        y_test = self.splits.y_test.values
        y_pred = self.y_test_pred

        ax.scatter(y_test, y_pred, alpha=0.4, edgecolors="k", linewidths=0.3)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted (Test)")
        ax.legend()

    def plot_predictions_timeline(self, axes_iter):
        ax = self._get_ax(axes_iter)
        if ax is None:
            return

        y_test = self.splits.y_test.values
        y_pred = self.y_test_pred
        idx = np.arange(len(y_test))

        ax.plot(idx, y_test, label="Actual", alpha=0.7)
        ax.plot(idx, y_pred, label="Predicted", alpha=0.7)
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Value")
        ax.set_title("Actual vs Predicted Over Samples (Test)")
        ax.legend()

    def plot_residuals(self, axes_iter):
        ax = self._get_ax(axes_iter)
        if ax is None:
            return

        y_test = self.splits.y_test.values
        y_pred = self.y_test_pred
        residuals = y_test - y_pred

        ax.scatter(y_pred, residuals, alpha=0.4, edgecolors="k", linewidths=0.3)
        ax.axhline(0, color="r", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual")
        ax.set_title("Residuals vs Predicted (Test)")

    def plot_coefficients(self, axes_iter):
        """Bar chart of model coefficients (Ridge, LinearRegression, LogisticRegression)."""
        if not hasattr(self.model, 'coef_'):
            return

        ax = self._get_ax(axes_iter)
        if ax is None:
            return
        
        if hasattr(self.model, 'feature_log_prob_'):
            coef = self.model.feature_log_prob_[1]
        elif hasattr(self.model, 'coef_'):
            coef = self.model.coef_
        else:
            return

        feat_names = self.splits.X_train.columns
        coef = self.model.coef_
        if coef.ndim > 1:
            coef = coef[0]

        sorted_idx = np.argsort(np.abs(coef))[::-1]
        top_n = min(20, len(sorted_idx))
        idx = sorted_idx[:top_n]

        colors = ["steelblue" if c >= 0 else "tomato" for c in coef[idx][::-1]]
        ax.barh(range(top_n), coef[idx][::-1], color=colors)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(feat_names[idx][::-1])
        ax.axvline(0, color="k", linewidth=0.8)
        ax.set_xlabel("Coefficient Value")
        ax.set_title(f"Top {top_n} Coefficients (by |value|)")


    def plot_sign_accuracy(self, axes_iter):
        """Confusion matrix of sign(pred) vs sign(actual) — the metric that matters for direction."""
        from sklearn.metrics import confusion_matrix

        ax = self._get_ax(axes_iter)
        if ax is None:
            return

        y_test = self.splits.y_test.values
        y_pred = self.y_test_pred
        threshold = 0.5 if (y_test.max() <= 1 and y_test.min() >= 0) else 0.0
        y_test_sign = (y_test > threshold).astype(int)
        y_pred_sign = (y_pred > threshold).astype(int)

        correct = (y_test_sign == y_pred_sign).mean()
        classes = [0, 1]
        cm = confusion_matrix(y_test_sign, y_pred_sign, labels=classes)

        im = ax.imshow(cm, cmap="Blues")
        ax.get_figure().colorbar(im, ax=ax)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(classes); ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted direction")
        ax.set_ylabel("Actual direction")
        ax.set_title(f"Sign Accuracy: {correct:.2%}")
        threshold_val = cm.max() / 2
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > threshold_val else "black")

    def plot_confusion_matrix(self, axes_iter):
        """Confusion matrix for classification models."""
        from sklearn.metrics import confusion_matrix

        ax = self._get_ax(axes_iter)
        if ax is None:
            return

        y_test = self.splits.y_test.values
        y_pred = self.y_test_pred
        classes = np.unique(y_test)
        cm = confusion_matrix(y_test, y_pred)

        im = ax.imshow(cm, cmap="Blues")
        ax.get_figure().colorbar(im, ax=ax)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix (Test)")

        threshold = cm.max() / 2
        for i in range(len(classes)):
            for j in range(len(classes)):
                bg_intensity = cm[i, j] / (cm.max() + 1e-8)
                text_color = "white" if bg_intensity > 0.5 else "black"
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color=text_color)

    def plot_feature_importances(self, axes_iter):
        """Horizontal bar chart of feature importances (RF, XGB, LGB)."""
        if not hasattr(self.model, 'feature_importances_'):
            return

        ax = self._get_ax(axes_iter)
        if ax is None:
            return

        importances = self.model.feature_importances_
        top_n = min(20, len(importances))
        top_indices = np.argsort(importances)[::-1][:top_n]
        feat_names = self.splits.X_train.columns[top_indices]

        ax.barh(range(top_n), importances[top_indices][::-1])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(feat_names[::-1])
        ax.set_xlabel("Importance")
        ax.set_title(f"Top {top_n} Feature Importances")