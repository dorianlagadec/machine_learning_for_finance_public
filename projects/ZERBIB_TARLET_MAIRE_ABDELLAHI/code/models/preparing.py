import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple, Union

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.feature_extraction.text import TfidfVectorizer

@dataclass
class SplitData:
    X_train: pd.DataFrame
    X_val: Optional[pd.DataFrame]
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: Optional[pd.Series]
    y_test: pd.Series


class ForwardFillImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in__ = np.asarray(X.columns, dtype=object)
        else:
            self.feature_names_in__ = None
        return self

    def transform(self, X):
        return X.ffill()
    
    def get_feature_names_out(self, input_features=None):
        if self.feature_names_in__ is not None:
            return self.feature_names_in__
        if input_features is None:
            return np.array([], dtype=object)
        return np.asarray(input_features, dtype=object)


class Dataset:
    def __init__(
        self,
        data: pd.DataFrame,
        target: str,
        features: Optional[List[str]] = None,
        timestamp_col: Optional[str] = "timestamp",
    ):
        self.data = data.copy()
        self.target = target
        self.timestamp_col = timestamp_col

        if self.timestamp_col not in self.data.columns:
            raise ValueError(f"timestamp_col='{timestamp_col}' not found in data.")

        self._target_encoder: Optional[LabelEncoder] = None
        self._preprocessor: Optional[ColumnTransformer] = None

        self._get_features(features)
        self._ensure_datetime_sorted()
        self._encode_target()


    ####################################################################################################################################################
    #                                                                   Builders                                                                       #
    ####################################################################################################################################################


    def _get_features(self, features) -> None:
        if features is None:
            features = [c for c in self.data.columns if c not in [self.target, self.timestamp_col]]
        else:
            if any(col not in self.data.columns for col in features):
                raise ValueError("Some specified features not found in data.")
        self.feature_names_in_ = list(features)

    def _ensure_datetime_sorted(self) -> None:
        self.data[self.timestamp_col] = pd.to_datetime(self.data[self.timestamp_col], errors="coerce")
        if self.data[self.timestamp_col].isna().any():
            raise ValueError("Some timestamp values could not be parsed to datetime.")
        self.data = self.data.sort_values(self.timestamp_col)

    def _encode_target(self) -> None:
        if self.target is None:
            raise ValueError("Target variable not specified.")
        if self.data[self.target].dtype in ["object", "category"]:
            self._target_encoder = LabelEncoder()
            self.data[self.target] = self._target_encoder.fit_transform(
                self.data[self.target].astype(str)
            )
        elif pd.api.types.is_float_dtype(self.data[self.target]):
            # Cast float targets that are integer-valued (e.g. 0.0/1.0) to int
            if (self.data[self.target].dropna() % 1 == 0).all():
                self.data[self.target] = self.data[self.target].astype(int)


    ####################################################################################################################################################
    #                                                                   Helpers                                                                        #
    ####################################################################################################################################################


    def split_dataset(
        self,
        test_size: float = 0.2,
        val_size: Optional[float] = None,
        shuffle: bool = False,
    ) -> SplitData:

        if test_size == 0:
            X_full = self.data[self.feature_names_in_]
            y_full = self.data[self.target]
            return SplitData(X_full, None, X_full.iloc[0:0], y_full, None, y_full.iloc[0:0])

        X_train_full, X_test, y_train_full, y_test = train_test_split(
            self.data[self.feature_names_in_],
            self.data[self.target],
            test_size=test_size,
            shuffle=shuffle,
        )

        if val_size:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full,
                y_train_full,
                test_size=val_size,
                shuffle=shuffle,
            )
            return SplitData(X_train, X_val, X_test, y_train, y_val, y_test)

        return SplitData(X_train_full, None, X_test, y_train_full, None, y_test)


    def build_preprocessor(
        self,
        scale_numeric: bool = False,
        use_knn_for_numeric: bool = False,
        text_cols: Optional[List[str]] =None,
    ) -> ColumnTransformer:

        if text_cols is None:
            text_cols = self._detect_text_columns() 
        
        X = self.data[self.feature_names_in_] 

        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X.select_dtypes(include=["object", "category", "bool"]
                                               ).columns if c not in (text_cols or [])]

    
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        if use_knn_for_numeric:
            num_steps = [("imputer", KNNImputer(n_neighbors=5, weights='distance'))]
        else:
            # Use constant-fill (0) by default: semantically correct for panel/cross-sectional
            # data where rows are different entities, not sequential time steps of one entity.
            # ForwardFill across rows would leak values between different stocks.
            num_steps = [("imputer", SimpleImputer(strategy='constant', fill_value=0))]

        if scale_numeric:
            num_steps.append(("scaler", StandardScaler()))
        num_pipe = Pipeline(steps=num_steps)

        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("ohe", OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)),
            ]
        )
        transformers = [
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
            ]
        for col in text_cols:
            text_pipe = Pipeline(steps=[
                ("tfidf", TfidfVectorizer(max_features=5000, stop_words='english'))
            ])
            transformers.append((f"text_{col}", text_pipe, col))

        self._preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=True,
        )
        return self._preprocessor


    def prepare_dataset(
        self,
        test_size: float = 0.2,
        val_size: Optional[float] = None,
        shuffle: bool = False,
        scale_numeric: bool = False,
        use_knn_for_numeric: bool = False,
        text_cols: Optional[List[str]] = None,
    ) -> None:

        splits = self.split_dataset(test_size=test_size, val_size=val_size, shuffle=shuffle)

        self.build_preprocessor(
            scale_numeric=scale_numeric,
            use_knn_for_numeric=use_knn_for_numeric,
            text_cols=text_cols,
        )

        if self._preprocessor is None:
            raise ValueError("Call build_preprocessor() and fit_transform() first.")

        X_train_t = self._preprocessor.fit_transform(splits.X_train)
        self.feature_names_out_ = self._preprocessor.get_feature_names_out()
        X_train_t = pd.DataFrame(X_train_t, columns=self.feature_names_out_, index=splits.X_train.index)

        X_test_t = self._preprocessor.transform(splits.X_test)
        X_test_t = pd.DataFrame(X_test_t, columns=self.feature_names_out_, index=splits.X_test.index)

        if splits.X_val is not None:
            X_val_t = self._preprocessor.transform(splits.X_val)
            X_val_t = pd.DataFrame(X_val_t, columns=self.feature_names_out_, index=splits.X_val.index)

        self.splits = SplitData(
            X_train_t,
            X_val_t if 'X_val_t' in locals() else None,
            X_test_t,
            splits.y_train,
            splits.y_val,
            splits.y_test
        )

        # Set attributes for direct access
        self.X_train = X_train_t
        self.X_test = X_test_t
        self.y_train = splits.y_train
        self.y_test = splits.y_test


    @classmethod
    def from_split(
        cls,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        features: Optional[List[str]] = None,
        timestamp_col: Optional[str] = None,
        scale_numeric: bool = False,
        use_knn_for_numeric: bool = False,
    ) -> "Dataset":
        """
        Create a Dataset from already-split data (competition / pre-split scenario).
        The preprocessor is fit on X_train only and then applied to X_test.
        Avoids the need to merge then re-split a single DataFrame.

        Parameters
        ----------
        X_train, y_train : training features and target
        X_test           : held-out / submission features (no labels)
        features         : subset of columns to use (default: all columns of X_train)
        timestamp_col    : name of the date column in X_train (kept aside for date-based CV)
        """
        ds = cls.__new__(cls)
        ds._target_encoder = None
        ds._preprocessor   = None
        ds.target          = "target"
        ds.timestamp_col   = timestamp_col

        if features is None:
            features = [c for c in X_train.columns if c != timestamp_col]
        ds.feature_names_in_ = list(features)

        # Store the date index for date-based CV (not used as a feature)
        if timestamp_col and timestamp_col in X_train.columns:
            ds.train_dates = X_train[timestamp_col].copy()
        else:
            ds.train_dates = None

        # Encode target
        y = y_train.copy()
        if y.dtype in ["object", "category"]:
            ds._target_encoder = LabelEncoder()
            y = pd.Series(
                ds._target_encoder.fit_transform(y.astype(str)), index=y.index
            )
        elif pd.api.types.is_float_dtype(y):
            if (y.dropna() % 1 == 0).all():
                y = y.astype(int)

        # Build preprocessor
        X = X_train[features]
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        if use_knn_for_numeric:
            num_steps = [("imputer", KNNImputer(n_neighbors=5, weights="distance"))]
        else:
            num_steps = [("imputer", SimpleImputer(strategy="constant", fill_value=0))]
        if scale_numeric:
            num_steps.append(("scaler", StandardScaler()))

        ds._preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline(steps=num_steps), num_cols),
                ("cat", Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
                ]), cat_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=True,
        )

        X_train_t = ds._preprocessor.fit_transform(X_train[features])
        ds.feature_names_out_ = ds._preprocessor.get_feature_names_out()
        X_train_df = pd.DataFrame(X_train_t, columns=ds.feature_names_out_, index=X_train.index)
        X_test_df  = pd.DataFrame(
            ds._preprocessor.transform(X_test[features]),
            columns=ds.feature_names_out_, index=X_test.index
        )

        empty_y = pd.Series(dtype=y.dtype)
        ds.splits  = SplitData(X_train_df, None, X_test_df, y, None, empty_y)
        ds.X_train = X_train_df
        ds.X_test  = X_test_df
        ds.y_train = y
        ds.y_test  = empty_y
        # store raw X_train for date-based CV (needed to build local train/val masks)
        ds._raw_X_train = X_train.copy()
        ds._raw_y_train = y_train.copy()
        return ds


    ####################################################################################################################################################
    #                                                                   Helpers                                                                        #
    ####################################################################################################################################################


    def basic_summary(self) -> None:
        print("=" * 80)
        print("DATASET SUMMARY (TIME-SERIES)")
        print("=" * 80)
        print(f"\nShape: {self.data.shape}")
        print(f"\nColumns: {self.data.columns.tolist()}")
        ts = self.data[self.timestamp_col]
        print(f"\nDate range: {ts.min()} to {ts.max()}")
        print("\nMissing values:")
        print(self.data.isnull().sum())
        print("\nBasic statistics (numeric):")
        print(self.data.select_dtypes(include=[np.number]).describe())
        print("=" * 80)


    ######################################################################################################################################################
    #                                                                   Other Methods                                                                    #
    ######################################################################################################################################################



    def add_datetime_features(
        self,
        drop_original: bool = False,
    ) -> None:
        col = self.timestamp_col
        ts = self.data[col]

        self.data[f"{col}_year"] = ts.dt.year
        self.data[f"{col}_month"] = ts.dt.month
        self.data[f"{col}_day"] = ts.dt.day
        self.data[f"{col}_dayofweek"] = ts.dt.dayofweek
        self.data[f"{col}_hour"] = ts.dt.hour
        self.data[f"{col}_is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

        for new_col in [
            f"{col}_year", f"{col}_month", f"{col}_day",
            f"{col}_dayofweek", f"{col}_hour", f"{col}_is_weekend"
        ]:
            if new_col not in self.feature_names_in_ and new_col != self.target:
                self.feature_names_in_.append(new_col)

        if drop_original:
            self.data.drop(columns=[col], inplace=True, errors="ignore")
            # Safe removal from feature_names_in_
            if col in self.feature_names_in_:
                self.feature_names_in_.remove(col)

    def add_lag_features(
        self,
        cols: List[str],
        lags: List[int],
        group_col: Optional[str] = None,
        drop_na_rows: bool = True,
    ) -> None:
        """
        Add lag features. For panel data (multiple entities per timestamp),
        pass group_col (e.g. stock ID) so lags are computed within each entity,
        not across different entities.
        """
        for c in cols:
            if c not in self.data.columns:
                raise ValueError(f"Column '{c}' not found.")
            for lag in lags:
                new_name = f"{c}_lag{lag}"
                if group_col is not None:
                    self.data[new_name] = self.data.groupby(group_col)[c].shift(lag)
                else:
                    self.data[new_name] = self.data[c].shift(lag)
                if new_name not in self.feature_names_in_ and new_name != self.target:
                    self.feature_names_in_.append(new_name)

        if drop_na_rows:
            new_lag_cols = [f"{c}_lag{lag}" for c in cols for lag in lags]
            self.data = self.data.dropna(subset=new_lag_cols)

    def add_rolling_features(
        self,
        cols: List[str],
        windows: List[int],
        stats: Tuple[str, ...] = ("mean", "std"),
        group_col: Optional[str] = None,
        drop_na_rows: bool = True,
    ) -> None:
        """
        Rolling window features: mean/std/etc over the past window.
        Uses past-only rolling (shift(1) before rolling to avoid leakage).
        For panel data (multiple entities per timestamp), pass group_col
        so rolling stats are computed within each entity, not across different ones.
        """
        rolling_features = {}

        for c in cols:
            if c not in self.data.columns:
                raise ValueError(f"Column '{c}' not found.")
            if group_col is not None:
                base = self.data.groupby(group_col)[c].shift(1)
            else:
                base = self.data[c].shift(1)
            for w in windows:
                r = base.rolling(window=w, min_periods=w)
                if "mean" in stats:
                    rolling_features[f"{c}_roll{w}_mean"] = r.mean()
                if "std" in stats:
                    rolling_features[f"{c}_roll{w}_std"] = r.std()
                if "min" in stats:
                    rolling_features[f"{c}_roll{w}_min"] = r.min()
                if "max" in stats:
                    rolling_features[f"{c}_roll{w}_max"] = r.max()

        rolling_df = pd.DataFrame(rolling_features, index=self.data.index)
        self.data = pd.concat([self.data, rolling_df], axis=1)
        self.feature_names_in_.extend(rolling_df.columns.tolist())

        if drop_na_rows:
            self.data.dropna(subset=rolling_df.columns, inplace=True)
            self.data.reset_index(drop=True, inplace=True)

    def drop_high_missing(self, threshold: float = 0.4) -> None:
        miss = self.data.isna().mean()
        to_drop = miss[miss > threshold].index.tolist()

        # Print columns being dropped due to high missing values
        if to_drop:
            print(f"Dropping columns due to high missing values (>{threshold*100}%): {to_drop}")

        self.data.drop(columns=to_drop, inplace=True, errors="ignore")
        self.feature_names_in_ = [f for f in self.feature_names_in_ if f not in to_drop]

    def clip_outliers(self, lower_q: float = 0.01, upper_q: float = 0.99) -> None:
        if lower_q >= upper_q:
            raise ValueError("lower_q must be < upper_q")

        num_cols = self.data[self.feature_names_in_].select_dtypes(include=[np.number]).columns
        for c in num_cols:
            lo = self.data[c].quantile(lower_q)
            hi = self.data[c].quantile(upper_q)
            self.data[c] = self.data[c].clip(lo, hi)

    def _detect_text_columns(self, threshold_unique: int = 100, min_avg_length: int = 20) -> List[str]:

        text_cols = []
        potential_cols = self.data[self.feature_names_in_].select_dtypes(include=['object', 'string'])
        
        for col in potential_cols.columns:
            unique_count = self.data[col].nunique()
            avg_length = self.data[col].str.len().mean()
            
            if unique_count > threshold_unique and avg_length > min_avg_length:
                text_cols.append(col)
                
        return text_cols