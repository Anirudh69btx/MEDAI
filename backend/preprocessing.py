"""MindCare AI - Data Preprocessing Module.

Handles all raw data cleaning operations before feature engineering:
- Duplicate row removal
- Column name normalisation (strip whitespace, lowercase)
- Numeric column coercion with NaN-safe imputation
- Categorical column imputation with mode
- Outlier clamping using the IQR method
- Target label encoding via sklearn LabelEncoder
- StandardScaler fitting and transformation

The ``DataPreprocessor`` class is designed for fit-on-train / transform-on-all
semantics, preventing data leakage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import config
from .logger import get_logger

logger: logging.Logger = get_logger(__name__)


class DataPreprocessor:
    """Stateful data cleaning and scaling pipeline.

    Call ``fit_transform`` on the training set, then ``transform`` on
    validation and test sets to prevent data leakage.

    Attributes
    ----------
    target_column : str
        The name of the target/label column in the DataFrame.
    label_encoder : LabelEncoder
        Fitted sklearn LabelEncoder for the target column.
    scaler : StandardScaler
        Fitted sklearn StandardScaler for numeric features.
    feature_columns : List[str]
        Ordered list of feature column names after preprocessing.
    numeric_cols : List[str]
        Numeric columns identified during fit.
    categorical_cols : List[str]
        Categorical (non-numeric, non-target) columns identified during fit.
    _col_medians : dict
        Median values of numeric columns used for NaN imputation.
    _col_modes : dict
        Mode values of categorical columns used for NaN imputation.
    """

    def __init__(self, target_column: str) -> None:
        """Initialise the preprocessor.

        Parameters
        ----------
        target_column : str
            Name of the target label column.
        """
        self.target_column = target_column
        self.label_encoder: LabelEncoder = LabelEncoder()
        self.scaler: StandardScaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self._col_medians: dict = {}
        self._col_modes: dict = {}
        self._fitted: bool = False

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Strip whitespace and convert column names to lowercase."""
        df = df.copy()
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        return df

    def _identify_column_types(self, df: pd.DataFrame) -> None:
        """Identify numeric and categorical feature columns (excluding target)."""
        feature_cols = [c for c in df.columns if c != self.target_column]
        self.numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = [
            c for c in feature_cols if c not in self.numeric_cols
        ]

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        if removed:
            logger.info("Removed %d duplicate rows.", removed)
        return df

    def _impute_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN in numeric columns with fitted medians."""
        for col in self.numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                if df[col].isnull().any():
                    df[col] = df[col].fillna(self._col_medians.get(col, 0.0))
        return df

    def _impute_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN in categorical columns with fitted modes."""
        for col in self.categorical_cols:
            if col in df.columns and df[col].isnull().any():
                df[col] = df[col].fillna(self._col_modes.get(col, "unknown"))
        return df

    def _clamp_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clamp extreme outliers in numeric columns using IQR method (fit values)."""
        for col in self.numeric_cols:
            if col not in df.columns:
                continue
            lower = self._iqr_bounds.get(col, {}).get("lower")
            upper = self._iqr_bounds.get(col, {}).get("upper")
            if lower is not None and upper is not None:
                df[col] = df[col].clip(lower=lower, upper=upper)
        return df

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """Fit the preprocessor on the training DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Training data including the target column.

        Returns
        -------
        DataPreprocessor
            Self, for method chaining.
        """
        df = self._normalise_columns(df)

        if self.target_column not in df.columns:
            raise ValueError(
                f"Target column '{self.target_column}' not found in DataFrame."
            )

        df = self._remove_duplicates(df)
        self._identify_column_types(df)

        # Fit imputation statistics on training data only
        for col in self.numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            self._col_medians[col] = float(df[col].median())

        for col in self.categorical_cols:
            mode_val = df[col].mode()
            self._col_modes[col] = mode_val.iloc[0] if not mode_val.empty else "unknown"

        # Fit IQR bounds for outlier clamping
        self._iqr_bounds: dict = {}
        for col in self.numeric_cols:
            q1 = float(df[col].quantile(0.25))
            q3 = float(df[col].quantile(0.75))
            iqr = q3 - q1
            self._iqr_bounds[col] = {
                "lower": q1 - 1.5 * iqr,
                "upper": q3 + 1.5 * iqr,
            }

        # Impute and clamp training data, then fit scaler and encoder
        df = self._impute_numeric(df)
        df = self._impute_categorical(df)
        df = self._clamp_outliers(df)

        # Fit LabelEncoder on target
        self.label_encoder.fit(df[self.target_column])

        # Fit StandardScaler on numeric feature columns
        if self.numeric_cols:
            self.scaler.fit(df[self.numeric_cols])

        self.feature_columns = self.numeric_cols + self.categorical_cols
        self._fitted = True

        logger.info(
            "DataPreprocessor fitted: %d numeric, %d categorical feature columns.",
            len(self.numeric_cols),
            len(self.categorical_cols),
        )
        return self

    def transform(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """Apply fitted transformations to a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Data split to transform. May or may not contain the target column.

        Returns
        -------
        Tuple[pd.DataFrame, Optional[np.ndarray]]
            (transformed_features_df, encoded_labels_or_None)
        """
        if not self._fitted:
            raise RuntimeError("DataPreprocessor must be fitted before calling transform().")

        df = self._normalise_columns(df)
        df = df.copy()

        # Impute and clamp
        df = self._impute_numeric(df)
        df = self._impute_categorical(df)
        df = self._clamp_outliers(df)

        # Extract and encode target if present
        y: Optional[np.ndarray] = None
        if self.target_column in df.columns:
            y = self.label_encoder.transform(df[self.target_column])
            df = df.drop(columns=[self.target_column])

        # Scale numeric features
        if self.numeric_cols:
            present_numeric = [c for c in self.numeric_cols if c in df.columns]
            df[present_numeric] = self.scaler.transform(df[present_numeric])

        # Keep only known feature columns in consistent order
        present_features = [c for c in self.feature_columns if c in df.columns]
        df = df[present_features]

        return df, y

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
        """Fit on *df* then return its transformation.

        Convenience method equivalent to ``fit(df).transform(df)``.
        """
        self.fit(df)
        return self.transform(df)

    def save(self, directory: Path) -> None:
        """Persist all fitted artifacts to *directory*.

        Parameters
        ----------
        directory : Path
            Destination directory (created if absent).
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.label_encoder, directory / config.LABEL_ENCODER_FILENAME)
        joblib.dump(self.scaler, directory / config.SCALER_FILENAME)
        joblib.dump(self.feature_columns, directory / config.FEATURE_NAMES_FILENAME)
        joblib.dump(
            {
                "numeric_cols": self.numeric_cols,
                "categorical_cols": self.categorical_cols,
                "col_medians": self._col_medians,
                "col_modes": self._col_modes,
                "iqr_bounds": self._iqr_bounds,
                "target_column": self.target_column,
            },
            directory / config.FEATURE_ENCODER_FILENAME,
        )
        logger.info("DataPreprocessor artifacts saved to %s", directory)

    @classmethod
    def load(cls, directory: Path) -> "DataPreprocessor":
        """Load a previously saved ``DataPreprocessor`` from *directory*.

        Parameters
        ----------
        directory : Path
            Directory containing the serialised artifacts.

        Returns
        -------
        DataPreprocessor
            Restored, ready-to-use preprocessor instance.
        """
        directory = Path(directory)
        meta = joblib.load(directory / config.FEATURE_ENCODER_FILENAME)

        instance = cls(target_column=meta["target_column"])
        instance.label_encoder = joblib.load(directory / config.LABEL_ENCODER_FILENAME)
        instance.scaler = joblib.load(directory / config.SCALER_FILENAME)
        instance.feature_columns = joblib.load(directory / config.FEATURE_NAMES_FILENAME)
        instance.numeric_cols = meta["numeric_cols"]
        instance.categorical_cols = meta["categorical_cols"]
        instance._col_medians = meta["col_medians"]
        instance._col_modes = meta["col_modes"]
        instance._iqr_bounds = meta["iqr_bounds"]
        instance._fitted = True

        logger.info("DataPreprocessor loaded from %s", directory)
        return instance


__all__ = ["DataPreprocessor"]
