"""MindCare AI - Feature Engineering Module.

Implements the ``FeatureEngineer`` class, which wraps the full preprocessing
and feature-transformation pipeline:

- Delegates raw cleaning to ``DataPreprocessor``
- Drops low-variance features
- One-hot encodes remaining categorical columns
- Returns a scaled, encoded numeric DataFrame suitable for all model types

The class implements fit-on-train / transform-on-all semantics to prevent
data leakage, and provides ``save()`` / ``load()`` methods for persistence so
the exact same pipeline can be reapplied during inference.

``trainer.py`` calls ``self.feature_engineer.save(config.MODELS_DIR)`` after
fitting; this module satisfies that contract.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from .config import config
from .logger import get_logger
from .preprocessing import DataPreprocessor

logger: logging.Logger = get_logger(__name__)

_FEATURE_ENGINEER_META_FILENAME = "feature_engineer_meta.pkl"
_OHE_COLUMNS_FILENAME = "ohe_columns.pkl"


class FeatureEngineer:
    """End-to-end feature engineering pipeline.

    Combines ``DataPreprocessor`` (cleaning + scaling) with one-hot encoding
    and low-variance feature removal.

    Parameters
    ----------
    target_column : str
        Name of the classification target column.
    variance_threshold : float, optional
        Features with variance below this threshold are dropped.
        Defaults to 0.0 (removes zero-variance features only).
    """

    def __init__(
        self,
        target_column: str,
        variance_threshold: float = 0.0,
    ) -> None:
        self.target_column = target_column
        self.variance_threshold = variance_threshold

        self._preprocessor: DataPreprocessor = DataPreprocessor(
            target_column=target_column
        )
        self._var_selector: Optional[VarianceThreshold] = None
        self._ohe_columns: List[str] = []      # column names after one-hot encoding
        self._feature_names: List[str] = []    # final feature names after VT
        self._categorical_cols: List[str] = []
        self._fitted: bool = False

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _one_hot_encode(
        self, df: pd.DataFrame, fit: bool = False
    ) -> pd.DataFrame:
        """Apply one-hot encoding to categorical columns.

        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame (target already removed).
        fit : bool
            When ``True`` record the resulting column names for later alignment.

        Returns
        -------
        pd.DataFrame
            DataFrame with categorical columns replaced by OHE dummies.
        """
        cat_cols = [
            c for c in self._categorical_cols if c in df.columns
        ]
        if not cat_cols:
            if fit:
                self._ohe_columns = list(df.columns)
            return df

        df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False)

        if fit:
            self._ohe_columns = list(df_encoded.columns)
        else:
            # Align to fitted columns, filling missing ones with 0
            for col in self._ohe_columns:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
            df_encoded = df_encoded[self._ohe_columns]

        return df_encoded

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        """Fit the full feature engineering pipeline on the training DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Training data including the target column.

        Returns
        -------
        FeatureEngineer
            Self, for method chaining.
        """
        # Step 1: Clean, scale, encode target
        self._preprocessor.fit(df)
        self._categorical_cols = list(self._preprocessor.categorical_cols)

        X_df, _ = self._preprocessor.transform(df)

        # Step 2: One-hot encode categorical columns
        X_df = self._one_hot_encode(X_df, fit=True)

        # Ensure all values are numeric (bool -> int)
        X_df = X_df.astype(float)

        # Step 3: Remove low-variance features
        self._var_selector = VarianceThreshold(threshold=self.variance_threshold)
        self._var_selector.fit(X_df.values)
        support_mask = self._var_selector.get_support()
        self._feature_names = [
            col for col, keep in zip(X_df.columns, support_mask) if keep
        ]

        self._fitted = True

        logger.info(
            "FeatureEngineer fitted: %d features after OHE + VarianceThreshold (threshold=%.4f).",
            len(self._feature_names),
            self.variance_threshold,
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted pipeline to *df* and return processed feature DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Data split to transform (may or may not include the target column).

        Returns
        -------
        pd.DataFrame
            Processed, numeric-only feature DataFrame.
        """
        if not self._fitted:
            raise RuntimeError(
                "FeatureEngineer must be fitted before calling transform()."
            )

        X_df, _ = self._preprocessor.transform(df)
        X_df = self._one_hot_encode(X_df, fit=False)
        X_df = X_df.astype(float)

        # Apply variance threshold using saved feature names
        X_df = X_df[[c for c in self._feature_names if c in X_df.columns]]

        return X_df.reset_index(drop=True)

    def get_feature_names(self) -> List[str]:
        """Return the ordered list of final feature names after fitting.

        Returns
        -------
        List[str]
            Feature column names.

        Raises
        ------
        RuntimeError
            If called before ``fit()``.
        """
        if not self._fitted:
            raise RuntimeError("FeatureEngineer has not been fitted yet.")
        return list(self._feature_names)

    def get_label_encoder(self):
        """Return the fitted ``LabelEncoder`` from the internal ``DataPreprocessor``."""
        return self._preprocessor.label_encoder

    def encode_labels(self, y: pd.Series) -> np.ndarray:
        """Encode a Series of raw target labels using the fitted LabelEncoder.

        Parameters
        ----------
        y : pd.Series
            Raw (string or integer) target labels.

        Returns
        -------
        np.ndarray
            Integer-encoded label array.
        """
        return self._preprocessor.label_encoder.transform(y)

    def decode_labels(self, y_encoded: np.ndarray) -> np.ndarray:
        """Reverse integer-encoded labels back to original class names.

        Parameters
        ----------
        y_encoded : np.ndarray
            Integer-encoded labels.

        Returns
        -------
        np.ndarray
            Original class label strings/values.
        """
        return self._preprocessor.label_encoder.inverse_transform(y_encoded)

    def save(self, directory: Path) -> None:
        """Persist the full FeatureEngineer pipeline to *directory*.

        Saves:
        - All ``DataPreprocessor`` artifacts (scaler, label encoder, etc.)
        - OHE column list
        - VarianceThreshold selector
        - Final feature names
        - Metadata dict

        Parameters
        ----------
        directory : Path
            Destination directory (created if absent).
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Persist inner preprocessor artifacts
        self._preprocessor.save(directory)

        # Persist OHE columns
        joblib.dump(self._ohe_columns, directory / _OHE_COLUMNS_FILENAME)

        # Persist meta (feature names, variance selector, config)
        joblib.dump(
            {
                "feature_names": self._feature_names,
                "categorical_cols": self._categorical_cols,
                "variance_threshold": self.variance_threshold,
                "var_selector": self._var_selector,
                "target_column": self.target_column,
            },
            directory / _FEATURE_ENGINEER_META_FILENAME,
        )

        logger.info("FeatureEngineer pipeline saved to %s", directory)

    @classmethod
    def load(cls, directory: Path) -> "FeatureEngineer":
        """Load a previously saved ``FeatureEngineer`` from *directory*.

        Parameters
        ----------
        directory : Path
            Directory containing serialised pipeline artifacts.

        Returns
        -------
        FeatureEngineer
            Fully restored pipeline instance ready for inference.
        """
        directory = Path(directory)

        meta = joblib.load(directory / _FEATURE_ENGINEER_META_FILENAME)
        instance = cls(
            target_column=meta["target_column"],
            variance_threshold=meta["variance_threshold"],
        )

        # Restore inner preprocessor
        instance._preprocessor = DataPreprocessor.load(directory)
        instance._ohe_columns = joblib.load(directory / _OHE_COLUMNS_FILENAME)
        instance._feature_names = meta["feature_names"]
        instance._categorical_cols = meta["categorical_cols"]
        instance._var_selector = meta["var_selector"]
        instance._fitted = True

        logger.info("FeatureEngineer pipeline loaded from %s", directory)
        return instance


__all__ = ["FeatureEngineer"]
