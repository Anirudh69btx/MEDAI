"""MindCare AI - Feature Engineering Module.

Complete feature-engineering pipeline:

Raw input
    -> DataPreprocessor
    -> One-hot encoding
    -> Exact training OHE schema
    -> VarianceThreshold
    -> Exact final feature schema

The same fitted pipeline is used during training and inference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from .logger import get_logger
from .preprocessing import DataPreprocessor

logger: logging.Logger = get_logger(__name__)

_FEATURE_ENGINEER_META_FILENAME = (
    "feature_engineer_meta.pkl"
)

_OHE_COLUMNS_FILENAME = (
    "ohe_columns.pkl"
)


class FeatureEngineer:
    """End-to-end feature-engineering pipeline."""

    def __init__(
        self,
        target_column: str,
        variance_threshold: float = 0.0,
    ) -> None:

        self.target_column = (
            str(target_column)
            .strip()
            .lower()
            .replace(" ", "_")
        )

        self.variance_threshold = (
            variance_threshold
        )

        self._preprocessor = DataPreprocessor(
            target_column=self.target_column
        )

        self._var_selector: Optional[
            VarianceThreshold
        ] = None

        self._ohe_columns: List[str] = []

        self._feature_names: List[str] = []

        self._categorical_cols: List[str] = []

        self._fitted = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_columns(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Normalise input column names."""

        df = df.copy()

        df.columns = [
            str(c)
            .strip()
            .lower()
            .replace(" ", "_")
            for c in df.columns
        ]

        return df

    def _one_hot_encode(
        self,
        df: pd.DataFrame,
        fit: bool = False,
    ) -> pd.DataFrame:
        """One-hot encode categorical columns."""

        df = df.copy()

        categorical_cols = [
            col
            for col in self._categorical_cols
            if col in df.columns
        ]

        if fit:

            df_encoded = pd.get_dummies(
                df,
                columns=categorical_cols,
                drop_first=False,
            )

            self._ohe_columns = list(
                df_encoded.columns
            )

        else:

            df_encoded = pd.get_dummies(
                df,
                columns=categorical_cols,
                drop_first=False,
            )

            # Exact training schema.
            df_encoded = df_encoded.reindex(
                columns=self._ohe_columns,
                fill_value=0,
            )

        return df_encoded

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
    ) -> "FeatureEngineer":
        """Fit feature engineering on training data."""

        df = self._normalise_columns(df)

        self._preprocessor.fit(df)

        # ALWAYS derive this from the fitted preprocessor.
        self._categorical_cols = list(
            self._preprocessor.categorical_cols
        )

        X_df, _ = self._preprocessor.transform(
            df
        )

        X_df = self._one_hot_encode(
            X_df,
            fit=True,
        )

        X_df = X_df.astype(float)

        self._var_selector = VarianceThreshold(
            threshold=self.variance_threshold
        )

        self._var_selector.fit(
            X_df.values
        )

        support = (
            self._var_selector.get_support()
        )

        self._feature_names = [
            col
            for col, keep
            in zip(
                X_df.columns,
                support,
            )
            if keep
        ]

        self._fitted = True

        logger.info(
            "FeatureEngineer fitted: "
            "%d OHE columns -> %d final features.",
            len(self._ohe_columns),
            len(self._feature_names),
        )

        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Transform data into exact trained feature schema."""

        if not self._fitted:

            raise RuntimeError(
                "FeatureEngineer must be fitted "
                "before calling transform()."
            )

        df = self._normalise_columns(df)

        X_df, _ = self._preprocessor.transform(
            df
        )

        # Use categorical schema learned by preprocessing.
        self._categorical_cols = list(
            self._preprocessor.categorical_cols
        )

        X_df = self._one_hot_encode(
            X_df,
            fit=False,
        )

        X_df = X_df.astype(float)

        missing = [
            col
            for col in self._feature_names
            if col not in X_df.columns
        ]

        if missing:

            raise RuntimeError(
                "Unable to reconstruct required training "
                f"features: {missing}"
            )

        # Exact final feature order.
        X_df = X_df[
            self._feature_names
        ]

        if X_df.shape[1] != len(
            self._feature_names
        ):

            raise RuntimeError(
                "Feature count mismatch after "
                "feature engineering. "
                f"Expected {len(self._feature_names)}, "
                f"got {X_df.shape[1]}."
            )

        if X_df.isnull().any().any():

            raise RuntimeError(
                "NaN values detected after "
                "feature engineering."
            )

        return X_df.reset_index(
            drop=True
        )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_feature_names(
        self,
    ) -> List[str]:
        """Return final feature names."""

        if not self._fitted:

            raise RuntimeError(
                "FeatureEngineer has not been fitted yet."
            )

        return list(
            self._feature_names
        )

    def get_label_encoder(self):
        """Return fitted target encoder."""

        return (
            self._preprocessor.label_encoder
        )

    def encode_labels(
        self,
        y: pd.Series,
    ) -> np.ndarray:
        """Encode labels."""

        return (
            self._preprocessor
            .label_encoder
            .transform(y)
        )

    def decode_labels(
        self,
        y_encoded: np.ndarray,
    ) -> np.ndarray:
        """Decode labels."""

        return (
            self._preprocessor
            .label_encoder
            .inverse_transform(
                y_encoded
            )
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        directory: Path,
    ) -> None:
        """Save fitted pipeline."""

        directory = Path(directory)

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._preprocessor.save(
            directory
        )

        joblib.dump(
            self._ohe_columns,
            directory / _OHE_COLUMNS_FILENAME,
        )

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

        logger.info(
            "FeatureEngineer pipeline saved to %s",
            directory,
        )

    @classmethod
    def load(
        cls,
        directory: Path,
    ) -> "FeatureEngineer":
        """Load fitted pipeline."""

        directory = Path(directory)

        meta = joblib.load(
            directory
            / _FEATURE_ENGINEER_META_FILENAME
        )

        instance = cls(
            target_column=meta[
                "target_column"
            ],
            variance_threshold=meta[
                "variance_threshold"
            ],
        )

        # Load preprocessor first.
        instance._preprocessor = (
            DataPreprocessor.load(
                directory
            )
        )

        # IMPORTANT:
        # The preprocessor is the authoritative source
        # for raw categorical columns.
        instance._categorical_cols = list(
            instance
            ._preprocessor
            .categorical_cols
        )

        instance._ohe_columns = joblib.load(
            directory
            / _OHE_COLUMNS_FILENAME
        )

        instance._feature_names = list(
            meta["feature_names"]
        )

        instance._var_selector = meta[
            "var_selector"
        ]

        instance._fitted = True

        logger.info(
            "FeatureEngineer loaded: "
            "%d OHE columns, %d final features.",
            len(instance._ohe_columns),
            len(instance._feature_names),
        )

        logger.info(
            "Categorical schema restored from "
            "DataPreprocessor: %s",
            instance._categorical_cols,
        )

        return instance


__all__ = [
    "FeatureEngineer",
]