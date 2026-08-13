"""MindCare AI - Data Preprocessing Module.

Handles raw data cleaning before feature engineering.

Responsibilities
----------------
- Column-name normalisation
- Duplicate removal
- Numeric type coercion
- Numeric median imputation
- Categorical mode imputation
- Missing raw-feature reconstruction during inference
- Numeric outlier clamping
- Target label encoding
- Numeric StandardScaler fitting/transformation

Important
---------
Categorical columns remain categorical in this module.

One-hot encoding belongs exclusively to FeatureEngineer.

Therefore DataPreprocessor.transform() returns:
    - scaled numeric columns
    - categorical columns as categorical values

FeatureEngineer performs one-hot encoding and converts the final
feature matrix to numeric form.
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


# ---------------------------------------------------------------------------
# IMPORTANT
# ---------------------------------------------------------------------------
# This filename belongs ONLY to DataPreprocessor.
#
# Do NOT use config.FEATURE_NAMES_FILENAME here because that file is also
# used later by model_saver for the final model feature names.
#
# Raw training schema = 15 features
# Final model schema  = 77 features
#
# They MUST be stored separately.
# ---------------------------------------------------------------------------

_RAW_FEATURE_COLUMNS_FILENAME = "raw_feature_columns.pkl"


class DataPreprocessor:
    """Stateful raw-data preprocessing pipeline."""

    def __init__(
        self,
        target_column: str,
    ) -> None:

        self.target_column = (
            str(target_column)
            .strip()
            .lower()
            .replace(" ", "_")
        )

        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

        # Exact raw training schema.
        self.feature_columns: List[str] = []

        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []

        self._col_medians: dict = {}
        self._col_modes: dict = {}
        self._iqr_bounds: dict = {}

        self._fitted = False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise_columns(
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Normalise DataFrame column names."""

        df = df.copy()

        df.columns = [
            str(c)
            .strip()
            .lower()
            .replace(" ", "_")
            for c in df.columns
        ]

        return df

    def _identify_column_types(
        self,
        df: pd.DataFrame,
    ) -> None:
        """Identify numeric and categorical feature columns."""

        excluded = {
            self.target_column,
            "timestamp",
        }

        feature_cols = [
            c
            for c in df.columns
            if c not in excluded
        ]

        self.numeric_cols = (
            df[feature_cols]
            .select_dtypes(
                include=[np.number]
            )
            .columns
            .tolist()
        )

        self.categorical_cols = [
            c
            for c in feature_cols
            if c not in self.numeric_cols
        ]

    def _remove_duplicates(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Remove duplicate rows."""

        before = len(df)

        df = df.drop_duplicates()

        removed = before - len(df)

        if removed:
            logger.info(
                "Removed %d duplicate rows.",
                removed,
            )

        return df

    def _impute_numeric(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Impute numeric columns using training medians."""

        df = df.copy()

        for col in self.numeric_cols:

            if col not in df.columns:
                continue

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

            df[col] = df[col].fillna(
                self._col_medians.get(
                    col,
                    0.0,
                )
            )

        return df

    def _impute_categorical(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Impute categorical columns using training modes."""

        df = df.copy()

        for col in self.categorical_cols:

            if col not in df.columns:
                continue

            default = self._col_modes.get(
                col,
                "unknown",
            )

            df[col] = df[col].fillna(
                default
            )

        return df

    def _restore_missing_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Restore missing raw training columns during inference."""

        df = df.copy()

        # --------------------------------------------------------------
        # Numeric columns
        # --------------------------------------------------------------

        for col in self.numeric_cols:

            if col not in df.columns:

                value = self._col_medians.get(
                    col,
                    0.0,
                )

                df[col] = value

                logger.debug(
                    "Restored missing numeric column '%s' "
                    "with median %s.",
                    col,
                    value,
                )

        # --------------------------------------------------------------
        # Categorical columns
        # --------------------------------------------------------------

        for col in self.categorical_cols:

            if col not in df.columns:

                value = self._col_modes.get(
                    col,
                    "unknown",
                )

                df[col] = value

                logger.debug(
                    "Restored missing categorical column '%s' "
                    "with mode '%s'.",
                    col,
                    value,
                )

        return df

    def _clamp_outliers(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Clamp numeric values using training IQR bounds."""

        df = df.copy()

        for col in self.numeric_cols:

            if col not in df.columns:
                continue

            bounds = self._iqr_bounds.get(
                col,
                {},
            )

            lower = bounds.get("lower")
            upper = bounds.get("upper")

            if (
                lower is not None
                and upper is not None
            ):

                df[col] = df[col].clip(
                    lower=lower,
                    upper=upper,
                )

        return df

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
    ) -> "DataPreprocessor":
        """Fit preprocessing statistics on training data."""

        df = self._normalise_columns(df)

        if self.target_column not in df.columns:

            raise ValueError(
                f"Target column '{self.target_column}' "
                "not found in DataFrame."
            )

        df = self._remove_duplicates(df)

        self._identify_column_types(df)

        # --------------------------------------------------------------
        # Numeric statistics
        # --------------------------------------------------------------

        for col in self.numeric_cols:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

            median = df[col].median()

            if pd.isna(median):
                median = 0.0

            self._col_medians[col] = float(
                median
            )

        # --------------------------------------------------------------
        # Categorical statistics
        # --------------------------------------------------------------

        for col in self.categorical_cols:

            mode = df[col].mode()

            if not mode.empty:

                self._col_modes[col] = (
                    mode.iloc[0]
                )

            else:

                self._col_modes[col] = "unknown"

        # --------------------------------------------------------------
        # IQR bounds
        # --------------------------------------------------------------

        self._iqr_bounds = {}

        for col in self.numeric_cols:

            q1 = float(
                df[col].quantile(0.25)
            )

            q3 = float(
                df[col].quantile(0.75)
            )

            iqr = q3 - q1

            self._iqr_bounds[col] = {
                "lower": q1 - 1.5 * iqr,
                "upper": q3 + 1.5 * iqr,
            }

        # --------------------------------------------------------------
        # Clean training data
        # --------------------------------------------------------------

        df = self._impute_numeric(df)

        df = self._impute_categorical(df)

        df = self._clamp_outliers(df)

        # --------------------------------------------------------------
        # Target encoder
        # --------------------------------------------------------------

        self.label_encoder.fit(
            df[self.target_column]
        )

        # --------------------------------------------------------------
        # Numeric scaler
        # --------------------------------------------------------------

        if self.numeric_cols:

            self.scaler.fit(
                df[self.numeric_cols]
            )

        # --------------------------------------------------------------
        # EXACT RAW TRAINING SCHEMA
        # --------------------------------------------------------------

        self.feature_columns = (
            self.numeric_cols
            + self.categorical_cols
        )

        self._fitted = True

        logger.info(
            "DataPreprocessor fitted: "
            "%d numeric, %d categorical.",
            len(self.numeric_cols),
            len(self.categorical_cols),
        )

        logger.info(
            "Raw training schema: %d features.",
            len(self.feature_columns),
        )

        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(
        self,
        df: pd.DataFrame,
    ) -> Tuple[
        pd.DataFrame,
        Optional[np.ndarray],
    ]:
        """Transform raw data while preserving categorical values."""

        if not self._fitted:

            raise RuntimeError(
                "DataPreprocessor must be fitted "
                "before calling transform()."
            )

        df = self._normalise_columns(df)

        df = df.copy()

        # --------------------------------------------------------------
        # Target
        # --------------------------------------------------------------

        y: Optional[np.ndarray] = None

        if self.target_column in df.columns:

            y = self.label_encoder.transform(
                df[self.target_column]
            )

            df = df.drop(
                columns=[
                    self.target_column
                ]
            )

        # --------------------------------------------------------------
        # Remove timestamp if supplied during inference
        # --------------------------------------------------------------

        if "timestamp" in df.columns:

            df = df.drop(
                columns=["timestamp"]
            )

        # --------------------------------------------------------------
        # Restore missing RAW features
        # --------------------------------------------------------------

        df = self._restore_missing_features(
            df
        )

        # --------------------------------------------------------------
        # Imputation
        # --------------------------------------------------------------

        df = self._impute_numeric(df)

        df = self._impute_categorical(df)

        # --------------------------------------------------------------
        # Outlier handling
        # --------------------------------------------------------------

        df = self._clamp_outliers(df)

        # --------------------------------------------------------------
        # Exact raw schema
        # --------------------------------------------------------------

        missing = [
            col
            for col in self.feature_columns
            if col not in df.columns
        ]

        if missing:

            raise RuntimeError(
                "Unable to reconstruct required raw "
                "training features: "
                f"{missing}"
            )

        df = df[
            self.feature_columns
        ]

        # --------------------------------------------------------------
        # Scale ONLY numeric columns
        # --------------------------------------------------------------

        if self.numeric_cols:

            df[self.numeric_cols] = (
                self.scaler.transform(
                    df[self.numeric_cols]
                )
            )

        # --------------------------------------------------------------
        # IMPORTANT
        # --------------------------------------------------------------
        # Do NOT convert the entire DataFrame to float.
        #
        # Categorical values must remain categorical until
        # FeatureEngineer performs one-hot encoding.
        # --------------------------------------------------------------

        return (
            df.reset_index(drop=True),
            y,
        )

    # ------------------------------------------------------------------
    # Fit-transform
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        df: pd.DataFrame,
    ) -> Tuple[
        pd.DataFrame,
        Optional[np.ndarray],
    ]:
        """Fit and transform."""

        self.fit(df)

        return self.transform(df)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        directory: Path,
    ) -> None:
        """Persist preprocessing artifacts."""

        directory = Path(directory)

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        # --------------------------------------------------------------
        # Target encoder
        # --------------------------------------------------------------

        joblib.dump(
            self.label_encoder,
            directory
            / config.LABEL_ENCODER_FILENAME,
        )

        # --------------------------------------------------------------
        # Numeric scaler
        # --------------------------------------------------------------

        joblib.dump(
            self.scaler,
            directory
            / config.SCALER_FILENAME,
        )

        # --------------------------------------------------------------
        # CRITICAL FIX
        #
        # Raw feature schema gets its OWN filename.
        #
        # This file contains 15 raw features.
        #
        # It must never be overwritten by model_saver's final
        # feature_names.pkl containing 77 model features.
        # --------------------------------------------------------------

        joblib.dump(
            self.feature_columns,
            directory
            / _RAW_FEATURE_COLUMNS_FILENAME,
        )

        # --------------------------------------------------------------
        # Preprocessing metadata
        # --------------------------------------------------------------

        joblib.dump(
            {
                "numeric_cols": self.numeric_cols,
                "categorical_cols": self.categorical_cols,
                "col_medians": self._col_medians,
                "col_modes": self._col_modes,
                "iqr_bounds": self._iqr_bounds,
                "target_column": self.target_column,
            },
            directory
            / config.FEATURE_ENCODER_FILENAME,
        )

        logger.info(
            "DataPreprocessor artifacts saved to %s",
            directory,
        )

        logger.info(
            "Saved raw feature schema: %d features -> %s",
            len(self.feature_columns),
            directory
            / _RAW_FEATURE_COLUMNS_FILENAME,
        )

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        directory: Path,
    ) -> "DataPreprocessor":
        """Load persisted preprocessing artifacts."""

        directory = Path(directory)

        # --------------------------------------------------------------
        # Load preprocessing metadata
        # --------------------------------------------------------------

        meta = joblib.load(
            directory
            / config.FEATURE_ENCODER_FILENAME
        )

        instance = cls(
            target_column=meta[
                "target_column"
            ]
        )

        # --------------------------------------------------------------
        # Load target encoder
        # --------------------------------------------------------------

        instance.label_encoder = joblib.load(
            directory
            / config.LABEL_ENCODER_FILENAME
        )

        # --------------------------------------------------------------
        # Load scaler
        # --------------------------------------------------------------

        instance.scaler = joblib.load(
            directory
            / config.SCALER_FILENAME
        )

        # --------------------------------------------------------------
        # CRITICAL FIX
        #
        # Load RAW schema from raw_feature_columns.pkl,
        # NOT feature_names.pkl.
        # --------------------------------------------------------------

        raw_schema_path = (
            directory
            / _RAW_FEATURE_COLUMNS_FILENAME
        )

        if not raw_schema_path.exists():

            raise RuntimeError(
                "Raw feature schema artifact is missing: "
                f"{raw_schema_path}. "
                "Please retrain the model using the corrected "
                "preprocessing pipeline."
            )

        instance.feature_columns = list(
            joblib.load(
                raw_schema_path
            )
        )

        # --------------------------------------------------------------
        # Load column metadata
        # --------------------------------------------------------------

        instance.numeric_cols = list(
            meta["numeric_cols"]
        )

        instance.categorical_cols = list(
            meta["categorical_cols"]
        )

        instance._col_medians = dict(
            meta["col_medians"]
        )

        instance._col_modes = dict(
            meta["col_modes"]
        )

        instance._iqr_bounds = dict(
            meta["iqr_bounds"]
        )

        instance._fitted = True

        logger.info(
            "DataPreprocessor loaded from %s",
            directory,
        )

        logger.info(
            "Loaded preprocessing schema: "
            "%d raw features.",
            len(instance.feature_columns),
        )

        logger.info(
            "Raw feature columns: %s",
            instance.feature_columns,
        )

        return instance


__all__ = [
    "DataPreprocessor",
]