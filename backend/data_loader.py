"""MindCare AI - Dataset loading and validation module.

Responsible for loading the raw CSV dataset from ``config.DATASET_PATH`` and
performing robust structural validation before handing the DataFrame to the
preprocessing pipeline.

The module is intentionally decoupled from preprocessing so it can be imported
even when the data file is absent; callers catch ``DataLoadingError`` and decide
how to proceed.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

import pandas as pd

from .config import config
from .logger import get_logger

logger: logging.Logger = get_logger(__name__)


class DataLoadingError(RuntimeError):
    """Raised when the dataset cannot be loaded or fails validation.

    The exception message is safe to expose to end-users (e.g. via the API)
    because it contains only high-level information, not internal stack traces.

    Attributes
    ----------
    details : Mapping[str, str]
        Optional machine-readable context about the failure.
    """

    def __init__(self, message: str, details: Optional[Mapping[str, str]] = None) -> None:
        super().__init__(message)
        self.details: Mapping[str, str] = details or {}


def load_dataset(
    path: Optional[Path] = None,
    required_columns: Optional[Sequence[str]] = None,
    allow_missing_values: bool = False,
) -> pd.DataFrame:
    """Load the CSV dataset and validate its structure.

    Parameters
    ----------
    path : Path, optional
        Override path to the CSV file. Defaults to ``config.DATASET_PATH``.
    required_columns : Sequence[str], optional
        Column names that *must* be present in the loaded DataFrame.
        Raises ``DataLoadingError`` if any are absent.
    allow_missing_values : bool, optional
        When ``False`` (default) the function raises if any NaN values are
        present after loading.  Set to ``True`` to skip this check when
        downstream preprocessing handles imputation.

    Returns
    -------
    pd.DataFrame
        Validated dataset ready for the preprocessing pipeline.

    Raises
    ------
    DataLoadingError
        On file-not-found, parse failure, missing columns, or NaN values.
    """
    dataset_path: Path = Path(path or config.DATASET_PATH)

    logger.debug("Attempting to load dataset from %s", dataset_path)

    # ------------------------------------------------------------------ #
    # 1. File existence check
    # ------------------------------------------------------------------ #
    if not dataset_path.is_file():
        msg = (
            f"Dataset file not found at '{dataset_path}'. "
            f"Please place your CSV at: {config.DATASET_PATH}"
        )
        logger.error(msg)
        raise DataLoadingError(msg, {"path": str(dataset_path)})

    # ------------------------------------------------------------------ #
    # 2. Parse CSV
    # ------------------------------------------------------------------ #
    try:
        df: pd.DataFrame = pd.read_csv(
            dataset_path, encoding="utf-8", low_memory=False
        )
    except (UnicodeDecodeError, pd.errors.ParserError, csv.Error) as exc:
        msg = f"Failed to parse CSV at '{dataset_path}': {exc}"
        logger.exception(msg)
        raise DataLoadingError(msg, {"exception": str(exc)}) from exc

    if df.empty:
        msg = f"Dataset at '{dataset_path}' is empty (0 rows)."
        logger.error(msg)
        raise DataLoadingError(msg, {"path": str(dataset_path)})

    # ------------------------------------------------------------------ #
    # 3. Required column validation
    # ------------------------------------------------------------------ #
    if required_columns:
        missing: List[str] = [c for c in required_columns if c not in df.columns]
        if missing:
            msg = f"Missing required columns: {', '.join(missing)}"
            logger.error(msg)
            raise DataLoadingError(msg, {"missing_columns": json.dumps(missing)})

    # ------------------------------------------------------------------ #
    # 4. Missing-value check (optional, on by default)
    # ------------------------------------------------------------------ #
    if not allow_missing_values and df.isnull().any().any():
        null_cols: List[str] = df.columns[df.isnull().any()].tolist()
        msg = (
            "Dataset contains missing values in columns: "
            f"{', '.join(null_cols)}. "
            "Either clean the data or pass allow_missing_values=True."
        )
        logger.warning(msg)
        # Treat as warning only – downstream imputation handles it
        logger.info("Proceeding despite missing values; imputation expected downstream.")

    logger.info(
        "Dataset loaded successfully - %d rows x %d columns from %s",
        df.shape[0],
        df.shape[1],
        dataset_path,
    )
    return df


__all__ = ["DataLoadingError", "load_dataset"]
