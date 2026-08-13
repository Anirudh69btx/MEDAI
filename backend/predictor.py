"""MindCare AI - Inference Predictor Module.

Loads the saved best model, feature engineering pipeline, and label encoder,
then runs end-to-end inference on raw input dictionaries or DataFrames.

Supports:
- Scikit-Learn model predictions
- PyTorch model predictions (via MindCarePyTorchClassifier)
- Probability / confidence scoring
- Recommendation generation via RecommendationEngine
- Graceful fallback between model types

Designed to be instantiated once and reused across API requests.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .config import config
from .feature_engineering import FeatureEngineer
from .logger import get_logger
from .model_saver import ModelSaver
from .pytorch_model import MindCarePyTorchClassifier
from .recommendation_engine import (
    RecommendationEngine,
    RecommendationResult,
)

logger: logging.Logger = get_logger(__name__)


class PredictionResult:
    """Container for a single prediction result.

    Attributes
    ----------
    predicted_class : str
        Human-readable class label.

    predicted_index : int
        Integer class index produced by the model.

    confidence : float
        Maximum class probability in [0, 1].

    probabilities : Dict[str, float]
        Mapping of class label to probability for all classes.

    recommendation : RecommendationResult
        Generated mental health recommendation.

    model_name : str
        Name of the model used for inference.
    """

    def __init__(
        self,
        predicted_class: str,
        predicted_index: int,
        confidence: float,
        probabilities: Dict[str, float],
        recommendation: RecommendationResult,
        model_name: str = "Unknown",
    ) -> None:
        self.predicted_class = predicted_class
        self.predicted_index = predicted_index
        self.confidence = confidence
        self.probabilities = probabilities
        self.recommendation = recommendation
        self.model_name = model_name

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the prediction result to a JSON-safe dictionary."""

        return {
            "predicted_class": self.predicted_class,
            "predicted_index": self.predicted_index,
            "confidence": round(
                self.confidence,
                4,
            ),
            "probabilities": {
                key: round(value, 4)
                for key, value in self.probabilities.items()
            },
            "recommendation": (
                self.recommendation.to_dict()
            ),
            "model_used": self.model_name,
        }


class MindCarePredictor:
    """End-to-end inference pipeline for MindCare AI.

    Loads saved artefacts from ``models/`` and exposes a single
    ``predict`` method that accepts raw feature data and returns a
    structured ``PredictionResult``.

    Parameters
    ----------
    models_dir : Path or str, optional
        Directory containing saved model artefacts.
        Defaults to ``config.MODELS_DIR``.

    use_pytorch : bool, optional
        When ``True``, use the PyTorch model if available.
        Falls back to the saved sklearn model if PyTorch is unavailable.
        Defaults to ``False``.
    """

    def __init__(
        self,
        models_dir: Optional[Union[Path, str]] = None,
        use_pytorch: bool = False,
    ) -> None:

        self.models_dir: Path = Path(
            models_dir or config.MODELS_DIR
        )

        self.use_pytorch = use_pytorch

        self._saver: ModelSaver = ModelSaver(
            self.models_dir
        )

        self._recommendation_engine: RecommendationEngine = (
            RecommendationEngine()
        )

        self._sklearn_model: Optional[Any] = None

        self._pytorch_model: Optional[
            MindCarePyTorchClassifier
        ] = None

        self._feature_engineer: Optional[
            FeatureEngineer
        ] = None

        self._label_classes: Optional[
            np.ndarray
        ] = None

        self._feature_names: Optional[
            List[str]
        ] = None

        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> "MindCarePredictor":
        """Load all saved model and preprocessing artefacts."""

        logger.info(
            "Loading MindCarePredictor artefacts from %s",
            self.models_dir,
        )

        # --------------------------------------------------------------
        # Load sklearn model
        # --------------------------------------------------------------

        try:

            self._sklearn_model = (
                self._saver.load_sklearn_model()
            )

        except FileNotFoundError as exc:

            raise RuntimeError(
                f"Cannot load sklearn model: {exc}. "
                "Run train.py first."
            ) from exc

        # --------------------------------------------------------------
        # Load PyTorch model if requested
        # --------------------------------------------------------------

        if self.use_pytorch:

            try:

                self._pytorch_model = (
                    self._saver.load_pytorch_model(
                        device=config.DEVICE
                    )
                )

                logger.info(
                    "PyTorch model loaded for inference."
                )

            except FileNotFoundError:

                logger.warning(
                    "PyTorch model not found. "
                    "Falling back to sklearn model."
                )

                self.use_pytorch = False

        # --------------------------------------------------------------
        # Load FeatureEngineer
        # --------------------------------------------------------------

        feature_engineer_meta = (
            self.models_dir
            / "feature_engineer_meta.pkl"
        )

        if feature_engineer_meta.is_file():

            try:

                self._feature_engineer = (
                    FeatureEngineer.load(
                        self.models_dir
                    )
                )

                self._label_classes = (
                    self._feature_engineer
                    .get_label_encoder()
                    .classes_
                )

                logger.info(
                    "FeatureEngineer pipeline loaded."
                )

            except Exception as exc:

                logger.warning(
                    "Failed to load FeatureEngineer: %s",
                    exc,
                )

        # --------------------------------------------------------------
        # Load final model feature names
        # --------------------------------------------------------------

        try:

            self._feature_names = (
                self._saver.load_feature_names()
            )

        except FileNotFoundError:

            if self._feature_engineer is not None:

                self._feature_names = (
                    self._feature_engineer
                    .get_feature_names()
                )

        # --------------------------------------------------------------
        # Load label encoder as fallback
        # --------------------------------------------------------------

        if self._label_classes is None:

            le = self._saver.load_label_encoder()

            if le is not None:

                self._label_classes = (
                    le.classes_
                )

        # --------------------------------------------------------------
        # Validate final feature schema
        # --------------------------------------------------------------

        if self._feature_names is None:

            raise RuntimeError(
                "Final model feature names could not "
                "be loaded."
            )

        if len(self._feature_names) == 0:

            raise RuntimeError(
                "Final model feature schema is empty."
            )

        logger.info(
            "Final inference schema: %d features.",
            len(self._feature_names),
        )

        self._loaded = True

        logger.info(
            "MindCarePredictor ready."
        )

        return self

    # ------------------------------------------------------------------
    # State validation
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Raise RuntimeError if artefacts are not loaded."""

        if not self._loaded:

            raise RuntimeError(
                "MindCarePredictor.load() must be called "
                "before predict()."
            )

    # ------------------------------------------------------------------
    # Input preprocessing
    # ------------------------------------------------------------------

    def _preprocess_input(
        self,
        input_data: Union[
            Dict[str, Any],
            pd.DataFrame,
        ],
    ) -> pd.DataFrame:
        """Convert raw input into the exact model feature schema.

        Parameters
        ----------
        input_data : dict or DataFrame
            Raw feature values.

        Returns
        -------
        pd.DataFrame
            Final 77-feature DataFrame ready for sklearn
            inference or conversion to NumPy for PyTorch.
        """

        # --------------------------------------------------------------
        # Convert dictionary to one-row DataFrame
        # --------------------------------------------------------------

        if isinstance(input_data, dict):

            df = pd.DataFrame(
                [input_data]
            )

        elif isinstance(input_data, pd.DataFrame):

            df = input_data.copy()

        else:

            raise TypeError(
                "input_data must be dict or DataFrame, "
                f"got {type(input_data)}"
            )

        # --------------------------------------------------------------
        # Feature engineering
        # --------------------------------------------------------------

        if self._feature_engineer is not None:

            df = (
                self._feature_engineer
                .transform(df)
            )

        elif self._feature_names is not None:

            # ----------------------------------------------------------
            # Best-effort fallback.
            #
            # Normally this branch should not be used because the
            # FeatureEngineer pipeline is required for raw inference.
            # ----------------------------------------------------------

            for col in self._feature_names:

                if col not in df.columns:

                    df[col] = 0.0

            df = df[
                self._feature_names
            ]

        else:

            raise RuntimeError(
                "No feature engineering pipeline or "
                "feature schema is available."
            )

        # --------------------------------------------------------------
        # Final schema validation
        # --------------------------------------------------------------

        if list(df.columns) != list(
            self._feature_names
        ):

            raise RuntimeError(
                "Inference feature schema mismatch. "
                f"Expected {len(self._feature_names)} "
                f"features, got {len(df.columns)}."
            )

        if df.shape[1] != len(
            self._feature_names
        ):

            raise RuntimeError(
                "Inference feature count mismatch. "
                f"Expected {len(self._feature_names)}, "
                f"got {df.shape[1]}."
            )

        if df.isnull().any().any():

            raise RuntimeError(
                "NaN values detected after "
                "feature engineering."
            )

        return df.reset_index(
            drop=True
        )

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        input_data: Union[
            Dict[str, Any],
            pd.DataFrame,
        ],
    ) -> PredictionResult:
        """Run end-to-end inference on raw input data."""

        self._ensure_loaded()

        try:

            # ----------------------------------------------------------
            # Raw input
            #     ↓
            # DataPreprocessor
            #     ↓
            # FeatureEngineer
            #     ↓
            # Exact 77-feature DataFrame
            # ----------------------------------------------------------

            X = self._preprocess_input(
                input_data
            )

            # ==========================================================
            # PYTORCH INFERENCE
            # ==========================================================

            if (
                self.use_pytorch
                and self._pytorch_model is not None
            ):

                model_name = (
                    "PyTorch Deep Learning"
                )

                # PyTorch expects NumPy float32.
                X_array = X.to_numpy(
                    dtype=np.float32
                )

                probs_matrix = (
                    self._pytorch_model
                    .predict_proba(
                        X_array,
                        device=config.DEVICE,
                    )
                )

                probs_1d = probs_matrix[0]

                predicted_index = int(
                    np.argmax(probs_1d)
                )

                confidence = float(
                    probs_1d[
                        predicted_index
                    ]
                )

            # ==========================================================
            # SCIKIT-LEARN INFERENCE
            # ==========================================================

            else:

                model_name = type(
                    self._sklearn_model
                ).__name__

                # ------------------------------------------------------
                # IMPORTANT
                #
                # Keep X as a pandas DataFrame.
                #
                # The sklearn model was trained with feature names.
                # Passing X.values would remove those names and produce:
                #
                # "X does not have valid feature names..."
                #
                # Therefore sklearn receives X directly.
                # ------------------------------------------------------

                if hasattr(
                    self._sklearn_model,
                    "predict_proba",
                ):

                    probs_matrix = (
                        self._sklearn_model
                        .predict_proba(X)
                    )

                    probs_1d = (
                        probs_matrix[0]
                    )

                    predicted_index = int(
                        np.argmax(probs_1d)
                    )

                    confidence = float(
                        probs_1d[
                            predicted_index
                        ]
                    )

                else:

                    pred = (
                        self._sklearn_model
                        .predict(X)
                    )

                    predicted_index = int(
                        pred[0]
                    )

                    class_count = (
                        len(
                            self._label_classes
                        )
                        if self._label_classes
                        is not None
                        else 2
                    )

                    probs_1d = np.zeros(
                        class_count,
                        dtype=float,
                    )

                    probs_1d[
                        predicted_index
                    ] = 1.0

                    confidence = 1.0

            # ==========================================================
            # DECODE CLASS LABEL
            # ==========================================================

            if (
                self._label_classes
                is not None
            ):

                if (
                    0
                    <= predicted_index
                    < len(
                        self._label_classes
                    )
                ):

                    predicted_class = str(
                        self._label_classes[
                            predicted_index
                        ]
                    )

                else:

                    predicted_class = str(
                        predicted_index
                    )

                probabilities = {
                    str(cls): float(prob)
                    for cls, prob in zip(
                        self._label_classes,
                        probs_1d,
                    )
                }

            else:

                predicted_class = str(
                    predicted_index
                )

                probabilities = {
                    predicted_class: confidence
                }

            # ==========================================================
            # RECOMMENDATION
            # ==========================================================

            recommendation = (
                self._recommendation_engine.generate(
                    predicted_class=predicted_class,
                    confidence=confidence,
                )
            )

            # ==========================================================
            # FINAL RESULT
            # ==========================================================

            return PredictionResult(
                predicted_class=predicted_class,
                predicted_index=predicted_index,
                confidence=confidence,
                probabilities=probabilities,
                recommendation=recommendation,
                model_name=model_name,
            )

        except Exception as exc:

            logger.exception(
                "Prediction failed: %s",
                exc,
            )

            raise RuntimeError(
                f"Prediction pipeline error: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------

    def predict_batch(
        self,
        input_df: pd.DataFrame,
    ) -> List[PredictionResult]:
        """Run batch inference on a DataFrame.

        Each row represents one raw input sample.
        """

        self._ensure_loaded()

        if not isinstance(
            input_df,
            pd.DataFrame,
        ):

            raise TypeError(
                "input_df must be a pandas DataFrame."
            )

        results: List[
            PredictionResult
        ] = []

        for _, row in input_df.iterrows():

            results.append(
                self.predict(
                    row.to_dict()
                )
            )

        return results


__all__ = [
    "MindCarePredictor",
    "PredictionResult",
]