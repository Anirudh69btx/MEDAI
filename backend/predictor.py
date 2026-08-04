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
from .recommendation_engine import RecommendationEngine, RecommendationResult

logger: logging.Logger = get_logger(__name__)


class PredictionResult:
    """Container for a single prediction result.

    Attributes
    ----------
    predicted_class : str
        Human-readable class label.
    predicted_index : int
        Integer class index as produced by the model.
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
        """Serialise to a JSON-safe dictionary."""
        return {
            "predicted_class": self.predicted_class,
            "predicted_index": self.predicted_index,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "recommendation": self.recommendation.to_dict(),
            "model_used": self.model_name,
        }


class MindCarePredictor:
    """End-to-end inference pipeline for MindCare AI.

    Loads saved artefacts from ``models/`` and exposes a single ``predict``
    method that accepts raw feature data and returns a structured
    ``PredictionResult``.

    Parameters
    ----------
    models_dir : Path or str, optional
        Directory containing saved model artefacts. Defaults to ``config.MODELS_DIR``.
    use_pytorch : bool, optional
        When ``True`` use the PyTorch model if available, falling back to the
        best sklearn model. Defaults to ``False`` (sklearn model preferred).
    """

    def __init__(
        self,
        models_dir: Optional[Union[Path, str]] = None,
        use_pytorch: bool = False,
    ) -> None:
        self.models_dir: Path = Path(models_dir or config.MODELS_DIR)
        self.use_pytorch = use_pytorch

        self._saver: ModelSaver = ModelSaver(self.models_dir)
        self._recommendation_engine: RecommendationEngine = RecommendationEngine()

        self._sklearn_model: Optional[Any] = None
        self._pytorch_model: Optional[MindCarePyTorchClassifier] = None
        self._feature_engineer: Optional[FeatureEngineer] = None
        self._label_classes: Optional[np.ndarray] = None
        self._feature_names: Optional[List[str]] = None
        self._loaded: bool = False

    def load(self) -> "MindCarePredictor":
        """Load all saved artefacts from disk.

        Returns
        -------
        MindCarePredictor
            Self, for method chaining.

        Raises
        ------
        RuntimeError
            If required artefacts cannot be loaded.
        """
        logger.info("Loading MindCarePredictor artefacts from %s", self.models_dir)

        # Load sklearn model (always required)
        try:
            self._sklearn_model = self._saver.load_sklearn_model()
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Cannot load sklearn model: {exc}. Run train.py first."
            ) from exc

        # Load PyTorch model (optional)
        if self.use_pytorch:
            try:
                self._pytorch_model = self._saver.load_pytorch_model(
                    device=config.DEVICE
                )
                logger.info("PyTorch model loaded for inference.")
            except FileNotFoundError:
                logger.warning(
                    "PyTorch model not found. Falling back to sklearn model."
                )
                self.use_pytorch = False

        # Load feature engineering pipeline
        feature_engineer_meta = self.models_dir / "feature_engineer_meta.pkl"
        if feature_engineer_meta.is_file():
            try:
                self._feature_engineer = FeatureEngineer.load(self.models_dir)
                self._label_classes = (
                    self._feature_engineer.get_label_encoder().classes_
                )
                logger.info("FeatureEngineer pipeline loaded.")
            except Exception as exc:
                logger.warning("Failed to load FeatureEngineer: %s", exc)

        # Load feature names
        try:
            self._feature_names = self._saver.load_feature_names()
        except FileNotFoundError:
            if self._feature_engineer is not None:
                self._feature_names = self._feature_engineer.get_feature_names()

        # Load label encoder (fallback if not via feature engineer)
        if self._label_classes is None:
            le = self._saver.load_label_encoder()
            if le is not None:
                self._label_classes = le.classes_

        self._loaded = True
        logger.info("MindCarePredictor ready.")
        return self

    def _ensure_loaded(self) -> None:
        """Raise RuntimeError if artefacts have not been loaded."""
        if not self._loaded:
            raise RuntimeError(
                "MindCarePredictor.load() must be called before predict()."
            )

    def _preprocess_input(
        self, input_data: Union[Dict[str, Any], pd.DataFrame]
    ) -> pd.DataFrame:
        """Convert raw input to a processed feature DataFrame.

        Parameters
        ----------
        input_data : dict or pd.DataFrame
            Raw features. If a dict, it is wrapped in a single-row DataFrame.

        Returns
        -------
        pd.DataFrame
            Processed, scaled feature DataFrame ready for model inference.
        """
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise TypeError(
                f"input_data must be dict or DataFrame, got {type(input_data)}"
            )

        if self._feature_engineer is not None:
            df = self._feature_engineer.transform(df)
        elif self._feature_names is not None:
            # Best-effort alignment if feature engineer not available
            for col in self._feature_names:
                if col not in df.columns:
                    df[col] = 0.0
            df = df[[c for c in self._feature_names if c in df.columns]]

        return df

    def predict(
        self,
        input_data: Union[Dict[str, Any], pd.DataFrame],
    ) -> PredictionResult:
        """Run end-to-end inference on *input_data*.

        Parameters
        ----------
        input_data : dict or pd.DataFrame
            Raw feature values. Keys must match training-time feature names.

        Returns
        -------
        PredictionResult
            Structured prediction with class label, probabilities, and recommendations.

        Raises
        ------
        RuntimeError
            If ``load()`` has not been called, or inference fails.
        """
        self._ensure_loaded()

        try:
            X = self._preprocess_input(input_data)
            X_array = X.values.astype(np.float32)

            if self.use_pytorch and self._pytorch_model is not None:
                model_name = "PyTorch Deep Learning"
                probs_matrix = self._pytorch_model.predict_proba(
                    X_array, device=config.DEVICE
                )
                probs_1d = probs_matrix[0]
                predicted_index = int(np.argmax(probs_1d))
                confidence = float(probs_1d[predicted_index])
            else:
                model_name = type(self._sklearn_model).__name__
                if hasattr(self._sklearn_model, "predict_proba"):
                    probs_matrix = self._sklearn_model.predict_proba(X_array)
                    probs_1d = probs_matrix[0]
                    predicted_index = int(np.argmax(probs_1d))
                    confidence = float(probs_1d[predicted_index])
                else:
                    pred = self._sklearn_model.predict(X_array)
                    predicted_index = int(pred[0])
                    probs_1d = np.zeros(
                        len(self._label_classes) if self._label_classes is not None else 2
                    )
                    probs_1d[predicted_index] = 1.0
                    confidence = 1.0

            # Decode class label
            if self._label_classes is not None:
                if predicted_index < len(self._label_classes):
                    predicted_class = str(self._label_classes[predicted_index])
                else:
                    predicted_class = str(predicted_index)
                probabilities = {
                    str(cls): float(prob)
                    for cls, prob in zip(self._label_classes, probs_1d)
                }
            else:
                predicted_class = str(predicted_index)
                probabilities = {predicted_class: confidence}

            recommendation = self._recommendation_engine.generate(
                predicted_class=predicted_class,
                confidence=confidence,
            )

            return PredictionResult(
                predicted_class=predicted_class,
                predicted_index=predicted_index,
                confidence=confidence,
                probabilities=probabilities,
                recommendation=recommendation,
                model_name=model_name,
            )

        except Exception as exc:
            logger.exception("Prediction failed: %s", exc)
            raise RuntimeError(f"Prediction pipeline error: {exc}") from exc

    def predict_batch(
        self,
        input_df: pd.DataFrame,
    ) -> List[PredictionResult]:
        """Run batch inference on a DataFrame.

        Parameters
        ----------
        input_df : pd.DataFrame
            DataFrame where each row is one sample.

        Returns
        -------
        List[PredictionResult]
            One ``PredictionResult`` per row.
        """
        self._ensure_loaded()
        results: List[PredictionResult] = []
        for _, row in input_df.iterrows():
            results.append(self.predict(row.to_dict()))
        return results


__all__ = ["MindCarePredictor", "PredictionResult"]
