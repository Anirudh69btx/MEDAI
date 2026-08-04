"""backend/__init__.py
================================
MindCare AI backend package initialiser.

Exposes the primary public API surface of the backend package.
All heavy imports are deferred to their respective modules to
keep this package import fast and avoid circular import issues.
"""

from .config import Config, config
from .data_loader import DataLoadingError, load_dataset
from .evaluation import ModelEvaluator
from .feature_engineering import FeatureEngineer
from .logger import get_logger
from .model_comparator import ModelComparator
from .model_saver import ModelSaver
from .predictor import MindCarePredictor, PredictionResult
from .preprocessing import DataPreprocessor
from .pytorch_model import MindCarePyTorchClassifier
from .pytorch_trainer import PyTorchTrainer
from .recommendation_engine import RecommendationEngine, RecommendationResult
from .trainer import ModelEvaluationResult, ModelTrainer
from .utils import (
    ensure_dir,
    format_timestamp,
    load_json,
    timed,
    validate_prediction_payload,
)

__all__ = [
    # Config
    "Config",
    "config",
    # Logging
    "get_logger",
    # Data
    "DataLoadingError",
    "load_dataset",
    # Preprocessing
    "DataPreprocessor",
    # Feature Engineering
    "FeatureEngineer",
    # Training
    "ModelTrainer",
    "ModelEvaluationResult",
    # PyTorch
    "MindCarePyTorchClassifier",
    "PyTorchTrainer",
    # Evaluation
    "ModelEvaluator",
    "ModelComparator",
    # Persistence
    "ModelSaver",
    # Inference
    "MindCarePredictor",
    "PredictionResult",
    # Recommendations
    "RecommendationEngine",
    "RecommendationResult",
    # Utilities
    "timed",
    "load_json",
    "format_timestamp",
    "ensure_dir",
    "validate_prediction_payload",
]
