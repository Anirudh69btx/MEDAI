"""MindCare AI - Model & Preprocessing Persistence Module.

Manages serialisation and deserialisation of all model artefacts:
- Best Scikit-Learn model
- PyTorch model (via MindCarePyTorchClassifier.save/load)
- LabelEncoder
- Feature names list
- Training metadata JSON

All save/load operations are centralised here so no other module
directly writes model files to disk (except ``EarlyStopping``, which
saves the PyTorch checkpoint via ``MindCarePyTorchClassifier.save``
during training for checkpoint safety).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

from .config import config
from .logger import get_logger
from .pytorch_model import MindCarePyTorchClassifier

logger: logging.Logger = get_logger(__name__)


class ModelSaver:
    """Centralised artifact serialisation and deserialisation handler.

    Parameters
    ----------
    models_dir : Path or str, optional
        Directory to store/load all artifacts. Defaults to ``config.MODELS_DIR``.
    """

    def __init__(self, models_dir: Optional[Path | str] = None) -> None:
        self.models_dir: Path = Path(models_dir or config.MODELS_DIR)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    #  Save methods
    # ------------------------------------------------------------------ #

    def save_sklearn_model(self, model: Any) -> Path:
        """Serialise the best Scikit-Learn model with joblib.

        Parameters
        ----------
        model : Any
            A fitted sklearn estimator.

        Returns
        -------
        Path
            Absolute path to the saved file.
        """
        path = self.models_dir / config.BEST_MODEL_FILENAME
        joblib.dump(model, path)
        logger.info("Saved best sklearn model to %s", path)
        return path

    def save_pytorch_model(self, model: MindCarePyTorchClassifier) -> Path:
        """Serialise the PyTorch model using its own ``save()`` method.

        Parameters
        ----------
        model : MindCarePyTorchClassifier
            A fitted PyTorch classifier instance.

        Returns
        -------
        Path
            Absolute path to the saved checkpoint.
        """
        path = self.models_dir / config.TORCH_MODEL_FILENAME
        saved_path = model.save(path)
        logger.info("Saved PyTorch model to %s", saved_path)
        return saved_path

    def save_label_encoder(self, label_encoder: Any) -> Path:
        """Persist the target LabelEncoder.

        Parameters
        ----------
        label_encoder : LabelEncoder
            Fitted sklearn LabelEncoder.

        Returns
        -------
        Path
            Absolute path to the saved file.
        """
        path = self.models_dir / config.LABEL_ENCODER_FILENAME
        joblib.dump(label_encoder, path)
        logger.info("Saved LabelEncoder to %s", path)
        return path

    def save_feature_names(self, feature_names: List[str]) -> Path:
        """Persist the ordered list of feature column names.

        Parameters
        ----------
        feature_names : List[str]
            Feature column names produced by ``FeatureEngineer``.

        Returns
        -------
        Path
            Absolute path to the saved file.
        """
        path = self.models_dir / config.FEATURE_NAMES_FILENAME
        joblib.dump(feature_names, path)
        logger.info("Saved feature names (%d cols) to %s", len(feature_names), path)
        return path

    def save_training_metadata(self, metadata: Dict[str, Any]) -> Path:
        """Write training run metadata to a JSON file.

        Parameters
        ----------
        metadata : Dict[str, Any]
            Arbitrary metadata dict (must be JSON-serialisable).

        Returns
        -------
        Path
            Absolute path to the saved JSON file.
        """
        path = self.models_dir / config.TRAINING_METADATA_FILENAME
        with path.open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=4, default=str)
        logger.info("Saved training metadata to %s", path)
        return path

    def save_all(
        self,
        sklearn_model: Any,
        pytorch_model: Optional[MindCarePyTorchClassifier],
        label_encoder: Any,
        feature_names: List[str],
        metadata: Dict[str, Any],
    ) -> Dict[str, Path]:
        """Save all artefacts in a single call.

        Parameters
        ----------
        sklearn_model : Any
            Best fitted sklearn estimator.
        pytorch_model : MindCarePyTorchClassifier or None
            Trained PyTorch model, or ``None`` if not used.
        label_encoder : Any
            Fitted LabelEncoder.
        feature_names : List[str]
            Ordered feature column names.
        metadata : Dict[str, Any]
            Training run metadata.

        Returns
        -------
        Dict[str, Path]
            Mapping of artefact key to saved path.
        """
        paths: Dict[str, Path] = {}
        paths["sklearn_model"] = self.save_sklearn_model(sklearn_model)
        if pytorch_model is not None:
            paths["pytorch_model"] = self.save_pytorch_model(pytorch_model)
        paths["label_encoder"] = self.save_label_encoder(label_encoder)
        paths["feature_names"] = self.save_feature_names(feature_names)
        paths["metadata"] = self.save_training_metadata(metadata)
        logger.info("All model artefacts saved successfully.")
        return paths

    # ------------------------------------------------------------------ #
    #  Load methods
    # ------------------------------------------------------------------ #

    def load_sklearn_model(self) -> Any:
        """Load and return the best sklearn model from disk.

        Returns
        -------
        Any
            Loaded sklearn estimator.

        Raises
        ------
        FileNotFoundError
            If the model file does not exist.
        """
        path = self.models_dir / config.BEST_MODEL_FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"Sklearn model file not found at {path}")
        model = joblib.load(path)
        logger.info("Loaded sklearn model from %s", path)
        return model

    def load_pytorch_model(
        self, device: str | None = None
    ) -> MindCarePyTorchClassifier:
        """Load and return the PyTorch model from disk.

        Parameters
        ----------
        device : str, optional
            Compute device override ('cpu' or 'cuda'). Defaults to ``config.DEVICE``.

        Returns
        -------
        MindCarePyTorchClassifier
            Loaded model in eval mode.

        Raises
        ------
        FileNotFoundError
            If the checkpoint file does not exist.
        """
        path = self.models_dir / config.TORCH_MODEL_FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"PyTorch model checkpoint not found at {path}")
        model = MindCarePyTorchClassifier.load(path, device=device or config.DEVICE)
        logger.info("Loaded PyTorch model from %s", path)
        return model

    def load_label_encoder(self) -> Any:
        """Load and return the LabelEncoder from disk.

        Returns ``None`` if the file does not exist (allows graceful fallback).
        """
        path = self.models_dir / config.LABEL_ENCODER_FILENAME
        if not path.is_file():
            logger.warning("LabelEncoder file not found at %s – returning None.", path)
            return None
        encoder = joblib.load(path)
        logger.info("Loaded LabelEncoder from %s", path)
        return encoder

    def load_feature_names(self) -> List[str]:
        """Load and return the feature names list from disk.

        Returns
        -------
        List[str]
            Ordered feature column names.

        Raises
        ------
        FileNotFoundError
            If the feature names file does not exist.
        """
        path = self.models_dir / config.FEATURE_NAMES_FILENAME
        if not path.is_file():
            raise FileNotFoundError(f"Feature names file not found at {path}")
        names: List[str] = joblib.load(path)
        logger.info("Loaded %d feature names from %s", len(names), path)
        return names

    def load_training_metadata(self) -> Dict[str, Any]:
        """Load and return training metadata from disk.

        Returns an empty dict if the file does not exist.
        """
        path = self.models_dir / config.TRAINING_METADATA_FILENAME
        if not path.is_file():
            logger.debug("No training metadata file found at %s.", path)
            return {}
        with path.open("r", encoding="utf-8") as fh:
            meta: Dict[str, Any] = json.load(fh)
        return meta


__all__ = ["ModelSaver"]
