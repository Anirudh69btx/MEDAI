"""MindCare AI - Central configuration module.

Defines a single ``Config`` dataclass aggregating all project-wide settings:
file system paths, random seeds, model-training hyper-parameters, and PyTorch
device selection. Directories are created on instantiation so downstream modules
can assume they exist.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Mapping

import numpy as np
import torch


def _mkdir(path: Path) -> None:
    """Create *path* if it does not already exist (idempotent)."""
    path.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class Config:
    """Project-wide configuration singleton.

    All path attributes are resolved relative to the repository root
    (``PROJECT_ROOT``) which is determined at runtime from this file's location.
    """

    # ------------------------------------------------------------------ #
    #  Core paths
    # ------------------------------------------------------------------ #
    PROJECT_ROOT: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1]
    )
    DATASET_PATH: Path = field(init=False)
    MODELS_DIR: Path = field(init=False)
    LOGS_DIR: Path = field(init=False)
    REPORTS_DIR: Path = field(init=False)
    PLOTS_DIR: Path = field(init=False)

    # ------------------------------------------------------------------ #
    #  Randomness & reproducibility
    # ------------------------------------------------------------------ #
    RANDOM_SEED: int = 42

    # ------------------------------------------------------------------ #
    #  Training hyper-parameters
    # ------------------------------------------------------------------ #
    DEFAULT_TARGET_COLUMN: str = "treatment"
    TEST_SIZE: float = 0.2
    VALID_SIZE: float = 0.2
    N_EPOCHS: int = 50
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 1e-3
    EARLY_STOPPING_PATIENCE: int = 5
    LR_SCHEDULER_FACTOR: float = 0.5
    LR_SCHEDULER_PATIENCE: int = 3

    # ------------------------------------------------------------------ #
    #  PyTorch hyper-parameters (accessed by pytorch_model / pytorch_trainer)
    # ------------------------------------------------------------------ #
    HYPERPARAMS: Dict[str, Any] = field(
        default_factory=lambda: {
            "hidden_dims": [128, 64, 32],
            "dropout_rate": 0.2,
            "activation": "leaky_relu",
            "num_workers": 0,
        }
    )

    # ------------------------------------------------------------------ #
    #  Artifact filenames
    # ------------------------------------------------------------------ #
    BEST_MODEL_FILENAME: str = "best_model.pkl"
    TORCH_MODEL_FILENAME: str = "torch_model.pth"
    LABEL_ENCODER_FILENAME: str = "label_encoder.pkl"
    FEATURE_ENCODER_FILENAME: str = "feature_encoder.pkl"
    SCALER_FILENAME: str = "scaler.pkl"
    FEATURE_NAMES_FILENAME: str = "feature_names.pkl"
    TRAINING_METADATA_FILENAME: str = "training_metadata.json"

    # ------------------------------------------------------------------ #
    #  PyTorch device
    # ------------------------------------------------------------------ #
    DEVICE: Literal["cpu", "cuda"] = field(init=False)

    def __post_init__(self) -> None:
        """Populate derived attributes and enforce reproducibility."""
        self.DATASET_PATH = self.PROJECT_ROOT / "dataset" / "data.csv"
        self.MODELS_DIR = self.PROJECT_ROOT / "models"
        self.LOGS_DIR = self.PROJECT_ROOT / "logs"
        self.REPORTS_DIR = self.PROJECT_ROOT / "reports"
        self.PLOTS_DIR = self.PROJECT_ROOT / "plots"

        for p in (self.MODELS_DIR, self.LOGS_DIR, self.REPORTS_DIR, self.PLOTS_DIR):
            _mkdir(p)

        self._set_seeds(self.RANDOM_SEED)
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _set_seeds(seed: int) -> None:
        """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def to_dict(self) -> Mapping[str, Any]:
        """Return a JSON-serialisable representation of the configuration."""
        return {
            "project_root": str(self.PROJECT_ROOT),
            "dataset_path": str(self.DATASET_PATH),
            "models_dir": str(self.MODELS_DIR),
            "logs_dir": str(self.LOGS_DIR),
            "reports_dir": str(self.REPORTS_DIR),
            "plots_dir": str(self.PLOTS_DIR),
            "random_seed": self.RANDOM_SEED,
            "default_target_column": self.DEFAULT_TARGET_COLUMN,
            "test_size": self.TEST_SIZE,
            "valid_size": self.VALID_SIZE,
            "n_epochs": self.N_EPOCHS,
            "batch_size": self.BATCH_SIZE,
            "learning_rate": self.LEARNING_RATE,
            "device": self.DEVICE,
            "hyperparams": self.HYPERPARAMS,
        }

    def save_metadata(self) -> None:
        """Write current configuration to ``models/training_metadata.json``."""
        path = self.MODELS_DIR / self.TRAINING_METADATA_FILENAME
        path.write_text(json.dumps(self.to_dict(), indent=4), encoding="utf-8")


# Module-level singleton used throughout the project
config = Config()

__all__ = ["Config", "config"]
