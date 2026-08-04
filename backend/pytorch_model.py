"""backend/pytorch_model.py
================================
PyTorch Deep Learning Classifier Architecture for MindCare AI.

Defines a modular, production-ready deep neural network architecture for tabular
classification tasks.

Features:
- Configurable hidden layer architecture via config.py
- Configurable activation functions (LeakyReLU, ReLU, GELU)
- Activation-specific weight initialization (Kaiming for ReLU/LeakyReLU, Xavier for GELU)
- Standardized multi-class / binary classification (N-output neurons + CrossEntropyLoss)
- Inference helpers: predict() and predict_proba()
- Model parameter counting & summary logging
- Built-in save() and load() state-dict serialization
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .config import config
from .logger import get_logger

logger: logging.Logger = get_logger(__name__)

ActivationType = Literal["leaky_relu", "relu", "gelu"]


class MindCarePyTorchClassifier(nn.Module):
    """Deep Learning Tabular Classifier Architecture in PyTorch.

    Classification Design Standard (Option B):
    - All classification problems (both binary with 2 classes and multi-class with N classes)
      use `num_classes` output logits.
    - Loss function: `nn.CrossEntropyLoss()`.
    - Probability activation: `torch.softmax(logits, dim=1)`.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Sequence[int] | None = None,
        dropout_rate: float = 0.2,
        activation: ActivationType = "leaky_relu",
    ) -> None:
        """Initialize the deep neural network classifier.

        Parameters
        ----------
        input_dim : int
            Number of input features.
        num_classes : int
            Number of target output classes (min 2 for classification).
        hidden_dims : Sequence[int], optional
            Dimensions of hidden layers. Defaults to config or (128, 64, 32).
        dropout_rate : float, optional
            Dropout probability for regularization. Defaults to 0.2.
        activation : ActivationType, optional
            Activation function name ('leaky_relu', 'relu', or 'gelu'). Defaults to 'leaky_relu'.
        """
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")

        if num_classes < 2:
            raise ValueError(f"num_classes must be at least 2 for classification, got {num_classes}")

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.activation_name = activation

        if hidden_dims is None:
            default_dims = config.HYPERPARAMS.get("hidden_dims", (128, 64, 32))
            if isinstance(default_dims, (list, tuple)):
                hidden_dims = tuple(default_dims)
            else:
                hidden_dims = (128, 64, 32)

        self.hidden_dims: Tuple[int, ...] = tuple(hidden_dims)

        layers: List[nn.Module] = []
        current_dim = input_dim

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))

            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(negative_slope=0.01))
            else:
                raise ValueError(f"Unsupported activation function: {activation}")

            layers.append(nn.Dropout(p=dropout_rate))
            current_dim = hidden_dim

        # Output classification head (N output neurons for both binary and multi-class)
        layers.append(nn.Linear(current_dim, num_classes))

        self.network = nn.Sequential(*layers)

        # Apply weight initialization automatically
        self.initialize_weights()

        total_p, train_p = self.get_num_parameters()
        logger.info(
            "MindCarePyTorchClassifier initialized | Input: %d | Classes: %d | Hidden: %s | Activation: %s | Params: %d (Trainable: %d)",
            self.input_dim,
            self.num_classes,
            self.hidden_dims,
            self.activation_name,
            total_p,
            train_p,
        )

    def _init_layer_weights(self, module: nn.Module) -> None:
        """Initialize weights of Linear and BatchNorm layers according to selected activation."""
        if isinstance(module, nn.Linear):
            if self.activation_name == "relu":
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif self.activation_name == "leaky_relu":
                nn.init.kaiming_normal_(module.weight, a=0.01, nonlinearity="leaky_relu")
            elif self.activation_name == "gelu":
                nn.init.xavier_normal_(module.weight)
            else:
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")

            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0.0)

    def initialize_weights(self) -> None:
        """Apply weight initialization to all linear and batchnorm modules recursively."""
        self.apply(self._init_layer_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the neural network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, num_classes).
        """
        return self.network(x)

    def predict_proba(
        self, x: Union[torch.Tensor, np.ndarray], device: str | torch.device = "cpu"
    ) -> np.ndarray:
        """Predict class probability distribution.

        Parameters
        ----------
        x : torch.Tensor or np.ndarray
            Input feature tensor or array of shape (batch_size, input_dim).
        device : str or torch.device
            Device to execute inference on.

        Returns
        -------
        np.ndarray
            Probability distribution matrix of shape (batch_size, num_classes).
        """
        self.eval()
        device_obj = torch.device(device)

        if isinstance(x, np.ndarray):
            tensor_x = torch.tensor(x, dtype=torch.float32)
        else:
            tensor_x = x.to(torch.float32)

        tensor_x = tensor_x.to(device_obj)

        with torch.no_grad():
            logits = self.forward(tensor_x)
            probs = torch.softmax(logits, dim=1)
            return probs.cpu().numpy()

    def predict(
        self, x: Union[torch.Tensor, np.ndarray], device: str | torch.device = "cpu"
    ) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        x : torch.Tensor or np.ndarray
            Input feature tensor or array.
        device : str or torch.device
            Device to execute inference on.

        Returns
        -------
        np.ndarray
            Class index predictions of shape (batch_size,).
        """
        probs = self.predict_proba(x, device=device)
        return np.argmax(probs, axis=1)

    def get_num_parameters(self) -> Tuple[int, int]:
        """Return parameter statistics.

        Returns
        -------
        Tuple of (total_parameters, trainable_parameters)
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

    def save(self, filepath: Path | str) -> Path:
        """Save model architecture state dict and configuration metadata."""
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "state_dict": self.state_dict(),
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "hidden_dims": self.hidden_dims,
            "dropout_rate": self.dropout_rate,
            "activation_name": self.activation_name,
        }
        torch.save(checkpoint, save_path)
        logger.info("Saved MindCarePyTorchClassifier checkpoint to %s", save_path)
        return save_path

    @classmethod
    def load(
        cls, filepath: Path | str, device: str | torch.device = "cpu"
    ) -> "MindCarePyTorchClassifier":
        """Load model instance from saved checkpoint file."""
        load_path = Path(filepath)
        if not load_path.is_file():
            raise FileNotFoundError(f"PyTorch model checkpoint not found at {load_path}")

        device_obj = torch.device(device)
        checkpoint: Dict = torch.load(load_path, map_location=device_obj)

        model = cls(
            input_dim=checkpoint["input_dim"],
            num_classes=checkpoint["num_classes"],
            hidden_dims=checkpoint.get("hidden_dims", (128, 64, 32)),
            dropout_rate=checkpoint.get("dropout_rate", 0.2),
            activation=checkpoint.get("activation_name", "leaky_relu"),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device_obj)
        model.eval()

        logger.info("Loaded MindCarePyTorchClassifier checkpoint from %s", load_path)
        return model


__all__ = ["MindCarePyTorchClassifier"]
