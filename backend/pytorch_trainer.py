"""backend/pytorch_trainer.py
================================
PyTorch Model Training and Execution Pipeline for MindCare AI.

Handles dataset wrapping, data loaders, training loops, validation monitoring,
early stopping, mixed precision acceleration, progress tracking, model checkpointing,
reproducible DataLoader shuffling, evaluation, and inference for PyTorch deep learning models.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import config
from .logger import get_logger
from .pytorch_model import MindCarePyTorchClassifier

logger: logging.Logger = get_logger(__name__)


class TabularDataset(Dataset):
    """PyTorch Dataset wrapper for tabular feature matrices and integer labels."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """Initialize Dataset.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target labels array of shape (n_samples,).
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class EarlyStopping:
    """Early stopping handler to halt training when validation loss stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        """Initialize EarlyStopping.

        Parameters
        ----------
        patience : int, optional
            Number of epochs to wait after last improvement. Defaults to 5.
        min_delta : float, optional
            Minimum change in validation loss to qualify as an improvement. Defaults to 1e-4.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_state_dict: Dict[str, Any] | None = None

    def __call__(self, val_loss: float, model: MindCarePyTorchClassifier) -> bool:
        """Check if validation loss improved, update best state, and trigger checkpoint save.

        Parameters
        ----------
        val_loss : float
            Current epoch validation loss.
        model : MindCarePyTorchClassifier
            Current PyTorch model instance.

        Returns
        -------
        bool
            True if early stopping criterion is met, False otherwise.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # Save checkpoint immediately on improvement using MindCarePyTorchClassifier format
            checkpoint_path = config.MODELS_DIR / config.TORCH_MODEL_FILENAME
            model.save(checkpoint_path)
            logger.info("Validation loss improved to %.4f. Saved checkpoint to %s", val_loss, checkpoint_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class PyTorchTrainer:
    """Orchestrates PyTorch model training, validation, evaluation, mixed precision, and persistence."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        device: str | None = None,
    ) -> None:
        """Initialize PyTorch Trainer.

        Parameters
        ----------
        input_dim : int
            Number of input features.
        num_classes : int
            Number of output target classes (min 2).
        device : str, optional
            Device name ('cuda' or 'cpu'). Defaults to config.DEVICE.
        """
        self.input_dim = input_dim
        self.num_classes = max(num_classes, 2)
        self.device = torch.device(device or config.DEVICE)
        self.model = MindCarePyTorchClassifier(
            input_dim=self.input_dim, num_classes=self.num_classes
        ).to(self.device)

        self.total_training_time_seconds = 0.0
        self.history: Dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        patience: int = 5,
    ) -> Dict[str, list[float]]:
        """Train the PyTorch neural network model across specified epochs with mixed precision.

        Returns
        -------
        Dict[str, list[float]]
            Training history dictionary containing loss and accuracy metrics.
        """
        logger.info("Starting PyTorch training pipeline on compute device: %s", self.device)

        train_dataset = TabularDataset(X_train, y_train)
        val_dataset = TabularDataset(X_val, y_val)

        # Reproducible random generator for DataLoader shuffling
        generator = torch.Generator().manual_seed(config.RANDOM_SEED)

        # High performance DataLoader parameters for GPU training
        is_cuda = self.device.type == "cuda"
        num_workers = int(config.HYPERPARAMS.get("num_workers", 0))
        pin_memory = is_cuda
        persistent_workers = num_workers > 0

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=config.LR_SCHEDULER_FACTOR, patience=config.LR_SCHEDULER_PATIENCE
        )
        early_stopping = EarlyStopping(patience=patience)

        # Mixed Precision Acceleration setup
        scaler = torch.amp.GradScaler("cuda", enabled=is_cuda)

        start_train_time = time.perf_counter()

        # Epoch loop wrapped in tqdm progress bar
        progress_bar = tqdm(range(1, epochs + 1), desc="PyTorch Training Epochs", leave=True)

        for epoch in progress_bar:
            # --- Training Epoch ---
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for X_b, y_b in train_loader:
                X_b, y_b = X_b.to(self.device, non_blocking=is_cuda), y_b.to(self.device, non_blocking=is_cuda)

                optimizer.zero_grad()

                # Autocast context for mixed precision
                with torch.amp.autocast(device_type=self.device.type, enabled=is_cuda):
                    logits = self.model(X_b)
                    loss = criterion(logits, y_b)

                if is_cuda:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * X_b.size(0)
                preds = torch.argmax(logits, dim=1)
                correct_train += (preds == y_b).sum().item()
                total_train += y_b.size(0)

            epoch_train_loss = float(running_loss / total_train)
            epoch_train_acc = float(correct_train / total_train)

            # --- Validation Epoch ---
            self.model.eval()
            val_running_loss = 0.0
            correct_val = 0
            total_val = 0

            with torch.no_grad():
                for X_b, y_b in val_loader:
                    X_b, y_b = X_b.to(self.device, non_blocking=is_cuda), y_b.to(self.device, non_blocking=is_cuda)

                    with torch.amp.autocast(device_type=self.device.type, enabled=is_cuda):
                        logits = self.model(X_b)
                        loss = criterion(logits, y_b)

                    val_running_loss += loss.item() * X_b.size(0)
                    preds = torch.argmax(logits, dim=1)
                    correct_val += (preds == y_b).sum().item()
                    total_val += y_b.size(0)

            epoch_val_loss = float(val_running_loss / total_val)
            epoch_val_acc = float(correct_val / total_val)

            scheduler.step(epoch_val_loss)

            self.history["train_loss"].append(epoch_train_loss)
            self.history["train_acc"].append(epoch_train_acc)
            self.history["val_loss"].append(epoch_val_loss)
            self.history["val_acc"].append(epoch_val_acc)

            progress_bar.set_postfix({
                "Train Loss": f"{epoch_train_loss:.4f}",
                "Val Loss": f"{epoch_val_loss:.4f}",
                "Val Acc": f"{epoch_val_acc:.4f}",
            })

            logger.info(
                "Epoch [%d/%d] | Train Loss: %.4f | Train Acc: %.4f | Val Loss: %.4f | Val Acc: %.4f",
                epoch,
                epochs,
                epoch_train_loss,
                epoch_train_acc,
                epoch_val_loss,
                epoch_val_acc,
            )

            if early_stopping(epoch_val_loss, self.model):
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

        self.total_training_time_seconds = time.perf_counter() - start_train_time
        logger.info("PyTorch training completed in %.3fs", self.total_training_time_seconds)

        if early_stopping.best_state_dict is not None:
            self.model.load_state_dict(early_stopping.best_state_dict)
            logger.info("Restored best PyTorch model weights with lowest validation loss: %.4f", early_stopping.best_loss)

        return self.history

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate trained PyTorch model on test set.

        Returns
        -------
        Dict[str, Any]
            Performance metrics dictionary compatible with ModelComparator.
        """
        self.model.eval()

        start_time = time.perf_counter()
        probs = self.model.predict_proba(X_test, device=self.device)
        preds = np.argmax(probs, axis=1)
        infer_time = time.perf_counter() - start_time

        average_mode = "weighted" if self.num_classes > 2 else "binary"

        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, average=average_mode, zero_division=0))
        rec = float(recall_score(y_test, preds, average=average_mode, zero_division=0))
        f1 = float(f1_score(y_test, preds, average=average_mode, zero_division=0))
        cm = confusion_matrix(y_test, preds)
        clf_rep_dict = classification_report(y_test, preds, output_dict=True, zero_division=0)
        clf_rep_text = classification_report(y_test, preds, zero_division=0)

        roc_auc = None
        try:
            if self.num_classes == 2:
                roc_auc = float(roc_auc_score(y_test, probs[:, 1]))
            else:
                roc_auc = float(roc_auc_score(y_test, probs, multi_class="ovr", average="weighted"))
        except Exception as roc_exc:
            logger.warning("ROC-AUC calculation skipped for PyTorch model: %s", roc_exc)

        return {
            "model_name": "PyTorch Deep Learning",
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": cm,
            "classification_report_dict": clf_rep_dict,
            "classification_report_text": clf_rep_text,
            "inference_time_seconds": infer_time,
            "training_time_seconds": float(self.total_training_time_seconds),
        }

    def save_model(self, filepath: Path | str | None = None) -> Path:
        """Save PyTorch model architecture and weights to disk."""
        save_path = Path(filepath or (config.MODELS_DIR / config.TORCH_MODEL_FILENAME))
        return self.model.save(save_path)

    def load_model(self, filepath: Path | str | None = None) -> None:
        """Load PyTorch model architecture and weights from disk."""
        load_path = Path(filepath or (config.MODELS_DIR / config.TORCH_MODEL_FILENAME))
        self.model = MindCarePyTorchClassifier.load(load_path, device=self.device)
        self.input_dim = self.model.input_dim
        self.num_classes = self.model.num_classes


__all__ = ["TabularDataset", "EarlyStopping", "PyTorchTrainer"]
