"""backend/evaluator.py
================================
Evaluation and Visual Metric Generation Module for MindCare AI.

Formats performance statistics, calculates comprehensive classification metrics
(Accuracy, Balanced Accuracy, Precision, Recall, F1, MCC, Cohen's Kappa, Log Loss, ROC-AUC,
Brier Score, Expected Calibration Error ECE), and generates matplotlib plots
(raw & normalized confusion matrices, OvR ROC curves, Precision-Recall curves)
saved to reports/ and plots/.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from .config import config
from .logger import get_logger

logger: logging.Logger = get_logger(__name__)


def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE)."""
    if y_prob.ndim == 2:
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
    else:
        confidences = y_prob
        predictions = (y_prob >= 0.5).astype(int)

    accuracies = (predictions == y_true).astype(float)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * (np.sum(in_bin) / total_samples)

    return float(ece)


class ModelEvaluator:
    """Evaluates classification models, computes advanced metrics, and renders Matplotlib plots."""

    def __init__(self, reports_dir: Path | str | None = None, plots_dir: Path | str | None = None) -> None:
        """Initialize ModelEvaluator.

        Parameters
        ----------
        reports_dir : Path or str, optional
            Destination directory for text reports. Defaults to config.REPORTS_DIR.
        plots_dir : Path or str, optional
            Destination directory for plots. Defaults to config.PLOTS_DIR.
        """
        self.reports_dir = Path(reports_dir or config.REPORTS_DIR)
        self.plots_dir = Path(plots_dir or config.PLOTS_DIR)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_predictions(
        self,
        y_true: Sequence[Any],
        y_pred: Sequence[Any],
        y_prob: np.ndarray | None = None,
        model_name: str = "Model",
        class_names: Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        """Compute full suite of classification metrics, calibration scores, and generate visual plots.

        Parameters
        ----------
        y_true : Sequence[Any]
            Ground truth target labels.
        y_pred : Sequence[Any]
            Predicted target labels.
        y_prob : np.ndarray, optional
            Predicted probability matrix of shape (n_samples, n_classes).
        model_name : str, optional
            Model identifier name. Defaults to "Model".
        class_names : Sequence[str], optional
            Display labels for confusion matrix and reports.

        Returns
        -------
        Dict[str, Any]
            Complete evaluation metrics and generated artifact filepaths.
        """
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)
        unique_classes = np.unique(y_true_arr)
        n_classes = len(unique_classes)
        average_mode = "weighted" if n_classes > 2 else "binary"

        # Core Metrics
        acc = float(accuracy_score(y_true_arr, y_pred_arr))
        bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred_arr))
        prec = float(precision_score(y_true_arr, y_pred_arr, average=average_mode, zero_division=0))
        rec = float(recall_score(y_true_arr, y_pred_arr, average=average_mode, zero_division=0))
        f1 = float(f1_score(y_true_arr, y_pred_arr, average=average_mode, zero_division=0))
        mcc = float(matthews_corrcoef(y_true_arr, y_pred_arr))
        kappa = float(cohen_kappa_score(y_true_arr, y_pred_arr))

        # Log Loss & Calibration Metrics
        l_loss: float | None = None
        brier_score: float | None = None
        ece_score: float | None = None

        if y_prob is not None:
            try:
                l_loss = float(log_loss(y_true_arr, y_prob))
            except Exception as ll_exc:
                logger.warning("Log loss calculation skipped for %s: %s", model_name, ll_exc)

            try:
                ece_score = _compute_ece(y_true_arr, y_prob)
            except Exception as ece_exc:
                logger.warning("ECE calculation skipped for %s: %s", model_name, ece_exc)

            if n_classes == 2 and y_prob.ndim == 2 and y_prob.shape[1] >= 2:
                try:
                    brier_score = float(brier_score_loss(y_true_arr, y_prob[:, 1]))
                except Exception as bs_exc:
                    logger.warning("Brier score calculation skipped for %s: %s", model_name, bs_exc)

        cm = confusion_matrix(y_true_arr, y_pred_arr)
        clf_dict = classification_report(y_true_arr, y_pred_arr, output_dict=True, zero_division=0)
        clf_text = classification_report(y_true_arr, y_pred_arr, zero_division=0)

        plot_paths: Dict[str, str | None] = {
            "confusion_matrix_path": None,
            "normalized_confusion_matrix_path": None,
            "roc_curve_path": None,
            "pr_curve_path": None,
            "classification_report_path": None,
            "metrics_json_path": None,
        }

        # 1. Save Raw & Normalized Confusion Matrices
        try:
            raw_path = self.plot_confusion_matrix(cm, model_name=model_name, class_names=class_names, normalize=False)
            norm_path = self.plot_confusion_matrix(cm, model_name=model_name, class_names=class_names, normalize=True)
            plot_paths["confusion_matrix_path"] = str(raw_path)
            plot_paths["normalized_confusion_matrix_path"] = str(norm_path)
        except Exception as cm_exc:
            logger.exception("Failed generating confusion matrix plot for %s: %s", model_name, cm_exc)

        # 2. Save Classification Report Text
        try:
            report_path = self.save_classification_report(clf_text, model_name=model_name)
            plot_paths["classification_report_path"] = str(report_path)
        except Exception as rep_exc:
            logger.exception("Failed saving classification report for %s: %s", model_name, rep_exc)

        # 3. Save ROC and Precision-Recall Curves if probabilities are available
        if y_prob is not None:
            try:
                roc_path = self.plot_roc_curve(y_true_arr, y_prob, model_name=model_name, class_names=class_names)
                if roc_path:
                    plot_paths["roc_curve_path"] = str(roc_path)
            except Exception as roc_exc:
                logger.exception("Failed generating ROC curve for %s: %s", model_name, roc_exc)

            try:
                pr_path = self.plot_precision_recall_curve(y_true_arr, y_prob, model_name=model_name, class_names=class_names)
                if pr_path:
                    plot_paths["pr_curve_path"] = str(pr_path)
            except Exception as pr_exc:
                logger.exception("Failed generating Precision-Recall curve for %s: %s", model_name, pr_exc)

        results_payload = {
            "model_name": model_name,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "mcc": mcc,
            "cohen_kappa": kappa,
            "log_loss": l_loss,
            "brier_score": brier_score,
            "ece_score": ece_score,
            "confusion_matrix": cm,
            "classification_report_dict": clf_dict,
            "classification_report_text": clf_text,
            "plot_paths": plot_paths,
        }

        # Export Machine-Readable metrics.json
        try:
            json_filename = f"metrics_{model_name.lower().replace(' ', '_')}.json"
            json_save_path = self.reports_dir / json_filename
            serializable_dict = {
                k: (v if k != "confusion_matrix" else v.tolist())
                for k, v in results_payload.items()
            }
            json_save_path.write_text(json.dumps(serializable_dict, indent=4), encoding="utf-8")
            plot_paths["metrics_json_path"] = str(json_save_path)
            logger.info("Saved machine-readable metrics JSON to %s", json_save_path)
        except Exception as json_exc:
            logger.warning("Failed saving metrics JSON for %s: %s", model_name, json_exc)

        return results_payload

    def plot_confusion_matrix(
        self,
        confusion_mat: np.ndarray,
        model_name: str,
        class_names: Sequence[str] | None = None,
        normalize: bool = False,
    ) -> Path:
        """Plot confusion matrix heatmap using Pure Matplotlib (with try/finally figure cleanup)."""
        suffix = "_normalized" if normalize else ""
        filename = f"confusion_matrix{suffix}_{model_name.lower().replace(' ', '_')}.png"
        save_path = self.plots_dir / filename

        cm_data = confusion_mat.astype("float")
        if normalize:
            row_sums = cm_data.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            cm_data = cm_data / row_sums

        fig, ax = plt.subplots(figsize=(6, 5))
        try:
            cax = ax.imshow(cm_data, interpolation="nearest", cmap=plt.cm.Blues)
            fig.colorbar(cax)

            n_classes = confusion_mat.shape[0]
            ticks = np.arange(n_classes)
            labels = [str(c) for c in class_names] if class_names else [str(i) for i in range(n_classes)]

            ax.set_xticks(ticks)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_yticks(ticks)
            ax.set_yticklabels(labels)

            thresh = cm_data.max() / 2.0
            for i in range(n_classes):
                for j in range(n_classes):
                    val_str = f"{cm_data[i, j]:.2f}" if normalize else f"{int(confusion_mat[i, j])}"
                    color = "white" if cm_data[i, j] > thresh else "black"
                    ax.text(j, i, val_str, horizontalalignment="center", color=color, fontweight="bold")

            title_type = "Normalized Confusion Matrix" if normalize else "Confusion Matrix"
            ax.set_title(f"{title_type}: {model_name}")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            plt.tight_layout()

            fig.savefig(save_path, dpi=300)
            logger.info("Saved %s plot to %s", title_type, save_path)
            return save_path
        finally:
            plt.close(fig)

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str,
        class_names: Sequence[str] | None = None,
    ) -> Path | None:
        """Plot ROC Curve using Pure Matplotlib (with try/finally figure cleanup)."""
        filename = f"roc_curve_{model_name.lower().replace(' ', '_')}.png"
        save_path = self.plots_dir / filename

        classes = np.unique(y_true)
        n_classes = len(classes)

        if y_prob.ndim < 2 or y_prob.shape[1] < n_classes:
            return None

        fig, ax = plt.subplots(figsize=(7, 5))
        try:
            if n_classes == 2:
                fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
                roc_auc_val = float(auc(fpr, tpr))
                ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc_val:.3f})")
            else:
                y_bin = label_binarize(y_true, classes=classes)
                for i in range(n_classes):
                    fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
                    cls_auc = float(auc(fpr, tpr))
                    label_name = class_names[i] if (class_names and i < len(class_names)) else f"Class {classes[i]}"
                    ax.plot(fpr, tpr, lw=2, label=f"ROC {label_name} (AUC = {cls_auc:.3f})")

            ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve: {model_name}")
            ax.legend(loc="lower right")
            ax.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()

            fig.savefig(save_path, dpi=300)
            logger.info("Saved ROC curve plot to %s", save_path)
            return save_path
        finally:
            plt.close(fig)

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str,
        class_names: Sequence[str] | None = None,
    ) -> Path | None:
        """Plot Precision-Recall Curve using Pure Matplotlib (with try/finally figure cleanup)."""
        filename = f"pr_curve_{model_name.lower().replace(' ', '_')}.png"
        save_path = self.plots_dir / filename

        classes = np.unique(y_true)
        n_classes = len(classes)

        if y_prob.ndim < 2 or y_prob.shape[1] < n_classes:
            return None

        fig, ax = plt.subplots(figsize=(7, 5))
        try:
            if n_classes == 2:
                prec, rec, _ = precision_recall_curve(y_true, y_prob[:, 1])
                avg_prec = float(average_precision_score(y_true, y_prob[:, 1]))
                ax.plot(rec, prec, color="purple", lw=2, label=f"PR curve (AP = {avg_prec:.3f})")
            else:
                y_bin = label_binarize(y_true, classes=classes)
                for i in range(n_classes):
                    prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
                    cls_ap = float(average_precision_score(y_bin[:, i], y_prob[:, i]))
                    label_name = class_names[i] if (class_names and i < len(class_names)) else f"Class {classes[i]}"
                    ax.plot(rec, prec, lw=2, label=f"PR {label_name} (AP = {cls_ap:.3f})")

            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f"Precision-Recall Curve: {model_name}")
            ax.legend(loc="lower left")
            ax.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()

            fig.savefig(save_path, dpi=300)
            logger.info("Saved Precision-Recall curve plot to %s", save_path)
            return save_path
        finally:
            plt.close(fig)

    def save_classification_report(
        self, classification_text: str, model_name: str
    ) -> Path:
        """Save text classification report to reports/ directory."""
        filename = f"classification_report_{model_name.lower().replace(' ', '_')}.txt"
        save_path = self.reports_dir / filename
        save_path.write_text(classification_text, encoding="utf-8")
        logger.info("Saved classification report to %s", save_path)
        return save_path


__all__ = ["ModelEvaluator"]
