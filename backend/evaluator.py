"""backend/evaluator.py
================================
Evaluation and Visual Metric Generation Module for MindCare AI.

Formats performance statistics, calculates comprehensive classification metrics
(Accuracy, Balanced Accuracy, Precision, Recall, F1, MCC, Cohen's Kappa,
Log Loss, ROC-AUC, Brier Score, Expected Calibration Error ECE), and generates
Matplotlib plots (raw & normalized confusion matrices, ROC curves,
Precision-Recall curves) saved to reports/ and plots/.

Supports both:
    - String labels: "No" / "Yes"
    - Encoded labels: 0 / 1
    - General multiclass labels
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Sequence

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


# ============================================================================ #
# Helper Functions
# ============================================================================ #


def _get_positive_label(classes: np.ndarray) -> Any:
    """Return the intended positive label for binary classification.

    MindCare AI's target is:
        No / Yes

    PyTorch uses:
        0 / 1

    Therefore:
        Yes -> positive class
        1   -> positive class

    For any other binary labels, the last sorted class is used.
    """
    if len(classes) != 2:
        return None

    if "Yes" in classes:
        return "Yes"

    if 1 in classes:
        return 1

    return classes[-1]


def _get_positive_class_index(
    classes: np.ndarray,
    positive_label: Any,
) -> int:
    """Return the probability-column index corresponding to positive_label."""
    matches = np.where(classes == positive_label)[0]

    if len(matches) > 0:
        return int(matches[0])

    # Defensive fallback for unexpected label/probability combinations.
    return 1 if len(classes) == 2 else 0


def _encode_labels_for_probability_metrics(
    y_true: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    """Encode arbitrary class labels to probability-column indices.

    This is critical because the evaluator receives either:

        ["No", "Yes", ...]
    or:
        [0, 1, ...]

    The encoded values must correspond to the columns of y_prob.
    """
    class_to_index = {
        class_value: index
        for index, class_value in enumerate(classes)
    }

    try:
        return np.asarray(
            [class_to_index[value] for value in y_true],
            dtype=int,
        )
    except KeyError as exc:
        raise ValueError(
            f"Found label {exc.args[0]!r} that is not present in "
            f"classes {classes.tolist()}."
        ) from exc


def _compute_ece(
    y_true_encoded: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calculate Expected Calibration Error (ECE).

    Supports:
        Binary probabilities with shape (n_samples, 2)
        Multiclass probabilities with shape (n_samples, n_classes)

    ECE compares confidence with correctness of the predicted class.
    """
    y_true_encoded = np.asarray(y_true_encoded)
    y_prob = np.asarray(y_prob)

    if len(y_true_encoded) == 0:
        return 0.0

    if y_prob.ndim == 1:
        confidences = y_prob
        predictions = (y_prob >= 0.5).astype(int)

    elif y_prob.ndim == 2:
        if y_prob.shape[1] == 1:
            confidences = y_prob[:, 0]
            predictions = (y_prob[:, 0] >= 0.5).astype(int)
        else:
            predictions = np.argmax(y_prob, axis=1)
            confidences = np.max(y_prob, axis=1)

    else:
        raise ValueError(
            f"Unsupported probability array shape for ECE: {y_prob.shape}"
        )

    accuracies = (predictions == y_true_encoded).astype(float)

    bin_boundaries = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    ece = 0.0
    total_samples = len(y_true_encoded)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # Include 0 in first bin and 1 in final bin.
        if i == 0:
            in_bin = (
                (confidences >= bin_lower)
                & (confidences <= bin_upper)
            )
        else:
            in_bin = (
                (confidences > bin_lower)
                & (confidences <= bin_upper)
            )

        count_in_bin = int(np.sum(in_bin))

        if count_in_bin == 0:
            continue

        accuracy_in_bin = float(
            np.mean(accuracies[in_bin])
        )

        avg_confidence_in_bin = float(
            np.mean(confidences[in_bin])
        )

        ece += (
            abs(
                accuracy_in_bin
                - avg_confidence_in_bin
            )
            * (count_in_bin / total_samples)
        )

    return float(ece)


# ============================================================================ #
# Model Evaluator
# ============================================================================ #


class ModelEvaluator:
    """Evaluate classification models and generate evaluation artifacts."""

    def __init__(
        self,
        reports_dir: Path | str | None = None,
        plots_dir: Path | str | None = None,
    ) -> None:
        """Initialize ModelEvaluator."""

        self.reports_dir = Path(
            reports_dir or config.REPORTS_DIR
        )

        self.plots_dir = Path(
            plots_dir or config.PLOTS_DIR
        )

        self.reports_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.plots_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ====================================================================== #
    # Main Evaluation
    # ====================================================================== #

    def evaluate_predictions(
        self,
        y_true: Sequence[Any],
        y_pred: Sequence[Any],
        y_prob: np.ndarray | None = None,
        model_name: str = "Model",
        class_names: Sequence[str] | None = None,
    ) -> Dict[str, Any]:
        """Compute comprehensive classification metrics.

        Supports both string and encoded labels.

        Examples:
            y_true = ["No", "Yes", "No", "Yes"]

        or:

            y_true = [0, 1, 0, 1]
        """

        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)

        if len(y_true_arr) != len(y_pred_arr):
            raise ValueError(
                "y_true and y_pred must contain the same number of samples."
            )

        unique_classes = np.unique(y_true_arr)
        n_classes = len(unique_classes)

        if n_classes < 2:
            raise ValueError(
                f"Evaluation requires at least 2 classes. "
                f"Found: {unique_classes.tolist()}"
            )

        is_binary = n_classes == 2

        # ------------------------------------------------------------------ #
        # Positive-class determination
        # ------------------------------------------------------------------ #

        positive_label = (
            _get_positive_label(unique_classes)
            if is_binary
            else None
        )

        positive_class_index = (
            _get_positive_class_index(
                unique_classes,
                positive_label,
            )
            if is_binary
            else None
        )

        # For MindCare:
        #
        #   ["No", "Yes"] -> positive = Yes
        #   [0, 1]        -> positive = 1
        #
        # This keeps sklearn and PyTorch evaluation consistent.

        if is_binary:
            average_mode = "binary"

            binary_kwargs = {
                "pos_label": positive_label
            }

        else:
            average_mode = "weighted"
            binary_kwargs = {}

        # ================================================================== #
        # Core Classification Metrics
        # ================================================================== #

        acc = float(
            accuracy_score(
                y_true_arr,
                y_pred_arr,
            )
        )

        bal_acc = float(
            balanced_accuracy_score(
                y_true_arr,
                y_pred_arr,
            )
        )

        prec = float(
            precision_score(
                y_true_arr,
                y_pred_arr,
                average=average_mode,
                zero_division=0,
                **binary_kwargs,
            )
        )

        rec = float(
            recall_score(
                y_true_arr,
                y_pred_arr,
                average=average_mode,
                zero_division=0,
                **binary_kwargs,
            )
        )

        f1 = float(
            f1_score(
                y_true_arr,
                y_pred_arr,
                average=average_mode,
                zero_division=0,
                **binary_kwargs,
            )
        )

        mcc = float(
            matthews_corrcoef(
                y_true_arr,
                y_pred_arr,
            )
        )

        kappa = float(
            cohen_kappa_score(
                y_true_arr,
                y_pred_arr,
            )
        )

        # ================================================================== #
        # Probability-Based Metrics
        # ================================================================== #

        l_loss: float | None = None
        brier_score: float | None = None
        ece_score: float | None = None
        roc_auc_score_value: float | None = None

        normalized_probabilities: np.ndarray | None = None

        if y_prob is not None:

            y_prob = np.asarray(
                y_prob,
                dtype=float,
            )

            # -------------------------------------------------------------- #
            # Validate probability shape
            # -------------------------------------------------------------- #

            if len(y_prob) != len(y_true_arr):
                logger.warning(
                    "Probability sample count mismatch for %s. "
                    "Expected %d, got %d.",
                    model_name,
                    len(y_true_arr),
                    len(y_prob),
                )

            else:

                # ---------------------------------------------------------- #
                # Probability-column count
                # ---------------------------------------------------------- #

                if y_prob.ndim == 2:

                    if y_prob.shape[1] < n_classes:
                        logger.warning(
                            "Probability array for %s has %d columns "
                            "but %d classes were detected.",
                            model_name,
                            y_prob.shape[1],
                            n_classes,
                        )

                    else:
                        normalized_probabilities = y_prob

                elif y_prob.ndim == 1:

                    # Binary 1-D probability vector.
                    if is_binary:
                        normalized_probabilities = np.column_stack(
                            [
                                1.0 - y_prob,
                                y_prob,
                            ]
                        )

                # ---------------------------------------------------------- #
                # Label encoding for probability metrics
                # ---------------------------------------------------------- #

                try:

                    y_encoded = _encode_labels_for_probability_metrics(
                        y_true_arr,
                        unique_classes,
                    )

                    # ------------------------------------------------------ #
                    # Log Loss
                    # ------------------------------------------------------ #

                    try:
                        if normalized_probabilities is not None:

                            l_loss = float(
                                log_loss(
                                    y_encoded,
                                    normalized_probabilities,
                                    labels=np.arange(
                                        normalized_probabilities.shape[1]
                                    ),
                                )
                            )

                    except Exception as exc:

                        logger.warning(
                            "Log loss calculation skipped for %s: %s",
                            model_name,
                            exc,
                        )

                    # ------------------------------------------------------ #
                    # ECE
                    # ------------------------------------------------------ #

                    try:

                        if normalized_probabilities is not None:

                            ece_score = _compute_ece(
                                y_encoded,
                                normalized_probabilities,
                            )

                    except Exception as exc:

                        logger.warning(
                            "ECE calculation skipped for %s: %s",
                            model_name,
                            exc,
                        )

                    # ------------------------------------------------------ #
                    # Binary Probability Metrics
                    # ------------------------------------------------------ #

                    if (
                        is_binary
                        and normalized_probabilities is not None
                        and normalized_probabilities.ndim == 2
                        and normalized_probabilities.shape[1] >= 2
                    ):

                        assert positive_class_index is not None

                        positive_probability = (
                            normalized_probabilities[
                                :,
                                positive_class_index,
                            ]
                        )

                        y_true_binary = (
                            y_encoded == positive_class_index
                        ).astype(int)

                        # -------------------------------------------------- #
                        # Brier Score
                        # -------------------------------------------------- #

                        try:

                            brier_score = float(
                                brier_score_loss(
                                    y_true_binary,
                                    positive_probability,
                                )
                            )

                        except Exception as exc:

                            logger.warning(
                                "Brier score calculation skipped for %s: %s",
                                model_name,
                                exc,
                            )

                        # -------------------------------------------------- #
                        # ROC-AUC
                        # -------------------------------------------------- #

                        try:

                            if (
                                len(np.unique(y_true_binary))
                                == 2
                            ):

                                roc_fpr, roc_tpr, _ = roc_curve(
                                    y_true_binary,
                                    positive_probability,
                                )

                                roc_auc_score_value = float(
                                    auc(
                                        roc_fpr,
                                        roc_tpr,
                                    )
                                )

                        except Exception as exc:

                            logger.warning(
                                "ROC-AUC calculation skipped for %s: %s",
                                model_name,
                                exc,
                            )

                    # ------------------------------------------------------ #
                    # Multiclass ROC-AUC
                    # ------------------------------------------------------ #

                    elif (
                        not is_binary
                        and normalized_probabilities is not None
                        and normalized_probabilities.ndim == 2
                    ):

                        try:

                            y_bin = label_binarize(
                                y_encoded,
                                classes=np.arange(
                                    n_classes
                                ),
                            )

                            auc_values: list[float] = []

                            for i in range(
                                min(
                                    n_classes,
                                    normalized_probabilities.shape[1],
                                )
                            ):

                                # ROC is undefined if a class has only
                                # one label in the evaluation set.
                                if (
                                    len(
                                        np.unique(
                                            y_bin[:, i]
                                        )
                                    )
                                    < 2
                                ):
                                    continue

                                fpr, tpr, _ = roc_curve(
                                    y_bin[:, i],
                                    normalized_probabilities[:, i],
                                )

                                auc_values.append(
                                    float(
                                        auc(
                                            fpr,
                                            tpr,
                                        )
                                    )
                                )

                            if auc_values:
                                roc_auc_score_value = float(
                                    np.mean(auc_values)
                                )

                        except Exception as exc:

                            logger.warning(
                                "Multiclass ROC-AUC calculation skipped "
                                "for %s: %s",
                                model_name,
                                exc,
                            )

                except Exception as exc:

                    logger.warning(
                        "Probability label encoding failed for %s: %s",
                        model_name,
                        exc,
                    )

        # ================================================================== #
        # Confusion Matrix
        # ================================================================== #

        cm = confusion_matrix(
            y_true_arr,
            y_pred_arr,
            labels=unique_classes,
        )

        # ================================================================== #
        # Classification Report
        # ================================================================== #

        clf_dict = classification_report(
            y_true_arr,
            y_pred_arr,
            labels=unique_classes,
            output_dict=True,
            zero_division=0,
        )

        clf_text = classification_report(
            y_true_arr,
            y_pred_arr,
            labels=unique_classes,
            zero_division=0,
        )

        # ================================================================== #
        # Plot Paths
        # ================================================================== #

        plot_paths: Dict[str, str | None] = {
            "confusion_matrix_path": None,
            "normalized_confusion_matrix_path": None,
            "roc_curve_path": None,
            "pr_curve_path": None,
            "classification_report_path": None,
            "metrics_json_path": None,
        }

        # ================================================================== #
        # Confusion Matrix Plots
        # ================================================================== #

        try:

            raw_path = self.plot_confusion_matrix(
                cm,
                model_name=model_name,
                class_names=(
                    class_names
                    if class_names is not None
                    else [str(c) for c in unique_classes]
                ),
                normalize=False,
            )

            norm_path = self.plot_confusion_matrix(
                cm,
                model_name=model_name,
                class_names=(
                    class_names
                    if class_names is not None
                    else [str(c) for c in unique_classes]
                ),
                normalize=True,
            )

            plot_paths[
                "confusion_matrix_path"
            ] = str(raw_path)

            plot_paths[
                "normalized_confusion_matrix_path"
            ] = str(norm_path)

        except Exception as exc:

            logger.exception(
                "Failed generating confusion matrix plots "
                "for %s: %s",
                model_name,
                exc,
            )

        # ================================================================== #
        # Classification Report File
        # ================================================================== #

        try:

            report_path = self.save_classification_report(
                clf_text,
                model_name=model_name,
            )

            plot_paths[
                "classification_report_path"
            ] = str(report_path)

        except Exception as exc:

            logger.exception(
                "Failed saving classification report "
                "for %s: %s",
                model_name,
                exc,
            )

        # ================================================================== #
        # ROC and Precision-Recall Plots
        # ================================================================== #

        if y_prob is not None:

            try:

                roc_path = self.plot_roc_curve(
                    y_true_arr,
                    y_prob,
                    model_name=model_name,
                    class_names=class_names,
                )

                if roc_path is not None:

                    plot_paths[
                        "roc_curve_path"
                    ] = str(roc_path)

            except Exception as exc:

                logger.exception(
                    "Failed generating ROC curve for %s: %s",
                    model_name,
                    exc,
                )

            try:

                pr_path = self.plot_precision_recall_curve(
                    y_true_arr,
                    y_prob,
                    model_name=model_name,
                    class_names=class_names,
                )

                if pr_path is not None:

                    plot_paths[
                        "pr_curve_path"
                    ] = str(pr_path)

            except Exception as exc:

                logger.exception(
                    "Failed generating Precision-Recall curve "
                    "for %s: %s",
                    model_name,
                    exc,
                )

        # ================================================================== #
        # Final Evaluation Payload
        # ================================================================== #

        results_payload: Dict[str, Any] = {
            "model_name": model_name,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "mcc": mcc,
            "cohen_kappa": kappa,
            "roc_auc": roc_auc_score_value,
            "log_loss": l_loss,
            "brier_score": brier_score,
            "ece_score": ece_score,
            "confusion_matrix": cm,
            "classification_report_dict": clf_dict,
            "classification_report_text": clf_text,
            "plot_paths": plot_paths,
        }

        # ================================================================== #
        # Machine-Readable JSON Report
        # ================================================================== #

        try:

            json_filename = (
                f"metrics_"
                f"{model_name.lower().replace(' ', '_')}.json"
            )

            json_save_path = (
                self.reports_dir / json_filename
            )

            serializable_dict = dict(
                results_payload
            )

            serializable_dict[
                "confusion_matrix"
            ] = cm.tolist()

            json_save_path.write_text(
                json.dumps(
                    serializable_dict,
                    indent=4,
                ),
                encoding="utf-8",
            )

            plot_paths[
                "metrics_json_path"
            ] = str(json_save_path)

            logger.info(
                "Saved machine-readable metrics JSON to %s",
                json_save_path,
            )

        except Exception as exc:

            logger.warning(
                "Failed saving metrics JSON for %s: %s",
                model_name,
                exc,
            )

        return results_payload

    # ====================================================================== #
    # Confusion Matrix Plot
    # ====================================================================== #

    def plot_confusion_matrix(
        self,
        confusion_mat: np.ndarray,
        model_name: str,
        class_names: Sequence[str] | None = None,
        normalize: bool = False,
    ) -> Path:
        """Generate and save a confusion matrix plot."""

        suffix = (
            "_normalized"
            if normalize
            else ""
        )

        filename = (
            f"confusion_matrix"
            f"{suffix}_"
            f"{model_name.lower().replace(' ', '_')}.png"
        )

        save_path = (
            self.plots_dir / filename
        )

        cm_data = confusion_mat.astype(
            float
        )

        if normalize:

            row_sums = cm_data.sum(
                axis=1,
                keepdims=True,
            )

            row_sums[
                row_sums == 0
            ] = 1.0

            cm_data = (
                cm_data / row_sums
            )

        fig, ax = plt.subplots(
            figsize=(6, 5)
        )

        try:

            cax = ax.imshow(
                cm_data,
                interpolation="nearest",
                cmap=plt.cm.Blues,
            )

            fig.colorbar(cax)

            n_classes = (
                confusion_mat.shape[0]
            )

            ticks = np.arange(
                n_classes
            )

            if class_names:

                labels = [
                    str(c)
                    for c in class_names
                ]

            else:

                labels = [
                    str(i)
                    for i in range(
                        n_classes
                    )
                ]

            # Ensure label count matches matrix size.
            if len(labels) != n_classes:

                labels = [
                    str(i)
                    for i in range(
                        n_classes
                    )
                ]

            ax.set_xticks(ticks)

            ax.set_xticklabels(
                labels,
                rotation=45,
                ha="right",
            )

            ax.set_yticks(ticks)

            ax.set_yticklabels(
                labels
            )

            thresh = (
                cm_data.max() / 2.0
            )

            for i in range(
                n_classes
            ):

                for j in range(
                    n_classes
                ):

                    if normalize:

                        val_str = (
                            f"{cm_data[i, j]:.2f}"
                        )

                    else:

                        val_str = (
                            f"{int(confusion_mat[i, j])}"
                        )

                    text_color = (
                        "white"
                        if cm_data[i, j]
                        > thresh
                        else "black"
                    )

                    ax.text(
                        j,
                        i,
                        val_str,
                        horizontalalignment="center",
                        color=text_color,
                        fontweight="bold",
                    )

            title_type = (
                "Normalized Confusion Matrix"
                if normalize
                else "Confusion Matrix"
            )

            ax.set_title(
                f"{title_type}: {model_name}"
            )

            ax.set_xlabel(
                "Predicted Label"
            )

            ax.set_ylabel(
                "True Label"
            )

            plt.tight_layout()

            fig.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight",
            )

            logger.info(
                "Saved %s plot to %s",
                title_type,
                save_path,
            )

            return save_path

        finally:

            plt.close(fig)

    # ====================================================================== #
    # ROC Curve
    # ====================================================================== #

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str,
        class_names: Sequence[str] | None = None,
    ) -> Path | None:
        """Generate and save ROC curve.

        Correctly handles:
            No / Yes
            0 / 1
            multiclass labels
        """

        filename = (
            f"roc_curve_"
            f"{model_name.lower().replace(' ', '_')}.png"
        )

        save_path = (
            self.plots_dir / filename
        )

        classes = np.unique(y_true)

        n_classes = len(classes)

        if (
            y_prob.ndim != 2
            or y_prob.shape[1] < n_classes
        ):

            logger.warning(
                "Cannot generate ROC curve for %s: "
                "y_prob shape=%s, classes=%d",
                model_name,
                y_prob.shape,
                n_classes,
            )

            return None

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        try:

            if n_classes == 2:

                positive_label = (
                    _get_positive_label(
                        classes
                    )
                )

                positive_index = (
                    _get_positive_class_index(
                        classes,
                        positive_label,
                    )
                )

                y_encoded = (
                    _encode_labels_for_probability_metrics(
                        y_true,
                        classes,
                    )
                )

                y_binary = (
                    y_encoded
                    == positive_index
                ).astype(int)

                positive_probability = (
                    y_prob[:, positive_index]
                )

                if (
                    len(
                        np.unique(
                            y_binary
                        )
                    )
                    < 2
                ):

                    logger.warning(
                        "ROC curve skipped for %s: "
                        "test set contains only one class.",
                        model_name,
                    )

                    return None

                fpr, tpr, _ = roc_curve(
                    y_binary,
                    positive_probability,
                )

                roc_auc_val = float(
                    auc(
                        fpr,
                        tpr,
                    )
                )

                positive_name = (
                    str(positive_label)
                )

                ax.plot(
                    fpr,
                    tpr,
                    lw=2,
                    label=(
                        f"ROC curve "
                        f"(positive={positive_name}, "
                        f"AUC = {roc_auc_val:.3f})"
                    ),
                )

            else:

                y_encoded = (
                    _encode_labels_for_probability_metrics(
                        y_true,
                        classes,
                    )
                )

                y_bin = label_binarize(
                    y_encoded,
                    classes=np.arange(
                        n_classes
                    ),
                )

                for i in range(
                    n_classes
                ):

                    if (
                        len(
                            np.unique(
                                y_bin[:, i]
                            )
                        )
                        < 2
                    ):

                        continue

                    fpr, tpr, _ = roc_curve(
                        y_bin[:, i],
                        y_prob[:, i],
                    )

                    cls_auc = float(
                        auc(
                            fpr,
                            tpr,
                        )
                    )

                    if (
                        class_names
                        and i < len(class_names)
                    ):

                        label_name = (
                            str(class_names[i])
                        )

                    else:

                        label_name = (
                            f"Class {classes[i]}"
                        )

                    ax.plot(
                        fpr,
                        tpr,
                        lw=2,
                        label=(
                            f"ROC {label_name} "
                            f"(AUC = {cls_auc:.3f})"
                        ),
                    )

            ax.plot(
                [0, 1],
                [0, 1],
                lw=2,
                linestyle="--",
            )

            ax.set_xlim(
                [0.0, 1.0]
            )

            ax.set_ylim(
                [0.0, 1.05]
            )

            ax.set_xlabel(
                "False Positive Rate"
            )

            ax.set_ylabel(
                "True Positive Rate"
            )

            ax.set_title(
                f"ROC Curve: {model_name}"
            )

            ax.legend(
                loc="lower right"
            )

            ax.grid(
                True,
                linestyle="--",
                alpha=0.5,
            )

            plt.tight_layout()

            fig.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight",
            )

            logger.info(
                "Saved ROC curve plot to %s",
                save_path,
            )

            return save_path

        finally:

            plt.close(fig)

    # ====================================================================== #
    # Precision-Recall Curve
    # ====================================================================== #

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str,
        class_names: Sequence[str] | None = None,
    ) -> Path | None:
        """Generate and save Precision-Recall curve."""

        filename = (
            f"pr_curve_"
            f"{model_name.lower().replace(' ', '_')}.png"
        )

        save_path = (
            self.plots_dir / filename
        )

        classes = np.unique(y_true)

        n_classes = len(classes)

        if (
            y_prob.ndim != 2
            or y_prob.shape[1] < n_classes
        ):

            logger.warning(
                "Cannot generate PR curve for %s: "
                "y_prob shape=%s, classes=%d",
                model_name,
                y_prob.shape,
                n_classes,
            )

            return None

        fig, ax = plt.subplots(
            figsize=(7, 5)
        )

        try:

            if n_classes == 2:

                positive_label = (
                    _get_positive_label(
                        classes
                    )
                )

                positive_index = (
                    _get_positive_class_index(
                        classes,
                        positive_label,
                    )
                )

                y_encoded = (
                    _encode_labels_for_probability_metrics(
                        y_true,
                        classes,
                    )
                )

                y_binary = (
                    y_encoded
                    == positive_index
                ).astype(int)

                positive_probability = (
                    y_prob[:, positive_index]
                )

                if (
                    len(
                        np.unique(
                            y_binary
                        )
                    )
                    < 2
                ):

                    logger.warning(
                        "PR curve skipped for %s: "
                        "test set contains only one class.",
                        model_name,
                    )

                    return None

                prec, rec, _ = (
                    precision_recall_curve(
                        y_binary,
                        positive_probability,
                    )
                )

                avg_prec = float(
                    average_precision_score(
                        y_binary,
                        positive_probability,
                    )
                )

                positive_name = (
                    str(positive_label)
                )

                ax.plot(
                    rec,
                    prec,
                    lw=2,
                    label=(
                        f"PR curve "
                        f"(positive={positive_name}, "
                        f"AP = {avg_prec:.3f})"
                    ),
                )

            else:

                y_encoded = (
                    _encode_labels_for_probability_metrics(
                        y_true,
                        classes,
                    )
                )

                y_bin = label_binarize(
                    y_encoded,
                    classes=np.arange(
                        n_classes
                    ),
                )

                for i in range(
                    n_classes
                ):

                    if (
                        len(
                            np.unique(
                                y_bin[:, i]
                            )
                        )
                        < 2
                    ):

                        continue

                    prec, rec, _ = (
                        precision_recall_curve(
                            y_bin[:, i],
                            y_prob[:, i],
                        )
                    )

                    cls_ap = float(
                        average_precision_score(
                            y_bin[:, i],
                            y_prob[:, i],
                        )
                    )

                    if (
                        class_names
                        and i < len(class_names)
                    ):

                        label_name = (
                            str(class_names[i])
                        )

                    else:

                        label_name = (
                            f"Class {classes[i]}"
                        )

                    ax.plot(
                        rec,
                        prec,
                        lw=2,
                        label=(
                            f"PR {label_name} "
                            f"(AP = {cls_ap:.3f})"
                        ),
                    )

            ax.set_xlim(
                [0.0, 1.0]
            )

            ax.set_ylim(
                [0.0, 1.05]
            )

            ax.set_xlabel(
                "Recall"
            )

            ax.set_ylabel(
                "Precision"
            )

            ax.set_title(
                f"Precision-Recall Curve: {model_name}"
            )

            ax.legend(
                loc="lower left"
            )

            ax.grid(
                True,
                linestyle="--",
                alpha=0.5,
            )

            plt.tight_layout()

            fig.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight",
            )

            logger.info(
                "Saved Precision-Recall curve plot to %s",
                save_path,
            )

            return save_path

        finally:

            plt.close(fig)

    # ====================================================================== #
    # Classification Report
    # ====================================================================== #

    def save_classification_report(
        self,
        classification_text: str,
        model_name: str,
    ) -> Path:
        """Save text classification report to reports/."""

        filename = (
            f"classification_report_"
            f"{model_name.lower().replace(' ', '_')}.txt"
        )

        save_path = (
            self.reports_dir / filename
        )

        save_path.write_text(
            classification_text,
            encoding="utf-8",
        )

        logger.info(
            "Saved classification report to %s",
            save_path,
        )

        return save_path


__all__ = ["ModelEvaluator"]
