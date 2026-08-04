"""backend/model_comparator.py
================================
Model Comparison and Best Model Selection Module for MindCare AI.

Compares metrics across all trained Scikit-Learn models and the PyTorch Deep Learning
model, generates structured comparison tables, applies intelligent multi-tier tie-breaking rules,
and logs selection rationale.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from .config import config
from .logger import get_logger

logger: logging.Logger = get_logger(__name__)


class ModelComparator:
    """Ranks candidate models and selects the optimal model based on business rules."""

    def __init__(self, reports_dir: Path | str | None = None) -> None:
        """Initialize ModelComparator.

        Parameters
        ----------
        reports_dir : Path or str, optional
            Destination directory for summary reports. Defaults to config.REPORTS_DIR.
        """
        self.reports_dir = Path(reports_dir or config.REPORTS_DIR)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def compare_and_select(
        self, evaluation_results: Dict[str, Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], pd.DataFrame, str]:
        """Compare candidate models and select the best overall performer.

        Multi-tier Selection Strategy:
        1. Compare by Accuracy.
        2. If two models have similar accuracy (within 0.5% / 0.005 tolerance), tie-break using:
           a) Higher Recall (crucial for mental health risk detection)
           b) Higher F1 Score
           c) Lower Training Time

        Parameters
        ----------
        evaluation_results : Dict[str, Dict[str, Any]]
            Map of model name -> metrics dictionary containing 'accuracy', 'recall',
            'f1_score', 'training_time_seconds', etc.

        Returns
        -------
        Tuple of (best_model_dict, comparison_dataframe, selection_reason)
        """
        if not evaluation_results:
            raise ValueError("No evaluation results provided for comparison.")

        rows: List[Dict[str, Any]] = []
        for name, metrics in evaluation_results.items():
            rows.append(
                {
                    "Model": name,
                    "Accuracy": float(metrics.get("accuracy", 0.0)),
                    "Balanced Accuracy": float(metrics.get("balanced_accuracy", metrics.get("accuracy", 0.0))),
                    "Precision": float(metrics.get("precision", 0.0)),
                    "Recall": float(metrics.get("recall", 0.0)),
                    "F1 Score": float(metrics.get("f1_score", 0.0)),
                    "ROC AUC": float(metrics.get("roc_auc", 0.0)) if metrics.get("roc_auc") is not None else None,
                    "Training Time (s)": float(metrics.get("training_time_seconds", 0.0)),
                    "Inference Time (s)": float(metrics.get("inference_time_seconds", 0.0)),
                }
            )

        df_comp = pd.DataFrame(rows)

        # Sort candidate models by multi-tier criteria
        candidates = list(evaluation_results.values())

        def sort_key(res: Dict[str, Any]) -> Tuple[float, float, float, float]:
            acc = float(res.get("accuracy", 0.0))
            rec = float(res.get("recall", 0.0))
            f1 = float(res.get("f1_score", 0.0))
            t_time = float(res.get("training_time_seconds", 0.0))
            # Negative t_time so lower training time ranks higher
            return (acc, rec, f1, -t_time)

        candidates.sort(key=sort_key, reverse=True)

        best_model = candidates[0]
        runner_up = candidates[1] if len(candidates) > 1 else None

        reason = ""
        if runner_up:
            best_acc = float(best_model.get("accuracy", 0.0))
            runner_acc = float(runner_up.get("accuracy", 0.0))
            acc_diff = abs(best_acc - runner_acc)

            if acc_diff <= 0.005:
                reason = (
                    f"Selected '{best_model['model_name']}' over '{runner_up['model_name']}' "
                    f"because accuracies were comparable ({best_acc:.4f} vs {runner_acc:.4f}), "
                    f"and '{best_model['model_name']}' demonstrated superior Recall "
                    f"({best_model.get('recall', 0.0):.4f} vs {runner_up.get('recall', 0.0):.4f}), "
                    f"F1 score ({best_model.get('f1_score', 0.0):.4f} vs {runner_up.get('f1_score', 0.0):.4f}), "
                    f"or faster training time ({best_model.get('training_time_seconds', 0.0):.3f}s)."
                )
            else:
                reason = (
                    f"Selected '{best_model['model_name']}' as the best performing model "
                    f"with the highest Accuracy of {best_acc:.4f} "
                    f"(Recall: {best_model.get('recall', 0.0):.4f}, F1: {best_model.get('f1_score', 0.0):.4f})."
                )
        else:
            reason = f"Selected '{best_model['model_name']}' as the sole evaluated candidate model."

        logger.info("Best Model Selected: %s", best_model["model_name"])
        logger.info("Selection Rationale: %s", reason)

        # Save model_comparison.csv
        csv_path = self.reports_dir / "model_comparison.csv"
        df_comp.to_csv(csv_path, index=False)
        logger.info("Saved model comparison table to %s", csv_path)

        # Save training summary text
        summary_path = self.reports_dir / "training_summary.txt"
        summary_text = (
            "===========================================================\n"
            "MINDCARE AI MODEL TRAINING & COMPARISON SUMMARY\n"
            "===========================================================\n\n"
            f"Best Model Selected: {best_model['model_name']}\n"
            f"Selection Rationale: {reason}\n\n"
            "Full Model Comparison Table:\n"
            f"{df_comp.to_string(index=False)}\n\n"
            "===========================================================\n"
        )
        summary_path.write_text(summary_text, encoding="utf-8")
        logger.info("Saved training summary to %s", summary_path)

        return best_model, df_comp, reason


__all__ = ["ModelComparator"]
