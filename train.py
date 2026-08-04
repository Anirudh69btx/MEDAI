"""train.py
================================
MindCare AI - Main Training Script.

Orchestrates the complete training pipeline:
1. Load and validate dataset
2. Prepare and split data (Train / Validation / Test)
3. Fit feature engineering pipeline and save it
4. Train all Scikit-Learn candidate models with 5-fold CV
5. Train PyTorch deep learning model
6. Evaluate all models using the Evaluator
7. Compare all models and select the best performer
8. Save all artefacts via ModelSaver
9. Write training metadata and configuration snapshot

Usage
-----
    python train.py
    python train.py --target-column Status
    python train.py --no-pytorch
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Ensure the project root is on sys.path when executed directly
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import config
from backend.data_loader import DataLoadingError, load_dataset
from backend.evaluator import ModelEvaluator
from backend.feature_engineering import FeatureEngineer
from backend.logger import get_logger
from backend.model_comparator import ModelComparator
from backend.model_saver import ModelSaver
from backend.pytorch_trainer import PyTorchTrainer
from backend.trainer import ModelTrainer
from backend.utils import format_timestamp

logger: logging.Logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MindCare AI Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target-column",
        type=str,
        default="mental_health_status",
        help="Name of the target column in the dataset CSV.",
    )
    parser.add_argument(
        "--no-pytorch",
        action="store_true",
        default=False,
        help="Skip PyTorch model training.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Override dataset path (default: config.DATASET_PATH).",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the full MindCare AI training pipeline."""
    args = parse_args()
    target_column: str = args.target_column
    run_pytorch: bool = not args.no_pytorch
    dataset_path = Path(args.dataset_path) if args.dataset_path else config.DATASET_PATH

    logger.info("=" * 60)
    logger.info("MindCare AI Training Pipeline Started")
    logger.info("Timestamp      : %s", format_timestamp())
    logger.info("Target Column  : %s", target_column)
    logger.info("Dataset Path   : %s", dataset_path)
    logger.info("Device         : %s", config.DEVICE)
    logger.info("PyTorch        : %s", "enabled" if run_pytorch else "disabled")
    logger.info("=" * 60)

    # ------------------------------------------------------------------ #
    # 1. Load Dataset
    # ------------------------------------------------------------------ #
    try:
        df = load_dataset(path=dataset_path, allow_missing_values=True)
        logger.info("Dataset loaded: %d rows x %d columns.", df.shape[0], df.shape[1])
    except DataLoadingError as exc:
        logger.error("Dataset loading failed: %s", exc)
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 2. Scikit-Learn Training Pipeline
    # ------------------------------------------------------------------ #
    trainer = ModelTrainer(target_column=target_column)

    try:
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(df)
    except (ValueError, RuntimeError) as exc:
        logger.error("Data preparation failed: %s", exc)
        sys.exit(1)

    try:
        best_sk_name, best_sk_model, sk_results = trainer.train_and_evaluate_all()
    except RuntimeError as exc:
        logger.error("Sklearn training failed: %s", exc)
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 3. Evaluate Scikit-Learn models and produce plots
    # ------------------------------------------------------------------ #
    evaluator = ModelEvaluator()
    all_evaluation_results: Dict[str, Dict[str, Any]] = {}

    for name, result in sk_results.items():
        y_pred = result.model_instance.predict(X_test.values)
        y_prob = None
        if hasattr(result.model_instance, "predict_proba"):
            try:
                y_prob = result.model_instance.predict_proba(X_test.values)
            except Exception:
                pass

        eval_metrics = evaluator.evaluate_predictions(
            y_true=y_test.values,
            y_pred=y_pred,
            y_prob=y_prob,
            model_name=name,
        )
        all_evaluation_results[name] = eval_metrics

    # ------------------------------------------------------------------ #
    # 4. PyTorch Training Pipeline (optional)
    # ------------------------------------------------------------------ #
    pytorch_trainer: PyTorchTrainer | None = None

    if run_pytorch:
        logger.info("Starting PyTorch training pipeline...")
        try:
            feature_engineer: FeatureEngineer = trainer.feature_engineer  # type: ignore[assignment]
            n_classes = int(y_train.nunique())

            pytorch_trainer = PyTorchTrainer(
                input_dim=X_train.shape[1],
                num_classes=max(n_classes, 2),
            )
            pytorch_trainer.train_model(
                X_train=X_train.values.astype(np.float32),
                y_train=feature_engineer.encode_labels(y_train),
                X_val=X_val.values.astype(np.float32),
                y_val=feature_engineer.encode_labels(y_val),
                epochs=config.N_EPOCHS,
                batch_size=config.BATCH_SIZE,
                learning_rate=config.LEARNING_RATE,
                patience=config.EARLY_STOPPING_PATIENCE,
            )

            pt_eval = pytorch_trainer.evaluate(
                X_test=X_test.values.astype(np.float32),
                y_test=feature_engineer.encode_labels(y_test),
            )

            # Generate full evaluation plots for PyTorch model
            pt_y_pred = np.argmax(
                pytorch_trainer.model.predict_proba(
                    X_test.values.astype(np.float32), device=config.DEVICE
                ),
                axis=1,
            )
            pt_y_prob = pytorch_trainer.model.predict_proba(
                X_test.values.astype(np.float32), device=config.DEVICE
            )

            pt_eval_full = evaluator.evaluate_predictions(
                y_true=feature_engineer.encode_labels(y_test),
                y_pred=pt_y_pred,
                y_prob=pt_y_prob,
                model_name="PyTorch Deep Learning",
            )
            all_evaluation_results["PyTorch Deep Learning"] = pt_eval_full

        except Exception as exc:
            logger.exception("PyTorch training failed: %s", exc)
            pytorch_trainer = None

    # ------------------------------------------------------------------ #
    # 5. Model Comparison & Selection
    # ------------------------------------------------------------------ #
    comparator = ModelComparator()
    best_overall, comparison_df, selection_reason = comparator.compare_and_select(
        evaluation_results=all_evaluation_results
    )
    logger.info("Overall Best Model: %s", best_overall.get("model_name"))
    logger.info("Reason: %s", selection_reason)

    # ------------------------------------------------------------------ #
    # 6. Save All Artefacts
    # ------------------------------------------------------------------ #
    saver = ModelSaver()
    feature_engineer = trainer.feature_engineer  # type: ignore[assignment]

    # Determine which sklearn model is best
    best_sk_model_instance = sk_results[best_sk_name].model_instance

    saved_paths = saver.save_all(
        sklearn_model=best_sk_model_instance,
        pytorch_model=pytorch_trainer.model if pytorch_trainer else None,
        label_encoder=feature_engineer.get_label_encoder(),
        feature_names=feature_engineer.get_feature_names(),
        metadata={
            "best_model_name": best_overall.get("model_name"),
            "best_sklearn_model": best_sk_name,
            "selection_reason": selection_reason,
            "training_timestamp": format_timestamp(),
            "config": config.to_dict(),
            "sklearn_results": {
                name: res.to_dict() for name, res in sk_results.items()
            },
        },
    )

    logger.info("Saved artefact paths:")
    for key, path in saved_paths.items():
        logger.info("  %s -> %s", key, path)

    logger.info("=" * 60)
    logger.info("MindCare AI Training Pipeline Completed Successfully.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
