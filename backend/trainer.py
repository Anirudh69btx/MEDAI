"""backend/trainer.py

Scikit-Learn Model Training Pipeline for MindCare AI.

This module orchestrates training, cross-validation, fitting, evaluation,
and performance tracking for five resource-conscious Scikit-Learn
classification algorithms:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Linear Support Vector Machine Classifier
5. Gaussian Naive Bayes Classifier

K-Nearest Neighbors is intentionally excluded because prediction on the
full 175K-row training split with 657 engineered features is prohibitively
memory-intensive on the target hardware.

Features:
- Stratified Train/Validation/Test splitting
- Stratified 3-Fold Cross-Validation
- Preprocessing pipeline saving via FeatureEngineer.save()
- Probability & ROC-AUC calculation when supported
- Comprehensive per-model exception handling
- Model persistence compatibility through ModelSaver
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from .config import config
from .feature_engineering import FeatureEngineer
from .logger import get_logger

logger: logging.Logger = get_logger(__name__)


@dataclass
class ModelEvaluationResult:
    """Dataclass holding evaluation metrics and performance data for a single model."""

    model_name: str
    model_instance: Any
    cv_scores: List[float]
    cv_mean_score: float
    cv_std_score: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float | None
    val_accuracy: float
    val_f1: float
    confusion_matrix: np.ndarray
    classification_report_dict: Dict[str, Any]
    classification_report_text: str
    training_time_seconds: float
    inference_time_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a dictionary representation compatible with JSON serialization."""
        return {
            "model_name": self.model_name,
            "cv_scores": [float(s) for s in self.cv_scores],
            "cv_mean_score": float(self.cv_mean_score),
            "cv_std_score": float(self.cv_std_score),
            "accuracy": float(self.accuracy),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "f1_score": float(self.f1_score),
            "roc_auc": float(self.roc_auc) if self.roc_auc is not None else None,
            "val_accuracy": float(self.val_accuracy),
            "val_f1": float(self.val_f1),
            "confusion_matrix": self.confusion_matrix.tolist(),
            "training_time_seconds": float(self.training_time_seconds),
            "inference_time_seconds": float(self.inference_time_seconds),
        }


class ModelTrainer:
    """Orchestrates Scikit-Learn multi-model training, CV, and evaluation."""

    def __init__(self, target_column: str) -> None:
        """Initialize the trainer with target column specification."""
        self.target_column = target_column
        self.feature_engineer: FeatureEngineer | None = None
        self.results: Dict[str, ModelEvaluationResult] = {}
        self.X_train: pd.DataFrame | None = None
        self.X_val: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_val: pd.Series | None = None
        self.y_test: pd.Series | None = None

    def prepare_data(
        self, df: pd.DataFrame
    ) -> Tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.Series,
        pd.Series,
        pd.Series,
    ]:
        """Split dataset, fit preprocessing on train only, and transform all splits."""
        if self.target_column not in df.columns:
            norm_target = str(self.target_column).strip().lower().replace(" ", "_")
            norm_map = {
                str(c).strip().lower().replace(" ", "_"): c for c in df.columns
            }
            if norm_target in norm_map:
                self.target_column = norm_map[norm_target]
            else:
                raise ValueError(
                    f"Target column '{self.target_column}' not found in dataset."
                )

        logger.info("Splitting dataset into Train, Validation, and Test sets...")

        test_size = config.TEST_SIZE
        val_size = config.VALID_SIZE
        seed = config.RANDOM_SEED

        y = df[self.target_column]
        X_df = df.copy()

        stratify = y if y.nunique() <= 20 else None

        df_train_val, df_test = train_test_split(
            X_df,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )

        relative_val_size = val_size / (1.0 - test_size)
        y_train_val = df_train_val[self.target_column]
        stratify_val = y_train_val if y_train_val.nunique() <= 20 else None

        df_train, df_val = train_test_split(
            df_train_val,
            test_size=relative_val_size,
            random_state=seed,
            stratify=stratify_val,
        )

        # Fit Feature Engineer on Train set only.
        self.feature_engineer = FeatureEngineer(target_column=self.target_column)
        self.feature_engineer.fit(df_train)

        self.feature_engineer.save(config.MODELS_DIR)
        logger.info(
            "Saved feature engineering pipeline to %s",
            config.MODELS_DIR,
        )

        # Transform all splits.
        self.X_train = self.feature_engineer.transform(df_train)
        self.X_val = self.feature_engineer.transform(df_val)
        self.X_test = self.feature_engineer.transform(df_test)

        self.y_train = df_train[self.target_column].reset_index(drop=True)
        self.y_val = df_val[self.target_column].reset_index(drop=True)
        self.y_test = df_test[self.target_column].reset_index(drop=True)

        logger.info(
            "Data split completed: Train=%d, Validation=%d, Test=%d",
            len(self.X_train),
            len(self.X_val),
            len(self.X_test),
        )

        return (
            self.X_train,
            self.X_val,
            self.X_test,
            self.y_train,
            self.y_val,
            self.y_test,
        )

    def _get_candidate_models(self) -> Dict[str, Any]:
        """Instantiate resource-conscious Scikit-Learn candidate models."""
        seed = config.RANDOM_SEED

        return {
            "Logistic Regression": LogisticRegression(
                random_state=seed,
                max_iter=1000,
                n_jobs=1,
            ),
            "Decision Tree": DecisionTreeClassifier(
                random_state=seed,
                max_depth=10,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=50,
                random_state=seed,
                n_jobs=1,
            ),
            "Support Vector Machine": LinearSVC(
                dual=False,
                random_state=seed,
                max_iter=2000,
            ),
            "Gaussian Naive Bayes": GaussianNB(),
        }

    def train_and_evaluate_all(
        self,
    ) -> Tuple[str, Any, Dict[str, ModelEvaluationResult]]:
        """Train candidates with 3-fold CV, evaluate, and select the best model."""
        if (
            self.X_train is None
            or self.X_val is None
            or self.X_test is None
            or self.y_train is None
            or self.y_val is None
            or self.y_test is None
        ):
            raise RuntimeError(
                "Data must be prepared via prepare_data() before training."
            )

        candidate_models = self._get_candidate_models()

        is_binary = self.y_train.nunique() == 2
        average_mode = "binary" if is_binary else "weighted"

        # Dataset target uses string labels ("Yes"/"No"), so explicitly
        # identify "Yes" as the positive class for binary metrics.
        pos_label = "Yes" if is_binary else None
        binary_kwargs = {"pos_label": pos_label} if is_binary else {}

        # Three folds substantially reduce repeated training cost while
        # retaining meaningful stratified cross-validation for this dataset.
        skf = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=config.RANDOM_SEED,
        )

        best_val_score = -1.0
        best_model_name = ""
        best_model_instance = None

        for name, model in candidate_models.items():
            logger.info("Starting training & evaluation for: %s", name)

            try:
                # 1. Stratified 3-Fold Cross-Validation on training data.
                cv_scores = cross_val_score(
                    model,
                    self.X_train,
                    self.y_train,
                    cv=skf,
                    scoring="accuracy",
                    n_jobs=1,
                )
                cv_scores_list = [float(s) for s in cv_scores]
                cv_mean = float(np.mean(cv_scores_list))
                cv_std = float(np.std(cv_scores_list))

                # 2. Fit model on full training set and record training time.
                start_train = time.perf_counter()
                model.fit(self.X_train, self.y_train)
                train_time = time.perf_counter() - start_train

                # 3. Validation evaluation used for model selection.
                val_preds = model.predict(self.X_val)
                val_acc = float(accuracy_score(self.y_val, val_preds))
                val_f1 = float(
                    f1_score(
                        self.y_val,
                        val_preds,
                        average=average_mode,
                        zero_division=0,
                        **binary_kwargs,
                    )
                )

                # 4. Final test evaluation and inference timing.
                start_infer = time.perf_counter()
                y_pred = model.predict(self.X_test)
                infer_time = time.perf_counter() - start_infer

                acc = float(accuracy_score(self.y_test, y_pred))
                prec = float(
                    precision_score(
                        self.y_test,
                        y_pred,
                        average=average_mode,
                        zero_division=0,
                        **binary_kwargs,
                    )
                )
                rec = float(
                    recall_score(
                        self.y_test,
                        y_pred,
                        average=average_mode,
                        zero_division=0,
                        **binary_kwargs,
                    )
                )
                f1 = float(
                    f1_score(
                        self.y_test,
                        y_pred,
                        average=average_mode,
                        zero_division=0,
                        **binary_kwargs,
                    )
                )

                cm = confusion_matrix(self.y_test, y_pred)

                clf_rep_dict = classification_report(
                    self.y_test,
                    y_pred,
                    output_dict=True,
                    zero_division=0,
                )
                clf_rep_text = classification_report(
                    self.y_test,
                    y_pred,
                    zero_division=0,
                )

                # 5. ROC-AUC when the estimator exposes probabilities.
                # LinearSVC does not expose predict_proba(), so its ROC-AUC
                # is intentionally left as None.
                roc_auc = None
                if hasattr(model, "predict_proba"):
                    try:
                        probs = model.predict_proba(self.X_test)
                        if is_binary:
                            roc_auc = float(
                                roc_auc_score(self.y_test, probs[:, 1])
                            )
                        else:
                            roc_auc = float(
                                roc_auc_score(
                                    self.y_test,
                                    probs,
                                    multi_class="ovr",
                                    average="weighted",
                                )
                            )
                    except Exception as roc_exc:
                        logger.warning(
                            "ROC-AUC calculation skipped for %s: %s",
                            name,
                            roc_exc,
                        )

                result = ModelEvaluationResult(
                    model_name=name,
                    model_instance=model,
                    cv_scores=cv_scores_list,
                    cv_mean_score=cv_mean,
                    cv_std_score=cv_std,
                    accuracy=acc,
                    precision=prec,
                    recall=rec,
                    f1_score=f1,
                    roc_auc=roc_auc,
                    val_accuracy=val_acc,
                    val_f1=val_f1,
                    confusion_matrix=cm,
                    classification_report_dict=clf_rep_dict,
                    classification_report_text=clf_rep_text,
                    training_time_seconds=train_time,
                    inference_time_seconds=infer_time,
                )

                self.results[name] = result

                logger.info(
                    "%s -> 3-Fold CV: %.4f (+/- %.4f) | "
                    "Val Acc: %.4f | Test Acc: %.4f | F1: %.4f | Fit Time: %.3fs",
                    name,
                    cv_mean,
                    cv_std,
                    val_acc,
                    acc,
                    f1,
                    train_time,
                )

                # Model selection uses validation accuracy first and
                # validation F1 as the tie-breaker.
                if val_acc > best_val_score or (
                    abs(val_acc - best_val_score) < 1e-4
                    and (
                        best_model_instance is None
                        or val_f1 > self.results[best_model_name].val_f1
                    )
                ):
                    best_val_score = val_acc
                    best_model_name = name
                    best_model_instance = model

            except Exception as exc:
                logger.exception(
                    "Failed training model '%s': %s",
                    name,
                    exc,
                )

        if not self.results or best_model_instance is None:
            raise RuntimeError("All candidate model training attempts failed.")

        logger.info(
            "Best Candidate Model selected based on Validation set: '%s' "
            "(Val Acc: %.4f)",
            best_model_name,
            best_val_score,
        )

        return best_model_name, best_model_instance, self.results


__all__ = ["ModelEvaluationResult", "ModelTrainer"]