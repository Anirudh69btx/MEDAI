"""backend/evaluation.py
================================
Public re-export alias for the MindCare AI evaluation module.

All evaluation functionality lives in ``backend/evaluator.py``.
This module exists as a clean public alias so callers can import from
either ``backend.evaluation`` or ``backend.evaluator`` without error.
"""

from .evaluator import ModelEvaluator  # noqa: F401

__all__ = ["ModelEvaluator"]
