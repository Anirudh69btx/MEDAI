"""app.py
================================
MindCare AI - Flask REST API Server.

Provides the following endpoints:

    GET  /health          - Health check
    GET  /api/info        - API metadata
    POST /api/predict     - Run mental health prediction + recommendations
    GET  /api/classes     - List supported class labels

The predictor is loaded once at startup and reused across all requests.
All errors are returned as JSON with an appropriate HTTP status code.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

# Ensure the project root is on sys.path when executed directly
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, Response, jsonify, request

from backend.config import config
from backend.logger import get_logger
from backend.predictor import MindCarePredictor
from backend.recommendation_engine import RecommendationEngine
from backend.utils import format_timestamp, validate_prediction_payload

# ---------------------------------------------------------------------------
#  App and logger setup
# ---------------------------------------------------------------------------
logger: logging.Logger = get_logger(__name__)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# ---------------------------------------------------------------------------
#  Predictor – loaded once at startup
# ---------------------------------------------------------------------------
_predictor: MindCarePredictor | None = None
_startup_error: str | None = None


def _load_predictor() -> None:
    """Attempt to load the trained predictor at application startup."""
    global _predictor, _startup_error
    try:
        _predictor = MindCarePredictor(use_pytorch=False)
        _predictor.load()
        logger.info("MindCarePredictor loaded successfully at startup.")
    except RuntimeError as exc:
        _startup_error = str(exc)
        logger.warning(
            "Predictor could not be loaded at startup (run train.py first): %s", exc
        )


# ---------------------------------------------------------------------------
#  Helper
# ---------------------------------------------------------------------------

def _error(message: str, status: int) -> Tuple[Response, int]:
    """Return a standardised JSON error response."""
    return jsonify({"success": False, "error": message, "timestamp": format_timestamp()}), status


def _get_predictor() -> MindCarePredictor:
    """Return the loaded predictor or raise a 503 if unavailable."""
    if _predictor is None:
        raise RuntimeError(
            f"Predictor not loaded. {_startup_error or 'Run train.py first.'}"
        )
    return _predictor


# ---------------------------------------------------------------------------
#  Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health() -> Tuple[Response, int]:
    """Health check endpoint.

    Returns
    -------
    200 OK
        ``{"status": "ok", "predictor_loaded": true|false}``
    """
    return jsonify({
        "status": "ok",
        "predictor_loaded": _predictor is not None,
        "device": config.DEVICE,
        "timestamp": format_timestamp(),
    }), 200


@app.route("/api/info", methods=["GET"])
def api_info() -> Tuple[Response, int]:
    """Return API metadata.

    Returns
    -------
    200 OK
        API description, version, and endpoint list.
    """
    return jsonify({
        "name": "MindCare AI API",
        "version": "1.0.0",
        "description": "Mental health classification and personalised recommendation API.",
        "endpoints": {
            "GET /health": "Health check",
            "GET /api/info": "API metadata",
            "POST /api/predict": "Predict mental health status and get recommendations",
            "GET /api/classes": "List supported class labels",
        },
        "timestamp": format_timestamp(),
    }), 200


@app.route("/api/classes", methods=["GET"])
def get_classes() -> Tuple[Response, int]:
    """Return the list of class labels supported by the recommendation engine.

    Returns
    -------
    200 OK
        ``{"classes": [...]}``
    """
    engine = RecommendationEngine()
    return jsonify({
        "classes": engine.get_supported_classes(),
        "timestamp": format_timestamp(),
    }), 200


@app.route("/api/predict", methods=["POST"])
def predict() -> Tuple[Response, int]:
    """Run mental health classification and return recommendations.

    Request Body (JSON)
    -------------------
    A JSON object containing feature key-value pairs matching the
    columns expected by the trained model.

    Returns
    -------
    200 OK
        ``{"success": true, "result": { ... }}``
    400 Bad Request
        When the request body is missing or malformed.
    503 Service Unavailable
        When the predictor is not loaded.
    500 Internal Server Error
        On unexpected inference errors.
    """
    # Validate predictor availability
    try:
        predictor = _get_predictor()
    except RuntimeError as exc:
        return _error(str(exc), 503)

    # Parse JSON body
    if not request.is_json:
        return _error("Request Content-Type must be application/json.", 400)

    payload: Dict[str, Any] | None = request.get_json(silent=True)
    if payload is None:
        return _error("Request body is missing or not valid JSON.", 400)

    if not isinstance(payload, dict):
        return _error("Request body must be a JSON object.", 400)

    if not payload:
        return _error("Request body must contain at least one feature.", 400)

    # Run inference
    try:
        result = predictor.predict(payload)
        return jsonify({
            "success": True,
            "timestamp": format_timestamp(),
            "result": result.to_dict(),
        }), 200
    except RuntimeError as exc:
        logger.exception("Prediction error: %s", exc)
        return _error(f"Prediction failed: {exc}", 500)
    except Exception as exc:
        logger.exception("Unexpected error during prediction: %s", exc)
        return _error("An unexpected error occurred.", 500)


# ---------------------------------------------------------------------------
#  Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(error: Any) -> Tuple[Response, int]:
    return _error("Endpoint not found.", 404)


@app.errorhandler(405)
def method_not_allowed(error: Any) -> Tuple[Response, int]:
    return _error("HTTP method not allowed for this endpoint.", 405)


@app.errorhandler(500)
def internal_server_error(error: Any) -> Tuple[Response, int]:
    return _error("Internal server error.", 500)


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting MindCare AI Flask API server...")
    _load_predictor()
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False,
    )
