"""app.py
================================
MindCare AI - Flask REST API Server.

Provides the following endpoints:

    GET  /health          - Health check
    POST /predict         - Run mental health prediction + recommendations
    POST /api/predict     - Alias for /predict
    GET  /api/info        - API metadata
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

from flask import Flask, Response, jsonify, request, send_from_directory

from backend.config import config
from backend.logger import get_logger
from backend.predictor import MindCarePredictor
from backend.recommendation_engine import RecommendationEngine
from backend.utils import format_timestamp

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
        _predictor = MindCarePredictor(use_pytorch=True)
        _predictor.load()
        logger.info("MindCarePredictor loaded successfully at startup.")
    except Exception as exc:
        _startup_error = str(exc)
        logger.warning(
            "Predictor could not be loaded at startup (run train.py first): %s", exc
        )


# ---------------------------------------------------------------------------
#  CORS & Helper
# ---------------------------------------------------------------------------

@app.after_request
def add_cors_headers(response: Response) -> Response:
    """Add standard CORS headers so browser frontends can connect seamlessly."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


@app.route("/predict", methods=["OPTIONS"])
@app.route("/api/predict", methods=["OPTIONS"])
def options_predict() -> Tuple[Response, int]:
    """Handle CORS preflight requests."""
    return jsonify({"status": "ok"}), 200


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

@app.route("/", methods=["GET"])
def index() -> Tuple[Response, int]:
    """Root landing endpoint providing API status and overview."""
    return jsonify({
        "name": "MindCare AI API",
        "version": "1.0.0",
        "status": "online",
        "predictor_loaded": _predictor is not None,
        "description": "Mental health classification and personalised recommendation API.",
        "endpoints": {
            "GET /": "API Overview & status",
            "GET /health": "Health check",
            "POST /predict": "Predict mental health status and get recommendations",
            "POST /api/predict": "Alias for /predict",
            "GET /api/info": "API metadata",
            "GET /api/classes": "List supported class labels",
        },
        "timestamp": format_timestamp(),
    }), 200


@app.route("/health", methods=["GET"])
def health() -> Tuple[Response, int]:
    """Health check endpoint.

    Returns
    -------
    200 OK
        ``{"status": "ok", "predictor_loaded": true|false, "device": "...", "timestamp": "..."}``
    """
    return jsonify({
        "status": "ok",
        "predictor_loaded": _predictor is not None,
        "device": config.DEVICE,
        "timestamp": format_timestamp(),
    }), 200


@app.route("/api/info", methods=["GET"])
@app.route("/info", methods=["GET"])
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
            "GET /": "API Overview & status",
            "GET /health": "Health check",
            "POST /predict": "Predict mental health status and get recommendations",
            "POST /api/predict": "Alias for /predict",
            "GET /api/info": "API metadata",
            "GET /api/classes": "List supported class labels",
        },
        "timestamp": format_timestamp(),
    }), 200


@app.route("/api/classes", methods=["GET"])
@app.route("/classes", methods=["GET"])
def get_classes() -> Tuple[Response, int]:
    """Return the list of class labels supported by the recommendation engine.

    Returns
    -------
    200 OK
        ``{"classes": [...], "timestamp": "..."}``
    """
    engine = RecommendationEngine()
    return jsonify({
        "classes": engine.get_supported_classes(),
        "timestamp": format_timestamp(),
    }), 200


@app.route("/api/metrics", methods=["GET"])
@app.route("/metrics", methods=["GET"])
def get_metrics() -> Tuple[Response, int]:
    """Return verified training metrics and benchmark metadata for models."""
    return jsonify({
        "production_model": "PyTorch Deep Learning",
        "pytorch": {
            "model_name": "PyTorch Deep Learning",
            "accuracy": 0.7778,
            "recall": 0.8397,
            "precision": 0.7501,
            "f1_score": 0.7924,
            "roc_auc": 0.8693,
        },
        "sklearn_baseline": {
            "model_name": "Decision Tree",
            "accuracy": 0.7584,
            "recall": 0.8234,
            "precision": 0.7316,
            "f1_score": 0.7748,
            "roc_auc": 0.8440,
        },
        "timestamp": format_timestamp(),
    }), 200


@app.route("/frontend/<path:filename>", methods=["GET"])
def serve_frontend(filename: str) -> Response:
    """Serve static frontend assets."""
    frontend_dir = PROJECT_ROOT / "frontend"
    return send_from_directory(str(frontend_dir), filename)


@app.route("/app", methods=["GET"])
def serve_app() -> Response:
    """Convenience route to serve the MindCare AI frontend application."""
    frontend_dir = PROJECT_ROOT / "frontend"
    return send_from_directory(str(frontend_dir), "index.html")


@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict() -> Tuple[Response, int]:
    """Run mental health classification and return recommendations.

    Request Body (JSON)
    -------------------
    A JSON object containing raw feature key-value pairs matching the
    columns expected by the trained model.

    Returns
    -------
    200 OK
        ``{"success": true, "prediction": { ... }, "timestamp": "..."}``
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
        pred_dict = result.to_dict()
        return jsonify({
            "success": True,
            "prediction": pred_dict,
            "result": pred_dict,
            "timestamp": format_timestamp(),
        }), 200
    except (ValueError, TypeError) as exc:
        logger.warning("Invalid input for prediction: %s", exc)
        return _error(f"Invalid input: {exc}", 400)
    except RuntimeError as exc:
        logger.exception("Prediction runtime error: %s", exc)
        return _error("Prediction failed due to an internal error.", 500)
    except Exception as exc:
        logger.exception("Unexpected error during prediction: %s", exc)
        return _error("An unexpected internal error occurred.", 500)


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

