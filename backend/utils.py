"""MindCare AI - Shared utility helpers.

Groups small, reusable helpers needed across many backend components:

* ``timed``                       - decorator that logs execution time.
* ``load_json``                   - safely parse a JSON string.
* ``format_timestamp``            - ISO-8601 UTC timestamp string.
* ``ensure_dir``                  - create a directory and return its Path.
* ``validate_prediction_payload`` - validate incoming API prediction payloads.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Tuple, TypeVar, Union

import logging

_F = TypeVar("_F", bound=Callable[..., Any])
LOGGER = logging.getLogger("mindcare.utils")


def timed(func: _F) -> _F:
    """Decorator that logs the wall-clock execution time of *func* at INFO level.

    Parameters
    ----------
    func : Callable
        The function to wrap.

    Returns
    -------
    Callable
        Wrapped function that records and logs its execution duration.
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        LOGGER.info("%s executed in %.3f s", func.__qualname__, elapsed)
        return result

    return wrapper  # type: ignore[return-value]


def load_json(json_string: str) -> Mapping[str, Any]:
    """Parse *json_string* and return a mapping.

    Parameters
    ----------
    json_string : str
        Raw JSON text to parse.

    Returns
    -------
    Mapping[str, Any]
        Parsed JSON object.

    Raises
    ------
    ValueError
        If *json_string* is not valid JSON or does not represent an object.
    """
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON payload: {exc.msg}") from exc
    if not isinstance(data, dict):
        raise ValueError("JSON payload must be a JSON object (key/value mapping).")
    return data


def format_timestamp() -> str:
    """Return the current UTC timestamp in ISO-8601 format with a ``Z`` suffix.

    Returns
    -------
    str
        Example: ``"2024-01-15T12:30:00Z"``
    """
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create *path* as a directory if it does not already exist.

    Parameters
    ----------
    path : str or Path
        Target directory path.

    Returns
    -------
    Path
        Absolute resolved path to the directory.
    """
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def validate_prediction_payload(
    payload: Mapping[str, Any],
    required_fields: Iterable[str],
) -> Tuple[bool, str]:
    """Validate that *payload* contains all *required_fields* with non-empty values.

    Parameters
    ----------
    payload : Mapping[str, Any]
        The incoming request payload dict.
    required_fields : Iterable[str]
        Field names that must be present and non-empty.

    Returns
    -------
    Tuple[bool, str]
        ``(True, "")`` when valid; ``(False, <reason>)`` on the first failure.
    """
    for field_name in required_fields:
        if field_name not in payload:
            return False, f"Missing required field: '{field_name}'."
        value = payload[field_name]
        if value is None or value == "" or value == [] or value == {}:
            return False, f"Field '{field_name}' cannot be empty."
    return True, ""


__all__ = [
    "timed",
    "load_json",
    "format_timestamp",
    "ensure_dir",
    "validate_prediction_payload",
]
