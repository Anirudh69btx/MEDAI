"""MindCare AI - Centralised logging configuration.

Provides a ``get_logger`` factory that returns a pre-configured
``logging.Logger`` writing to both the console (INFO) and a rotating file
handler (DEBUG).  All modules import ``get_logger`` instead of constructing
their own loggers, keeping configuration consistent across the code-base.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import config


def _ensure_log_dir() -> None:
    """Create the logs directory if it does not already exist."""
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """Return a consistently configured logger for *name*.

    The logger is created only once per unique *name* – subsequent calls return
    the cached instance, mirroring the behaviour of ``logging.getLogger``.

    Parameters
    ----------
    name : str
        Logger name, typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
        Configured logger with console (INFO) and file (DEBUG) handlers.
    """
    _ensure_log_dir()
    logger = logging.getLogger(name)

    # Guard against duplicate handler registration on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler – INFO and above
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Rotating file handler – DEBUG and above, 5 MiB per file, 3 backups
    log_file: Path = config.LOGS_DIR / "mindcare.log"
    file_handler = RotatingFileHandler(
        filename=log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


__all__ = ["get_logger"]
