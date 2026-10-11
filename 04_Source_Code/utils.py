"""
utils.py
========
Small shared helper functions: logging setup, directory helpers, safe
optional-import helper, and simple timing decorator. Kept dependency-free
(standard library only) so it never breaks even in minimal environments.
"""

from __future__ import annotations
import logging
import time
import functools
from pathlib import Path
from typing import Callable, Any


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that prints readable, leveled messages."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                                 datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def ensure_dir(path: Path) -> Path:
    """Create a directory (and parents) if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def try_import(module_name: str):
    """
    Attempt to import an optional dependency by name.
    Returns the module object, or None if it is not installed.
    Used throughout the project to provide graceful fallback behaviour
    (e.g. XGBoost -> GradientBoostingClassifier, SHAP -> permutation importance).
    """
    try:
        return __import__(module_name)
    except ImportError:
        return None


def timed(func: Callable) -> Callable:
    """Decorator that logs how long a pipeline step took to run."""
    logger = get_logger(func.__module__)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.time()
        logger.info(f"START  {func.__name__}")
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"DONE   {func.__name__} ({elapsed:.2f}s)")
        return result
    return wrapper
