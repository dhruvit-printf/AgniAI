"""
ollama_settings.py
==================
Shared timeout policy for Ollama report generation.
"""

from __future__ import annotations

import os
from typing import Tuple


def _read_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    if value <= 0:
        raise ValueError(f"Timeout must be positive, got: {value}")
    return value


def get_ollama_timeout() -> Tuple[float, float]:
    connect = _read_float("OLLAMA_CONNECT_TIMEOUT", 5.0)
    read = _read_float("OLLAMA_READ_TIMEOUT", 30.0)
    return connect, read
