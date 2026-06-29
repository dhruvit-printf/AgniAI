"""
dotnet_adapter.py
=================
Canonical .NET response normalization helpers.

This module is a compatibility shim around normalized_models.py.
"""

from __future__ import annotations

from typing import Any, Dict, List

from normalized_models import extract_records, normalize_dotnet_response

__all__ = ["normalize_dotnet_response", "extract_records"]
