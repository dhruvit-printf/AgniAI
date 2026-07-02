"""
cache_manager.py
================
Query response caching manager for AgniAI. Handles query normalization, MD5 query hashing,
TTL expiration (60-300 seconds), and optional Redis caching with thread-safe memory fallback.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class CacheManager:
    """Compatibility shim with caching disabled."""

    def __init__(self, default_ttl: int = 300) -> None:
        self.default_ttl = default_ttl

    def get_query_hash(self, query: str, scope: Optional[dict[str, Any]] = None) -> str:
        """Compute MD5 hash of normalized query and any scope-affecting filters."""
        normalized = query.lower().strip().rstrip("!?.,;")
        # Normalize whitespace
        normalized = " ".join(normalized.split())
        scope_blob = ""
        if scope:
            scope_blob = json.dumps(
                scope, sort_keys=True, separators=(",", ":"), default=str
            )
        cache_key = f"{normalized}|{scope_blob}" if scope_blob else normalized
        return hashlib.md5(cache_key.encode("utf-8")).hexdigest()

    def get(self, query_hash: str) -> Optional[Any]:
        return None

    def get_ttl_for_category(self, category: Optional[str]) -> int:
        if not category:
            return self.default_ttl
        return self.TTL_BY_CATEGORY.get(category, self.default_ttl)

    def set(
        self,
        query_hash: str,
        value: Any,
        ttl: Optional[int] = None,
        category: Optional[str] = None,
    ) -> None:
        """Caching is disabled; keep the API as a no-op."""
        return None

    def is_cacheable_category(self, category: Optional[str]) -> bool:
        """Caching is disabled for all categories."""
        return False


# Global cache manager instance
cache_manager = CacheManager()
