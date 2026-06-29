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
    """Thread-safe Cache Manager implementing memory and optional Redis cache."""

    TTL_BY_CATEGORY = {
        "Attendance": 120,
        "Leave": 60,
        "Performance": 600,
        "Medical": 300,
        "Training": 600,
    }

    def __init__(self, default_ttl: int = 300) -> None:
        self._lock = threading.Lock()
        self._memory_cache: dict[str, tuple[Any, float]] = {}
        self.default_ttl = default_ttl
        self._redis_client = None

        if REDIS_AVAILABLE:
            redis_host = os.getenv("REDIS_HOST", "127.0.0.1")
            redis_port = int(os.getenv("REDIS_PORT", "6379"))
            redis_password = os.getenv("REDIS_PASSWORD", None)
            try:
                self._redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    password=redis_password,
                    socket_timeout=0.5,
                    socket_connect_timeout=0.5,
                    decode_responses=True
                )
                self._redis_client.ping()
                logger.info("Connected to Redis cache at %s:%d", redis_host, redis_port)
            except Exception as e:
                logger.info("Redis cache not available, using memory cache only. Detail: %s", e)
                self._redis_client = None
        else:
            logger.info("redis package not installed, memory cache only enabled.")

    def get_query_hash(self, query: str, scope: Optional[dict[str, Any]] = None) -> str:
        """Compute MD5 hash of normalized query and any scope-affecting filters."""
        normalized = query.lower().strip().rstrip("!?.,;")
        # Normalize whitespace
        normalized = " ".join(normalized.split())
        scope_blob = ""
        if scope:
            scope_blob = json.dumps(scope, sort_keys=True, separators=(",", ":"), default=str)
        cache_key = f"{normalized}|{scope_blob}" if scope_blob else normalized
        return hashlib.md5(cache_key.encode("utf-8")).hexdigest()

    def get(self, query_hash: str) -> Optional[Any]:
        """Retrieve cached value. Returns None if not found or expired."""
        # Try Redis GET
        if self._redis_client:
            try:
                cached = self._redis_client.get(query_hash)
                if cached:
                    logger.info("Cache HIT (Redis) for hash %s", query_hash)
                    value = json.loads(cached)
                    with self._lock:
                        self._memory_cache[query_hash] = (value, time.time() + self.default_ttl)
                    return value
            except Exception as e:
                logger.warning("Redis GET failed: %s", e)

        # Fallback memory GET
        with self._lock:
            cached_item = self._memory_cache.get(query_hash)
            if cached_item:
                val, expires_at = cached_item
                if expires_at > time.time():
                    logger.info("Cache HIT (Memory) for hash %s", query_hash)
                    return val
                else:
                    del self._memory_cache[query_hash]
                    logger.info("Memory cache item expired for hash %s", query_hash)
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
        """Cache a value with a specific TTL (seconds)."""
        expiry = ttl or self.get_ttl_for_category(category)
        
        # Redis SET
        if self._redis_client:
            try:
                self._redis_client.setex(query_hash, expiry, json.dumps(value))
                logger.info("Cache SET (Redis) for hash %s, TTL %d", query_hash, expiry)
            except Exception as e:
                logger.warning("Redis SET failed: %s", e)

        # Memory SET
        with self._lock:
            expires_at = time.time() + expiry
            self._memory_cache[query_hash] = (value, expires_at)
            logger.info("Cache SET (Memory) for hash %s, TTL %d", query_hash, expiry)

    def is_cacheable_category(self, category: Optional[str]) -> bool:
        """Identify if category is cacheable (e.g. Top performers, Leave, Attendance, Performance)."""
        if not category:
            return False
        cat_lower = category.lower()
        return any(
            target in cat_lower
            for target in ("performance", "leave", "attendance", "verification", "skills", "medical")
        )

# Global cache manager instance
cache_manager = CacheManager()
