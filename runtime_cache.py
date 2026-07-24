"""TTL In-memory Cache implementation for AgniAI."""

from __future__ import annotations

import time
from typing import Dict, Generic, Optional, Tuple, TypeVar

T = TypeVar("T")


class TTLCache(Generic[T]):
    """Thread-safe, simple TTL in-memory cache."""

    def __init__(self, maxsize: int = 1024, ttl: float = 300.0) -> None:
        self.maxsize = maxsize
        self.ttl = ttl
        self._store: Dict[str, Tuple[float, T]] = {}

    def get(self, key: str) -> Optional[T]:
        now = time.time()
        entry = self._store.get(key)
        if entry is None:
            return None
        created_at, val = entry
        if now - created_at > self.ttl:
            self._store.pop(key, None)
            return None
        return val

    def set(self, key: str, value: T) -> None:
        now = time.time()
        if len(self._store) >= self.maxsize and key not in self._store:
            # Evict oldest entry
            oldest_key = min(self._store.keys(), key=lambda k: self._store[k][0])
            self._store.pop(oldest_key, None)
        self._store[key] = (now, value)

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        now = time.time()
        # Clean up expired items during len check
        expired = [k for k, (ts, _) in self._store.items() if now - ts > self.ttl]
        for k in expired:
            self._store.pop(k, None)
        return len(self._store)
