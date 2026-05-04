"""Thread-safe in-memory TTL caches for AgniAI."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class TTLCache(Generic[T]):
    def __init__(self, maxsize: int = 1024, ttl: int = 300) -> None:
        self.maxsize = maxsize
        self.ttl = ttl
        self._data: OrderedDict[str, tuple[float, T]] = OrderedDict()
        self._lock = threading.RLock()
        self._write_count = 0

    def _purge_locked(self) -> None:
        now = time.time()
        expired = [key for key, (ts, _) in self._data.items() if now - ts > self.ttl]
        for key in expired:
            self._data.pop(key, None)
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def get(self, key: str) -> Optional[T]:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            ts, value = item
            if time.time() - ts > self.ttl:
                self._data.pop(key, None)
                return None
            return value

    def set(self, key: str, value: T) -> None:
        with self._lock:
            self._data[key] = (time.time(), value)
            self._data.move_to_end(key)
            self._write_count += 1
            if self._write_count % 64 == 0:
                self._purge_locked()

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            self._purge_locked()
            return len(self._data)
