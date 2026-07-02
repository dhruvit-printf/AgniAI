"""Compatibility shim for cache APIs with caching disabled."""

from __future__ import annotations

from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class TTLCache(Generic[T]):
    def __init__(self, maxsize: int = 1024, ttl: int = 300) -> None:
        self.maxsize = maxsize
        self.ttl = ttl

    def get(self, key: str) -> Optional[T]:
        return None

    def set(self, key: str, value: T) -> None:
        return None

    def clear(self) -> None:
        return None

    def __len__(self) -> int:
        return 0
