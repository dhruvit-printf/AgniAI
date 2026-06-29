"""Shared TTL cache and .NET fetch helpers for entity resolution."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import requests as _requests

from dotnet_security import resolve_dotnet_verify_ssl
from settings import get_dotnet_config

logger = logging.getLogger(__name__)

_DOTNET_BASE = get_dotnet_config().BASE_URL.rstrip("/")
_DOTNET_API_KEY = os.getenv("DOTNET_API_KEY", "")
_TIMEOUT = int(os.getenv("DOTNET_TIMEOUT", "15"))
_TTL_SECONDS = int(os.getenv("ENTITY_CACHE_TTL", "600"))

_AGNIVEER_URL = f"{_DOTNET_BASE}/api/Agniveer/GetAgniveerDetails"
_COMPANY_URL = f"{_DOTNET_BASE}/api/CompanyDetails/Get"
_PLATOON_URL = f"{_DOTNET_BASE}/api/PlatoonDetails/Get"

_VERIFY_SSL = resolve_dotnet_verify_ssl(logger)


def _headers() -> Dict[str, str]:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if _DOTNET_API_KEY:
        headers["X-Api-Key"] = _DOTNET_API_KEY
    return headers


def _extract_list(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "Data", "items", "Items", "result", "Result"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


@dataclass
class _CacheEntry:
    fetched_at: float = 0.0
    data: Optional[List[Dict[str, Any]]] = None
    refreshing: bool = False


class EntityCache:
    def __init__(self) -> None:
        self._session = _requests.Session()
        self._lock = threading.RLock()
        self._entries: Dict[str, _CacheEntry] = {
            "agniveers": _CacheEntry(data=[]),
            "companies": _CacheEntry(data=[]),
            "platoons": _CacheEntry(data=[]),
        }

    def _is_fresh(self, key: str) -> bool:
        entry = self._entries[key]
        return bool(entry.fetched_at) and (time.time() - entry.fetched_at) < _TTL_SECONDS

    def _fetch(self, url: str, trace_id: Optional[str] = None) -> List[Dict[str, Any]]:
        start = time.time()
        resp = self._session.get(
            url,
            headers=_headers(),
            timeout=_TIMEOUT,
            verify=_VERIFY_SSL,
        )
        resp.raise_for_status()
        data = _extract_list(resp.json())
        logger.info(
            {
                "trace_id": trace_id or "N/A",
                "stage": "entity_fetch",
                "url": url,
                "status_code": resp.status_code,
                "duration_ms": round((time.time() - start) * 1000, 2),
                "record_count": len(data),
            }
        )
        return data

    def _get(self, key: str, url: str, *, force_refresh: bool = False, trace_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            entry = self._entries[key]
            if not force_refresh and self._is_fresh(key):
                return list(entry.data or [])

        try:
            data = self._fetch(url, trace_id=trace_id)
        except Exception as exc:
            with self._lock:
                entry = self._entries[key]
                logger.warning(
                    "Entity cache fetch failed for %s: %s (stale_age=%.0fs)",
                    key, exc, time.time() - entry.fetched_at
                )
                return list(entry.data or [])

        with self._lock:
            entry = self._entries[key]
            entry.fetched_at = time.time()
            entry.data = list(data)
            entry.refreshing = False
            return list(entry.data)

    def get_agniveers(self, *, force_refresh: bool = False, trace_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._get("agniveers", _AGNIVEER_URL, force_refresh=force_refresh, trace_id=trace_id)

    def get_companies(self, *, force_refresh: bool = False, trace_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._get("companies", _COMPANY_URL, force_refresh=force_refresh, trace_id=trace_id)

    def get_platoons(self, *, force_refresh: bool = False, trace_id: Optional[str] = None) -> List[Dict[str, Any]]:
        return self._get("platoons", _PLATOON_URL, force_refresh=force_refresh, trace_id=trace_id)

    def preload(self, *, trace_id: Optional[str] = None) -> Dict[str, int]:
        agniveers = self.get_agniveers(force_refresh=True, trace_id=trace_id)
        companies = self.get_companies(force_refresh=True, trace_id=trace_id)
        platoons = self.get_platoons(force_refresh=True, trace_id=trace_id)
        return {
            "agniveers": len(agniveers),
            "companies": len(companies),
            "platoons": len(platoons),
        }

    def refresh_all(self, *, trace_id: Optional[str] = None) -> Dict[str, int]:
        return self.preload(trace_id=trace_id)

    def invalidate(self) -> None:
        with self._lock:
            for entry in self._entries.values():
                entry.fetched_at = 0.0
                entry.data = []

    def snapshot(self) -> Dict[str, List[Dict[str, Any]]]:
        with self._lock:
            return {key: list(entry.data or []) for key, entry in self._entries.items()}


ENTITY_CACHE = EntityCache()


def fetch_agniveers(*, force_refresh: bool = False, trace_id: Optional[str] = None) -> List[Dict[str, Any]]:
    return ENTITY_CACHE.get_agniveers(force_refresh=force_refresh, trace_id=trace_id)


def fetch_companies(*, force_refresh: bool = False, trace_id: Optional[str] = None) -> List[Dict[str, Any]]:
    return ENTITY_CACHE.get_companies(force_refresh=force_refresh, trace_id=trace_id)


def fetch_platoons(*, force_refresh: bool = False, trace_id: Optional[str] = None) -> List[Dict[str, Any]]:
    return ENTITY_CACHE.get_platoons(force_refresh=force_refresh, trace_id=trace_id)


def preload_entities(*, trace_id: Optional[str] = None) -> Dict[str, int]:
    return ENTITY_CACHE.preload(trace_id=trace_id)


def refresh_all_entities(*, trace_id: Optional[str] = None) -> Dict[str, int]:
    return ENTITY_CACHE.refresh_all(trace_id=trace_id)


def invalidate_cache() -> None:
    ENTITY_CACHE.invalidate()
