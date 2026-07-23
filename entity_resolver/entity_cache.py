"""Shared fetch helpers for entity resolution, backed by a TTL in-memory cache."""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
    TimeoutError as FutureTimeout,
)
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import requests as _requests

from settings import get_dotnet_config

logger = logging.getLogger(__name__)

_DOTNET_BASE = get_dotnet_config().BASE_URL.rstrip("/")
_DOTNET_API_KEY = os.getenv("DOTNET_API_KEY", "")
# Reduced from 15s to 3s — if the .NET tunnel is down we fail fast and
# serve stale cache rather than blocking the pipeline for 30 seconds.
_TIMEOUT = int(os.getenv("DOTNET_TIMEOUT", "3"))
_AGNIVEER_URL = f"{_DOTNET_BASE}/api/Agniveer/GetAgniveerDetails"
_COMPANY_URL = f"{_DOTNET_BASE}/api/CompanyDetails/Get"
_PLATOON_URL = f"{_DOTNET_BASE}/api/PlatoonDetails/Get"

# SSL verification default (dotnet_security module removed with .NET decommission)
_VERIFY_SSL = os.getenv("DOTNET_VERIFY_SSL", "true").lower() not in ("0", "false", "no")

_DEFAULT_TTL_SECONDS = int(os.getenv("ENTITY_CACHE_TTL_SECONDS", "300"))


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
    data: List[Dict[str, Any]] = field(default_factory=list)
    fetched_at: float = 0.0

    def is_fresh(self, ttl_seconds: float) -> bool:
        return (time.time() - self.fetched_at) < ttl_seconds


class EntityCache:
    """TTL in-memory cache for entity directories fetched from .NET.

    Each directory (agniveers/companies/platoons) is cached independently for
    ``ttl_seconds``. Access to the cache table is guarded by a lock; the
    underlying HTTP fetch itself is not, so a slow .NET call cannot stall
    other cache reads/writes. On fetch failure, a still-held stale entry is
    served instead of an empty list so a transient .NET blip doesn't make
    entity resolution behave as if the directory were empty.
    """

    def __init__(self, ttl_seconds: int = _DEFAULT_TTL_SECONDS) -> None:
        self._session = _requests.Session()
        self.ttl_seconds = ttl_seconds
        self._entries: Dict[str, _CacheEntry] = {}
        self._lock = threading.Lock()

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

    def _get(
        self,
        key: str,
        url: str,
        *,
        force_refresh: bool = False,
        trace_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not force_refresh:
            with self._lock:
                entry = self._entries.get(key)
                if entry is not None and entry.is_fresh(self.ttl_seconds):
                    return list(entry.data)

        try:
            data = list(self._fetch(url, trace_id=trace_id))
        except Exception as exc:
            logger.warning("Entity fetch failed for %s: %s", key, exc)
            with self._lock:
                stale = self._entries.get(key)
            if stale is not None:
                logger.warning(
                    "Serving stale cached %s (%d records) after fetch failure",
                    key,
                    len(stale.data),
                )
                return list(stale.data)
            return []

        with self._lock:
            self._entries[key] = _CacheEntry(data=data, fetched_at=time.time())
        return list(data)

    def get_agniveers(
        self, *, force_refresh: bool = False, trace_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        return self._get(
            "agniveers", _AGNIVEER_URL, force_refresh=force_refresh, trace_id=trace_id
        )

    def get_companies(
        self, *, force_refresh: bool = False, trace_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        return self._get(
            "companies", _COMPANY_URL, force_refresh=force_refresh, trace_id=trace_id
        )

    def get_platoons(
        self, *, force_refresh: bool = False, trace_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        return self._get(
            "platoons", _PLATOON_URL, force_refresh=force_refresh, trace_id=trace_id
        )

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
            self._entries.clear()

    def snapshot(self) -> Dict[str, List[Dict[str, Any]]]:
        with self._lock:
            return {key: list(entry.data) for key, entry in self._entries.items()}


ENTITY_CACHE = EntityCache()


def fetch_agniveers(
    *, force_refresh: bool = False, trace_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    return ENTITY_CACHE.get_agniveers(force_refresh=force_refresh, trace_id=trace_id)


def fetch_companies(
    *, force_refresh: bool = False, trace_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    return ENTITY_CACHE.get_companies(force_refresh=force_refresh, trace_id=trace_id)


def fetch_platoons(
    *, force_refresh: bool = False, trace_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    return ENTITY_CACHE.get_platoons(force_refresh=force_refresh, trace_id=trace_id)


def preload_entities(*, trace_id: Optional[str] = None) -> Dict[str, int]:
    return ENTITY_CACHE.preload(trace_id=trace_id)


def refresh_all_entities(*, trace_id: Optional[str] = None) -> Dict[str, int]:
    return ENTITY_CACHE.refresh_all(trace_id=trace_id)


def invalidate_cache() -> None:
    ENTITY_CACHE.invalidate()
