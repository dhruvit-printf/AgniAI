"""Background and manual refresh helpers for the entity cache."""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

from .entity_cache import ENTITY_CACHE, preload_entities, refresh_all_entities

logger = logging.getLogger(__name__)


class EntityRefreshService:
    def __init__(self, interval_seconds: int = 600) -> None:
        self.interval_seconds = max(60, int(interval_seconds))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_lock = threading.Lock()

    def preload(self, trace_id: Optional[str] = None) -> Dict[str, int]:
        return preload_entities(trace_id=trace_id)

    def refresh_now(self, trace_id: Optional[str] = None) -> Dict[str, int]:
        return refresh_all_entities(trace_id=trace_id)

    def start(self, trace_id: Optional[str] = None) -> None:
        with self._start_lock:
            if self._thread and self._thread.is_alive():
                return

            self._stop_event.clear()

            def _runner() -> None:
                import uuid as _uuid
                logger.info("Entity refresh service started.")
                while not self._stop_event.wait(self.interval_seconds):
                    cycle_trace_id = _uuid.uuid4().hex
                    try:
                        refresh_all_entities(trace_id=cycle_trace_id)
                    except Exception as exc:  # pragma: no cover - defensive
                        logger.warning("Entity refresh cycle failed: %s", exc)

            self._thread = threading.Thread(target=_runner, name="entity-refresh-service", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()


_SERVICE = EntityRefreshService()


def start_entity_refresh_service(interval_seconds: int = 600, trace_id: Optional[str] = None) -> EntityRefreshService:
    _SERVICE.interval_seconds = max(60, int(interval_seconds))
    _SERVICE.start(trace_id=trace_id)
    return _SERVICE


def refresh_entities_now(trace_id: Optional[str] = None) -> Dict[str, int]:
    return _SERVICE.refresh_now(trace_id=trace_id)
