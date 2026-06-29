"""
admin_context.py
================
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)

_MAX_SESSIONS = int(os.getenv("ADMIN_SESSION_CACHE_SIZE", "1000"))


def _extract_ids_from_result(data: Any) -> Set[int]:
    ids: Set[int] = set()
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                for key in ("agniveerId", "AgniveerId", "AgniVeerId", "id", "Id"):
                    val = item.get(key)
                    if val is not None:
                        try:
                            ids.add(int(val))
                        except (ValueError, TypeError):
                            pass
                        break
            else:
                try:
                    ids.add(int(item))
                except (ValueError, TypeError):
                    pass
    elif isinstance(data, dict):
        for key in (
            "data",
            "Data",
            "result",
            "Result",
            "records",
            "Records",
            "personnel",
            "persons",
            "teams",
            "Teams",
            "members",
            "Members",
        ):
            val = data.get(key)
            if val is not None:
                ids.update(_extract_ids_from_result(val))

        teams = data.get("teams") or data.get("Teams")
        if isinstance(teams, list):
            for team in teams:
                members = team.get("members") or team.get("Members")
                if isinstance(members, list):
                    ids.update(_extract_ids_from_result(members))

        for key in ("agniveerId", "AgniveerId", "AgniVeerId", "id", "Id"):
            val = data.get(key)
            if val is not None:
                try:
                    ids.add(int(val))
                except (ValueError, TypeError):
                    pass
                break
    return ids


class AdminSessionContext:
    """Stores the single most recent interaction per session_id."""

    def __init__(self) -> None:
        self._history: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def is_followup_query(self, session_id: str, query_text: str) -> bool:
        with self._lock:
            if session_id not in self._history:
                return False

            q = (query_text or "").lower()
            indicators = {
                "them",
                "those",
                "these",
                "their",
                "they",
                "him",
                "her",
                "which of",
                "who among",
                "any of",
                "some of",
                "each of",
            }
            for ind in indicators:
                if re.search(r"\b" + re.escape(ind) + r"\b", q):
                    return True
            return False

    def update(
        self,
        session_id: str,
        query_text: str,
        intent_dict: Dict[str, Any],
        result_data: Any,
    ) -> None:
        ids = _extract_ids_from_result(result_data)
        with self._lock:
            self._history[session_id] = {
                "query": query_text,
                "intent": intent_dict,
                "ids": ids,
            }
            self._history.move_to_end(session_id)
            while len(self._history) > _MAX_SESSIONS:
                evicted = next(iter(self._history))
                del self._history[evicted]
                logger.debug(
                    "AdminSessionContext: evicted session %s (cap=%d)",
                    evicted,
                    _MAX_SESSIONS,
                )
        logger.debug("Updated session %s history: %d ids stored", session_id, len(ids))

    def get_previous_ids(self, session_id: str) -> Optional[Set[int]]:
        with self._lock:
            if session_id in self._history:
                return self._history[session_id]["ids"]
            return None

    def clear(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._history:
                del self._history[session_id]
                logger.debug("Cleared session %s history", session_id)
