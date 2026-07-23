"""
admin_context.py
================
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)

_MAX_SESSIONS = int(os.getenv("ADMIN_SESSION_CACHE_SIZE", "1000"))
# Matches context_engine's own candidate-interaction TTL so the two stores
# agree on how long "the previous turn" stays eligible for carry-forward.
_HISTORY_TTL_SECONDS = int(os.getenv("ADMIN_SESSION_TTL_SECONDS", "300"))


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
                "who among",
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
        ast: Optional[Any] = None,
    ) -> None:
        ids = _extract_ids_from_result(result_data)
        with self._lock:
            self._history[session_id] = {
                "query": query_text,
                "intent": intent_dict,
                "ids": ids,
                "ast": ast,
                "timestamp": time.time(),
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

    def get_recent(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return the stored interaction for session_id, or None if missing/expired."""
        with self._lock:
            entry = self._history.get(session_id)
            if not entry:
                return None
            age = time.time() - entry.get("timestamp", 0)
            if age > _HISTORY_TTL_SECONDS:
                return None
            return entry

    def get_previous_ids(self, session_id: str) -> Optional[Set[int]]:
        with self._lock:
            if session_id in self._history:
                return self._history[session_id]["ids"]
            return None

    def get_last_ast(self, session_id: str) -> Optional[Any]:
        with self._lock:
            if session_id in self._history:
                return self._history[session_id].get("ast")
            return None

    def clear(self, session_id: str) -> None:
        with self._lock:
            if session_id in self._history:
                del self._history[session_id]
                logger.debug("Cleared session %s history", session_id)


class ASTPatchEngine:
    """
    Merges new follow-up intents into the previous execution's AST without rebuilding the entire query.
    """

    @staticmethod
    def merge_intent(last_ast: Any, new_intent: Dict[str, Any]) -> Any:
        import copy

        from ast_models import ConditionGroupNode, WhereNode

        patched = copy.deepcopy(last_ast)
        filters = new_intent.get("filters", {})

        # In a full implementation, we'd traverse the ConditionGroupNode tree.
        # For this prototype, we'll append new conditions using AND.
        new_conditions = []
        for k, v in filters.items():
            new_conditions.append(WhereNode(column=k, operator="=", value=v))

        if new_conditions:
            if not patched.where:
                patched.where = new_conditions
            else:
                existing = patched.where
                patched.where = [
                    ConditionGroupNode(
                        operator="AND", conditions=existing + new_conditions
                    )
                ]

        # If the user asks to sort differently
        if new_intent.get("sort_by"):
            from ast_models import OrderByNode

            patched.order_by = [
                OrderByNode(column=new_intent["sort_by"], descending=True)
            ]

        return patched
