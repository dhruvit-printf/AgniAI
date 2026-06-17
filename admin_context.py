"""
admin_context.py
================
Provides session tracking and follow-up query context detection for the AgniAI Admin Chatbot.
"""

from __future__ import annotations
import logging
import re
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)


def _extract_ids_from_result(data: Any) -> Set[int]:
    """
    Recursively extract all agniveerIds found in any list/dict .NET response wrapper structure.
    """
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
            "data", "Data", "result", "Result", "records", "Records",
            "personnel", "persons", "teams", "Teams", "members", "Members"
        ):
            val = data.get(key)
            if val is not None:
                ids.update(_extract_ids_from_result(val))
        
        # Flatten teams structure if found
        teams = data.get("teams") or data.get("Teams")
        if isinstance(teams, list):
            for team in teams:
                members = team.get("members") or team.get("Members")
                if isinstance(members, list):
                    ids.update(_extract_ids_from_result(members))
                    
        # Check direct dict keys for agniveerId
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
    def __init__(self) -> None:
        # Maps session_id -> dict with previous query details
        self._history: Dict[str, Dict[str, Any]] = {}

    def is_followup_query(self, session_id: str, query_text: str) -> bool:
        """
        Determine if the current query is a follow-up to the previous query in this session.
        A query is a follow-up if we have previous results in this session and the current query
        contains one or more follow-up pronouns/indicators (e.g. "them", "these", "those", "their", "they", "which of").
        """
        if session_id not in self._history:
            return False

        q = (query_text or "").lower()
        indicators = {
            "them", "those", "these", "their", "they", "him", "her",
            "which of", "who among", "any of", "some of", "each of"
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
        """
        Store context history for the current session.
        """
        ids = _extract_ids_from_result(result_data)
        self._history[session_id] = {
            "query": query_text,
            "intent": intent_dict,
            "ids": ids,
        }
        logger.debug(
            "Updated session %s history: %d ids stored", session_id, len(ids)
        )

    def get_previous_ids(self, session_id: str) -> Optional[Set[int]]:
        """
        Get the set of Agniveer IDs from the previous result in this session.
        """
        if session_id in self._history:
            return self._history[session_id]["ids"]
        return None

    def clear(self, session_id: str) -> None:
        """
        Clear history for the given session.
        """
        if session_id in self._history:
            del self._history[session_id]
            logger.debug("Cleared session %s history", session_id)
