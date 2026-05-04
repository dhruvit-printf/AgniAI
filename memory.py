"""Conversation memory utilities."""

from __future__ import annotations

from collections import OrderedDict, deque
from threading import RLock
from typing import Deque, Dict, List, Optional

from config import MEMORY_MAX_MESSAGES

MAX_SESSIONS = 500


class ConversationMemory:
    """Thread-safe, session-aware sliding-window conversation memory."""

    def __init__(self, max_messages: int = MEMORY_MAX_MESSAGES) -> None:
        self.max_messages = max_messages
        self._sessions: "OrderedDict[str, Deque[Dict[str, str]]]" = OrderedDict()
        self._lock = RLock()

    def add(self, role: str, content: str, session_id: Optional[str] = None) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError(f"Invalid role: {role!r}. Must be 'user' or 'assistant'.")
        with self._lock:
            key = session_id or "default"
            bucket = self._sessions.get(key)
            if bucket is None:
                bucket = deque(maxlen=self.max_messages)
                self._sessions[key] = bucket
            else:
                self._sessions.move_to_end(key)
            bucket.append({"role": role, "content": content})
            while len(self._sessions) > MAX_SESSIONS:
                self._sessions.popitem(last=False)

    def clear(self, session_id: Optional[str] = None) -> None:
        with self._lock:
            if session_id is None:
                self._sessions.clear()
            else:
                self._sessions.pop(session_id, None)

    def history(self, session_id: Optional[str] = None) -> List[Dict[str, str]]:
        with self._lock:
            return list(self._sessions.get(session_id or "default", ()))

    def __len__(self) -> int:
        with self._lock:
            return sum(len(bucket) for bucket in self._sessions.values())
