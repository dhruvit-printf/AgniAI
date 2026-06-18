"""
websocket_manager.py
====================
Manages active WebSocket connections for AgniAI.

Responsibilities:
  - Register / unregister clients by session ID.
  - Send messages safely (swallows disconnect errors).
  - Handle disconnects gracefully.

This module is purely infrastructure — no business logic lives here.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class WebSocketManager:
    """Thread-safe registry of active Socket.IO session IDs."""

    def __init__(self) -> None:
        self._clients: Dict[str, str] = {}   # sid → session_id
        self._lock = threading.RLock()
        self._socketio = None                 # set via init()

    def init(self, socketio) -> None:
        """Attach the Flask-SocketIO instance (called once at startup)."""
        self._socketio = socketio

    # ── Registration ──────────────────────────────────────────────────────

    def register(self, sid: str, session_id: Optional[str] = None) -> None:
        """Record a newly connected client."""
        with self._lock:
            self._clients[sid] = session_id or sid
        logger.debug("WebSocket client registered: sid=%s session=%s", sid, session_id)

    def unregister(self, sid: str) -> None:
        """Remove a disconnected client."""
        with self._lock:
            self._clients.pop(sid, None)
        logger.debug("WebSocket client unregistered: sid=%s", sid)

    # ── Messaging ─────────────────────────────────────────────────────────

    def send(self, sid: str, event: str, data: Any) -> bool:
        """
        Emit *event* with *data* to a single client identified by *sid*.

        Returns True on success, False if the client has disconnected or
        any other error occurs (the caller need not handle exceptions).
        """
        if self._socketio is None:
            logger.warning("WebSocketManager.send called before init().")
            return False
        try:
            self._socketio.emit(event, data, to=sid)
            return True
        except Exception as exc:
            logger.debug("WebSocket send failed (sid=%s): %s", sid, exc)
            return False

    def send_json(self, sid: str, payload: Dict[str, Any]) -> bool:
        """
        Emit a single 'message' event whose data is *payload*.

        The frontend reads `event.data` and routes by `payload["type"]`.
        """
        return self.send(sid, "message", payload)

    # ── Introspection ─────────────────────────────────────────────────────

    @property
    def active_count(self) -> int:
        with self._lock:
            return len(self._clients)

    def is_connected(self, sid: str) -> bool:
        with self._lock:
            return sid in self._clients


# Singleton — imported by websocket_routes.py and app.py
ws_manager = WebSocketManager()