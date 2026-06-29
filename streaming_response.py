"""WebSocket response chunk helpers."""

from __future__ import annotations

from typing import Any, Dict, Generator


def stream_response_chunks(
    payload: Dict[str, Any],
) -> Generator[Dict[str, Any], None, None]:
    for key in (
        "status",
        "message",
        "formattedData",
        "suggestedQuestions",
        "dotnetPayload",
        "metadata",
    ):
        if key in payload:
            yield {key: payload[key]}
