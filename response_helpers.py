"""Small shared helpers for inspecting response payloads."""

from __future__ import annotations

from typing import Any


def extract_primary_widget_title(formatted_data: Any) -> str:
    """Return the first widget title from a list payload or a single widget dict."""
    if isinstance(formatted_data, list):
        for widget in formatted_data:
            if isinstance(widget, dict):
                title = widget.get("title")
                if title:
                    return str(title)
        return ""

    if isinstance(formatted_data, dict):
        title = formatted_data.get("title")
        return str(title) if title else ""

    return ""
