"""
query_normalizer.py
===================

Shared query normalization helpers for admin pipeline compatibility.
"""

from __future__ import annotations

import re


def clean_query(query: str) -> str:
    """Normalize query text before intent classification."""
    if not query:
        return ""
    text = str(query)
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"(?is)```.*?```", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"(?im)^(?:ws:|socket:|event:|message:)\s*", "", text)
    text = re.sub(
        r"(?i)\b(?:agni\s*ai|agniai)\b.*?\bmay make mistakes\b(?:\.\s*|\s*)?",
        " ",
        text,
    )
    text = re.sub(r"(?i)\bverify important information\b\.?", " ", text)
    text = re.sub(r"(?i)\bplease verify before use\b\.?", " ", text)
    text = re.sub(r"(?i)\bverify before use\b\.?", " ", text)
    text = re.sub(r"(?i)agniai can make mistakes\.?\s*please verify before use\.?", " ", text)
    text = re.sub(r"(?m)^#{1,6}\s*", "", text)
    text = re.sub(r"[â€œâ€]", '"', text)
    text = re.sub(r"[â€˜â€™]", "'", text)
    text = re.sub(r"[^\S\r\n]+", " ", text)
    text = re.sub(r"\s*([!?.,;:])\s*", r"\1 ", text)
    text = re.sub(r"([!?.,;:]){2,}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def admin_normalize_query(query: str) -> str:
    """Backward-compatible alias for legacy pipeline normalization."""
    return clean_query(query)
