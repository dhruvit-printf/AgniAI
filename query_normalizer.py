"""
query_normalizer.py
===================

Shared query normalization helpers for admin intent classification.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Dict, Iterable

from intent_engine.intent_schema import FUZZY_VOCAB

_INVISIBLE_CHARS = ("\u200b", "\ufeff", "\u200e", "\u200f")
_BANNER_PATTERNS = (
    r"(?i)\b(?:agni\s*ai|agniai)\b.*?\bmay make mistakes\b(?:\.\s*|\s*)?",
    r"(?i)\bverify important information\b\.?",
    r"(?i)\bplease verify before use\b\.?",
    r"(?i)\bverify before use\b\.?",
    r"(?i)agniai can make mistakes\.?\s*please verify before use\.?",
)
_MESSAGE_PREFIXES = r"(?im)^(?:ws:|socket:|event:|message:)\s*"


def _collapse_spaces(text: str) -> str:
    text = re.sub(r"[^\S\r\n]+", " ", text)
    text = re.sub(r"\s*([!?.,;:])\s*", r"\1 ", text)
    text = re.sub(r"([!?.,;:]){2,}", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _apply_fuzzy_vocab(text: str) -> str:
    if not text:
        return text

    corrected = text
    for misspelling, canonical in sorted(FUZZY_VOCAB.items(), key=lambda item: len(item[0]), reverse=True):
        if " " in misspelling:
            pattern = re.compile(rf"(?<!\w){re.escape(misspelling)}(?!\w)", re.IGNORECASE)
            corrected = pattern.sub(canonical, corrected)
            continue
        pattern = re.compile(rf"(?<!\w){re.escape(misspelling)}(?!\w)", re.IGNORECASE)
        corrected = pattern.sub(canonical, corrected)
    return corrected


def _strip_banner_text(text: str) -> str:
    stripped = text
    for pattern in _BANNER_PATTERNS:
        stripped = re.sub(pattern, " ", stripped)
    return stripped


def clean_query(query: str) -> str:
    """Normalize query text before intent classification."""
    if not query:
        return ""

    text = str(query)
    for char in _INVISIBLE_CHARS:
        text = text.replace(char, "")
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?is)```.*?```", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    text = re.sub(_MESSAGE_PREFIXES, "", text)
    text = _strip_banner_text(text)
    text = re.sub(r"(?m)^#{1,6}\s*", "", text)
    text = text.replace("â€™", "'").replace("â€œ", '"').replace("â€", '"')
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = _apply_fuzzy_vocab(text)
    text = _collapse_spaces(text)
    return text


def admin_normalize_query(query: str) -> str:
    """Backward-compatible alias for legacy pipeline normalization."""
    return clean_query(query)
