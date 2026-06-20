"""
grounding_utils.py
===================
Shared text grounding helpers used by report/prediction engines.
"""

from __future__ import annotations

import re
from typing import Set


def extract_numbers_from_text(text: str) -> Set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", text or ""))


def ground_and_sanitize(text: str, grounded_text: str) -> str:
    if not text:
        return ""
    grounded_numbers = extract_numbers_from_text(grounded_text)
    sentences = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    kept = []
    for sentence in sentences:
        sentence_numbers = extract_numbers_from_text(sentence)
        if sentence_numbers and not sentence_numbers.issubset(grounded_numbers):
            continue
        kept.append(sentence)
    return " ".join(kept).strip()
