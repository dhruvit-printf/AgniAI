"""
conversation_detector.py
========================
Lightweight conversational query detection used to bypass the analytical
pipeline for greetings, small talk, and other non-admin chatter.
"""

from __future__ import annotations

import re
from typing import Any, Dict

_CONVERSATIONAL_PHRASES = (
    "hello",
    "hi",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "thank you",
    "thanks",
    "who are you",
    "what are you",
    "how are you",
    "tell me a joke",
    "joke",
    "weather",
)

_DISCLAIMER_PHRASES = (
    "may make mistakes",
    "verify important information",
    "please verify before use",
    "verify before use",
    "as an ai",
)

_ADMIN_CONTEXT_WORDS = {
    "current",
    "currently",
    "today",
    "today's",
    "present",
    "attendance",
    "medical",
    "hospital",
    "hospitalized",
    "hospitalised",
    "obese",
    "overweight",
    "underweight",
    "blood",
    "compare",
    "comparison",
    "distribution",
    "trend",
    "summary",
    "performance",
    "score",
    "marks",
    "platoon",
    "company",
    "batch",
    "section",
    "subsection",
    "firing",
    "bpet",
    "ppt",
    "leave",
    "absconded",
    "absent",
    "agniveer",
    "agniveers",
    "unit",
    "class",
    "sport",
    "sports",
    "cricket",
    "football",
    "equipment",
    "verification",
    "strength",
    "grading",
    "bmi",
    "fever",
    "malaria",
    "injury",
    "illness",
    "sick",
    "disease",
    "active",
    "items",
    "item",
    "issue",
    "issued",
    "procured",
}


_MULTI_WORD_CONVERSATIONAL_PHRASES = tuple(
    phrase for phrase in _CONVERSATIONAL_PHRASES if " " in phrase
)
_SINGLE_WORD_CONVERSATIONAL_PATTERNS = tuple(
    re.compile(rf"\b{re.escape(phrase)}\b")
    for phrase in _CONVERSATIONAL_PHRASES
    if " " not in phrase
)
_TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _contains_conversational_phrase(text: str) -> bool:
    for phrase in _MULTI_WORD_CONVERSATIONAL_PHRASES:
        if phrase in text:
            return True
    for pattern in _SINGLE_WORD_CONVERSATIONAL_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _contains_admin_signal(text: str) -> bool:
    tokens = _TOKEN_PATTERN.findall(text)
    return any(token in _ADMIN_CONTEXT_WORDS for token in tokens)


def _contains_disclaimer_phrase(text: str) -> bool:
    return any(phrase in text for phrase in _DISCLAIMER_PHRASES)


def is_conversational_query(text: str) -> bool:
    cleaned = normalize_text(text).rstrip("!?.,;")
    if not cleaned:
        return True
    if _contains_conversational_phrase(cleaned) and not _contains_admin_signal(cleaned):
        return True
    if _contains_disclaimer_phrase(cleaned):
        return True
    return False


def build_conversational_response(
    message: str,
    *,
    session_id: str = "admin-default",
    query_type: str = "conversation",
) -> Dict[str, Any]:
    intro = (message or "").strip() or "I can help with data, reports, and analysis."
    payload: Dict[str, Any] = {
        "status": True,
        "message": intro,
        "formattedData": [
            {
                "id": "conversation_message",
                "type": "MESSAGE",
                "title": "Conversation",
                "data": {"text": intro},
                "analysis": {"summary": "", "insights": [], "statistics": {}},
                "prediction": {},
                "conclusion": {"summary": "", "bullets": []},
            }
        ],
        # Root-level narrative fields (required by universal response contract)
        "analysis": "",
        "prediction": "",
        "conclusion": "",
        "suggestedQuestions": [],
        # No .NET query was executed — explicitly None per contract
        "dotnetPayload": None,
        "metadata": {
            "sessionId": session_id or "",
            "confidence": 1.0,
            "queryType": query_type,
            "operationCount": 0,
            "timings": {
                "plannerMs": 0,
                "intentMs": 0,
                "dotnetMs": 0,
                "combinerMs": 0,
                "reportMs": 0,
                "totalMs": 0,
            },
            "executionTimeMs": 0,
        },
        "overallConfidence": 1.0,
        "partialFailure": False,
        "failedSections": [],
    }
    return payload
