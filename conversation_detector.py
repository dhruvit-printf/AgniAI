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

_ADMIN_SIGNAL_WORDS = {
    "performance",
    "leave",
    "attendance",
    "medical",
    "equipment",
    "verification",
    "distribution",
    "skills",
    "roster",
    "strength",
    "overall",
    "score",
    "marks",
    "compare",
    "comparison",
    "trend",
    "distribution",
    "ranking",
    "rank",
    "top",
    "bottom",
    "highest",
    "lowest",
    "best",
    "worst",
    "average",
    "percentage",
    "count",
    "group",
    "grouping",
    "filter",
    "filters",
    "intersection",
    "show",
    "list",
    "display",
    "view",
    "find",
    "analyze",
    "analysis",
    "report",
    "records",
    "record",
    "candidate",
    "candidates",
    "results",
    "metric",
    "section",
    "platoon",
    "batch",
    "company",
    "sport",
    "class",
    "agniveer",
    "agniveers",
}

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
    "firing",
    "bpet",
    "ppt",
    "leave",
    "absconded",
    "absent",
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _contains_conversational_phrase(text: str) -> bool:
    for phrase in _CONVERSATIONAL_PHRASES:
        if " " in phrase:
            if phrase in text:
                return True
        else:
            if re.search(rf"\b{re.escape(phrase)}\b", text):
                return True
    return False


def _contains_admin_signal(text: str) -> bool:
    tokens = re.findall(r"[a-z0-9']+", text)
    return any(token in _ADMIN_SIGNAL_WORDS for token in tokens)


def _contains_disclaimer_phrase(text: str) -> bool:
    return any(phrase in text for phrase in _DISCLAIMER_PHRASES)


def is_conversational_query(text: str) -> bool:
    cleaned = normalize_text(text).rstrip("!?.,;")
    if not cleaned:
        return True
    if _contains_conversational_phrase(cleaned):
        return True
    if _contains_disclaimer_phrase(cleaned):
        return True
    if _contains_admin_signal(cleaned):
        return False
    tokens = re.findall(r"[a-z0-9']+", cleaned)
    has_admin_context = any(token in _ADMIN_CONTEXT_WORDS for token in tokens)
    if len(tokens) <= 3 and not has_admin_context:
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
        "sessionId": session_id or "admin-default",
        "message": intro,
        "widget": "conversation",
        "formattedData": {
            "type": "MESSAGE",
            "title": "Conversation",
            "data": {"text": intro},
            "analysis": {"summary": "", "insights": [], "statistics": {}},
            "prediction": None,
            "conclusion": {"summary": "", "bullets": []},
        },
        "suggestedQuestions": [],
        "dotnetPayload": {},
        "metadata": {
            "sessionId": session_id or "admin-default",
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
    }
    return payload


def conversational_reply(query: str, body: Dict[str, Any] | None = None) -> str:
    body = body or {}
    name = (
        body.get("fullName")
        or body.get("adminName")
        or body.get("userName")
        or body.get("name")
        or "Officer"
    )
    cleaned = normalize_text(query).rstrip("!?.,;")
    if cleaned in {"hi", "hello", "hey"}:
        return f"Hello, {name}. How can I help you today?"
    if "thank" in cleaned:
        return "You're welcome. I'm here if you need anything else."
    if "who are you" in cleaned or "what are you" in cleaned:
        return "I am AgniAI, your assistant for data review and reporting."
    if "how are you" in cleaned:
        return "I am ready to help with your questions."
    if "weather" in cleaned:
        return "I can help with data and reports, but I cannot check the weather."
    if "joke" in cleaned:
        return "I am focused on data work, but I can help with reports or analysis."
    return "I can help with data, reports, and analysis."
