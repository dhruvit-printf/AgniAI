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

_ADMIN_PHRASES = (
    "top performer",
    "top performers",
    "highest performer",
    "highest performers",
    "best performer",
    "best performers",
    "football players",
    "cricket players",
    "players currently on leave",
    "who plays",
    "who is on leave",
    "currently on leave",
    "active medical",
    "grade distribution",
    "grading summary",
    "monthly attendance",
    "weekly attendance",
    "pass percentage",
    "fail percentage",
    "compare",
    "bpet",
    "ppt",
    "firing",
    "drill",
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
    return any(token in _ADMIN_CONTEXT_WORDS for token in tokens)


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
        "overallConfidence": 1.0,
        "partialFailure": False,
        "failedSections": [],
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
