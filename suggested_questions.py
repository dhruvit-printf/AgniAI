"""
suggested_questions.py
======================
Compatibility shim for older imports.

The active implementation now lives in suggested_question_engine.py.
"""

from __future__ import annotations

from typing import Any, Dict, List

from suggested_question_engine import generate_suggested_questions as _generate

__all__ = ["generate_suggested_questions"]


def generate_suggested_questions(
    query_type: str,
    intent: Dict[str, Any],
    combined_result: Any,
) -> List[str]:
    return _generate(query_type, intent, combined_result)
