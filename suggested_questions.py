"""
suggested_questions.py
======================
Generates deterministic, template-driven next-step questions based on the intent category.
"""

from typing import Any, Dict, List


def generate_suggested_questions(
    query_type: str,
    intent: Dict[str, Any],
    combined_result: Any,
) -> List[str]:
    """
    Return 3–5 relevant next-step questions. Deterministic, template-driven, fast.
    """
    category = (intent.get("category") or "").strip()
    if not category:
        return []

    templates: Dict[str, List[str]] = {
        "Performance": [
            "Who are the lowest performers in this section?",
            "Which of these top performers are on leave?",
            "Compare this section with another.",
            "Show the grade distribution for this section.",
        ],
        "Leave": [
            "Who took the most leaves this month?",
            "Show currently absent personnel.",
            "Are there any absconded cases?",
            "Show attendance for today.",
        ],
        "Medical": [
            "Show BMI outliers and fitness analysis.",
            "What is the active medical case count?",
            "List the most common diseases.",
            "Who is currently hospitalized?",
        ],
        "Attendance": [
            "Show monthly attendance statistics.",
            "Who is present today?",
            "Show platoon-wise strength breakdown.",
            "List the absentees for today.",
        ],
        "Equipment": [
            "Show overdue equipment returns.",
            "What items were issued today?",
            "List unassigned items in inventory.",
            "Show poor condition returned items.",
        ],
        "Verification": [
            "Show pending verification list.",
            "How many verifications are completed?",
            "List all unverified documents.",
            "Show verification rate by platoon.",
        ],
        "Distribution": [
            "Show latest distribution stats.",
            "Show distribution by unit.",
            "List unassigned trainees.",
            "Which unit received the most items?",
        ],
        "Skills": [
            "Show roster by sport.",
            "List cricket and football players.",
            "Show blood group distribution.",
            "Show class distribution breakdown.",
        ],
    }

    # Return template list if category is matched
    if category in templates:
        return templates[category]

    return []
