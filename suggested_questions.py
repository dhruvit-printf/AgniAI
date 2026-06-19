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

    # Unrecognized or greeting category
    if category.lower() in ("greeting", "unknown", "none"):
        return []

    subcategory = (intent.get("subcategory") or "").strip()
    qtype_normalized = (query_type or "").strip().lower()

    # 1. Specific (category, subcategory) overrides
    category_subcategory_overrides = {
        ("Performance", "TopPerformers"): [
            "Who are the lowest performers in this section?",
            "Compare this section's top performers with another section.",
            "Which of these top performers are currently on leave?",
            "What is the average score for these top performers?",
        ],
        ("Performance", "LowestPerformers"): [
            "Who are the top performers in this section?",
            "Show the overall pass percentage.",
            "What is the average score for these lowest performers?",
            "Compare this section's lowest performers with another section.",
        ],
        ("Leave", "CurrentLeaveStatus"): [
            "Who has taken the most leaves this month?",
            "Show currently absent personnel.",
            "Are there any trainees hospitalized today?",
            "Show overall attendance for today.",
        ],
        ("Medical", "ActiveCases"): [
            "Who is currently hospitalized?",
            "What are the top diseases this month?",
            "Show the active medical case breakdown.",
            "What is the active medical case count?",
        ],
    }

    key_pair = (category, subcategory)
    if key_pair in category_subcategory_overrides:
        return category_subcategory_overrides[key_pair]

    # 2. Specific query_type overrides
    query_type_overrides = {
        "comparison": [
            "Show the absolute score difference.",
            "Compare this with the overall average.",
            "Compare another pair of sections.",
            "Show side-by-side metric distributions.",
        ],
        "cross_filter": [
            "How many matching records are on leave?",
            "Show the performance of these filtered trainees.",
            "Export this intersection roster.",
            "Compare this filtered subset with overall metrics.",
        ],
    }

    if qtype_normalized in query_type_overrides:
        return query_type_overrides[qtype_normalized]

    # 3. Base category-level templates
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

    if category in templates:
        return templates[category]

    return []
