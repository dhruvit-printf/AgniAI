"""
suggested_question_engine.py
============================
Generates dynamic, data-aware next-step questions based on query intent and answer data.
"""

from typing import Any, Dict, List

def generate_suggested_questions(
    query_type: str,
    intent: Dict[str, Any],
    answer: Dict[str, Any],
) -> List[str]:
    """
    Return 3-5 relevant next-step questions. Dynamic and data-aware.
    """
    category = (intent.get("category") or "").strip()
    if not category or category.lower() in ("greeting", "unknown", "none"):
        return []

    subcategory = (intent.get("subcategory") or "").strip()
    qtype_normalized = (query_type or "").strip().lower()

    # Extract dynamic properties from answer/intent
    sections = answer.get("sections") or []
    section_name = ""
    if sections:
        section_name = sections[0].get("label") or ""
    if not section_name:
        section_name = intent.get("section") or intent.get("sub_section") or "this section"

    sport_name = intent.get("sport") or "Cricket"

    # Base templates with placeholders replaced by actual data
    if category == "Performance":
        if subcategory == "TopPerformers":
            return [
                f"Who are the lowest performers in {section_name}?",
                f"Compare {section_name}'s top performers with another section.",
                f"Which of these top performers in {section_name} are currently on leave?",
                f"What is the average score for these top performers?"
            ]
        elif subcategory == "LowestPerformers":
            return [
                f"Who are the top performers in {section_name}?",
                f"Show the overall pass percentage in {section_name}.",
                f"What is the average score for these lowest performers?",
                f"Compare {section_name}'s lowest performers with another section."
            ]
        else:
            return [
                f"Who are the lowest performers in {section_name}?",
                f"Which of these top performers in {section_name} are on leave?",
                f"Compare {section_name} with another section.",
                f"Show the grade distribution for {section_name}."
            ]
    elif category == "Leave":
        return [
            f"Who took the most leaves in {section_name} this month?",
            f"Show currently absent personnel in {section_name}.",
            f"Are there any absconded cases in {section_name}?",
            f"Show overall attendance for today."
        ]
    elif category == "Medical":
        return [
            f"Show BMI outliers and fitness analysis for {section_name}.",
            f"What is the active medical case count in {section_name}?",
            f"List the most common diseases in {section_name}.",
            f"Who from {section_name} is currently hospitalized?"
        ]
    elif category == "Attendance":
        return [
            f"Show monthly attendance statistics for {section_name}.",
            f"Who is present in {section_name} today?",
            f"Show platoon-wise strength breakdown for {section_name}.",
            f"List the absentees for today."
        ]
    elif category == "Equipment":
        return [
            "Show overdue equipment returns.",
            "What items were issued today?",
            "List unassigned items in inventory.",
            "Show poor condition returned items."
        ]
    elif category == "Verification":
        return [
            "Show pending verification list.",
            "How many verifications are completed?",
            "List all unverified documents.",
            "Show verification rate by platoon."
        ]
    elif category == "Skills":
        return [
            f"Show roster by sport for {sport_name}.",
            f"List cricket and football players.",
            f"Show blood group distribution in {section_name}.",
            f"Show class distribution breakdown."
        ]

    # Fallback to query type templates
    if qtype_normalized == "compare":
        left_label = answer.get("left", {}).get("label") or "Side A"
        right_label = answer.get("right", {}).get("label") or "Side B"
        return [
            f"Show the absolute score difference between {left_label} and {right_label}.",
            f"Compare {left_label} and {right_label} with the overall average.",
            "Compare another pair of sections.",
            "Show side-by-side metric distributions."
        ]
    elif qtype_normalized == "cross_filter":
        return [
            "How many matching records are on leave?",
            "Show the performance of these filtered trainees.",
            "Export this intersection roster.",
            "Compare this filtered subset with overall metrics."
        ]

    return [
        f"Show average score for {category}.",
        f"List all records in {category}.",
        f"Show section-wise distribution for {category}."
    ]
