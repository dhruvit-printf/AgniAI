"""
suggested_question_engine.py
============================
Generates dynamic, data-aware next-step questions based on query intent and answer data.
"""

from typing import Any, Dict, List

def generate_suggested_questions(
    query_type: str,
    intent: Dict[str, Any],
    combined_result: Any,
) -> List[str]:
    """
    Return 3-5 relevant next-step questions from the canonical suggested question list.
    """
    category = (intent.get("category") or "").strip()
    if not category or category.lower() in ("greeting", "unknown", "none", "unclear"):
        return []

    subcategory = (intent.get("subcategory") or "").strip()
    qtype_normalized = (query_type or "").strip().lower()

    # Extract dynamic properties from combined_result/intent
    sections = (combined_result or {}).get("sections") or [] if isinstance(combined_result, dict) else []
    section_name = ""
    if sections:
        section_name = sections[0].get("label") or ""
    if not section_name:
        section_name = intent.get("section") or intent.get("sub_section") or "this section"

    sport_name = intent.get("sport") or "the sport"

    # ── Override: Fallback to query type templates for complex modes ──
    if qtype_normalized == "compare" or qtype_normalized == "comparison":
        left_label = ((combined_result or {}).get("left", {}).get("label") or "Side A") if isinstance(combined_result, dict) else "Side A"
        right_label = ((combined_result or {}).get("right", {}).get("label") or "Side B") if isinstance(combined_result, dict) else "Side B"
        return [
            "Compare Drill and Firing performance.",
            "Compare BEPT and PPT scores.",
            "Show side-by-side metric distributions.",
            f"Compare {left_label} and {right_label} with the overall average."
        ]
    elif qtype_normalized == "cross_filter":
        return [
            "Show Class A agniveers with Excellent grading in the BEPT section.",
            "List overweight agniveers from Batch 15.",
            "Show cricket players belonging to Class B.",
            "How many matching records are on leave?"
        ]
    elif qtype_normalized == "multi_independent":
        return [
            "Show the top 5 performers and the top 5 leave takers.",
            "Show active medical cases and current attendance summary.",
            "List overdue equipment and pending police verifications.",
            "Show monthly attendance statistics and blood group distribution."
        ]

    # ── Category Specific Question Lists ──
    if category == "Performance":
        if subcategory == "TopPerformers":
            return [
                "Who are the top 10 performers overall?",
                "Show the lowest scoring agniveers.",
                f"Who are the lowest performers in {section_name}?",
                f"Compare {section_name}'s top performers with another section."
            ]
        elif subcategory == "LowestPerformers":
            return [
                "Who scored highest in BEPT?",
                "Show the top 5 performers in Firing.",
                f"Who are the top performers in {section_name}?",
                f"Compare {section_name}'s lowest performers with another section."
            ]
        else:
            return [
                "Who are the top 10 performers overall?",
                "Show the lowest scoring agniveers.",
                f"Who are the lowest performers in {section_name}?",
                f"Compare {section_name} with another section."
            ]
    elif category == "Leave":
        return [
            "Who has taken the most leave?",
            "Which agniveers are currently on leave?",
            f"Who took the most leaves in {section_name} this month?",
            "Show all absconded leave records."
        ]
    elif category == "Medical":
        return [
            "Show active medical cases.",
            "Which agniveers are obese?",
            f"Show BMI outliers and fitness analysis for {section_name}.",
            "Give blood group distribution."
        ]
    elif category == "Attendance":
        return [
            "Show monthly attendance statistics.",
            "Give weekly attendance report.",
            f"Show monthly attendance statistics for {section_name}.",
            "Show today's attendance."
        ]
    elif category == "Equipment":
        return [
            "Show overall equipment statistics.",
            "Which equipment is overdue for return?",
            "Show returned equipment.",
            "Show overdue equipment returns."
        ]
    elif category == "Verification":
        return [
            "Which agniveers are pending verification?",
            "Show sent verification records.",
            "Show verified agniveers.",
            "Show pending verification list."
        ]
    elif category in ("Skills", "Roster"):
        return [
            "Show all sports players.",
            "List cricket players.",
            f"Show roster by sport for {sport_name}.",
            "List all Class A agniveers."
        ]
    elif category == "Strength":
        return [
            "Show complete strength breakdown.",
            "Give company-wise strength report.",
            "Show platoon-wise strength.",
            "Show present and absent counts."
        ]
    elif category == "Overall":
        return [
            "Show top overall performers.",
            "Rank agniveers using composite score.",
            "Which batch has the best overall performance?",
            "Show top 20 overall agniveers."
        ]

    return [
        f"Show average score for {category}.",
        f"List all records in {category}.",
        f"Show section-wise distribution for {category}."
    ]
