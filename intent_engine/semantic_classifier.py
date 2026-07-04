"""
semantic_classifier.py
======================

Stage 2 (embedding-based) and Stage 3 (constrained Ollama JSON) classification
for the AgniAI admin intent pipeline.

Stage 2:
  - At startup, embed a catalog of canonical English descriptions for every
    (category, operation) pair in OPERATIONS_BY_CATEGORY.
  - If keyword confidence < 0.45, run cosine-similarity search against the catalog.
  - Accept if top similarity >= 0.60 AND margin over the next DIFFERENT
    (category, operation) >= 0.08.
  - If margin < 0.08, return needs_clarification asking the user to choose.

Stage 3 (optional, only when Stage 2 is ambiguous):
  - Ask Ollama to return JSON with {category, operation} constrained to
    OPERATIONS_BY_CATEGORY.  Validate strictly; on any invalid output → clarification.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .intent_schema import OPERATIONS_BY_CATEGORY

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Catalog: canonical descriptions per (category, operation)
# ---------------------------------------------------------------------------

_CATALOG: List[Tuple[str, str, str]] = [
    # (category, operation, description)
    # Performance
    ("Performance", "Top", "show top performers in BPET"),
    ("Performance", "Top", "who scored highest in firing"),
    ("Performance", "Top", "best performers in the batch"),
    ("Performance", "Top", "list top 10 agniveers by score"),
    ("Performance", "Bottom", "who scored lowest in drill"),
    ("Performance", "Bottom", "worst performers in PPT"),
    ("Performance", "Bottom", "bottom 5 agniveers in training"),
    ("Performance", "Grading", "show grading distribution for BPET"),
    ("Performance", "Grading", "how many got excellent grade"),
    ("Performance", "Grading", "agniveers graded good in firing"),
    ("Performance", "GradingSummary", "grade summary for the company"),
    ("Performance", "GradingSummary", "how many passed and failed overall"),
    ("Performance", "Average", "what is the average score in PPT"),
    ("Performance", "Average", "average marks for company 2 in BPET"),
    ("Performance", "Improvement", "who improved between attempts"),
    ("Performance", "Improvement", "which agniveers showed improvement in drill"),
    ("Performance", "Drop", "whose score dropped between attempts"),
    ("Performance", "Drop", "who declined in performance"),
    ("Performance", "AttemptWise", "show attempt wise breakdown for firing"),
    ("Performance", "AttemptWise", "scores by attempt for BPET"),
    ("Performance", "BestAttempt", "show best attempt score for each agniveer"),
    ("Performance", "BestAttempt", "personal best in PPT"),
    ("Performance", "Trend", "performance trend over attempts for the batch"),
    ("Performance", "Trend", "how has the batch been performing over time"),
    # Leave
    ("Leave", "Most", "who took the most leave"),
    ("Leave", "Most", "top leave takers in the company"),
    ("Leave", "Least", "agniveers with least days absent"),
    ("Leave", "Least", "who is most regular with attendance"),
    ("Leave", "Current", "who is currently on leave"),
    ("Leave", "Current", "show leave status for today"),
    ("Leave", "Absconded", "list absconded agniveers"),
    ("Leave", "Absconded", "who went AWOL"),
    # Medical
    ("Medical", "BMI", "show BMI distribution of the company"),
    ("Medical", "BMI", "who is overweight or underweight"),
    ("Medical", "BloodGroup", "blood group distribution of the battalion"),
    ("Medical", "BloodGroup", "how many have blood group O positive"),
    ("Medical", "Disease", "most common diseases in the unit"),
    ("Medical", "Disease", "agniveers with malaria or dengue"),
    ("Medical", "Individual", "medical record of agniveer 12345"),
    ("Medical", "Individual", "health status of a specific agniveer"),
    # Attendance
    ("Attendance", "Monthly", "monthly attendance for company 3"),
    ("Attendance", "Monthly", "attendance report for January"),
    ("Attendance", "Weekly", "weekly attendance this week"),
    ("Attendance", "Daily", "daily attendance for today"),
    ("Attendance", "Present", "how many agniveers are present today"),
    ("Attendance", "Present", "who is on campus right now"),
    ("Attendance", "Summary", "attendance summary for the batch"),
    ("Attendance", "Summary", "overall attendance overview"),
    # Strength
    ("Strength", "StrengthBreakdown", "strength breakdown by section"),
    ("Strength", "StrengthBreakdown", "how many agniveers in each platoon"),
    ("Strength", "StrengthBreakdown", "current headcount of the unit"),
    # Verification
    ("Verification", "Pending", "show pending verification requests"),
    ("Verification", "Sent", "list verification requests that were sent"),
    ("Verification", "NotResponded", "who has not responded to verification"),
    ("Verification", "Completed", "completed verification count"),
    ("Verification", "Rejected", "rejected verification cases"),
    # Equipment
    ("Equipment", "Stats", "equipment inventory summary"),
    ("Equipment", "Stats", "how many items are issued overall"),
    ("Equipment", "Search", "find which agniveers have combat coat"),
    ("Equipment", "Search", "search equipment by name"),
    ("Equipment", "Returned", "equipment returned in poor condition"),
    ("Equipment", "Returned", "damaged items handed back"),
    ("Equipment", "Holding", "who is currently holding overdue equipment"),
    ("Equipment", "Holding", "items that have not been returned"),
    ("Equipment", "AgniveerWise", "equipment issued to each agniveer"),
    # Distribution
    ("Distribution", "Latest", "latest distribution of agniveers to units"),
    ("Distribution", "ByUnit", "agniveers distributed by unit"),
    ("Distribution", "ByUnit", "how many agniveers are in alpha unit"),
    ("Distribution", "Unassigned", "agniveers not yet assigned to any unit"),
    ("Distribution", "TopUnit", "which unit received the most agniveers"),
    # Skills
    ("Skills", "BySport", "who plays volleyball"),
    ("Skills", "BySport", "give me agniveers with cricket as a skill"),
    ("Skills", "BySport", "sports roster for the company"),
    ("Skills", "BySport", "list all football players"),
    ("Skills", "BySport", "agniveers who play badminton"),
    ("Skills", "ByClass", "show agniveers by class"),
    ("Skills", "ByClass", "how many sikh soldiers in the unit"),
    ("Skills", "ByClass", "class wise distribution of agniveers"),
    # Overall
    ("Overall", "OverallPerformance", "overall performance ranking of all agniveers"),
    ("Overall", "OverallPerformance", "composite score across all criteria"),
    ("Overall", "OverallPerformance", "holistic performance of the batch"),
    # Schedule
    ("Schedule", "Today", "what is today's training schedule"),
    ("Schedule", "Today", "show today's agenda for the company"),
    ("Schedule", "Company", "schedule for company 2 this week"),
    ("Schedule", "Date", "training schedule for a specific date"),
    ("Schedule", "Agniveer", "schedule for a particular agniveer"),
    # personaldetail
    ("personaldetail", "info", "personal details of agniveer 12345"),
    ("personaldetail", "info", "contact information and profile of an agniveer"),
    ("personaldetail", "info", "family background of the soldier"),
    # disqualified
    ("disqualified", "removed", "list disqualified agniveers"),
    ("disqualified", "removed", "who was expelled from the program"),
    ("disqualified", "removed", "disqualification reasons for removed soldiers"),
]

# ---------------------------------------------------------------------------
# Singleton state
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_catalog_embeddings: Optional[np.ndarray] = None  # shape (N, D)
_catalog_labels: List[Tuple[str, str]] = []        # (category, operation) per row


def _get_model():
    try:
        from rag import load_embedding_model
        return load_embedding_model()
    except Exception as exc:
        logger.warning("semantic_classifier: could not load embedding model: %s", exc)
        return None


def _build_catalog() -> None:
    global _catalog_embeddings, _catalog_labels
    model = _get_model()
    if model is None:
        return
    texts = [desc for _, _, desc in _CATALOG]
    labels = [(cat, op) for cat, op, _ in _CATALOG]
    try:
        vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        with _lock:
            _catalog_embeddings = vecs
            _catalog_labels = labels
        logger.info(
            "semantic_classifier: built catalog with %d entries", len(texts)
        )
    except Exception as exc:
        logger.error("semantic_classifier: catalog build failed: %s", exc)


def _ensure_catalog() -> bool:
    global _catalog_embeddings
    if _catalog_embeddings is None:
        _build_catalog()
    return _catalog_embeddings is not None


# ---------------------------------------------------------------------------
# Stage 2 — cosine similarity lookup
# ---------------------------------------------------------------------------

_SIMILARITY_THRESHOLD = 0.60
_MARGIN_THRESHOLD = 0.08


def classify_semantic(
    query: str,
) -> Dict[str, Any]:
    """
    Stage 2 classification.

    Returns a dict with keys:
      - category, operation: best match (or None)
      - confidence: cosine similarity score
      - needs_clarification: bool
      - clarification_question: str or None
      - stage: "semantic"
    """
    if not _ensure_catalog():
        return {"category": None, "operation": None, "confidence": 0.0,
                "needs_clarification": False, "clarification_question": None,
                "stage": "semantic_unavailable"}

    model = _get_model()
    if model is None:
        return {"category": None, "operation": None, "confidence": 0.0,
                "needs_clarification": False, "clarification_question": None,
                "stage": "semantic_unavailable"}

    try:
        q_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    except Exception as exc:
        logger.warning("semantic_classifier: encode failed: %s", exc)
        return {"category": None, "operation": None, "confidence": 0.0,
                "needs_clarification": False, "clarification_question": None,
                "stage": "semantic_error"}

    with _lock:
        emb = _catalog_embeddings
        labels = list(_catalog_labels)

    if emb is None:
        return {"category": None, "operation": None, "confidence": 0.0,
                "needs_clarification": False, "clarification_question": None,
                "stage": "semantic_unavailable"}

    sims = emb @ q_vec  # cosine similarity (embeddings are normalised)

    # Aggregate per (category, operation): take the maximum similarity
    best_per_label: Dict[Tuple[str, str], float] = {}
    for idx, label in enumerate(labels):
        prev = best_per_label.get(label, -1.0)
        if sims[idx] > prev:
            best_per_label[label] = float(sims[idx])

    ranked = sorted(best_per_label.items(), key=lambda x: x[1], reverse=True)
    if not ranked:
        return {"category": None, "operation": None, "confidence": 0.0,
                "needs_clarification": False, "clarification_question": None,
                "stage": "semantic"}

    top_label, top_sim = ranked[0]
    top_cat, top_op = top_label

    if top_sim < _SIMILARITY_THRESHOLD:
        return {"category": None, "operation": None, "confidence": top_sim,
                "needs_clarification": False, "clarification_question": None,
                "stage": "semantic_below_threshold"}

    # Find the best match in a DIFFERENT category
    second_diff = next(
        ((lbl, s) for lbl, s in ranked[1:] if lbl[0] != top_cat),
        None,
    )
    margin = (top_sim - second_diff[1]) if second_diff else 1.0

    if margin < _MARGIN_THRESHOLD:
        # Ambiguous — ask user to clarify
        second_label = second_diff[0] if second_diff else (top_cat, top_op)
        question = (
            f"Do you mean {top_cat}/{top_op} or {second_label[0]}/{second_label[1]}?"
        )
        logger.info(
            "semantic_classifier: ambiguous | top=(%s/%s, %.3f) | "
            "second=(%s/%s, %.3f) | margin=%.3f",
            top_cat, top_op, top_sim,
            second_label[0], second_label[1],
            second_diff[1] if second_diff else 0.0,
            margin,
        )
        return {
            "category": None,
            "operation": None,
            "confidence": top_sim,
            "needs_clarification": True,
            "clarification_question": question,
            "stage": "semantic_ambiguous",
        }

    logger.info(
        "semantic_classifier: accepted (%s/%s) sim=%.3f margin=%.3f",
        top_cat, top_op, top_sim, margin,
    )
    return {
        "category": top_cat,
        "operation": top_op,
        "confidence": top_sim,
        "needs_clarification": False,
        "clarification_question": None,
        "stage": "semantic",
    }


# ---------------------------------------------------------------------------
# Stage 3 — constrained Ollama JSON (only when Stage 2 is ambiguous)
# ---------------------------------------------------------------------------

def classify_ollama_constrained(query: str) -> Dict[str, Any]:
    """
    Stage 3: ask Ollama to classify the query into a valid (category, operation)
    pair.  Returns the same shape as classify_semantic().
    On any failure or invalid output → needs_clarification.
    """
    valid_pairs = [
        f"{cat}/{op}"
        for cat, ops in OPERATIONS_BY_CATEGORY.items()
        for op in sorted(ops)
    ]
    pairs_str = "\n".join(f"  {p}" for p in valid_pairs)
    prompt = (
        f"Classify the following admin query into exactly one category/operation "
        f"from this list:\n{pairs_str}\n\n"
        f'Query: "{query}"\n\n'
        f'Respond with ONLY valid JSON in this format: '
        f'{{"category": "...", "operation": "..."}}'
    )

    try:
        import requests  # type: ignore
        from config import OLLAMA_URL  # type: ignore
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": "mistral", "prompt": prompt, "stream": False},
            timeout=10,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        # Extract JSON block
        import re
        m = re.search(r"\{.*?\}", raw, re.DOTALL)
        if not m:
            raise ValueError(f"no JSON in response: {raw!r}")
        data = json.loads(m.group(0))
        cat = data.get("category", "")
        op = data.get("operation", "")
        # Validate
        if cat not in OPERATIONS_BY_CATEGORY:
            raise ValueError(f"unknown category: {cat!r}")
        if op not in OPERATIONS_BY_CATEGORY[cat]:
            raise ValueError(f"unknown operation {op!r} for {cat}")
        logger.info("semantic_classifier stage3: accepted (%s/%s)", cat, op)
        return {
            "category": cat,
            "operation": op,
            "confidence": 0.65,
            "needs_clarification": False,
            "clarification_question": None,
            "stage": "ollama_constrained",
        }
    except Exception as exc:
        logger.warning("semantic_classifier stage3 failed: %s", exc)
        return {
            "category": None,
            "operation": None,
            "confidence": 0.0,
            "needs_clarification": True,
            "clarification_question": (
                "I'm not sure what you're asking about. "
                "Could you clarify the category (e.g. Performance, Leave, Medical)?"
            ),
            "stage": "ollama_failed",
        }


# ---------------------------------------------------------------------------
# Public entry point used by intent_classifier.py
# ---------------------------------------------------------------------------

def classify_admin_intent_semantic(
    query: str,
    keyword_confidence: float,
) -> Optional[Dict[str, Any]]:
    """
    Called from classify_intent() when keyword_confidence < 0.45.

    Returns a result dict (same shape as classify_semantic) if Stage 2 accepts,
    or None if the query should remain unclassified.
    """
    result = classify_semantic(query)

    if result.get("needs_clarification"):
        # Stage 2 ambiguous → try Stage 3
        stage3 = classify_ollama_constrained(query)
        if not stage3.get("needs_clarification"):
            return stage3
        return result  # propagate clarification

    if result.get("category"):
        return result

    return None


# Trigger catalog build in background when this module is first imported
_build_thread = threading.Thread(target=_build_catalog, daemon=True)
_build_thread.start()
