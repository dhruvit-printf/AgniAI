"""
sql_query_plan.py
==================
Thin orchestrator that drives sql_executor.execute_sql_query() according to
the query plan produced by intent_engine/query_planner.py, then hands the
results to the EXISTING result_combiner.combine_results() so rendering
(grounding, widgets, suggested questions) is identical to the .NET path.

This module does not talk to the database directly and does not generate
SQL — it only decides *how many* execute_sql_query() calls to make and how
to label them, mirroring the .NET fan-out in admin_pipeline.py.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from intent_engine.query_planner import QueryPlan, SubOperation
from result_combiner import combine_results
from sql_executor import execute_sql_query

logger = logging.getLogger(__name__)

_ALL_SUBQUERIES_FAILED = "All sub-queries could not be answered from the database."


def _label_for(op: SubOperation, index: int) -> str:
    return (
        op.intent_result.get("section")
        or op.intent_result.get("category")
        or f"Query {index + 1}"
    )


def _run_one(op: SubOperation, fallback_question: str) -> Tuple[Any, Optional[str]]:
    question = op.raw_fragment or fallback_question
    return execute_sql_query(question=question, intent=op.intent_result)


def _run_simple(
    plan: QueryPlan, question: str, intent: Optional[Dict[str, Any]]
) -> Tuple[Any, Optional[str]]:
    op = plan.operations[0] if plan.operations else None
    hint = op.intent_result if op else intent
    fragment = op.raw_fragment if op else question
    section, err = execute_sql_query(question=fragment or question, intent=hint)
    if err:
        return None, err

    label = _label_for(op, 0) if op else "Result"
    qtype_str = plan.query_type.value
    combined = combine_results([section], [(label, section)], qtype_str, intent or {})
    return combined, None


def _run_cross_filter(
    plan: QueryPlan, question: str, intent: Optional[Dict[str, Any]]
) -> Tuple[Any, Optional[str]]:
    # HARD RULE R4: a cross-filter is ONE query whose CTEs each emit DISTINCT
    # AgniveerId for one condition, intersected in the final SELECT. We
    # therefore issue a single execute_sql_query() call, feeding every
    # sub-fragment as one combined question so the generator has all
    # conditions in view and can build the CTE-per-condition shape.
    fragments = [op.raw_fragment for op in plan.operations if op.raw_fragment]
    combined_question = " AND ".join(fragments) if fragments else question

    hint: Dict[str, Any] = dict(intent or {})
    hint["query_type"] = "cross_filter"
    hint["conditions"] = fragments or [question]

    section, err = execute_sql_query(question=combined_question, intent=hint)
    if err:
        return None, err

    label = (
        plan.operations[0].intent_result.get("category")
        if plan.operations
        else "Result"
    )
    combined = combine_results(
        [section], [(label or "Result", section)], "cross_filter", intent or {}
    )
    return combined, None


def _run_compare(
    plan: QueryPlan, question: str, intent: Optional[Dict[str, Any]]
) -> Tuple[Any, Optional[str]]:
    raw_results: List[Any] = []
    labeled_results: List[Tuple[str, Any]] = []

    for idx, op in enumerate(plan.operations):
        section, err = _run_one(op, question)
        if err:
            return None, err
        label = _label_for(op, idx)
        raw_results.append(section)
        labeled_results.append((label, section))

    combined = combine_results(raw_results, labeled_results, "compare", intent or {})
    return combined, None


def _run_multi_independent(
    plan: QueryPlan, question: str, intent: Optional[Dict[str, Any]]
) -> Tuple[Any, Optional[str]]:
    raw_results: List[Any] = []
    labeled_results: List[Tuple[str, Any]] = []
    failed_labels: List[str] = []

    for idx, op in enumerate(plan.operations):
        label = _label_for(op, idx)
        section, err = _run_one(op, question)
        if err:
            failed_labels.append(label)
            placeholder = {"unavailable": True}
            raw_results.append(placeholder)
            labeled_results.append((label, placeholder))
        else:
            raw_results.append(section)
            labeled_results.append((label, section))

    if failed_labels and len(failed_labels) == len(plan.operations):
        return None, _ALL_SUBQUERIES_FAILED

    combined = combine_results(
        raw_results, labeled_results, "multi_independent", intent or {}
    )
    if failed_labels:
        combined["degraded"] = True
        combined["failedSections"] = failed_labels
    return combined, None


def run_sql_plan(
    plan: QueryPlan,
    question: str,
    intent: Optional[Dict[str, Any]] = None,
) -> Tuple[Any, Optional[str]]:
    """
    Drive sql_executor per the query plan's type, then combine via the
    existing result_combiner so the shape matches the .NET path exactly.

    Returns (combined_result, None) on success, or (None, error) when the
    SQL backend could not answer the query at all — the caller (the hybrid
    branch in admin_pipeline.py) should degrade to the existing RAG fallback
    in that case, never raise, never 500.
    """
    if not plan.operations:
        section, err = execute_sql_query(question=question, intent=intent)
        if err:
            return None, err
        combined = combine_results(
            [section], [("Result", section)], "simple", intent or {}
        )
        return combined, None

    qtype = plan.query_type.value

    if qtype == "cross_filter" and len(plan.operations) >= 2:
        return _run_cross_filter(plan, question, intent)
    if qtype == "compare" and len(plan.operations) >= 2:
        return _run_compare(plan, question, intent)
    if qtype == "multi_independent" and len(plan.operations) >= 2:
        return _run_multi_independent(plan, question, intent)

    return _run_simple(plan, question, intent)
