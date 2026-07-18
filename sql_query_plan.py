"""
sql_query_plan.py
==================
Thin orchestrator that drives sql_executor.execute_sql_query() according to
the query plan produced by intent_engine/query_planner.py, and returns raw +
labeled results in the SAME shape admin_pipeline.py already builds for the
.NET branches. admin_pipeline.py owns the single call to
result_combiner.combine_results() — this module must never call it, or the
result gets combined twice (once here, once there).

This module does not talk to the database directly and does not generate
SQL — it only decides *how many* execute_sql_query() calls to make and how
to label them, mirroring the .NET fan-out in admin_pipeline.py.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from intent_engine.query_planner import QueryPlan, SubOperation
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


_RawAndLabeled = Tuple[List[Any], List[Tuple[str, Any]], Optional[str]]


def _fetch_simple(
    plan: QueryPlan, question: str, intent: Optional[Dict[str, Any]]
) -> _RawAndLabeled:
    op = plan.operations[0] if plan.operations else None
    hint = op.intent_result if op else intent
    fragment = op.raw_fragment if op else question
    section, err = execute_sql_query(question=fragment or question, intent=hint)
    if err:
        return [], [], err

    label = _label_for(op, 0) if op else "Result"
    return [section], [(label, section)], None


def _fetch_cross_filter(
    plan: QueryPlan, question: str, intent: Optional[Dict[str, Any]]
) -> _RawAndLabeled:
    # Cross-filter queries are executed leg-by-leg so the combiner can
    # intersect the returned record sets. Each leg still runs through the
    # normal SQL executor with its own intent hint.
    raw_results: List[Any] = []
    labeled_results: List[Tuple[str, Any]] = []

    for idx, op in enumerate(plan.operations):
        section, err = _run_one(op, question)
        if err:
            return [], [], err
        raw_results.append(section)
        labeled_results.append((_label_for(op, idx), section))

    return raw_results, labeled_results, None


def _fetch_compare(
    plan: QueryPlan, question: str, intent: Optional[Dict[str, Any]]
) -> _RawAndLabeled:
    raw_results: List[Any] = []
    labeled_results: List[Tuple[str, Any]] = []

    for idx, op in enumerate(plan.operations):
        section, err = _run_one(op, question)
        if err:
            return [], [], err
        label = _label_for(op, idx)
        raw_results.append(section)
        labeled_results.append((label, section))

    return raw_results, labeled_results, None


def _fetch_multi_independent(
    plan: QueryPlan, question: str, intent: Optional[Dict[str, Any]]
) -> _RawAndLabeled:
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
        return [], [], _ALL_SUBQUERIES_FAILED

    return raw_results, labeled_results, None


def fetch_sql_results(
    plan: QueryPlan,
    question: str,
    intent: Optional[Dict[str, Any]] = None,
) -> _RawAndLabeled:
    """
    Drive sql_executor per the query plan's type. Returns
    (raw_results, labeled_results, error) — the SAME shape admin_pipeline.py
    already builds for the (hybrid) .NET branches, so its single
    combine_results() call, partial-failure tracking, and cross-filter
    display-limit re-trim need ZERO changes to consume this.

    On error the SQL backend could not answer the query at all — the caller
    (the hybrid branch in admin_pipeline.py) should degrade to the existing
    RAG fallback in that case, never raise, never 500.
    """
    if not plan.operations:
        section, err = execute_sql_query(question=question, intent=intent)
        if err:
            return [], [], err
        return [section], [("Result", section)], None

    qtype = plan.query_type.value

    if qtype == "cross_filter" and len(plan.operations) >= 2:
        return _fetch_cross_filter(plan, question, intent)
    if qtype == "compare" and len(plan.operations) >= 2:
        return _fetch_compare(plan, question, intent)
    if qtype == "multi_independent" and len(plan.operations) >= 2:
        return _fetch_multi_independent(plan, question, intent)

    return _fetch_simple(plan, question, intent)
