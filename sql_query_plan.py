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
    intent = (op.intent_result or {}) if op else {}
    company = intent.get("company_name") or (f"Company {intent.get('company_id')}" if intent.get("company_id") is not None else None)
    platoon = intent.get("platoon_name") or (f"Platoon {intent.get('platoon_id')}" if intent.get("platoon_id") is not None else None)
    agniveer = intent.get("agniveer_no")
    section = intent.get("section")
    
    parts = []
    if company:
        parts.append(str(company))
    if platoon:
        parts.append(str(platoon))
    if agniveer:
        parts.append(str(agniveer))
    if section and not parts:
        parts.append(str(section))
        
    if parts:
        return " - ".join(parts)
        
    return (
        intent.get("section")
        or intent.get("category")
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
        section, err = _run_one(op, question)
        label = _label_for(op, idx)
        if err:
            logger.warning(
                "sql_query_plan: sub-operation %d (%s) failed: %s", idx, label, err
            )
            failed_labels.append(label)
            continue
        raw_results.append(section)
        labeled_results.append((label, section))

    if not labeled_results and failed_labels:
        return [], [], _ALL_SUBQUERIES_FAILED

    return raw_results, labeled_results, None


def fetch_sql_results(
    plan: QueryPlan, question: str, intent: Optional[Dict[str, Any]] = None
) -> _RawAndLabeled:
    """Execute plan against sql_executor and return raw/labeled result sets."""
    qtype = plan.query_type
    qtype_str = str(getattr(qtype, "value", qtype)).lower()

    if qtype_str == "cross_filter":
        return _fetch_cross_filter(plan, question, intent)
    if qtype_str in ("compare", "comparison"):
        return _fetch_compare(plan, question, intent)
    if qtype_str in ("multi_independent", "multi_query"):
        return _fetch_multi_independent(plan, question, intent)

    return _fetch_simple(plan, question, intent)
