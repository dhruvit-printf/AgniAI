"""
sql_executor.py
===============
Text-to-SQL execution backend for AgniAI (DB_Agni).

This is the *swappable middle* of the pipeline. It occupies the same slot as
`dotnet_executor._call_dotnet` and returns the SAME contract:

    execute_sql_query(...) -> Tuple[data, error]
        (rows, None)  on success
        (None, msg)   on failure

so it drops into the existing call sites in admin_pipeline.py. Everything
downstream (result_combiner, report_generator grounding, widget_engine,
suggested_question_engine) is reused unchanged — this module only replaces
"how the data is fetched", not "how the answer is built".

Flow (as actually wired in execute_sql_query today):
    intent -> [golden fast-path if GOLDEN_QUERIES has this (category,
    operation) AND every filter in the intent is one the golden template
    can express — see _golden_query_can_satisfy] -> otherwise
    query_planner_v2.plan_query() builds a deterministic AST from the
    intent's filters, which sql_builder.build() compiles to SQL
    -> sql_validator.validate_sql() (defense-in-depth gate)
    -> run read-only -> rows.

`generate_sql()` below (free-text LLM SQL generation carrying
business_rules.LLM_HARD_RULES) serves as the Tier 2 LLM fallback path inside
execute_sql_query() when AST planning/compilation fails or faces capability gaps.

SAFETY MODEL (do not weaken any of these):
  1. Connect with a READ-ONLY SQL login (db_datareader only, DENY on
     UserMaster.Password / LoginToken). The login is the real wall; the
     validator below is defense-in-depth, not the primary control.
  2. Only a single SELECT / WITH...SELECT statement is ever executed.
  3. Column/table allowlist derived from the schema card. Sensitive columns
     are hard-denied even if a grant slips — enforced in sql_validator.py's
     validate_sql(), the validator execute_sql_query actually calls (this
     module's own validate_sql() below is not on the live path — see its
     docstring).
  4. Row cap (SET ROWCOUNT) + command timeout on every execution.
  5. Every number in the final narrative is grounded downstream by
     grounding_utils.ground_and_sanitize against these rows.
"""

from __future__ import annotations

import datetime
import json
import decimal
import logging
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sql_validator import sql_validator

logger = logging.getLogger(__name__)

# ── Config (env-overridable, production-safe defaults) ──────────────────────
SQL_SERVER_2008_COMPAT = os.getenv("SQL_2008_COMPAT", "1").strip() == "1"
SQL_MAX_ROWS = int(os.getenv("SQL_MAX_ROWS", "500"))
SQL_COMMAND_TIMEOUT_S = int(os.getenv("SQL_COMMAND_TIMEOUT_S", "15"))
# Read-only connection string. MUST point at a db_datareader-only login.
SQL_READONLY_CONN = os.getenv("SQL_READONLY_CONN", "")

# ── Hard denylist: columns the SQL layer must NEVER return ─────────────────
DENIED_COLUMNS = {
    "usermaster.password",
    "logintoken.token",
    "logintoken.refreshtoken",
}
DENIED_TABLES = {"logintoken", "defaultlog"}


def _jsonable(v: Any) -> Any:
    """Normalize a pyodbc column value to a JSON-safe Python type.

    pyodbc returns native Python types for SQL Server columns that do NOT have
    a direct JSON representation:

    * ``datetime2`` / ``date`` / ``datetime`` columns → ``datetime.datetime`` or
      ``datetime.date`` objects, which Flask's DefaultJSONProvider serializes via
      ``http_date()`` (RFC 822 e.g. ``"Mon, 14 May 2001 00:00:00 GMT"``).  .NET's
      ``System.Text.Json`` default is ISO 8601 (``"2001-05-14"`` /
      ``"2001-05-14T00:00:00"``), which is what the frontend and widget_engine
      expect.

    * ``decimal(18,2)`` columns → ``decimal.Decimal`` objects, serialized by Flask
      as a JSON **string** (``"87.50"``).  .NET emits these as JSON **numbers**
      (``87.5``), which is what all downstream numeric consumers
      (``utils.safe_float``, ``analysis_engine``, ``widget_engine`` sorting) expect.
      ``Decimal`` values also fail the ``isinstance(v, (int, float))`` check in
      ``utils.numeric_distribution_breakdown``, silently dropping grading-summary
      count fields.

    This function is called per-value in ``_camel_case_row`` so every row that
    exits ``run_readonly`` → ``_to_section`` is already JSON-safe.
    """
    if isinstance(v, datetime.datetime):
        # datetime first — it is a subclass of date, so this branch must come first.
        return v.isoformat()
    if isinstance(v, datetime.date):
        return v.isoformat()  # e.g. "2001-05-14"
    if isinstance(v, decimal.Decimal):
        # float conversion is safe for the precision used by DB_Agni schema
        # (decimal(18,2) — at most 2 decimal places). Round to 10 dp to suppress
        # IEEE 754 representation noise without losing meaningful precision.
        return round(float(v), 10)
    return v


class CapabilityGapError(Exception):
    """Raised when the AST planner cannot express the query (route to Tier 2)."""

    pass


class ValidatorRejectionError(Exception):
    """Raised when AST or SQL validation fails (route to Tier 2)."""

    pass


class DatabaseExecutionError(Exception):
    """Raised when pyodbc DB execution fails (DO NOT fallback)."""

    pass


# Golden fast-path: (category, operation) -> parameterized SQL. `subcategory`
# is deliberately not part of the key — it's fully derived from (category,
# operation) via CATEGORY_OPERATION_TO_SUBCATEGORY in intent_schema.py, so
# The hardcoded GOLDEN_QUERIES dictionary has been retired in favor of the AST-First (+ LLM Fallback) pipeline.

_GENERATION_SYSTEM = (
    "You are a T-SQL generator for a read-only reporting API. "
    "Given the schema and a question, output ONE SQL Server SELECT statement "
    "and NOTHING else — no prose, no markdown, no comments, no semicolons. "
    "Obey every HARD RULE. If the question cannot be answered from the schema, "
    "output exactly: CANNOT_ANSWER.\n\n"
    "Example 1:\n"
    "QUESTION: Top 10 performers in BPET\n"
    "SQL: SELECT TOP 10 a.AgniveerNo, a.FullName, SUM(sa.MarksObtained) AS BestTotal FROM AgniveerMaster a INNER JOIN AgniveerScoreAttempt sa ON a.Id = sa.AgniveerId INNER JOIN ScoreSubItemMaster si ON si.Id = sa.SubItemId INNER JOIN ScoreSectionMaster sec ON sec.Id = si.SectionId WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL) AND sa.IsBestAttempt = 1 AND sec.SectionName = 'BPET' GROUP BY a.AgniveerNo, a.FullName ORDER BY BestTotal DESC\n\n"
    "Example 2:\n"
    "QUESTION: Which Agniveers play volleyball?\n"
    "SQL: SELECT a.AgniveerNo, a.FullName FROM AgniveerMaster a WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL) AND a.Sports = 'Volleyball'\n\n"
    "Example 3:\n"
    "QUESTION: PPT grades for Alpha company\n"
    "SQL: WITH AttemptedMax AS (SELECT DISTINCT sa.AgniveerId, si.SectionId, SUM(si.MaxMarks) AS DynamicMax FROM AgniveerScoreAttempt sa INNER JOIN ScoreSubItemMaster si ON si.Id = sa.SubItemId GROUP BY sa.AgniveerId, si.SectionId), BestTotals AS (SELECT sa.AgniveerId, si.SectionId, SUM(sa.MarksObtained) AS BestTotal FROM AgniveerScoreAttempt sa INNER JOIN ScoreSubItemMaster si ON si.Id = sa.SubItemId WHERE sa.IsBestAttempt = 1 GROUP BY sa.AgniveerId, si.SectionId), Scored AS (SELECT bt.AgniveerId, sec.SectionName, CASE WHEN dm.DynamicMax IS NULL OR dm.DynamicMax = 0 THEN NULL WHEN 100.0 * bt.BestTotal / dm.DynamicMax >= 90 THEN 'Exceptionally Well' WHEN 100.0 * bt.BestTotal / dm.DynamicMax >= 75 THEN 'Excellent' WHEN 100.0 * bt.BestTotal / dm.DynamicMax >= 60 THEN 'Good' WHEN 100.0 * bt.BestTotal / dm.DynamicMax >= 45 THEN 'SAT' ELSE 'Fail' END AS Grade FROM BestTotals bt INNER JOIN ScoreSectionMaster sec ON sec.Id = bt.SectionId LEFT JOIN AttemptedMax dm ON dm.AgniveerId = bt.AgniveerId AND dm.SectionId = bt.SectionId) SELECT a.AgniveerNo, a.FullName, sg.Grade FROM AgniveerMaster a INNER JOIN PlatoonMaster p ON a.PlatoonId = p.Id INNER JOIN CompanyMaster c ON p.CompanyId = c.Id INNER JOIN Scored sg ON a.Id = sg.AgniveerId WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL) AND sg.SectionName = 'PPT' AND c.Name = 'Alpha'\n\n"
    "Example 4:\n"
    "QUESTION: failed firing and still have issued equipment\n"
    "SQL: WITH AttemptedMax AS (SELECT DISTINCT sa.AgniveerId, si.SectionId, SUM(si.MaxMarks) AS DynamicMax FROM AgniveerScoreAttempt sa INNER JOIN ScoreSubItemMaster si ON si.Id = sa.SubItemId GROUP BY sa.AgniveerId, si.SectionId), BestTotals AS (SELECT sa.AgniveerId, si.SectionId, SUM(sa.MarksObtained) AS BestTotal FROM AgniveerScoreAttempt sa INNER JOIN ScoreSubItemMaster si ON si.Id = sa.SubItemId WHERE sa.IsBestAttempt = 1 GROUP BY sa.AgniveerId, si.SectionId), Scored AS (SELECT bt.AgniveerId, sec.SectionName, CASE WHEN dm.DynamicMax IS NULL OR dm.DynamicMax = 0 THEN NULL WHEN 100.0 * bt.BestTotal / dm.DynamicMax >= 90 THEN 'Exceptionally Well' WHEN 100.0 * bt.BestTotal / dm.DynamicMax >= 75 THEN 'Excellent' WHEN 100.0 * bt.BestTotal / dm.DynamicMax >= 60 THEN 'Good' WHEN 100.0 * bt.BestTotal / dm.DynamicMax >= 45 THEN 'SAT' ELSE 'Fail' END AS Grade FROM BestTotals bt INNER JOIN ScoreSectionMaster sec ON sec.Id = bt.SectionId LEFT JOIN AttemptedMax dm ON dm.AgniveerId = bt.AgniveerId AND dm.SectionId = bt.SectionId), FiringFail AS (SELECT AgniveerId FROM Scored WHERE SectionName = 'Firing' AND Grade = 'Fail'), HasEq AS (SELECT AgniveerId FROM AgniveerEquipment WHERE ReturnDateTime IS NULL) SELECT a.AgniveerNo, a.FullName FROM AgniveerMaster a INNER JOIN FiringFail f ON a.Id = f.AgniveerId INNER JOIN HasEq e ON a.Id = e.AgniveerId WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)\n\n"
    "Example 5:\n"
    "QUESTION: Rank volleyball players by total marks\n"
    "SQL: SELECT a.AgniveerNo, a.FullName, SUM(sa.MarksObtained) AS BestTotal FROM AgniveerMaster a INNER JOIN AgniveerScoreAttempt sa ON a.Id = sa.AgniveerId WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL) AND sa.IsBestAttempt = 1 AND a.Sports = 'Volleyball' GROUP BY a.AgniveerNo, a.FullName ORDER BY BestTotal DESC\n\n"
    "Example 6:\n"
    "QUESTION: Show obese agniveers\n"
    "SQL: WITH LatestMedical AS (SELECT AgniveerId, Height, Weight, ROW_NUMBER() OVER (PARTITION BY AgniveerId ORDER BY VisitDate DESC) AS rn FROM MedicalRecordMaster WHERE Height IS NOT NULL AND Weight IS NOT NULL), Vitals AS (SELECT a.Id AS AgniveerId, a.AgniveerNo, a.FullName, COALESCE(lm.Height, a.Height) AS EffHeight, COALESCE(lm.Weight, a.Weight) AS EffWeight FROM AgniveerMaster a LEFT JOIN LatestMedical lm ON lm.AgniveerId = a.Id AND lm.rn = 1 WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)), Bmi AS (SELECT AgniveerNo, FullName, EffWeight / POWER(EffHeight / 100.0, 2) AS BmiValue FROM Vitals WHERE EffHeight IS NOT NULL AND EffWeight IS NOT NULL AND EffHeight > 0) SELECT AgniveerNo, FullName FROM Bmi WHERE BmiValue >= 30.0"
)


# ── Strength breakdown (deterministic — Tier 0, no AST/LLM involved) ───────
# "Strength breakdown" is a fixed-shape dashboard card (single row of named
# counts), always as-of-today — not a filterable row list — so it doesn't fit
# query_planner_v2's one-condition-set-per-query AST model. Each count is a
# scalar subquery so the whole card is one round trip.
@lru_cache(maxsize=1)
def _load_business_ontology() -> Dict[str, Any]:
    """Load the business ontology that maps concepts to tables and default filters."""
    ontology_path = Path(__file__).with_name("business_ontology.json")
    try:
        return json.loads(ontology_path.read_text(encoding="utf-8"))
    except Exception:
        return {"concepts": {}}


def _build_project_overview_context() -> str:
    """Build the compact prompt context injected into text2sql fallback."""
    ontology = _load_business_ontology().get("concepts", {})

    lines = [
        "PROJECT OVERVIEW:",
        "  User question -> intent classification -> deterministic executor if available -> SQL fallback -> validation -> read-only execution -> grounded answer.",
        "",
        "DETERMINISTIC EXECUTORS IN CODE:",
        "  - Performance / Overall: performance_executor.execute_performance_query handles Top, Bottom, OverallPerformance, BestAttempt, Average, AttemptWise, Trend, Improvement, Drop, Grading, GradingSummary.",
        "  - Medical / BMI: compute per agniveer from the latest MedicalRecordMaster row with fallback to AgniveerMaster Height/Weight, using BMI = Weight / POWER(Height / 100.0, 2).",
        "  - Strength: sql_executor._build_strength_breakdown_sql returns the fixed dashboard card for active/leave/disqualified counts.",
        "  - Verification: sql_executor has explicit SQL branches for Pending, NotResponded, Verified/Completed, Rejected, Sent, and the default status listing.",
        "  - Leave: sql_executor has explicit SQL branches for Current, Most, Least, and Threshold leave queries.",
        "  - Attendance: sql_executor builds the daily/weekly/monthly attendance calendar directly for a named agniveer.",
        "  - Everything else: the AST planner and sql_builder compile the intent before any text-to-SQL fallback is used.",
        "",
        "BUSINESS CONCEPT TO TABLE MAP:",
    ]

    for concept, meta in ontology.items():
        table = meta.get("table", "unknown_table")
        concept_type = meta.get("type", "unknown")
        default_date = meta.get("default_date_column") or "none"
        implicit_filters = meta.get("implicit_filters") or {}
        if implicit_filters:
            implicit_text = ", ".join(f"{k}={v}" for k, v in implicit_filters.items())
        else:
            implicit_text = "none"
        lines.append(
            f"  - {concept}: {table} [{concept_type}], default_date={default_date}, implicit_filters={implicit_text}"
        )

    lines.extend(
        [
            "",
            "LLM FALLBACK RULES:",
            "  - Prefer the exact tables, joins, and filters used by the codebase.",
            "  - Do not invent columns, aliases, or business meanings that conflict with the executor logic.",
            "  - If a query maps to a supported executor above, mirror that executor's semantics exactly.",
        ]
    )
    return "\n".join(lines)


def _build_strength_breakdown_sql() -> str:
    today_in_range = (
        "CAST(GETDATE() AS DATE) BETWEEN CAST(FromDate AS DATE) AND CAST(ToDate AS DATE)"
    )
    # Same ">90% leave taken" definition as the Leave/Threshold operation
    # above: total leave days >= 55, or any single leave >= 40 days.
    leave_threshold_sql = (
        "(SELECT COUNT(*) FROM ("
        "SELECT AgniveerId FROM AgniveerLeaveMaster GROUP BY AgniveerId "
        "HAVING SUM(DATEDIFF(day, FromDate, ToDate) + 1) >= 55 "
        "OR MAX(DATEDIFF(day, FromDate, ToDate) + 1) >= 40"
        ") t) AS LeaveThresholdCrossed"
    )
    return (
        "SELECT "
        "(SELECT COUNT(*) FROM AgniveerMaster WHERE IsActive = 1) "
        "- (SELECT COUNT(DISTINCT lm.AgniveerId) FROM AgniveerLeaveMaster lm "
        "INNER JOIN AgniveerMaster am ON am.Id = lm.AgniveerId "
        f"WHERE am.IsActive = 1 AND CAST(GETDATE() AS DATE) BETWEEN CAST(lm.FromDate AS DATE) AND CAST(lm.ToDate AS DATE)) "
        "AS ActiveAgniveer, "
        f"(SELECT COUNT(DISTINCT AgniveerId) FROM AgniveerLeaveMaster WHERE OnMedicalLeave = 1 AND {today_in_range}) AS MedicalLeave, "
        f"(SELECT COUNT(DISTINCT AgniveerId) FROM AgniveerLeaveMaster WHERE OnAnnualLeave = 1 AND {today_in_range}) AS AnnualLeave, "
        f"(SELECT COUNT(DISTINCT AgniveerId) FROM AgniveerLeaveMaster WHERE IsHospitalized = 1 AND {today_in_range}) AS Hospitalized, "
        f"(SELECT COUNT(DISTINCT AgniveerId) FROM AgniveerLeaveMaster WHERE [OnATTN'C'] = 1 AND {today_in_range}) AS AttnC, "
        f"(SELECT COUNT(DISTINCT AgniveerId) FROM AgniveerLeaveMaster WHERE [OnEX PPG] = 1 AND {today_in_range}) AS ExPpg, "
        f"(SELECT COUNT(DISTINCT AgniveerId) FROM AgniveerLeaveMaster WHERE IsAbscondedLeave = 1 AND {today_in_range}) AS Absconded, "
        f"{leave_threshold_sql}, "
        "(SELECT COUNT(*) FROM AgniveerMaster WHERE IsDisqualified = 1) AS DisqualifiedAgniveers"
    )


# ── Per-agniveer attendance calendar (deterministic — Tier 0) ──────────────
# Monthly/Weekly/Daily attendance for a NAMED agniveer is derived purely from
# AgniveerLeaveMaster, not AgniveerAttendanceMaster: every day in the
# requested range is Present unless a leave record's FromDate/ToDate spans
# that day, in which case it's Absent. SQL Server 2008 has no date-series
# generator, so the calendar is built with a recursive CTE.
_AGNIVEER_LOOKUP_SQL = "SELECT TOP (1) Id, AgniveerNo, FullName FROM AgniveerMaster WHERE AgniveerNo = ?"


def _build_attendance_calendar_sql() -> str:
    return (
        "WITH Dates AS ("
        "SELECT CAST(? AS DATE) AS AttendanceDate "
        "UNION ALL "
        "SELECT DATEADD(day, 1, AttendanceDate) FROM Dates WHERE AttendanceDate < CAST(? AS DATE)"
        ") "
        "SELECT AttendanceDate, "
        "CASE WHEN EXISTS ("
        "SELECT 1 FROM AgniveerLeaveMaster lm "
        "WHERE lm.AgniveerId = ? "
        "AND AttendanceDate BETWEEN CAST(lm.FromDate AS DATE) AND CAST(lm.ToDate AS DATE)"
        ") THEN 'Absent' ELSE 'Present' END AS Status "
        "FROM Dates "
        "OPTION (MAXRECURSION 366)"
    )


def _build_attendance_summary_sql() -> str:
    return (
        "WITH LeaveDays AS ("
        "SELECT DISTINCT lm.AgniveerId, d.AttendanceDate "
        "FROM AgniveerLeaveMaster lm "
        "CROSS APPLY ("
        "SELECT CAST(lm.FromDate AS DATE) AS AttendanceDate "
        "UNION ALL "
        "SELECT CAST(DATEADD(day, v.number, CAST(lm.FromDate AS DATE)) AS DATE) "
        "FROM master..spt_values v "
        "WHERE v.type = 'P' "
        "AND v.number > 0 "
        "AND CAST(DATEADD(day, v.number, CAST(lm.FromDate AS DATE)) AS DATE) <= CAST(lm.ToDate AS DATE)"
        ") d"
        ") "
        "SELECT COUNT(*) AS LeaveDayCount FROM LeaveDays"
    )


def _build_medical_bmi_sql(
    *,
    top_n: Optional[int],
    bmi_category: str = "",
    batch_id: Optional[int] = None,
    platoon_id: Optional[int] = None,
    company_id: Optional[int] = None,
    company_name: Optional[str] = None,
    agniveer_no: Optional[str] = None,
    agniveer_class: Optional[str] = None,
) -> Tuple[str, List[Any]]:
    """Build a deterministic per-agniveer BMI query from raw medical records."""
    clauses = ["(a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)"]
    params: List[Any] = []

    if batch_id is not None:
        clauses.append("a.BatchId = ?")
        params.append(int(batch_id))
    if agniveer_class is not None:
        clauses.append("LOWER(a.Class) = LOWER(?)")
        params.append(str(agniveer_class))
    if platoon_id is not None:
        clauses.append("a.PlatoonId = ?")
        params.append(int(platoon_id))
    if company_id is not None:
        clauses.append("c.Id = ?")
        params.append(int(company_id))
    if company_name is not None:
        clauses.append("LOWER(c.Name) = LOWER(?)")
        params.append(str(company_name))
    if agniveer_no is not None:
        clauses.append("LOWER(a.AgniveerNo) LIKE '%' + LOWER(?) + '%'")
        params.append(str(agniveer_no))

    category = bmi_category.strip().lower()
    category_filter = ""
    if category == "underweight":
        category_filter = "WHERE BmiValue < 18.5"
    elif category == "normal":
        category_filter = "WHERE BmiValue >= 18.5 AND BmiValue < 25.0"
    elif category == "overweight":
        category_filter = "WHERE BmiValue >= 25.0 AND BmiValue < 30.0"
    elif category == "obese":
        category_filter = "WHERE BmiValue >= 30.0"
    elif category == "unfit":
        # Unfit == Overweight or Obese combined — a contiguous BMI >= 25 range.
        category_filter = "WHERE BmiValue >= 25.0"

    if category:
        final_select = (
            f"SELECT {_top_clause(top_n)} "
            "AgniveerNo, FullName, "
            "CASE "
            "WHEN BmiValue IS NULL THEN NULL "
            "WHEN BmiValue < 18.5 THEN 'Underweight' "
            "WHEN BmiValue < 25.0 THEN 'Normal' "
            "WHEN BmiValue < 30.0 THEN 'Overweight' "
            "ELSE 'Obese' "
            "END AS BmiCategory "
            "FROM Scored "
            f"{category_filter} "
            "ORDER BY AgniveerNo ASC"
        )
    else:
        final_select = (
            f"SELECT {_top_clause(top_n)} "
            "AgniveerNo, FullName, BmiValue, "
            "CASE "
            "WHEN BmiValue IS NULL THEN NULL "
            "WHEN BmiValue < 18.5 THEN 'Underweight' "
            "WHEN BmiValue < 25.0 THEN 'Normal' "
            "WHEN BmiValue < 30.0 THEN 'Overweight' "
            "ELSE 'Obese' "
            "END AS BmiCategory "
            "FROM Scored "
            "WHERE BmiValue IS NOT NULL "
            "ORDER BY BmiValue DESC, AgniveerNo ASC"
        )

    sql = f"""
    WITH LatestMedical AS (
        SELECT
            mr.AgniveerId,
            mr.Height,
            mr.Weight,
            ROW_NUMBER() OVER (
                PARTITION BY mr.AgniveerId
                ORDER BY mr.VisitDate DESC, mr.Id DESC
            ) AS rn
        FROM MedicalRecordMaster mr
        WHERE mr.Height IS NOT NULL
            AND mr.Weight IS NOT NULL
    ),
    Vitals AS (
        SELECT
            a.Id AS AgniveerId,
            a.AgniveerNo,
            a.FullName,
            a.BatchId,
            a.Class,
            p.Id AS PlatoonId,
            p.CompanyId,
            c.Name AS CompanyName,
            COALESCE(lm.Height, a.Height) AS EffHeight,
            COALESCE(lm.Weight, a.Weight) AS EffWeight
        FROM AgniveerMaster a
            LEFT JOIN LatestMedical lm
                ON lm.AgniveerId = a.Id
                AND lm.rn = 1
            LEFT JOIN PlatoonMaster p
                ON p.Id = a.PlatoonId
            LEFT JOIN CompanyMaster c
                ON c.Id = p.CompanyId
        WHERE {" AND ".join(clauses)}
    ),
    Scored AS (
        SELECT
            AgniveerNo,
            FullName,
            CASE
                WHEN EffHeight IS NULL OR EffWeight IS NULL OR EffHeight <= 0 THEN NULL
                ELSE CAST(EffWeight / POWER(EffHeight / 100.0, 2) AS DECIMAL(10, 2))
            END AS BmiValue
        FROM Vitals
    )
    {final_select}
    """
    return sql, params


def _build_medical_blood_group_sql(
    *,
    top_n: Optional[int],
    report_mode: bool,
    blood_group: str = "",
    batch_id: Optional[int] = None,
    platoon_id: Optional[int] = None,
    company_id: Optional[int] = None,
    company_name: Optional[str] = None,
    platoon_name: Optional[str] = None,
    agniveer_no: Optional[str] = None,
    agniveer_class: Optional[str] = None,
) -> Tuple[str, List[Any]]:
    """Build a deterministic blood-group query from AgniveerMaster."""
    clauses = ["(a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)"]
    params: List[Any] = []

    if batch_id is not None:
        clauses.append("a.BatchId = ?")
        params.append(int(batch_id))
    if agniveer_class is not None:
        clauses.append("LOWER(a.Class) = LOWER(?)")
        params.append(str(agniveer_class))
    if platoon_id is not None:
        clauses.append("a.PlatoonId = ?")
        params.append(int(platoon_id))
    if company_id is not None:
        clauses.append("c.Id = ?")
        params.append(int(company_id))
    if company_name is not None:
        clauses.append("LOWER(c.Name) = LOWER(?)")
        params.append(str(company_name))
    if platoon_name is not None:
        clauses.append("LOWER(p.Name) = LOWER(?)")
        params.append(str(platoon_name))
    if agniveer_no is not None:
        clauses.append("LOWER(a.AgniveerNo) LIKE '%' + LOWER(?) + '%'")
        params.append(str(agniveer_no))
    if blood_group:
        clauses.append("LOWER(COALESCE(NULLIF(a.BloodGroup, ''), 'Unknown')) = LOWER(?)")
        params.append(blood_group)

    sql = f"""
    WITH Scoped AS (
        SELECT
            a.AgniveerNo,
            a.FullName,
            COALESCE(NULLIF(a.BloodGroup, ''), 'Unknown') AS BloodGroup
        FROM AgniveerMaster a
            LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
            LEFT JOIN CompanyMaster c ON c.Id = p.CompanyId
        WHERE {" AND ".join(clauses)}
    )
    """

    if report_mode:
        sql += f"""
        SELECT
            BloodGroup,
            COUNT(*) AS AgniveerCount
        FROM Scoped
        GROUP BY BloodGroup
        ORDER BY AgniveerCount DESC, BloodGroup ASC
        """
    else:
        sql += f"""
        SELECT {_top_clause(top_n)}
            AgniveerNo,
            FullName,
            BloodGroup
        FROM Scoped
        ORDER BY AgniveerNo ASC
        """

    return sql, params


# ── Company Schedule (deterministic — Tier 0) ──────────────────────────────
# CompanySchedule only carries a CompanyId column (no PlatoonId/AgniveerId),
# so a schedule asked "for a platoon" or "for an agniveer" must first resolve
# up the chain to the company that platoon/agniveer belongs to:
#   AgniveerMaster.AgniveerNo -> AgniveerMaster.PlatoonId
#   -> PlatoonMaster.Id/CompanyId -> CompanyMaster.Id -> CompanySchedule.CompanyId
_COMPANY_ID_BY_NAME_SQL = (
    "SELECT TOP (1) Id AS CompanyId FROM CompanyMaster WHERE LOWER(Name) = LOWER(?)"
)
_COMPANY_ID_BY_PLATOON_ID_SQL = (
    "SELECT TOP (1) CompanyId FROM PlatoonMaster WHERE Id = ?"
)
_COMPANY_ID_BY_PLATOON_NAME_SQL = (
    "SELECT TOP (1) CompanyId FROM PlatoonMaster WHERE LOWER(Name) = LOWER(?)"
)
_COMPANY_ID_BY_AGNIVEER_NO_SQL = (
    "SELECT TOP (1) p.CompanyId AS CompanyId "
    "FROM AgniveerMaster a "
    "LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId "
    "WHERE LOWER(a.AgniveerNo) = LOWER(?)"
)


def resolve_company_id_from_agniveer(agniveer_no: str) -> Optional[int]:
    """Resolve company ID from Agniveer number."""
    sql = """
    SELECT TOP (1) p.CompanyId AS CompanyId
    FROM AgniveerMaster a
    LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
    WHERE LOWER(a.AgniveerNo) = LOWER(?)
    """
    rows, err = run_readonly(sql, [agniveer_no])
    if err or not rows:
        return None
    return rows[0].get("CompanyId")


def resolve_company_id_from_platoon(platoon_id: int) -> Optional[int]:
    """Resolve company ID from platoon ID."""
    sql = "SELECT TOP (1) CompanyId FROM PlatoonMaster WHERE Id = ?"
    rows, err = run_readonly(sql, [platoon_id])
    if err or not rows:
        return None
    return rows[0].get("CompanyId")


def resolve_company_id_from_name(company_name: str) -> Optional[int]:
    """Resolve company ID from company name."""
    sql = "SELECT TOP (1) Id AS CompanyId FROM CompanyMaster WHERE LOWER(Name) = LOWER(?)"
    rows, err = run_readonly(sql, [company_name])
    if err or not rows:
        return None
    return rows[0].get("CompanyId")


def build_schedule_sql(
    company_id: Optional[int] = None,
    date: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    top_n: Optional[int] = None
) -> Tuple[str, List[Any]]:
    """Build schedule SQL with proper filters."""
    
    clauses = []
    params = []

    # ── Company Filter ──────────────────────────────────────────────────
    if company_id is not None:
        clauses.append("s.CompanyId = ?")
        params.append(int(company_id))

    # ── Date Filters ──────────────────────────────────────────────────
    if from_date or to_date:
        if from_date:
            clauses.append("CAST(s.ScheduleDate AS DATE) >= CAST(? AS DATE)")
            params.append(str(from_date)[:10])
        if to_date:
            clauses.append("CAST(s.ScheduleDate AS DATE) <= CAST(? AS DATE)")
            params.append(str(to_date)[:10])
    elif date:
        clauses.append("CAST(s.ScheduleDate AS DATE) = CAST(? AS DATE)")
        params.append(str(date)[:10])

    where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    sql = f"""
    SELECT {_top_clause(top_n)}
        s.Id AS ScheduleId,
        s.CompanyId,
        c.Name AS CompanyName,
        s.ScheduleDate,
        s.Pd,
        s.TimeRange,
        s.Code,
        s.Type,
        s.Details,
        s.Location,
        s.Resp
    FROM CompanySchedule s
    LEFT JOIN CompanyMaster c ON c.Id = s.CompanyId
    {where_clause}
    ORDER BY c.Name ASC, s.ScheduleDate ASC, s.Pd ASC
    """
    return sql, params


def _execute_schedule_query(intent: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
    """Dispatch schedule query based on intent."""
    
    operation = str(intent.get("operation") or "bytoday").lower()
    
    # ── Resolve Company ID ─────────────────────────────────────────────
    company_id = None
    
    # Priority: Direct ID > AgniveerNo > PlatoonId > CompanyName
    if intent.get("company_id") or intent.get("companyId"):
        company_id = int(intent.get("company_id") or intent.get("companyId"))
    elif intent.get("agniveer_no") or intent.get("agniveerNo"):
        ag_no = intent.get("agniveer_no") or intent.get("agniveerNo")
        company_id = resolve_company_id_from_agniveer(ag_no)
    elif intent.get("platoon_id") or intent.get("platoonId"):
        pl_id = int(intent.get("platoon_id") or intent.get("platoonId"))
        company_id = resolve_company_id_from_platoon(pl_id)
    elif intent.get("company_name") or intent.get("companyName"):
        name = intent.get("company_name") or intent.get("companyName")
        company_id = resolve_company_id_from_name(name)
    
    # ── Resolve Date ──────────────────────────────────────────────────
    from_date = intent.get("from_date") or intent.get("fromDate")
    to_date = intent.get("to_date") or intent.get("toDate")
    date = intent.get("date")
    
    # Default to today for "bytoday" operation
    if operation in ("bytoday", "today", "now", "current"):
        if not date and not from_date and not to_date:
            date = datetime.date.today().isoformat()
    
    # ── Build and Execute SQL ──────────────────────────────────────────
    top_n = _get_top_n(intent)
    sql, params = build_schedule_sql(
        company_id=company_id,
        date=date,
        from_date=from_date,
        to_date=to_date,
        top_n=top_n,
    )
    
    rows, err = run_readonly(sql, params, max_rows=_row_cap(top_n))
    if err:
        return None, err

    return _to_section(rows or [], intent, sql=sql), None


# ── Disqualified Agniveers ──────────────────────────────────────────────────

def _build_disqualified_base_query(intent: Dict[str, Any]) -> Tuple[str, List[Any]]:
    """
    Build the base WHERE clause for disqualified queries.
    Returns (where_clause, params).
    """
    clauses = ["a.IsDisqualified = 1"]
    params: List[Any] = []

    # ── Scope filters ────────────────────────────────────────────────────────
    agniveer_no = intent.get("agniveer_no") or intent.get("agniveerNo")
    batch_id = intent.get("batch_id") or intent.get("batchId")
    platoon_id = intent.get("platoon_id") or intent.get("platoonId")
    company_id = intent.get("company_id") or intent.get("companyId")
    from_date = intent.get("from_date") or intent.get("fromDate")
    to_date = intent.get("to_date") or intent.get("toDate")
    
    if agniveer_no:
        clauses.append("LOWER(a.AgniveerNo) LIKE '%' + LOWER(?) + '%'")
        params.append(str(agniveer_no))
    
    if batch_id is not None:
        clauses.append("a.BatchId = ?")
        params.append(int(batch_id))
    
    if platoon_id is not None:
        clauses.append("a.PlatoonId = ?")
        params.append(int(platoon_id))
    
    if company_id is not None:
        clauses.append("EXISTS (SELECT 1 FROM PlatoonMaster p WHERE p.Id = a.PlatoonId AND p.CompanyId = ?)")
        params.append(int(company_id))
    
    if from_date:
        clauses.append("CAST(a.DisqualifiedDate AS DATE) >= CAST(? AS DATE)")
        params.append(str(from_date)[:10])
    
    if to_date:
        clauses.append("CAST(a.DisqualifiedDate AS DATE) <= CAST(? AS DATE)")
        params.append(str(to_date)[:10])
    
    # ── Leave filter ──────────────────────────────────────────────────────────
    leave_type = intent.get("leave_type") or intent.get("leaveType")
    if leave_type and str(leave_type).lower() in ("leave", "on leave", "any"):
        clauses.append("""
            EXISTS (
                SELECT 1 FROM AgniveerLeaveMaster l
                WHERE l.AgniveerId = a.Id
                    AND l.FromDate IS NOT NULL
            )
        """)
    
    where_clause = " AND ".join(clauses)
    return where_clause, params


def _build_disqualified_select_clause(detailed: bool = False) -> str:
    """
    Build the SELECT clause for disqualified queries.
    """
    if detailed:
        return """
            a.Id,
            a.AgniveerNo,
            a.FullName,
            a.DateOfBirth,
            a.DateOfJoining,
            a.PhotoPath,
            a.Address,
            a.EroName,
            a.NextOfKin,
            a.MobileNo,
            a.Class,
            p.Name AS PlatoonName,
            c.Name AS CompanyName,
            b.BatchName,
            a.DisqualifiedDate,
            a.Remarks
        """
    return "COUNT(*) AS TotalDisqualified"


def execute_disqualified_query(intent: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
    """
    Execute Disqualified Agniveers query.
    
    Returns:
        - Summary: {"totalDisqualified": count}
        - Detailed: List of disqualified Agniveers with full details
    """
    try:
        raw_q = str(intent.get("raw_query") or "").lower()
        response_type = str(intent.get("responseType") or intent.get("response_type") or intent.get("operation") or "").lower()
        
        # Check if user asks for count/summary vs detailed list
        wants_count = (
            response_type in ("summary", "count", "disqualifiedcount")
            or any(kw in raw_q for kw in ("how many", "count", "number of", "total disqualified", "total number"))
        )
        detailed = not wants_count if response_type != "detailed" else True
        top_n = _get_top_n(intent)
        
        # ── Build WHERE clause ──────────────────────────────────────────────────
        where_clause, params = _build_disqualified_base_query(intent)
        
        # ── Build SELECT clause ─────────────────────────────────────────────────
        select_clause = _build_disqualified_select_clause(detailed)
        
        # ── Build JOINs ─────────────────────────────────────────────────────────
        joins = """
            LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
            LEFT JOIN CompanyMaster c ON c.Id = p.CompanyId
            LEFT JOIN BatchMaster b ON b.Id = a.BatchId
        """
        
        # ── Build final SQL ─────────────────────────────────────────────────────
        if detailed:
            sql = f"""
            SELECT {_top_clause(top_n)}
                {select_clause}
            FROM AgniveerMaster a
            {joins}
            WHERE {where_clause}
            ORDER BY a.AgniveerNo ASC
            """
        else:
            sql = f"""
            SELECT {select_clause}
            FROM AgniveerMaster a
            WHERE {where_clause}
            """
        
        # ── Validate ────────────────────────────────────────────────────────────
        is_valid, err = sql_validator.validate_sql(sql)
        if not is_valid:
            return None, f"Disqualified SQL validation failed: {err}"
        
        # ── Execute ─────────────────────────────────────────────────────────────
        rows, run_err = run_readonly(sql, params, max_rows=_row_cap(top_n))
        if run_err:
            return None, f"Disqualified execution failed: {run_err}"
        
        # ── Build response ──────────────────────────────────────────────────────
        if not detailed:
            # Summary: return count
            count = rows[0].get("TotalDisqualified", 0) if rows else 0
            result = {"totalDisqualified": count}
            return _to_section([result], intent, sql=sql), None
        
        # Detailed: return list
        return _to_section(rows or [], intent, sql=sql), None
        
    except Exception as exc:
        logger.error("Disqualified query failed: %s", exc, exc_info=True)
        return None, str(exc)


# ── Verification (Police Verification) ──────────────────────────────────────

def build_verification_sql(
    status: str,
    agniveer_no: Optional[str] = None,
    batch_id: Optional[int] = None,
    platoon_id: Optional[int] = None,
    company_id: Optional[int] = None,
    detailed: bool = False,
    top_n: Optional[int] = None
) -> Tuple[str, List[Any]]:
    """
    Build Verification SQL based on status.
    status: Pending, Sent, NotResponded, Verified, Rejected, Summary
    """
    
    # ── Base Agniveer scope ──────────────────────────────────────────────────
    scope_clauses = ["(a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)", "a.IsActive = 1"]
    scope_params: List[Any] = []
    
    if agniveer_no:
        scope_clauses.append("LOWER(a.AgniveerNo) LIKE '%' + LOWER(?) + '%'")
        scope_params.append(str(agniveer_no))
    if batch_id is not None:
        scope_clauses.append("a.BatchId = ?")
        scope_params.append(int(batch_id))
    if platoon_id is not None:
        scope_clauses.append("a.PlatoonId = ?")
        scope_params.append(int(platoon_id))
    if company_id is not None:
        scope_clauses.append("EXISTS (SELECT 1 FROM PlatoonMaster p WHERE p.Id = a.PlatoonId AND p.CompanyId = ?)")
        scope_params.append(int(company_id))
    
    scope_where = " AND ".join(scope_clauses)
    status_lower = str(status or "").lower().replace(" ", "").replace("_", "")
    
    # ── Build SQL based on status ─────────────────────────────────────────────
    if status_lower in ("pending", "pendingverification", "unverified", "awaitingverification", "notverified", "notverifiedyet", "waitingforverification"):
        # Pending = No record OR Rejected
        sql = f"""
        SELECT {_top_clause(top_n)}
            a.Id,
            a.AgniveerNo,
            a.FullName,
            a.PhotoPath,
            a.Class,
            p.Name AS PlatoonName,
            c.Name AS CompanyName,
            b.BatchName,
            pv.PoliceStation,
            pv.SentDate,
            pv.ReceivedDate,
            pv.Status
        FROM AgniveerMaster a
        LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
        LEFT JOIN CompanyMaster c ON c.Id = p.CompanyId
        LEFT JOIN BatchMaster b ON b.Id = a.BatchId
        LEFT JOIN PoliceVerificationMaster pv ON pv.AgniveerId = a.Id
        WHERE {scope_where}
            AND (pv.Status = 'Rejected' OR pv.AgniveerId IS NULL)
        ORDER BY a.AgniveerNo ASC
        """
        return sql, scope_params
    
    elif status_lower in ("sent", "sentverification", "dispatched", "dispatchedforverification", "requestsent", "verificationrequested"):
        # Sent = Has a record (not Pending)
        sql = f"""
        WITH LatestVerification AS (
            SELECT
                v.AgniveerId,
                v.PoliceStation,
                v.SentDate,
                v.ReceivedDate,
                v.Status,
                ROW_NUMBER() OVER (PARTITION BY v.AgniveerId ORDER BY v.SentDate DESC, v.Id DESC) AS rn
            FROM PoliceVerificationMaster v
        )
        SELECT {_top_clause(top_n)}
            a.AgniveerNo,
            a.FullName,
            a.PhotoPath,
            a.Class,
            p.Name AS PlatoonName,
            c.Name AS CompanyName,
            b.BatchName,
            lv.PoliceStation,
            lv.SentDate,
            lv.ReceivedDate,
            lv.Status
        FROM AgniveerMaster a
        LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
        LEFT JOIN CompanyMaster c ON c.Id = p.CompanyId
        LEFT JOIN BatchMaster b ON b.Id = a.BatchId
        INNER JOIN LatestVerification lv ON lv.AgniveerId = a.Id AND lv.rn = 1
        WHERE {scope_where}
            AND lv.Status != 'Pending'
        ORDER BY lv.SentDate DESC
        """
        return sql, scope_params
    
    elif status_lower in ("notresponded", "noresponse", "unresponded", "awaitingresponse", "pendingresponse", "responsepending", "noreply", "noreplyyet", "notrespondedverification", "unresponsive"):
        sql = f"""
        WITH LatestVerification AS (
            SELECT
                v.AgniveerId,
                v.PoliceStation,
                v.SentDate,
                v.ReceivedDate,
                v.Status,
                ROW_NUMBER() OVER (PARTITION BY v.AgniveerId ORDER BY v.SentDate DESC, v.Id DESC) AS rn
            FROM PoliceVerificationMaster v
        )
        SELECT {_top_clause(top_n)}
            a.AgniveerNo,
            a.FullName,
            a.PhotoPath,
            a.Class,
            p.Name AS PlatoonName,
            c.Name AS CompanyName,
            b.BatchName,
            lv.PoliceStation,
            lv.SentDate,
            lv.ReceivedDate,
            DATEDIFF(DAY, lv.SentDate, GETDATE()) AS DaysSinceSent
        FROM AgniveerMaster a
        LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
        LEFT JOIN CompanyMaster c ON c.Id = p.CompanyId
        LEFT JOIN BatchMaster b ON b.Id = a.BatchId
        INNER JOIN LatestVerification lv ON lv.AgniveerId = a.Id AND lv.rn = 1
        WHERE {scope_where}
            AND lv.Status = 'Sent'
            AND lv.ReceivedDate IS NULL
        ORDER BY lv.SentDate ASC
        """
        return sql, scope_params
    
    elif status_lower in ("verified", "completed", "completedverification", "verifiedverification", "allverified", "fullyverified", "verificationdone", "cleared"):
        sql = f"""
        WITH LatestVerification AS (
            SELECT
                v.AgniveerId,
                v.PoliceStation,
                v.SentDate,
                v.ReceivedDate,
                v.Status,
                ROW_NUMBER() OVER (PARTITION BY v.AgniveerId ORDER BY v.SentDate DESC, v.Id DESC) AS rn
            FROM PoliceVerificationMaster v
        )
        SELECT {_top_clause(top_n)}
            a.AgniveerNo,
            a.FullName,
            a.PhotoPath,
            a.Class,
            p.Name AS PlatoonName,
            c.Name AS CompanyName,
            b.BatchName,
            lv.PoliceStation,
            lv.SentDate,
            lv.ReceivedDate,
            DATEDIFF(DAY, lv.SentDate, lv.ReceivedDate) AS DaysToRespond
        FROM AgniveerMaster a
        LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
        LEFT JOIN CompanyMaster c ON c.Id = p.CompanyId
        LEFT JOIN BatchMaster b ON b.Id = a.BatchId
        INNER JOIN LatestVerification lv ON lv.AgniveerId = a.Id AND lv.rn = 1
        WHERE {scope_where}
            AND lv.Status = 'Verified'
        ORDER BY lv.ReceivedDate DESC
        """
        return sql, scope_params
    
    elif status_lower in ("rejected", "rejectedverification", "failedverification", "verificationfailed", "denied", "verificationdenied"):
        sql = f"""
        WITH LatestVerification AS (
            SELECT
                v.AgniveerId,
                v.PoliceStation,
                v.SentDate,
                v.ReceivedDate,
                v.Status,
                ROW_NUMBER() OVER (PARTITION BY v.AgniveerId ORDER BY v.SentDate DESC, v.Id DESC) AS rn
            FROM PoliceVerificationMaster v
        )
        SELECT {_top_clause(top_n)}
            a.AgniveerNo,
            a.FullName,
            a.PhotoPath,
            a.Class,
            p.Name AS PlatoonName,
            c.Name AS CompanyName,
            b.BatchName,
            lv.PoliceStation,
            lv.SentDate,
            lv.ReceivedDate
        FROM AgniveerMaster a
        LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
        LEFT JOIN CompanyMaster c ON c.Id = p.CompanyId
        LEFT JOIN BatchMaster b ON b.Id = a.BatchId
        INNER JOIN LatestVerification lv ON lv.AgniveerId = a.Id AND lv.rn = 1
        WHERE {scope_where}
            AND lv.Status = 'Rejected'
        ORDER BY lv.ReceivedDate DESC
        """
        return sql, scope_params
    
    else:
        # Default: Summary
        sql = f"""
        SELECT
            COUNT(*) AS TotalAgniveers,
            SUM(CASE 
                WHEN NOT EXISTS (SELECT 1 FROM PoliceVerificationMaster pv2 WHERE pv2.AgniveerId = a.Id)
                    OR EXISTS (SELECT 1 FROM PoliceVerificationMaster pv2 WHERE pv2.AgniveerId = a.Id AND pv2.Status = 'Rejected')
                THEN 1 ELSE 0 
            END) AS PendingCount,
            COUNT(pv.Id) AS SentCount,
            SUM(CASE WHEN pv.Status = 'Sent' AND pv.ReceivedDate IS NULL THEN 1 ELSE 0 END) AS NotRespondedCount,
            SUM(CASE WHEN pv.Status = 'Verified' THEN 1 ELSE 0 END) AS VerifiedCount,
            SUM(CASE WHEN pv.Status = 'Rejected' THEN 1 ELSE 0 END) AS RejectedCount
        FROM AgniveerMaster a
        LEFT JOIN PoliceVerificationMaster pv ON pv.AgniveerId = a.Id
        WHERE {scope_where}
        """
        return sql, scope_params


def execute_verification_query(intent: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
    """
    Execute Verification query based on status.
    """
    try:
        status = intent.get("operation") or intent.get("verification_status") or intent.get("verificationStatus") or "Summary"
        agniveer_no = intent.get("agniveer_no") or intent.get("agniveerNo")
        batch_id = intent.get("batch_id") or intent.get("batchId")
        platoon_id = intent.get("platoon_id") or intent.get("platoonId")
        company_id = intent.get("company_id") or intent.get("companyId")
        response_type = intent.get("responseType") or intent.get("response_type") or "Summary"
        top_n = _get_top_n(intent)
        
        detailed = str(response_type).lower() == "detailed"
        
        sql, params = build_verification_sql(
            status=status,
            agniveer_no=agniveer_no,
            batch_id=batch_id,
            platoon_id=platoon_id,
            company_id=company_id,
            detailed=detailed,
            top_n=top_n
        )
        
        is_valid, err = sql_validator.validate_sql(sql)
        if not is_valid:
            return None, f"Verification SQL validation failed: {err}"
        
        rows, run_err = run_readonly(sql, params, max_rows=_row_cap(top_n))
        if run_err:
            return None, f"Verification execution failed: {run_err}"
        
        return _to_section(rows or [], intent, sql=sql), None
        
    except Exception as exc:
        logger.error("Verification query failed: %s", exc, exc_info=True)
        return None, str(exc)


# ── Leave Category ──────────────────────────────────────────────────────────

LEAVE_COLUMN_MAP = {
    "annual": "OnAnnualLeave",
    "medical": "OnMedicalLeave",
    "sick": "OnSickLeave",
    "hospitalized": "IsHospitalized",
    "absconded": "IsAbscondedLeave",
    "attnc": "OnATTN'C'",
    "exppg": "OnEX PPG",
}


def _get_leave_column(leave_type: str) -> str:
    """Get the column name for a leave type."""
    if leave_type:
        column = LEAVE_COLUMN_MAP.get(str(leave_type).lower())
        if column:
            return column
    return ""


def _calculate_leave_count(is_exppg: bool, from_date: Any, to_date: Any) -> int:
    """Calculate leave count based on leave type."""
    if from_date is None or to_date is None:
        return 0
    days = (to_date - from_date).days + 1
    return days // 4 if is_exppg else days


def _build_leave_base_query(intent: Dict[str, Any]) -> Tuple[str, List[Any]]:
    """
    Build the base WHERE clause for leave queries.
    Returns (where_clause, params).
    """
    clauses = ["(a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)", "a.IsActive = 1"]
    params: List[Any] = []

    # ── Scope filters ──────────────────────────────────────────────────────
    agniveer_no = intent.get("agniveer_no") or intent.get("agniveerNo")
    batch_id = intent.get("batch_id") or intent.get("batchId")
    platoon_id = intent.get("platoon_id") or intent.get("platoonId")
    company_id = intent.get("company_id") or intent.get("companyId")
    from_date = intent.get("from_date") or intent.get("fromDate")
    to_date = intent.get("to_date") or intent.get("toDate")

    if agniveer_no:
        clauses.append("LOWER(a.AgniveerNo) LIKE '%' + LOWER(?) + '%'")
        params.append(str(agniveer_no))

    if batch_id is not None:
        clauses.append("a.BatchId = ?")
        params.append(int(batch_id))

    if platoon_id is not None:
        clauses.append("a.PlatoonId = ?")
        params.append(int(platoon_id))

    if company_id is not None:
        clauses.append("EXISTS (SELECT 1 FROM PlatoonMaster p WHERE p.Id = a.PlatoonId AND p.CompanyId = ?)")
        params.append(int(company_id))

    # ── Date range filters ────────────────────────────────────────────────
    if from_date:
        clauses.append("CAST(l.FromDate AS DATE) >= CAST(? AS DATE)")
        params.append(str(from_date)[:10])
    if to_date:
        clauses.append("CAST(l.ToDate AS DATE) <= CAST(? AS DATE)")
        params.append(str(to_date)[:10])

    # ── Leave type filter ──────────────────────────────────────────────────
    leave_type = intent.get("leave_type") or intent.get("leaveType")
    if leave_type and str(leave_type).lower() not in ("threshold", "noleave"):
        column = _get_leave_column(leave_type)
        if column:
            col_ref = f"l.[{column}]" if (" " in column or "'" in column) else f"l.{column}"
            clauses.append(f"{col_ref} = 1")

    where_clause = " AND ".join(clauses)
    return where_clause, params


def _execute_leave_current(intent: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Execute Current Leave query.
    Returns Agniveers currently on leave (FromDate <= today <= ToDate).
    Excludes absconded by default.
    """
    try:
        top_n = _get_top_n(intent)
        response_type = str(intent.get("responseType") or intent.get("response_type") or "Summary")
        detailed = response_type.lower() == "detailed"

        # ── Base scope ──────────────────────────────────────────────────────
        base_where, base_params = _build_leave_base_query(intent)

        # ── Leave type filter ──────────────────────────────────────────────
        leave_type = intent.get("leave_type") or intent.get("leaveType")
        leave_clause = ""
        leave_params: List[Any] = []
        
        if leave_type and str(leave_type).lower() not in ("threshold", "noleave"):
            column = _get_leave_column(leave_type)
            if column:
                col_ref = f"l.[{column}]" if (" " in column or "'" in column) else f"l.{column}"
                leave_clause = f" AND {col_ref} = 1"

        # ── Current date filter ─────────────────────────────────────────────
        date_filter = "AND CAST(GETDATE() AS DATE) BETWEEN CAST(l.FromDate AS DATE) AND CAST(l.ToDate AS DATE)"
        
        from_date = intent.get("from_date") or intent.get("fromDate")
        to_date = intent.get("to_date") or intent.get("toDate")
        if from_date and to_date:
            date_filter = "AND CAST(l.FromDate AS DATE) <= CAST(? AS DATE) AND CAST(l.ToDate AS DATE) >= CAST(? AS DATE)"
            leave_params = [str(to_date)[:10], str(from_date)[:10]] + leave_params

        # ── Short/Summary ──────────────────────────────────────────────────
        if not detailed:
            sql = f"""
            SELECT
                COUNT(*) AS OnLeaveCount,
                SUM(CASE WHEN l.OnAnnualLeave = 1 THEN 1 ELSE 0 END) AS AnnualLeave,
                SUM(CASE WHEN l.OnMedicalLeave = 1 THEN 1 ELSE 0 END) AS MedicalLeave,
                SUM(CASE WHEN l.OnSickLeave = 1 THEN 1 ELSE 0 END) AS SickLeave,
                SUM(CASE WHEN l.IsHospitalized = 1 THEN 1 ELSE 0 END) AS Hospitalized,
                SUM(CASE WHEN l.[OnATTN'C'] = 1 THEN 1 ELSE 0 END) AS ATTNC,
                SUM(CASE WHEN l.[OnEX PPG] = 1 THEN 1 ELSE 0 END) AS EXPPG
            FROM AgniveerLeaveMaster l
            INNER JOIN AgniveerMaster a ON a.Id = l.AgniveerId
            WHERE {base_where}
                AND l.FromDate IS NOT NULL
                AND l.ToDate IS NOT NULL
                AND ISNULL(l.IsAbscondedLeave, 0) != 1
                {date_filter}
                AND (l.OnAnnualLeave = 1 OR l.OnMedicalLeave = 1 OR l.OnSickLeave = 1
                     OR l.IsHospitalized = 1 OR l.[OnATTN'C'] = 1 OR l.[OnEX PPG] = 1)
                {leave_clause}
            """
            
            is_valid, err = sql_validator.validate_sql(sql)
            if not is_valid:
                return None, f"Current Leave SQL validation failed: {err}"
            
            all_params = base_params + leave_params
            rows, run_err = run_readonly(sql, all_params, max_rows=_row_cap(top_n))
            if run_err:
                return None, f"Current Leave execution failed: {run_err}"
            
            row = rows[0] if rows else {}
            result = {
                "onLeaveCount": row.get("OnLeaveCount") or 0,
                "annualLeave": row.get("AnnualLeave") or 0,
                "medicalLeave": row.get("MedicalLeave") or 0,
                "sickLeave": row.get("SickLeave") or 0,
                "hospitalized": row.get("Hospitalized") or 0,
                "attnc": row.get("ATTNC") or 0,
                "exppg": row.get("EXPPG") or 0,
            }
            return _to_section([result], intent, sql=sql), None

        # ── Detailed ────────────────────────────────────────────────────────
        sql = f"""
        SELECT {_top_clause(top_n)}
            a.AgniveerNo,
            a.FullName,
            a.PhotoPath,
            a.Class,
            p.Name AS PlatoonName,
            c.Name AS CompanyName,
            b.BatchName,
            l.FromDate,
            l.ToDate,
            DATEDIFF(DAY, l.FromDate, l.ToDate) + 1 AS LeaveDays,
            CASE
                WHEN l.OnAnnualLeave = 1 THEN 'Annual'
                WHEN l.OnMedicalLeave = 1 THEN 'Medical'
                WHEN l.OnSickLeave = 1 THEN 'Sick'
                WHEN l.IsHospitalized = 1 THEN 'Hospitalized'
                WHEN l.[OnATTN'C'] = 1 THEN 'ATTNC'
                WHEN l.[OnEX PPG] = 1 THEN 'EX PPG'
                ELSE 'Unknown'
            END AS LeaveType,
            l.Remarks
        FROM AgniveerLeaveMaster l
        INNER JOIN AgniveerMaster a ON a.Id = l.AgniveerId
        LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
        LEFT JOIN CompanyMaster c ON c.Id = p.CompanyId
        LEFT JOIN BatchMaster b ON b.Id = a.BatchId
        WHERE {base_where}
            AND l.FromDate IS NOT NULL
            AND l.ToDate IS NOT NULL
            AND ISNULL(l.IsAbscondedLeave, 0) != 1
            {date_filter}
            AND (l.OnAnnualLeave = 1 OR l.OnMedicalLeave = 1 OR l.OnSickLeave = 1
                 OR l.IsHospitalized = 1 OR l.[OnATTN'C'] = 1 OR l.[OnEX PPG] = 1)
            {leave_clause}
        ORDER BY a.AgniveerNo ASC
        """
        
        is_valid, err = sql_validator.validate_sql(sql)
        if not is_valid:
            return None, f"Current Leave SQL validation failed: {err}"
        
        all_params = base_params + leave_params
        rows, run_err = run_readonly(sql, all_params, max_rows=_row_cap(top_n))
        if run_err:
            return None, f"Current Leave execution failed: {run_err}"
        
        return _to_section(rows or [], intent, sql=sql), None

    except Exception as exc:
        logger.error("Current Leave failed: %s", exc, exc_info=True)
        return None, str(exc)


def _execute_leave_most(intent: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Execute Most Leave Taken query.
    Returns Agniveers with highest leave counts.
    """
    try:
        top_n = _get_top_n(intent)
        response_type = str(intent.get("responseType") or intent.get("response_type") or "Summary")
        detailed = response_type.lower() == "detailed"

        # ── Base scope ──────────────────────────────────────────────────────
        base_where, base_params = _build_leave_base_query(intent)

        # ── Leave type filter ──────────────────────────────────────────────
        leave_type = intent.get("leave_type") or intent.get("leaveType")
        leave_clause = ""
        leave_params: List[Any] = []
        if leave_type and str(leave_type).lower() not in ("threshold", "noleave"):
            column = _get_leave_column(leave_type)
            if column:
                col_ref = f"l.[{column}]" if (" " in column or "'" in column) else f"l.{column}"
                leave_clause = f" AND {col_ref} = 1"

        # ── Short/Summary ──────────────────────────────────────────────────
        if not detailed:
            sql = f"""
            SELECT
                COUNT(DISTINCT l.AgniveerId) AS TotalAgniveers,
                SUM(
                    CASE
                        WHEN l.[OnEX PPG] = 1 THEN (DATEDIFF(DAY, l.FromDate, l.ToDate) + 1) / 4
                        ELSE DATEDIFF(DAY, l.FromDate, l.ToDate) + 1
                    END
                ) AS TotalLeaveDays
            FROM AgniveerLeaveMaster l
            INNER JOIN AgniveerMaster a ON a.Id = l.AgniveerId
            WHERE {base_where}
                AND l.FromDate IS NOT NULL
                AND l.ToDate IS NOT NULL
                {leave_clause}
            """
            
            is_valid, err = sql_validator.validate_sql(sql)
            if not is_valid:
                return None, f"Most Leave SQL validation failed: {err}"
            
            all_params = base_params + leave_params
            rows, run_err = run_readonly(sql, all_params, max_rows=_row_cap(top_n))
            if run_err:
                return None, f"Most Leave execution failed: {run_err}"
            
            row = rows[0] if rows else {}
            result = {
                "totalAgniveers": row.get("TotalAgniveers") or 0,
                "totalLeaveDays": row.get("TotalLeaveDays") or 0,
            }
            return _to_section([result], intent, sql=sql), None

        # ── Detailed ────────────────────────────────────────────────────────
        sql = f"""
        SELECT {_top_clause(top_n)}
            a.AgniveerNo,
            a.FullName,
            a.PhotoPath,
            a.Class,
            p.Name AS PlatoonName,
            c.Name AS CompanyName,
            b.BatchName,
            SUM(
                CASE
                    WHEN l.[OnEX PPG] = 1 THEN (DATEDIFF(DAY, l.FromDate, l.ToDate) + 1) / 4
                    ELSE DATEDIFF(DAY, l.FromDate, l.ToDate) + 1
                END
            ) AS TotalLeaveDays
        FROM AgniveerLeaveMaster l
        INNER JOIN AgniveerMaster a ON a.Id = l.AgniveerId
        LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
        LEFT JOIN CompanyMaster c ON c.Id = p.CompanyId
        LEFT JOIN BatchMaster b ON b.Id = a.BatchId
        WHERE {base_where}
            AND l.FromDate IS NOT NULL
            AND l.ToDate IS NOT NULL
            {leave_clause}
        GROUP BY a.AgniveerNo, a.FullName, a.PhotoPath, a.Class, p.Name, c.Name, b.BatchName
        ORDER BY TotalLeaveDays DESC
        """
        
        is_valid, err = sql_validator.validate_sql(sql)
        if not is_valid:
            return None, f"Most Leave SQL validation failed: {err}"
        
        all_params = base_params + leave_params
        rows, run_err = run_readonly(sql, all_params, max_rows=_row_cap(top_n))
        if run_err:
            return None, f"Most Leave execution failed: {run_err}"
        
        return _to_section(rows or [], intent, sql=sql), None

    except Exception as exc:
        logger.error("Most Leave failed: %s", exc, exc_info=True)
        return None, str(exc)


def _execute_leave_least(intent: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Execute Least Leave Taken query.
    Returns Agniveers with lowest leave counts (excluding zero).
    """
    try:
        top_n = _get_top_n(intent)
        response_type = str(intent.get("responseType") or intent.get("response_type") or "Summary")
        detailed = response_type.lower() == "detailed"

        # ── Base scope ──────────────────────────────────────────────────────
        base_where, base_params = _build_leave_base_query(intent)

        # ── Leave type filter ──────────────────────────────────────────────
        leave_type = intent.get("leave_type") or intent.get("leaveType")
        leave_clause = ""
        leave_params: List[Any] = []
        
        if leave_type and str(leave_type).lower() == "noleave":
            # No Leave mode: return Agniveers with zero leave
            sql = f"""
            SELECT {_top_clause(top_n)}
                a.AgniveerNo,
                a.FullName,
                a.PhotoPath,
                a.Class,
                p.Name AS PlatoonName,
                c.Name AS CompanyName,
                b.BatchName
            FROM AgniveerMaster a
            LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
            LEFT JOIN CompanyMaster c ON c.Id = p.CompanyId
            LEFT JOIN BatchMaster b ON b.Id = a.BatchId
            WHERE {base_where}
                AND NOT EXISTS (
                    SELECT 1 FROM AgniveerLeaveMaster l
                    WHERE l.AgniveerId = a.Id
                        AND l.FromDate IS NOT NULL
                        AND l.ToDate IS NOT NULL
                )
            ORDER BY a.AgniveerNo ASC
            """
            
            is_valid, err = sql_validator.validate_sql(sql)
            if not is_valid:
                return None, f"Least Leave SQL validation failed: {err}"
            
            rows, run_err = run_readonly(sql, base_params, max_rows=_row_cap(top_n))
            if run_err:
                return None, f"Least Leave execution failed: {run_err}"
            
            return _to_section(rows or [], intent, sql=sql), None
        
        if leave_type and str(leave_type).lower() not in ("threshold", "noleave"):
            column = _get_leave_column(leave_type)
            if column:
                col_ref = f"l.[{column}]" if (" " in column or "'" in column) else f"l.{column}"
                leave_clause = f" AND {col_ref} = 1"

        # ── Short/Summary ──────────────────────────────────────────────────
        if not detailed:
            sql = f"""
            SELECT
                COUNT(DISTINCT l.AgniveerId) AS TotalAgniveers,
                SUM(
                    CASE
                        WHEN l.[OnEX PPG] = 1 THEN (DATEDIFF(DAY, l.FromDate, l.ToDate) + 1) / 4
                        ELSE DATEDIFF(DAY, l.FromDate, l.ToDate) + 1
                    END
                ) AS TotalLeaveDays
            FROM AgniveerLeaveMaster l
            INNER JOIN AgniveerMaster a ON a.Id = l.AgniveerId
            WHERE {base_where}
                AND l.FromDate IS NOT NULL
                AND l.ToDate IS NOT NULL
                {leave_clause}
            """
            
            is_valid, err = sql_validator.validate_sql(sql)
            if not is_valid:
                return None, f"Least Leave SQL validation failed: {err}"
            
            all_params = base_params + leave_params
            rows, run_err = run_readonly(sql, all_params, max_rows=_row_cap(top_n))
            if run_err:
                return None, f"Least Leave execution failed: {run_err}"
            
            row = rows[0] if rows else {}
            result = {
                "totalAgniveers": row.get("TotalAgniveers") or 0,
                "totalLeaveDays": row.get("TotalLeaveDays") or 0,
            }
            return _to_section([result], intent, sql=sql), None

        # ── Detailed ────────────────────────────────────────────────────────
        sql = f"""
        SELECT {_top_clause(top_n)}
            a.AgniveerNo,
            a.FullName,
            a.PhotoPath,
            a.Class,
            p.Name AS PlatoonName,
            c.Name AS CompanyName,
            b.BatchName,
            SUM(
                CASE
                    WHEN l.[OnEX PPG] = 1 THEN (DATEDIFF(DAY, l.FromDate, l.ToDate) + 1) / 4
                    ELSE DATEDIFF(DAY, l.FromDate, l.ToDate) + 1
                END
            ) AS TotalLeaveDays
        FROM AgniveerLeaveMaster l
        INNER JOIN AgniveerMaster a ON a.Id = l.AgniveerId
        LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
        LEFT JOIN CompanyMaster c ON c.Id = p.CompanyId
        LEFT JOIN BatchMaster b ON b.Id = a.BatchId
        WHERE {base_where}
            AND l.FromDate IS NOT NULL
            AND l.ToDate IS NOT NULL
            {leave_clause}
        GROUP BY a.AgniveerNo, a.FullName, a.PhotoPath, a.Class, p.Name, c.Name, b.BatchName
        HAVING SUM(
            CASE
                WHEN l.[OnEX PPG] = 1 THEN (DATEDIFF(DAY, l.FromDate, l.ToDate) + 1) / 4
                ELSE DATEDIFF(DAY, l.FromDate, l.ToDate) + 1
            END
        ) > 0
        ORDER BY TotalLeaveDays ASC
        """
        
        is_valid, err = sql_validator.validate_sql(sql)
        if not is_valid:
            return None, f"Least Leave SQL validation failed: {err}"
        
        all_params = base_params + leave_params
        rows, run_err = run_readonly(sql, all_params, max_rows=_row_cap(top_n))
        if run_err:
            return None, f"Least Leave execution failed: {run_err}"
        
        return _to_section(rows or [], intent, sql=sql), None

    except Exception as exc:
        logger.error("Least Leave failed: %s", exc, exc_info=True)
        return None, str(exc)


def _execute_leave_absconded(intent: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Execute Absconded Leave query.
    Returns Agniveers marked as absconded.
    """
    try:
        top_n = _get_top_n(intent)
        response_type = str(intent.get("responseType") or intent.get("response_type") or "Summary")
        detailed = response_type.lower() == "detailed"

        # ── Base scope ──────────────────────────────────────────────────────
        base_where, base_params = _build_leave_base_query(intent)

        # ── Short/Summary ──────────────────────────────────────────────────
        if not detailed:
            sql = f"""
            SELECT COUNT(*) AS TotalAbsconded
            FROM AgniveerLeaveMaster l
            INNER JOIN AgniveerMaster a ON a.Id = l.AgniveerId
            WHERE {base_where}
                AND l.IsAbscondedLeave = 1
                AND l.FromDate IS NOT NULL
                AND l.ToDate IS NOT NULL
            """
            
            is_valid, err = sql_validator.validate_sql(sql)
            if not is_valid:
                return None, f"Absconded Leave SQL validation failed: {err}"
            
            rows, run_err = run_readonly(sql, base_params, max_rows=_row_cap(top_n))
            if run_err:
                return None, f"Absconded Leave execution failed: {run_err}"
            
            row = rows[0] if rows else {}
            total_abs = row.get("TotalAbsconded") if row.get("TotalAbsconded") is not None else row.get("totalAbsconded")
            result = {"totalAbsconded": total_abs or 0}
            return _to_section([result], intent, sql=sql), None

        # ── Detailed ────────────────────────────────────────────────────────
        sql = f"""
        SELECT {_top_clause(top_n)}
            a.AgniveerNo,
            a.FullName,
            a.PhotoPath,
            a.Class,
            p.Name AS PlatoonName,
            c.Name AS CompanyName,
            b.BatchName,
            l.FromDate,
            l.ToDate,
            DATEDIFF(DAY, l.FromDate, l.ToDate) + 1 AS LeaveDays,
            l.Remarks
        FROM AgniveerLeaveMaster l
        INNER JOIN AgniveerMaster a ON a.Id = l.AgniveerId
        LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
        LEFT JOIN CompanyMaster c ON c.Id = p.CompanyId
        LEFT JOIN BatchMaster b ON b.Id = a.BatchId
        WHERE {base_where}
            AND l.IsAbscondedLeave = 1
            AND l.FromDate IS NOT NULL
            AND l.ToDate IS NOT NULL
        ORDER BY a.AgniveerNo ASC
        """
        
        is_valid, err = sql_validator.validate_sql(sql)
        if not is_valid:
            return None, f"Absconded Leave SQL validation failed: {err}"
        
        rows, run_err = run_readonly(sql, base_params, max_rows=_row_cap(top_n))
        if run_err:
            return None, f"Absconded Leave execution failed: {run_err}"
        
        return _to_section(rows or [], intent, sql=sql), None

    except Exception as exc:
        logger.error("Absconded Leave failed: %s", exc, exc_info=True)
        return None, str(exc)


def _execute_leave_threshold(intent: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Execute Threshold Leave query.
    Returns Agniveers with continuous 40-44 days OR total 55-59 days.
    """
    try:
        top_n = _get_top_n(intent)
        response_type = str(intent.get("responseType") or intent.get("response_type") or "Summary")
        detailed = response_type.lower() == "detailed"

        # ── Base scope ──────────────────────────────────────────────────────
        base_where, base_params = _build_leave_base_query(intent)

        # ── Threshold SQL (Union of Continuous and Total) ──────────────────
        sql = f"""
        WITH ContinuousThreshold AS (
            SELECT
                a.AgniveerNo,
                a.FullName,
                a.PhotoPath,
                a.Class,
                p.Name AS PlatoonName,
                c.Name AS CompanyName,
                b.BatchName,
                DATEDIFF(DAY, l.FromDate, l.ToDate) + 1 AS LeaveDays,
                'Continuous 40-44 days' AS Reason
            FROM AgniveerLeaveMaster l
            INNER JOIN AgniveerMaster a ON a.Id = l.AgniveerId
            LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
            LEFT JOIN CompanyMaster c ON c.Id = p.CompanyId
            LEFT JOIN BatchMaster b ON b.Id = a.BatchId
            WHERE {base_where}
                AND l.FromDate IS NOT NULL
                AND l.ToDate IS NOT NULL
                AND ISNULL(l.IsAbscondedLeave, 0) != 1
                AND DATEDIFF(DAY, l.FromDate, l.ToDate) + 1 BETWEEN 40 AND 44
        ),
        TotalThreshold AS (
            SELECT
                a.AgniveerNo,
                a.FullName,
                a.PhotoPath,
                a.Class,
                p.Name AS PlatoonName,
                c.Name AS CompanyName,
                b.BatchName,
                SUM(DATEDIFF(DAY, l.FromDate, l.ToDate) + 1) AS TotalLeaveDays,
                'Total 55-59 days' AS Reason
            FROM AgniveerLeaveMaster l
            INNER JOIN AgniveerMaster a ON a.Id = l.AgniveerId
            LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
            LEFT JOIN CompanyMaster c ON c.Id = p.CompanyId
            LEFT JOIN BatchMaster b ON b.Id = a.BatchId
            WHERE {base_where}
                AND l.FromDate IS NOT NULL
                AND l.ToDate IS NOT NULL
                AND ISNULL(l.IsAbscondedLeave, 0) != 1
            GROUP BY a.AgniveerNo, a.FullName, a.PhotoPath, a.Class, p.Name, c.Name, b.BatchName
            HAVING SUM(DATEDIFF(DAY, l.FromDate, l.ToDate) + 1) BETWEEN 55 AND 59
        )
        SELECT * FROM ContinuousThreshold
        UNION
        SELECT * FROM TotalThreshold
        ORDER BY AgniveerNo ASC
        """
        
        is_valid, err = sql_validator.validate_sql(sql)
        if not is_valid:
            return None, f"Threshold Leave SQL validation failed: {err}"
        
        rows, run_err = run_readonly(sql, base_params, max_rows=_row_cap(top_n))
        if run_err:
            return None, f"Threshold Leave execution failed: {run_err}"
        
        if not detailed:
            # Summary: return counts
            continuous_ids = {r.get("AgniveerNo") for r in rows if r.get("Reason") == "Continuous 40-44 days"}
            total_ids = {r.get("AgniveerNo") for r in rows if r.get("Reason") == "Total 55-59 days"}
            
            result = {
                "thresholdCount": len(continuous_ids | total_ids),
                "continuous40to44Count": len(continuous_ids),
                "total55to59Count": len(total_ids),
            }
            return _to_section([result], intent, sql=sql), None
        
        return _to_section(rows or [], intent, sql=sql), None

    except Exception as exc:
        logger.error("Threshold Leave failed: %s", exc, exc_info=True)
        return None, str(exc)


def _execute_leave_history(intent: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Execute Leave History query for a specific Agniveer.
    """
    try:
        agniveer_no = intent.get("agniveer_no") or intent.get("agniveerNo")
        
        if not agniveer_no:
            return None, "AgniveerNo required for leave history"

        # ── Base scope ──────────────────────────────────────────────────────
        base_where, base_params = _build_leave_base_query(intent)

        # ── Build SQL ──────────────────────────────────────────────────────
        sql = f"""
        SELECT
            l.Id,
            l.FromDate,
            l.ToDate,
            DATEDIFF(DAY, l.FromDate, l.ToDate) + 1 AS LeaveDays,
            CASE
                WHEN l.[OnEX PPG] = 1 THEN (DATEDIFF(DAY, l.FromDate, l.ToDate) + 1) / 4
                ELSE DATEDIFF(DAY, l.FromDate, l.ToDate) + 1
            END AS LeaveCount,
            CASE
                WHEN l.OnAnnualLeave = 1 THEN 'Annual'
                WHEN l.OnMedicalLeave = 1 THEN 'Medical'
                WHEN l.OnSickLeave = 1 THEN 'Sick'
                WHEN l.IsHospitalized = 1 THEN 'Hospitalized'
                WHEN l.IsAbscondedLeave = 1 THEN 'Absconded'
                WHEN l.[OnATTN'C'] = 1 THEN 'ATTNC'
                WHEN l.[OnEX PPG] = 1 THEN 'EX PPG'
                ELSE 'Unknown'
            END AS LeaveType,
            l.Remarks
        FROM AgniveerLeaveMaster l
        INNER JOIN AgniveerMaster a ON a.Id = l.AgniveerId
        WHERE {base_where}
            AND l.FromDate IS NOT NULL
            AND l.ToDate IS NOT NULL
        ORDER BY l.FromDate DESC
        """
        
        is_valid, err = sql_validator.validate_sql(sql)
        if not is_valid:
            return None, f"Leave History SQL validation failed: {err}"
        
        rows, run_err = run_readonly(sql, base_params)
        if run_err:
            return None, f"Leave History execution failed: {run_err}"
        
        return _to_section(rows or [], intent, sql=sql), None

    except Exception as exc:
        logger.error("Leave History failed: %s", exc, exc_info=True)
        return None, str(exc)


def execute_leave_query(intent: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
    """
    Dispatch Leave queries based on operation.
    """
    operation = str(intent.get("operation") or "Current").lower()
    
    if operation == "current":
        return _execute_leave_current(intent)
    elif operation in ("most", "highest"):
        return _execute_leave_most(intent)
    elif operation in ("least", "lowest"):
        return _execute_leave_least(intent)
    elif operation == "absconded":
        return _execute_leave_absconded(intent)
    elif operation == "threshold":
        return _execute_leave_threshold(intent)
    elif operation in ("history", "records"):
        return _execute_leave_history(intent)
    else:
        # Default to Current
        return _execute_leave_current(intent)


# ── Medical Category ────────────────────────────────────────────────────────

BMI_THRESHOLDS = {
    "underweight": {"max": 18.5, "label": "Underweight"},
    "normal": {"min": 18.5, "max": 25.0, "label": "Normal"},
    "overweight": {"min": 25.0, "max": 30.0, "label": "Overweight"},
    "obese": {"min": 30.0, "label": "Obese"},
    "unfit": {"min": 25.0, "label": "Unfit"},  # Overweight + Obese combined
}


def _build_medical_base_scope(intent: Dict[str, Any]) -> Tuple[str, List[Any]]:
    """Build the base WHERE clause for medical queries."""
    clauses = ["(a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)", "a.IsActive = 1"]
    params: List[Any] = []

    agniveer_no = intent.get("agniveer_no") or intent.get("agniveerNo")
    batch_id = intent.get("batch_id") or intent.get("batchId")
    platoon_id = intent.get("platoon_id") or intent.get("platoonId")
    company_id = intent.get("company_id") or intent.get("companyId")
    class_name = intent.get("class") or intent.get("class_")

    if agniveer_no:
        clauses.append("LOWER(a.AgniveerNo) LIKE '%' + LOWER(?) + '%'")
        params.append(str(agniveer_no))
    if batch_id is not None:
        clauses.append("a.BatchId = ?")
        params.append(int(batch_id))
    if platoon_id is not None:
        clauses.append("a.PlatoonId = ?")
        params.append(int(platoon_id))
    if company_id is not None:
        clauses.append("EXISTS (SELECT 1 FROM PlatoonMaster p WHERE p.Id = a.PlatoonId AND p.CompanyId = ?)")
        params.append(int(company_id))
    if class_name:
        clauses.append("LOWER(a.Class) = LOWER(?)")
        params.append(str(class_name))

    return " AND ".join(clauses), params


def _execute_medical_bmi(intent: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Execute BMI query.
    Returns BMI distribution or individual BMI values.
    """
    try:
        top_n = _get_top_n(intent)
        bmi_category = intent.get("bmi_category") or intent.get("bmiCategory")
        response_type = str(intent.get("responseType") or intent.get("response_type") or "Summary")
        detailed = response_type.lower() == "detailed"
        blood_group = intent.get("blood_group") or intent.get("bloodGroup")

        base_where, base_params = _build_medical_base_scope(intent)

        # ── Blood group filter ──────────────────────────────────────────────
        blood_filter = ""
        blood_params: List[Any] = []
        if blood_group:
            blood_filter = "AND UPPER(REPLACE(a.BloodGroup, ' ', '')) = UPPER(REPLACE(?, ' ', ''))"
            blood_params = [str(blood_group)]

        # ── Build BMI CTE ─────────────────────────────────────────────────────
        bmi_cte = f"""
        WITH LatestMedical AS (
            SELECT
                mr.AgniveerId,
                mr.Height,
                mr.Weight,
                ROW_NUMBER() OVER (PARTITION BY mr.AgniveerId ORDER BY mr.VisitDate DESC, mr.Id DESC) AS rn
            FROM MedicalRecordMaster mr
            WHERE mr.Height IS NOT NULL
                AND mr.Weight IS NOT NULL
        ),
        Vitals AS (
            SELECT
                a.Id AS AgniveerId,
                a.AgniveerNo,
                a.FullName,
                a.Class,
                a.BloodGroup,
                p.Name AS PlatoonName,
                c.Name AS CompanyName,
                b.BatchName,
                COALESCE(lm.Height, a.Height) AS EffHeight,
                COALESCE(lm.Weight, a.Weight) AS EffWeight
            FROM AgniveerMaster a
            LEFT JOIN LatestMedical lm ON lm.AgniveerId = a.Id AND lm.rn = 1
            LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
            LEFT JOIN CompanyMaster c ON c.Id = p.CompanyId
            LEFT JOIN BatchMaster b ON b.Id = a.BatchId
            WHERE {base_where}
                {blood_filter}
        ),
        Scored AS (
            SELECT
                AgniveerNo,
                FullName,
                Class,
                BloodGroup,
                PlatoonName,
                CompanyName,
                BatchName,
                EffHeight,
                EffWeight,
                CASE
                    WHEN EffHeight IS NULL OR EffWeight IS NULL OR EffHeight <= 0 THEN NULL
                    ELSE CAST(EffWeight / POWER(EffHeight / 100.0, 2) AS DECIMAL(10, 2))
                END AS BmiValue,
                CASE
                    WHEN EffHeight IS NULL OR EffWeight IS NULL OR EffHeight <= 0 THEN NULL
                    WHEN EffWeight / POWER(EffHeight / 100.0, 2) < 18.5 THEN 'Underweight'
                    WHEN EffWeight / POWER(EffHeight / 100.0, 2) < 25.0 THEN 'Normal'
                    WHEN EffWeight / POWER(EffHeight / 100.0, 2) < 30.0 THEN 'Overweight'
                    ELSE 'Obese'
                END AS BmiCategory
            FROM Vitals
        )
        """

        # ── Short/Summary ────────────────────────────────────────────────────
        if not detailed:
            if bmi_category:
                # Specific category count
                sql = f"""
                {bmi_cte}
                SELECT COUNT(*) AS Count
                FROM Scored
                WHERE BmiCategory = ?
                """
                is_valid, err = sql_validator.validate_sql(sql)
                if not is_valid:
                    return None, f"BMI SQL validation failed: {err}"
                
                params = base_params + blood_params + [str(bmi_category)]
                rows, run_err = run_readonly(sql, params, max_rows=_row_cap(top_n))
                if run_err:
                    return None, f"BMI execution failed: {run_err}"
                
                row = rows[0] if rows else {}
                result = {"bmiCategory": bmi_category, "count": row.get("Count") if row.get("Count") is not None else (row.get("count") or 0)}
                return _to_section([result], intent, sql=sql), None
            else:
                # Distribution by category
                sql = f"""
                {bmi_cte}
                SELECT
                    BmiCategory,
                    COUNT(*) AS AgniveerCount
                FROM Scored
                WHERE BmiCategory IS NOT NULL
                GROUP BY BmiCategory
                ORDER BY BmiCategory ASC
                """
                is_valid, err = sql_validator.validate_sql(sql)
                if not is_valid:
                    return None, f"BMI SQL validation failed: {err}"
                
                params = base_params + blood_params
                rows, run_err = run_readonly(sql, params, max_rows=_row_cap(top_n))
                if run_err:
                    return None, f"BMI execution failed: {run_err}"
                
                return _to_section(rows or [], intent, sql=sql), None

        # ── Detailed ──────────────────────────────────────────────────────────
        category_filter = ""
        category_params: List[Any] = []
        if bmi_category:
            category_filter = "WHERE BmiCategory = ?"
            category_params = [str(bmi_category)]

        sql = f"""
        {bmi_cte}
        SELECT {_top_clause(top_n)}
            AgniveerNo,
            FullName,
            Class,
            BloodGroup,
            PlatoonName,
            CompanyName,
            BatchName,
            EffHeight AS Height,
            EffWeight AS Weight,
            BmiValue,
            BmiCategory
        FROM Scored
        {category_filter}
        ORDER BY BmiValue DESC
        """
        
        is_valid, err = sql_validator.validate_sql(sql)
        if not is_valid:
            return None, f"BMI SQL validation failed: {err}"
        
        params = base_params + blood_params + category_params
        rows, run_err = run_readonly(sql, params, max_rows=_row_cap(top_n))
        if run_err:
            return None, f"BMI execution failed: {run_err}"
        
        return _to_section(rows or [], intent, sql=sql), None

    except Exception as exc:
        logger.error("BMI query failed: %s", exc, exc_info=True)
        return None, str(exc)


def _execute_medical_blood_group(intent: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Execute Blood Group query.
    Returns distribution or details for specific group.
    """
    try:
        top_n = _get_top_n(intent)
        blood_group = intent.get("blood_group") or intent.get("bloodGroup")
        response_type = str(intent.get("responseType") or intent.get("response_type") or "Summary")
        detailed = response_type.lower() == "detailed"

        base_where, base_params = _build_medical_base_scope(intent)

        # ── Report/Distribution ─────────────────────────────────────────────
        if not blood_group and not detailed:
            sql = f"""
            SELECT
                COALESCE(NULLIF(a.BloodGroup, ''), 'Unknown') AS BloodGroup,
                COUNT(*) AS AgniveerCount
            FROM AgniveerMaster a
            LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
            LEFT JOIN CompanyMaster c ON c.Id = p.CompanyId
            LEFT JOIN BatchMaster b ON b.Id = a.BatchId
            WHERE {base_where}
            GROUP BY COALESCE(NULLIF(a.BloodGroup, ''), 'Unknown')
            ORDER BY AgniveerCount DESC, BloodGroup ASC
            """
            
            is_valid, err = sql_validator.validate_sql(sql)
            if not is_valid:
                return None, f"Blood Group SQL validation failed: {err}"
            
            rows, run_err = run_readonly(sql, base_params, max_rows=_row_cap(top_n))
            if run_err:
                return None, f"Blood Group execution failed: {run_err}"
            
            return _to_section(rows or [], intent, sql=sql), None

        # ── Detailed with specific blood group ─────────────────────────────
        if blood_group or detailed:
            blood_clause = ""
            params = list(base_params)
            if blood_group:
                blood_clause = "AND UPPER(REPLACE(a.BloodGroup, ' ', '')) = UPPER(REPLACE(?, ' ', ''))"
                params.append(str(blood_group))

            sql = f"""
            SELECT {_top_clause(top_n)}
                a.AgniveerNo,
                a.FullName,
                a.PhotoPath,
                a.Class,
                p.Name AS PlatoonName,
                c.Name AS CompanyName,
                b.BatchName,
                a.BloodGroup
            FROM AgniveerMaster a
            LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
            LEFT JOIN CompanyMaster c ON c.Id = p.CompanyId
            LEFT JOIN BatchMaster b ON b.Id = a.BatchId
            WHERE {base_where}
                {blood_clause}
            ORDER BY a.AgniveerNo ASC
            """
            
            is_valid, err = sql_validator.validate_sql(sql)
            if not is_valid:
                return None, f"Blood Group SQL validation failed: {err}"
            
            rows, run_err = run_readonly(sql, params, max_rows=_row_cap(top_n))
            if run_err:
                return None, f"Blood Group execution failed: {run_err}"
            
            return _to_section(rows or [], intent, sql=sql), None

        return None, "Blood Group query not configured"

    except Exception as exc:
        logger.error("Blood Group query failed: %s", exc, exc_info=True)
        return None, str(exc)


def _execute_medical_disease(intent: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Execute Disease query.
    Returns disease statistics or details.
    """
    try:
        top_n = _get_top_n(intent)
        diagnose = intent.get("diagnose") or intent.get("diagnosis")
        days = intent.get("days")
        response_type = str(intent.get("responseType") or intent.get("response_type") or "Summary")
        detailed = response_type.lower() == "detailed"

        base_where, base_params = _build_medical_base_scope(intent)

        # ── Date range filter ──────────────────────────────────────────────
        date_filter = ""
        date_params: List[Any] = []
        if days and int(days) > 0:
            date_filter = "AND CAST(mr.VisitDate AS DATE) >= DATEADD(DAY, -?, CAST(GETDATE() AS DATE))"
            date_params = [int(days)]
        elif intent.get("from_date") and intent.get("to_date"):
            date_filter = "AND CAST(mr.VisitDate AS DATE) >= CAST(? AS DATE) AND CAST(mr.VisitDate AS DATE) <= CAST(? AS DATE)"
            date_params = [
                str(intent.get("from_date") or "")[:10],
                str(intent.get("to_date") or "")[:10]
            ]

        # ── Specific disease ──────────────────────────────────────────────────
        if diagnose:
            sql = f"""
            SELECT {_top_clause(top_n)}
                a.AgniveerNo,
                a.FullName,
                a.PhotoPath,
                a.Class,
                p.Name AS PlatoonName,
                c.Name AS CompanyName,
                b.BatchName,
                mr.VisitDate,
                mr.Diagnosis,
                mr.Status,
                mr.HospitalNameLocation,
                mr.FollowUpDate
            FROM MedicalRecordMaster mr
            INNER JOIN AgniveerMaster a ON a.Id = mr.AgniveerId
            LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
            LEFT JOIN CompanyMaster c ON c.Id = p.CompanyId
            LEFT JOIN BatchMaster b ON b.Id = a.BatchId
            WHERE {base_where}
                AND LOWER(mr.Diagnosis) LIKE '%' + LOWER(?) + '%'
                {date_filter}
            ORDER BY mr.VisitDate DESC
            """
            
            is_valid, err = sql_validator.validate_sql(sql)
            if not is_valid:
                return None, f"Disease SQL validation failed: {err}"
            
            params = base_params + [str(diagnose)] + date_params
            rows, run_err = run_readonly(sql, params, max_rows=_row_cap(top_n))
            if run_err:
                return None, f"Disease execution failed: {run_err}"
            
            return _to_section(rows or [], intent, sql=sql), None

        # ── Short: Disease statistics ──────────────────────────────────────
        if not detailed:
            sql = f"""
            SELECT {_top_clause(top_n)}
                mr.Diagnosis,
                COUNT(*) AS TotalCount,
                COUNT(DISTINCT mr.AgniveerId) AS AgniveerCount
            FROM MedicalRecordMaster mr
            INNER JOIN AgniveerMaster a ON a.Id = mr.AgniveerId
            LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
            LEFT JOIN CompanyMaster c ON c.Id = p.CompanyId
            LEFT JOIN BatchMaster b ON b.Id = a.BatchId
            WHERE {base_where}
                AND mr.Diagnosis IS NOT NULL
                AND mr.Diagnosis != ''
                {date_filter}
            GROUP BY mr.Diagnosis
            ORDER BY TotalCount DESC
            """
            
            is_valid, err = sql_validator.validate_sql(sql)
            if not is_valid:
                return None, f"Disease SQL validation failed: {err}"
            
            params = base_params + date_params
            rows, run_err = run_readonly(sql, params, max_rows=_row_cap(top_n))
            if run_err:
                return None, f"Disease execution failed: {run_err}"
            
            return _to_section(rows or [], intent, sql=sql), None

        # ── Detailed: Disease with per-agniveer breakdown ──────────────────
        sql = f"""
        SELECT {_top_clause(top_n)}
            mr.Diagnosis,
            mr.AgniveerId,
            a.AgniveerNo,
            a.FullName,
            a.PhotoPath,
            a.Class,
            p.Name AS PlatoonName,
            c.Name AS CompanyName,
            b.BatchName,
            mr.VisitDate,
            mr.Status,
            mr.HospitalNameLocation,
            mr.FollowUpDate
        FROM MedicalRecordMaster mr
        INNER JOIN AgniveerMaster a ON a.Id = mr.AgniveerId
        LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
        LEFT JOIN CompanyMaster c ON c.Id = p.CompanyId
        LEFT JOIN BatchMaster b ON b.Id = a.BatchId
        WHERE {base_where}
            AND mr.Diagnosis IS NOT NULL
            AND mr.Diagnosis != ''
            {date_filter}
        ORDER BY mr.Diagnosis ASC, mr.VisitDate DESC
        """
        
        is_valid, err = sql_validator.validate_sql(sql)
        if not is_valid:
            return None, f"Disease SQL validation failed: {err}"
        
        params = base_params + date_params
        rows, run_err = run_readonly(sql, params, max_rows=_row_cap(top_n))
        if run_err:
            return None, f"Disease execution failed: {run_err}"
        
        return _to_section(rows or [], intent, sql=sql), None

    except Exception as exc:
        logger.error("Disease query failed: %s", exc, exc_info=True)
        return None, str(exc)


def _execute_medical_individual(intent: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Execute Individual Medical Report query.
    Returns complete medical history for a single Agniveer.
    """
    try:
        agniveer_no = intent.get("agniveer_no") or intent.get("agniveerNo")
        
        if not agniveer_no:
            return None, "AgniveerNo required for individual medical report"

        # ── Base scope ──────────────────────────────────────────────────────
        base_where, base_params = _build_medical_base_scope(intent)

        # ── Build SQL ──────────────────────────────────────────────────────
        sql = f"""
        SELECT
            a.AgniveerNo,
            a.FullName,
            a.PhotoPath,
            a.Class,
            a.BloodGroup,
            a.Height,
            a.Weight,
            p.Name AS PlatoonName,
            c.Name AS CompanyName,
            b.BatchName,
            mr.Id AS MedicalId,
            u.FullName AS DoctorName,
            mr.Type,
            mr.VisitDate,
            mr.FollowUpDate,
            mr.HospitalNameLocation,
            mr.AdmitDate,
            mr.DischargeDate,
            mr.Diagnosis,
            mr.TreatmentGiven,
            mr.Prescriptions,
            mr.Status,
            mr.Remarks,
            mr.BloodPressure,
            mr.HeartRate,
            mr.Weight AS MedicalWeight,
            mr.Height AS MedicalHeight,
            mr.EyeSight,
            mr.LeaveType,
            mr.FromDate,
            mr.ToDate
        FROM AgniveerMaster a
        LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
        LEFT JOIN CompanyMaster c ON c.Id = p.CompanyId
        LEFT JOIN BatchMaster b ON b.Id = a.BatchId
        INNER JOIN MedicalRecordMaster mr ON mr.AgniveerId = a.Id
        LEFT JOIN UserMaster u ON u.Id = mr.DoctorId
        WHERE {base_where}
        ORDER BY mr.VisitDate DESC
        """
        
        is_valid, err = sql_validator.validate_sql(sql)
        if not is_valid:
            return None, f"Individual Medical SQL validation failed: {err}"
        
        rows, run_err = run_readonly(sql, base_params)
        if run_err:
            return None, f"Individual Medical execution failed: {run_err}"
        
        return _to_section(rows or [], intent, sql=sql), None

    except Exception as exc:
        logger.error("Individual Medical failed: %s", exc, exc_info=True)
        return None, str(exc)


def _execute_medical_followup(intent: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Execute Follow-Up query.
    Returns Agniveers with follow-up appointments.
    """
    try:
        top_n = _get_top_n(intent)
        agniveer_no = intent.get("agniveer_no") or intent.get("agniveerNo")

        # ── Base scope ──────────────────────────────────────────────────────
        base_where, base_params = _build_medical_base_scope(intent)

        # ── Date filter ──────────────────────────────────────────────────────
        date_filter = "mr.FollowUpDate >= CAST(GETDATE() AS DATE)"
        date_params: List[Any] = []
        
        if intent.get("from_date") and intent.get("to_date"):
            date_filter = "CAST(mr.FollowUpDate AS DATE) >= CAST(? AS DATE) AND CAST(mr.FollowUpDate AS DATE) <= CAST(? AS DATE)"
            date_params = [
                str(intent.get("from_date") or "")[:10],
                str(intent.get("to_date") or "")[:10]
            ]

        # ── Build SQL ──────────────────────────────────────────────────────
        sql = f"""
        SELECT {_top_clause(top_n)}
            a.AgniveerNo,
            a.FullName,
            a.PhotoPath,
            a.Class,
            p.Name AS PlatoonName,
            c.Name AS CompanyName,
            b.BatchName,
            mr.FollowUpDate,
            mr.VisitDate,
            mr.Diagnosis,
            mr.HospitalNameLocation,
            mr.Status
        FROM MedicalRecordMaster mr
        INNER JOIN AgniveerMaster a ON a.Id = mr.AgniveerId
        LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
        LEFT JOIN CompanyMaster c ON c.Id = p.CompanyId
        LEFT JOIN BatchMaster b ON b.Id = a.BatchId
        WHERE {base_where}
            AND mr.FollowUpDate IS NOT NULL
            AND {date_filter}
        ORDER BY mr.FollowUpDate ASC
        """
        
        is_valid, err = sql_validator.validate_sql(sql)
        if not is_valid:
            return None, f"Follow-Up SQL validation failed: {err}"
        
        params = base_params + date_params
        rows, run_err = run_readonly(sql, params, max_rows=_row_cap(top_n))
        if run_err:
            return None, f"Follow-Up execution failed: {run_err}"
        
        return _to_section(rows or [], intent, sql=sql), None

    except Exception as exc:
        logger.error("Follow-Up failed: %s", exc, exc_info=True)
        return None, str(exc)


def _execute_medical_hospital_stats(intent: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Execute Hospital Statistics query.
    Returns hospitals with most Agniveer visits.
    """
    try:
        top_n = _get_top_n(intent)
        base_where, base_params = _build_medical_base_scope(intent)

        sql = f"""
        SELECT {_top_clause(top_n)}
            mr.HospitalNameLocation,
            COUNT(DISTINCT mr.AgniveerId) AS AgniveerCount
        FROM MedicalRecordMaster mr
        INNER JOIN AgniveerMaster a ON a.Id = mr.AgniveerId
        LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
        LEFT JOIN CompanyMaster c ON c.Id = p.CompanyId
        LEFT JOIN BatchMaster b ON b.Id = a.BatchId
        WHERE {base_where}
            AND mr.HospitalNameLocation IS NOT NULL
            AND TRIM(mr.HospitalNameLocation) != ''
        GROUP BY mr.HospitalNameLocation
        ORDER BY AgniveerCount DESC
        """
        
        is_valid, err = sql_validator.validate_sql(sql)
        if not is_valid:
            return None, f"Hospital Stats SQL validation failed: {err}"
        
        rows, run_err = run_readonly(sql, base_params, max_rows=_row_cap(top_n))
        if run_err:
            return None, f"Hospital Stats execution failed: {run_err}"
        
        return _to_section(rows or [], intent, sql=sql), None

    except Exception as exc:
        logger.error("Hospital Stats failed: %s", exc, exc_info=True)
        return None, str(exc)


def execute_medical_query(intent: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
    """
    Dispatch Medical queries based on operation.
    """
    operation = str(intent.get("operation") or intent.get("subcategory") or "BMI").lower()
    
    if operation in ("bmi", "bmianalysis"):
        return _execute_medical_bmi(intent)
    elif operation in ("bloodgroup", "blood_group"):
        return _execute_medical_blood_group(intent)
    elif operation in ("disease", "diseasestatistics", "diagnosed"):
        return _execute_medical_disease(intent)
    elif operation in ("individual", "individualmedical"):
        return _execute_medical_individual(intent)
    elif operation in ("followup", "follow_up"):
        return _execute_medical_followup(intent)
    elif operation in ("hospitalstats", "hospital_stats"):
        return _execute_medical_hospital_stats(intent)
    else:
        # Default to BMI
        return _execute_medical_bmi(intent)


# ── Attendance Category ───────────────────────────────────────────────────

ATTENDANCE_DATE_RANGE = {
    "daily": {"default_months": 1},
    "weekly": {"default_weeks": 4},
    "monthly": {"default_months": 3},
}


def _build_attendance_base_scope(intent: Dict[str, Any]) -> Tuple[str, List[Any]]:
    """Build the base WHERE clause for attendance queries."""
    clauses = ["(a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)", "a.IsActive = 1"]
    params: List[Any] = []

    agniveer_no = intent.get("agniveer_no") or intent.get("agniveerNo")
    batch_id = intent.get("batch_id") or intent.get("batchId")
    platoon_id = intent.get("platoon_id") or intent.get("platoonId")
    company_id = intent.get("company_id") or intent.get("companyId")

    if agniveer_no:
        clauses.append("LOWER(a.AgniveerNo) LIKE '%' + LOWER(?) + '%'")
        params.append(str(agniveer_no))
    if batch_id is not None:
        clauses.append("a.BatchId = ?")
        params.append(int(batch_id))
    if platoon_id is not None:
        clauses.append("a.PlatoonId = ?")
        params.append(int(platoon_id))
    if company_id is not None:
        clauses.append("EXISTS (SELECT 1 FROM PlatoonMaster p WHERE p.Id = a.PlatoonId AND p.CompanyId = ?)")
        params.append(int(company_id))

    return " AND ".join(clauses), params


def _resolve_attendance_dates(operation: str, intent: Dict[str, Any]) -> Tuple[str, str]:
    """
    Resolve date range for attendance queries.
    Returns (from_date, to_date) as ISO date strings.
    """
    import datetime
    
    # If explicit dates provided, use them
    date = intent.get("date")
    from_date = intent.get("from_date") or intent.get("fromDate")
    to_date = intent.get("to_date") or intent.get("toDate")
    
    if from_date and to_date:
        return str(from_date)[:10], str(to_date)[:10]
    if date:
        return str(date)[:10], str(date)[:10]
    
    # Default ranges based on operation
    today = datetime.date.today()
    
    if operation == "daily":
        # Current month
        start = datetime.date(today.year, today.month, 1)
        end = datetime.date(today.year, today.month, 1).replace(
            day=28
        ) + datetime.timedelta(days=4)
        end = end - datetime.timedelta(days=end.day)
        return start.isoformat(), end.isoformat()
    
    elif operation == "weekly":
        # Last 4 weeks from Monday
        monday = today - datetime.timedelta(days=today.weekday())
        start = monday - datetime.timedelta(weeks=3)
        end = monday + datetime.timedelta(days=6)
        return start.isoformat(), end.isoformat()
    
    else:  # monthly
        # Last 3 months
        start = datetime.date(today.year, today.month, 1) - datetime.timedelta(days=90)
        end = datetime.date(today.year, today.month, 1).replace(
            day=28
        ) + datetime.timedelta(days=4)
        end = end - datetime.timedelta(days=end.day)
        return start.isoformat(), end.isoformat()


def _execute_attendance_summary(intent: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Execute Attendance Summary query.
    Returns present/absent counts for today.
    """
    try:
        response_type = str(intent.get("responseType") or intent.get("response_type") or "Summary")
        detailed = response_type.lower() == "detailed"
        
        base_where, base_params = _build_attendance_base_scope(intent)
        
        # ── Get date ──────────────────────────────────────────────────────────
        import datetime
        today = datetime.date.today().isoformat()
        date = intent.get("date") or today

        # ── Build SQL ──────────────────────────────────────────────────────────
        if not detailed:
            # Summary: counts only
            sql = f"""
            WITH AgniveerScope AS (
                SELECT
                    a.Id,
                    a.AgniveerNo,
                    a.FullName,
                    a.DateOfJoining
                FROM AgniveerMaster a
                LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
                LEFT JOIN CompanyMaster c ON c.Id = p.CompanyId
                LEFT JOIN BatchMaster b ON b.Id = a.BatchId
                WHERE {base_where}
            ),
            TodayLeaves AS (
                SELECT
                    l.AgniveerId,
                    l.FromDate,
                    l.ToDate,
                    l.IsAbscondedLeave
                FROM AgniveerLeaveMaster l
                WHERE l.AgniveerId IN (SELECT Id FROM AgniveerScope)
                    AND l.FromDate IS NOT NULL
                    AND l.FromDate <= CAST(? AS DATE)
                    AND (l.ToDate IS NULL OR l.ToDate >= CAST(? AS DATE))
            )
            SELECT
                COUNT(*) AS TotalActive,
                SUM(
                    CASE
                        WHEN EXISTS (
                            SELECT 1 FROM TodayLeaves tl
                            WHERE tl.AgniveerId = a.Id
                                AND tl.ToDate IS NOT NULL
                                AND CAST(? AS DATE) BETWEEN tl.FromDate AND tl.ToDate
                        ) THEN 0
                        WHEN EXISTS (
                            SELECT 1 FROM TodayLeaves tl
                            WHERE tl.AgniveerId = a.Id
                                AND tl.ToDate IS NULL
                                AND tl.IsAbscondedLeave = 1
                                AND CAST(? AS DATE) >= tl.FromDate
                        ) THEN 0
                        ELSE 1
                    END
                ) AS PresentCount
            FROM AgniveerScope a
            """
            
            is_valid, err = sql_validator.validate_sql(sql)
            if not is_valid:
                return None, f"Attendance Summary SQL validation failed: {err}"
            
            params = base_params + [date, date, date, date]
            rows, run_err = run_readonly(sql, params)
            if run_err:
                return None, f"Attendance Summary execution failed: {run_err}"
            
            row = rows[0] if rows else {}
            total = row.get("TotalActive") if row.get("TotalActive") is not None else (row.get("totalActive") or 0)
            present = row.get("PresentCount") if row.get("PresentCount") is not None else (row.get("presentCount") or 0)
            absent = total - present
            pct = round((present / total * 100), 2) if total > 0 else 0
            
            result = {
                "date": date,
                "totalActive": total,
                "presentCount": present,
                "absentCount": absent,
                "presentPct": pct,
            }
            return _to_section([result], intent, sql=sql), None
        
        # ── Detailed: list of Agniveers with status ──────────────────────────
        sql = f"""
        WITH AgniveerScope AS (
            SELECT
                a.Id,
                a.AgniveerNo,
                a.FullName,
                a.PhotoPath,
                a.Class,
                a.DateOfJoining,
                p.Name AS PlatoonName,
                c.Name AS CompanyName,
                b.BatchName
            FROM AgniveerMaster a
            LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
            LEFT JOIN CompanyMaster c ON c.Id = p.CompanyId
            LEFT JOIN BatchMaster b ON b.Id = a.BatchId
            WHERE {base_where}
        ),
        TodayLeaves AS (
            SELECT
                l.AgniveerId,
                l.FromDate,
                l.ToDate,
                l.IsAbscondedLeave
            FROM AgniveerLeaveMaster l
            WHERE l.AgniveerId IN (SELECT Id FROM AgniveerScope)
                AND l.FromDate IS NOT NULL
                AND l.FromDate <= CAST(? AS DATE)
                AND (l.ToDate IS NULL OR l.ToDate >= CAST(? AS DATE))
        )
        SELECT
            a.Id,
            a.AgniveerNo,
            a.FullName,
            a.PhotoPath,
            a.Class,
            a.PlatoonName,
            a.CompanyName,
            a.BatchName,
            a.DateOfJoining,
            CASE
                WHEN a.DateOfJoining IS NOT NULL AND CAST(? AS DATE) < a.DateOfJoining THEN NULL
                WHEN EXISTS (
                    SELECT 1 FROM TodayLeaves tl
                    WHERE tl.AgniveerId = a.Id
                        AND tl.ToDate IS NOT NULL
                        AND CAST(? AS DATE) BETWEEN tl.FromDate AND tl.ToDate
                ) THEN 0
                WHEN EXISTS (
                    SELECT 1 FROM TodayLeaves tl
                    WHERE tl.AgniveerId = a.Id
                        AND tl.ToDate IS NULL
                        AND tl.IsAbscondedLeave = 1
                        AND CAST(? AS DATE) >= tl.FromDate
                ) THEN 0
                ELSE 1
            END AS IsPresent
        FROM AgniveerScope a
        ORDER BY a.AgniveerNo ASC
        """
        
        is_valid, err = sql_validator.validate_sql(sql)
        if not is_valid:
            return None, f"Attendance Summary SQL validation failed: {err}"
        
        params = base_params + [date, date, date, date, date]
        rows, run_err = run_readonly(sql, params)
        if run_err:
            return None, f"Attendance Summary execution failed: {run_err}"
        
        return _to_section(rows or [], intent, sql=sql), None

    except Exception as exc:
        logger.error("Attendance Summary failed: %s", exc, exc_info=True)
        return None, str(exc)


def _execute_attendance_daily(intent: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Execute Daily Attendance query.
    Returns day-by-day calendar for Agniveers.
    """
    try:
        agniveer_no = intent.get("agniveer_no") or intent.get("agniveerNo")
        
        if not agniveer_no:
            return None, "AgniveerNo required for daily attendance calendar"

        base_where, base_params = _build_attendance_base_scope(intent)
        
        # ── Resolve date range ──────────────────────────────────────────────
        from_date, to_date = _resolve_attendance_dates("daily", intent)

        # ── Build SQL ──────────────────────────────────────────────────────────
        sql = f"""
        WITH AgniveerInfo AS (
            SELECT
                a.Id,
                a.AgniveerNo,
                a.FullName,
                a.PhotoPath,
                a.DateOfJoining,
                p.Name AS PlatoonName,
                c.Name AS CompanyName
            FROM AgniveerMaster a
            LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
            LEFT JOIN CompanyMaster c ON c.Id = p.CompanyId
            WHERE {base_where}
        ),
        LeaveRecords AS (
            SELECT
                l.FromDate,
                l.ToDate,
                l.IsAbscondedLeave
            FROM AgniveerLeaveMaster l
            WHERE l.AgniveerId = (SELECT Id FROM AgniveerInfo)
                AND l.FromDate IS NOT NULL
                AND l.FromDate <= CAST(? AS DATE)
                AND (l.ToDate IS NULL OR l.ToDate >= CAST(? AS DATE))
        ),
        DateRange AS (
            SELECT CAST(? AS DATE) AS AttendanceDate
            UNION ALL
            SELECT DATEADD(DAY, 1, AttendanceDate)
            FROM DateRange
            WHERE AttendanceDate < CAST(? AS DATE)
        )
        SELECT
            d.AttendanceDate,
            CASE
                WHEN d.AttendanceDate < a.DateOfJoining THEN NULL
                WHEN EXISTS (
                    SELECT 1 FROM LeaveRecords l
                    WHERE l.ToDate IS NOT NULL
                        AND d.AttendanceDate BETWEEN l.FromDate AND l.ToDate
                ) THEN 0
                WHEN EXISTS (
                    SELECT 1 FROM LeaveRecords l
                    WHERE l.ToDate IS NULL
                        AND l.IsAbscondedLeave = 1
                        AND d.AttendanceDate >= l.FromDate
                ) THEN 0
                ELSE 1
            END AS IsPresent,
            a.AgniveerNo,
            a.FullName,
            a.PhotoPath
        FROM DateRange d
        CROSS JOIN AgniveerInfo a
        OPTION (MAXRECURSION 366)
        """
        
        is_valid, err = sql_validator.validate_sql(sql)
        if not is_valid:
            return None, f"Daily Attendance SQL validation failed: {err}"
        
        params = base_params + [to_date, from_date, from_date, to_date]
        rows, run_err = run_readonly(sql, params)
        if run_err:
            return None, f"Daily Attendance execution failed: {run_err}"
        
        return _to_section(rows or [], intent, sql=sql), None

    except Exception as exc:
        logger.error("Daily Attendance failed: %s", exc, exc_info=True)
        return None, str(exc)


def _execute_attendance_weekly(intent: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Execute Weekly Attendance query.
    Returns weekly attendance summary for Agniveers.
    """
    try:
        base_where, base_params = _build_attendance_base_scope(intent)
        
        # ── Resolve date range ──────────────────────────────────────────────
        from_date, to_date = _resolve_attendance_dates("weekly", intent)

        # ── Build SQL ──────────────────────────────────────────────────────────
        sql = f"""
        WITH AgniveerScope AS (
            SELECT
                a.Id,
                a.AgniveerNo,
                a.FullName,
                a.PhotoPath,
                a.DateOfJoining,
                p.Name AS PlatoonName,
                c.Name AS CompanyName,
                b.BatchName
            FROM AgniveerMaster a
            LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
            LEFT JOIN CompanyMaster c ON c.Id = p.CompanyId
            LEFT JOIN BatchMaster b ON b.Id = a.BatchId
            WHERE {base_where}
        ),
        DateRange AS (
            SELECT CAST(? AS DATE) AS AttendanceDate
            UNION ALL
            SELECT DATEADD(DAY, 1, AttendanceDate)
            FROM DateRange
            WHERE AttendanceDate < CAST(? AS DATE)
        ),
        LeaveRecords AS (
            SELECT
                l.AgniveerId,
                l.FromDate,
                l.ToDate,
                l.IsAbscondedLeave
            FROM AgniveerLeaveMaster l
            WHERE l.AgniveerId IN (SELECT Id FROM AgniveerScope)
                AND l.FromDate IS NOT NULL
                AND l.FromDate <= CAST(? AS DATE)
                AND (l.ToDate IS NULL OR l.ToDate >= CAST(? AS DATE))
        ),
        WeeklyStatus AS (
            SELECT
                a.Id AS AgniveerId,
                a.AgniveerNo,
                a.FullName,
                a.PhotoPath,
                a.PlatoonName,
                a.CompanyName,
                a.BatchName,
                d.AttendanceDate,
                DATEADD(WEEK, DATEDIFF(WEEK, 0, d.AttendanceDate), 0) AS WeekStart,
                CASE
                    WHEN d.AttendanceDate < a.DateOfJoining THEN NULL
                    WHEN EXISTS (
                        SELECT 1 FROM LeaveRecords l
                        WHERE l.AgniveerId = a.Id
                            AND l.ToDate IS NOT NULL
                            AND d.AttendanceDate BETWEEN l.FromDate AND l.ToDate
                    ) THEN 0
                    WHEN EXISTS (
                        SELECT 1 FROM LeaveRecords l
                        WHERE l.AgniveerId = a.Id
                            AND l.ToDate IS NULL
                            AND l.IsAbscondedLeave = 1
                            AND d.AttendanceDate >= l.FromDate
                    ) THEN 0
                    ELSE 1
                END AS IsPresent
            FROM AgniveerScope a
            CROSS JOIN DateRange d
        )
        SELECT
            AgniveerNo,
            FullName,
            PhotoPath,
            PlatoonName,
            CompanyName,
            BatchName,
            WeekStart,
            SUM(CASE WHEN IsPresent = 1 THEN 1 ELSE 0 END) AS Present,
            SUM(CASE WHEN IsPresent = 0 THEN 1 ELSE 0 END) AS Absent
        FROM WeeklyStatus
        WHERE IsPresent IS NOT NULL
        GROUP BY AgniveerNo, FullName, PhotoPath, PlatoonName, CompanyName, BatchName, WeekStart
        ORDER BY AgniveerNo ASC, WeekStart ASC
        OPTION (MAXRECURSION 366)
        """
        
        is_valid, err = sql_validator.validate_sql(sql)
        if not is_valid:
            return None, f"Weekly Attendance SQL validation failed: {err}"
        
        params = base_params + [from_date, to_date, to_date, from_date]
        rows, run_err = run_readonly(sql, params)
        if run_err:
            return None, f"Weekly Attendance execution failed: {run_err}"
        
        return _to_section(rows or [], intent, sql=sql), None

    except Exception as exc:
        logger.error("Weekly Attendance failed: %s", exc, exc_info=True)
        return None, str(exc)


def _execute_attendance_monthly(intent: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Execute Monthly Attendance query.
    Returns monthly attendance summary for Agniveers.
    """
    try:
        base_where, base_params = _build_attendance_base_scope(intent)
        
        # ── Resolve date range ──────────────────────────────────────────────
        from_date, to_date = _resolve_attendance_dates("monthly", intent)

        # ── Build SQL ──────────────────────────────────────────────────────────
        sql = f"""
        WITH AgniveerScope AS (
            SELECT
                a.Id,
                a.AgniveerNo,
                a.FullName,
                a.PhotoPath,
                a.DateOfJoining,
                p.Name AS PlatoonName,
                c.Name AS CompanyName,
                b.BatchName
            FROM AgniveerMaster a
            LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
            LEFT JOIN CompanyMaster c ON c.Id = p.CompanyId
            LEFT JOIN BatchMaster b ON b.Id = a.BatchId
            WHERE {base_where}
        ),
        DateRange AS (
            SELECT CAST(? AS DATE) AS AttendanceDate
            UNION ALL
            SELECT DATEADD(DAY, 1, AttendanceDate)
            FROM DateRange
            WHERE AttendanceDate < CAST(? AS DATE)
        ),
        LeaveRecords AS (
            SELECT
                l.AgniveerId,
                l.FromDate,
                l.ToDate,
                l.IsAbscondedLeave
            FROM AgniveerLeaveMaster l
            WHERE l.AgniveerId IN (SELECT Id FROM AgniveerScope)
                AND l.FromDate IS NOT NULL
                AND l.FromDate <= CAST(? AS DATE)
                AND (l.ToDate IS NULL OR l.ToDate >= CAST(? AS DATE))
        ),
        MonthlyStatus AS (
            SELECT
                a.Id AS AgniveerId,
                a.AgniveerNo,
                a.FullName,
                a.PhotoPath,
                a.PlatoonName,
                a.CompanyName,
                a.BatchName,
                d.AttendanceDate,
                FORMAT(d.AttendanceDate, 'MM-yyyy') AS Month,
                CASE
                    WHEN d.AttendanceDate < a.DateOfJoining THEN NULL
                    WHEN EXISTS (
                        SELECT 1 FROM LeaveRecords l
                        WHERE l.AgniveerId = a.Id
                            AND l.ToDate IS NOT NULL
                            AND d.AttendanceDate BETWEEN l.FromDate AND l.ToDate
                    ) THEN 0
                    WHEN EXISTS (
                        SELECT 1 FROM LeaveRecords l
                        WHERE l.AgniveerId = a.Id
                            AND l.ToDate IS NULL
                            AND l.IsAbscondedLeave = 1
                            AND d.AttendanceDate >= l.FromDate
                    ) THEN 0
                    ELSE 1
                END AS IsPresent
            FROM AgniveerScope a
            CROSS JOIN DateRange d
        )
        SELECT
            AgniveerNo,
            FullName,
            PhotoPath,
            PlatoonName,
            CompanyName,
            BatchName,
            Month,
            SUM(CASE WHEN IsPresent = 1 THEN 1 ELSE 0 END) AS Present,
            SUM(CASE WHEN IsPresent = 0 THEN 1 ELSE 0 END) AS Absent
        FROM MonthlyStatus
        WHERE IsPresent IS NOT NULL
        GROUP BY AgniveerNo, FullName, PhotoPath, PlatoonName, CompanyName, BatchName, Month
        ORDER BY AgniveerNo ASC, Month ASC
        OPTION (MAXRECURSION 366)
        """
        
        is_valid, err = sql_validator.validate_sql(sql)
        if not is_valid:
            return None, f"Monthly Attendance SQL validation failed: {err}"
        
        params = base_params + [from_date, to_date, to_date, from_date]
        rows, run_err = run_readonly(sql, params)
        if run_err:
            return None, f"Monthly Attendance execution failed: {run_err}"
        
        return _to_section(rows or [], intent, sql=sql), None

    except Exception as exc:
        logger.error("Monthly Attendance failed: %s", exc, exc_info=True)
        return None, str(exc)


def _execute_attendance_individual(intent: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Execute Individual Attendance History query.
    Returns full attendance history for a single Agniveer.
    """
    try:
        agniveer_no = intent.get("agniveer_no") or intent.get("agniveerNo")
        
        if not agniveer_no:
            return None, "AgniveerNo required for individual attendance history"

        base_where, base_params = _build_attendance_base_scope(intent)
        
        # ── Resolve date range ──────────────────────────────────────────────
        from_date, to_date = _resolve_attendance_dates("monthly", intent)

        # ── Build SQL ──────────────────────────────────────────────────────────
        sql = f"""
        WITH AgniveerInfo AS (
            SELECT
                a.Id,
                a.AgniveerNo,
                a.FullName,
                a.PhotoPath,
                a.DateOfJoining,
                p.Name AS PlatoonName,
                c.Name AS CompanyName
            FROM AgniveerMaster a
            LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
            LEFT JOIN CompanyMaster c ON c.Id = p.CompanyId
            WHERE {base_where}
        ),
        DateRange AS (
            SELECT CAST(? AS DATE) AS AttendanceDate
            UNION ALL
            SELECT DATEADD(DAY, 1, AttendanceDate)
            FROM DateRange
            WHERE AttendanceDate < CAST(? AS DATE)
        ),
        LeaveRecords AS (
            SELECT
                l.FromDate,
                l.ToDate,
                l.IsAbscondedLeave
            FROM AgniveerLeaveMaster l
            WHERE l.AgniveerId = (SELECT Id FROM AgniveerInfo)
                AND l.FromDate IS NOT NULL
        ),
        DailyStatus AS (
            SELECT
                d.AttendanceDate,
                CASE
                    WHEN d.AttendanceDate < a.DateOfJoining THEN NULL
                    WHEN EXISTS (
                        SELECT 1 FROM LeaveRecords l
                        WHERE l.ToDate IS NOT NULL
                            AND d.AttendanceDate BETWEEN l.FromDate AND l.ToDate
                    ) THEN 0
                    WHEN EXISTS (
                        SELECT 1 FROM LeaveRecords l
                        WHERE l.ToDate IS NULL
                            AND l.IsAbscondedLeave = 1
                            AND d.AttendanceDate >= l.FromDate
                    ) THEN 0
                    ELSE 1
                END AS IsPresent
            FROM DateRange d
            CROSS JOIN AgniveerInfo a
        )
        SELECT
            AttendanceDate AS Date,
            IsPresent,
            a.AgniveerNo,
            a.FullName,
            a.PhotoPath,
            a.PlatoonName,
            a.CompanyName
        FROM DailyStatus d
        CROSS JOIN AgniveerInfo a
        ORDER BY AttendanceDate ASC
        OPTION (MAXRECURSION 366)
        """
        
        is_valid, err = sql_validator.validate_sql(sql)
        if not is_valid:
            return None, f"Individual Attendance SQL validation failed: {err}"
        
        params = base_params + [from_date, to_date]
        rows, run_err = run_readonly(sql, params)
        if run_err:
            return None, f"Individual Attendance execution failed: {run_err}"
        
        return _to_section(rows or [], intent, sql=sql), None

    except Exception as exc:
        logger.error("Individual Attendance failed: %s", exc, exc_info=True)
        return None, str(exc)


def execute_attendance_query(intent: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
    """
    Dispatch Attendance queries based on operation.
    """
    operation = str(intent.get("operation") or intent.get("subcategory") or "Summary").lower()
    
    if operation == "summary":
        return _execute_attendance_summary(intent)
    elif operation == "daily":
        return _execute_attendance_daily(intent)
    elif operation == "weekly":
        return _execute_attendance_weekly(intent)
    elif operation == "monthly":
        return _execute_attendance_monthly(intent)
    elif operation in ("present", "absent"):
        return _execute_attendance_summary(intent)  # Same as summary
    elif operation == "individual":
        return _execute_attendance_individual(intent)
    else:
        # Default to Summary
        return _execute_attendance_summary(intent)


# ── Distribution Category ───────────────────────────────────────────────────

DISTRIBUTION_GROUP_FIELDS = {
    "unit": ["TeamId", "TeamName"],
    "company": ["CompanyName"],
    "batch": ["BatchName"],
}


def _build_distribution_base_scope(intent: Dict[str, Any]) -> Tuple[str, List[Any]]:
    """Build the base WHERE clause for distribution queries."""
    clauses = ["(a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)", "a.IsActive = 1"]
    params: List[Any] = []

    agniveer_no = intent.get("agniveer_no") or intent.get("agniveerNo")
    batch_id = intent.get("batch_id") or intent.get("batchId")
    platoon_id = intent.get("platoon_id") or intent.get("platoonId")
    company_id = intent.get("company_id") or intent.get("companyId")

    if agniveer_no:
        clauses.append("LOWER(a.AgniveerNo) LIKE '%' + LOWER(?) + '%'")
        params.append(str(agniveer_no))
    if batch_id is not None:
        clauses.append("a.BatchId = ?")
        params.append(int(batch_id))
    if platoon_id is not None:
        clauses.append("a.PlatoonId = ?")
        params.append(int(platoon_id))
    if company_id is not None:
        clauses.append("EXISTS (SELECT 1 FROM PlatoonMaster p WHERE p.Id = a.PlatoonId AND p.CompanyId = ?)")
        params.append(int(company_id))

    return " AND ".join(clauses), params


def _get_latest_distribution_id() -> Optional[int]:
    """Get the latest distribution ID from DistributionHistoryMaster."""
    sql = "SELECT MAX(DistributionId) AS DistributionId FROM DistributionHistoryMaster"
    rows, err = run_readonly(sql, [])
    if err or not rows:
        return None
    row = rows[0] if rows else {}
    return row.get("DistributionId") if row.get("DistributionId") is not None else row.get("distributionId")


def _execute_distribution_latest(intent: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Execute Latest Distribution query.
    Returns the most recent distribution with team breakdown.
    
    C# Equivalent: Cmd17_LatestUnitDistribution
    """
    try:
        response_type = str(intent.get("responseType") or intent.get("response_type") or "Summary")
        detailed = response_type.lower() == "detailed"
        
        base_where, base_params = _build_distribution_base_scope(intent)
        
        # ── Get latest distribution ID ──────────────────────────────────────
        latest_id = _get_latest_distribution_id()
        if latest_id is None:
            return _to_section([], intent), "No distribution events found."

        # ── Short: Team summary ─────────────────────────────────────────────
        if not detailed:
            sql = f"""
            SELECT
                dm.Id AS TeamId,
                dm.Name AS TeamName,
                COUNT(h.AgniveerId) AS MemberCount
            FROM DistributionHistoryMaster h
            INNER JOIN DistributionMaster dm ON dm.Id = h.TeamId
            INNER JOIN AgniveerMaster a ON a.Id = h.AgniveerId
            WHERE h.DistributionId = ?
                AND {base_where}
                AND h.TeamId IS NOT NULL
            GROUP BY dm.Id, dm.Name
            ORDER BY MIN(h.Rank) ASC
            """
            
            is_valid, err = sql_validator.validate_sql(sql)
            if not is_valid:
                return None, f"Latest Distribution SQL validation failed: {err}"
            
            params = [latest_id] + base_params
            rows, run_err = run_readonly(sql, params)
            if run_err:
                return None, f"Latest Distribution execution failed: {run_err}"
            
            # Get distribution date
            date_sql = "SELECT TOP 1 InsertedDate AS DistributionDate FROM DistributionHistoryMaster WHERE DistributionId = ?"
            date_rows, _ = run_readonly(date_sql, [latest_id])
            dist_date = date_rows[0].get("DistributionDate") if date_rows else None
            
            result = {
                "distributionId": latest_id,
                "distributionDate": dist_date,
                "teams": rows or [],
            }
            return _to_section([result], intent, sql=sql), None

        # ── Detailed: Full member list ──────────────────────────────────────
        sql = f"""
        SELECT
            h.DistributionId,
            h.InsertedDate AS DistributionDate,
            h.TeamId,
            dm.Name AS TeamName,
            a.AgniveerNo,
            a.FullName,
            a.PhotoPath,
            a.Class,
            p.Name AS PlatoonName,
            c.Name AS CompanyName,
            b.BatchName,
            h.Rank
        FROM DistributionHistoryMaster h
        INNER JOIN AgniveerMaster a ON a.Id = h.AgniveerId
        LEFT JOIN DistributionMaster dm ON dm.Id = h.TeamId
        LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
        LEFT JOIN CompanyMaster c ON c.Id = p.CompanyId
        LEFT JOIN BatchMaster b ON b.Id = a.BatchId
        WHERE h.DistributionId = ?
            AND {base_where}
            AND h.TeamId IS NOT NULL
        ORDER BY h.TeamId ASC, h.Rank ASC
        """
        
        is_valid, err = sql_validator.validate_sql(sql)
        if not is_valid:
            return None, f"Latest Distribution SQL validation failed: {err}"
        
        params = [latest_id] + base_params
        rows, run_err = run_readonly(sql, params)
        if run_err:
            return None, f"Latest Distribution execution failed: {run_err}"
        
        return _to_section(rows or [], intent, sql=sql), None

    except Exception as exc:
        logger.error("Latest Distribution failed: %s", exc, exc_info=True)
        return None, str(exc)


def _execute_distribution_by_unit(intent: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Execute By Unit query.
    Returns Agniveers assigned to a specific unit.
    
    C# Equivalent: Cmd19_AgniveersInUnit
    """
    try:
        unit_name = intent.get("unit_name") or intent.get("unitName") or intent.get("team_name") or intent.get("teamName")
        
        if not unit_name:
            return None, "UnitName required for by unit query"

        # ── Resolve unit ID ──────────────────────────────────────────────────
        lookup_sql = """
        SELECT TOP 1 Id AS UnitId
        FROM DistributionMaster
        WHERE LOWER(Name) LIKE '%' + LOWER(?) + '%'
        """
        is_valid, err = sql_validator.validate_sql(lookup_sql)
        if not is_valid:
            return None, f"Unit lookup SQL validation failed: {err}"
            
        rows, run_err = run_readonly(lookup_sql, [str(unit_name)])
        if run_err:
            return None, f"Unit lookup failed: {run_err}"
        if not rows:
            return _to_section([], intent), f"Unit '{unit_name}' not found"

        unit_id = rows[0].get("UnitId") if rows[0].get("UnitId") is not None else rows[0].get("unitId")
        base_where, base_params = _build_distribution_base_scope(intent)

        # ── Build SQL ──────────────────────────────────────────────────────────
        sql = f"""
        SELECT
            a.AgniveerNo,
            a.FullName,
            a.PhotoPath,
            a.Class,
            p.Name AS PlatoonName,
            c.Name AS CompanyName,
            b.BatchName,
            h.Rank,
            h.DistributionId,
            h.InsertedDate AS DistributionDate
        FROM DistributionHistoryMaster h
        INNER JOIN AgniveerMaster a ON a.Id = h.AgniveerId
        LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
        LEFT JOIN CompanyMaster c ON c.Id = p.CompanyId
        LEFT JOIN BatchMaster b ON b.Id = a.BatchId
        WHERE h.TeamId = ?
            AND {base_where}
        ORDER BY h.Rank ASC
        """
        
        is_valid, err = sql_validator.validate_sql(sql)
        if not is_valid:
            return None, f"By Unit SQL validation failed: {err}"
        
        params = [unit_id] + base_params
        rows, run_err = run_readonly(sql, params)
        if run_err:
            return None, f"By Unit execution failed: {run_err}"
        
        return _to_section(rows or [], intent, sql=sql), None

    except Exception as exc:
        logger.error("By Unit failed: %s", exc, exc_info=True)
        return None, str(exc)


def _execute_distribution_unassigned(intent: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Execute Unassigned query.
    Returns Agniveers not assigned to any unit.
    
    C# Equivalent: Cmd20_UnassignedAgniveers
    """
    try:
        base_where, base_params = _build_distribution_base_scope(intent)

        sql = f"""
        SELECT
            a.AgniveerNo,
            a.FullName,
            a.PhotoPath,
            a.Class,
            p.Name AS PlatoonName,
            c.Name AS CompanyName,
            b.BatchName
        FROM AgniveerMaster a
        LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
        LEFT JOIN CompanyMaster c ON c.Id = p.CompanyId
        LEFT JOIN BatchMaster b ON b.Id = a.BatchId
        WHERE {base_where}
            AND NOT EXISTS (
                SELECT 1 FROM DistributionHistoryMaster h
                WHERE h.AgniveerId = a.Id
            )
        ORDER BY a.AgniveerNo ASC
        """
        
        is_valid, err = sql_validator.validate_sql(sql)
        if not is_valid:
            return None, f"Unassigned SQL validation failed: {err}"
        
        rows, run_err = run_readonly(sql, base_params)
        if run_err:
            return None, f"Unassigned execution failed: {run_err}"
        
        return _to_section(rows or [], intent, sql=sql), None

    except Exception as exc:
        logger.error("Unassigned failed: %s", exc, exc_info=True)
        return None, str(exc)


def _execute_distribution_top_unit(intent: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Execute Top Unit query.
    Returns the unit with most Agniveers in latest distribution.
    
    C# Equivalent: Cmd21_TopUnitLatestDistribution
    """
    try:
        latest_id = _get_latest_distribution_id()
        if latest_id is None:
            return _to_section([], intent), "No distribution events found."

        sql = f"""
        SELECT TOP 1
            h.TeamId,
            dm.Name AS TeamName,
            COUNT(h.AgniveerId) AS AgniveerCount,
            ? AS DistributionEventId,
            (SELECT TOP 1 InsertedDate FROM DistributionHistoryMaster WHERE DistributionId = ?) AS DistributionDate
        FROM DistributionHistoryMaster h
        INNER JOIN AgniveerMaster a ON a.Id = h.AgniveerId
        LEFT JOIN DistributionMaster dm ON dm.Id = h.TeamId
        WHERE h.DistributionId = ?
            AND h.TeamId IS NOT NULL
            AND (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
            AND a.IsActive = 1
        GROUP BY h.TeamId, dm.Name
        ORDER BY AgniveerCount DESC
        """
        
        is_valid, err = sql_validator.validate_sql(sql)
        if not is_valid:
            return None, f"Top Unit SQL validation failed: {err}"
        
        params = [latest_id, latest_id, latest_id]
        rows, run_err = run_readonly(sql, params)
        if run_err:
            return None, f"Top Unit execution failed: {run_err}"
        
        return _to_section(rows or [], intent, sql=sql), None

    except Exception as exc:
        logger.error("Top Unit failed: %s", exc, exc_info=True)
        return None, str(exc)


def _execute_distribution_history(intent: Dict[str, Any]) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Execute Distribution History for a specific Agniveer.
    """
    try:
        agniveer_no = intent.get("agniveer_no") or intent.get("agniveerNo")
        
        if not agniveer_no:
            return None, "AgniveerNo required for distribution history"

        # ── Get Agniveer ID ──────────────────────────────────────────────────
        lookup_sql = """
        SELECT TOP 1 Id 
        FROM AgniveerMaster 
        WHERE LOWER(AgniveerNo) = LOWER(?)
            AND (IsDisqualified <> 1 OR IsDisqualified IS NULL)
            AND IsActive = 1
        """
        rows, err = run_readonly(lookup_sql, [str(agniveer_no)])
        if err or not rows:
            return _to_section([], intent), f"Agniveer '{agniveer_no}' not found"
        
        agniveer_id = rows[0].get("Id") if rows[0].get("Id") is not None else rows[0].get("id")

        sql = f"""
        SELECT
            h.DistributionId,
            h.InsertedDate AS DistributionDate,
            dm.Name AS UnitName,
            h.Rank,
            h.Location,
            h.UpdateCount
        FROM DistributionHistoryMaster h
        LEFT JOIN DistributionMaster dm ON dm.Id = h.TeamId
        WHERE h.AgniveerId = ?
        ORDER BY h.InsertedDate DESC
        """
        
        is_valid, err = sql_validator.validate_sql(sql)
        if not is_valid:
            return None, f"Distribution History SQL validation failed: {err}"
        
        rows, run_err = run_readonly(sql, [agniveer_id])
        if run_err:
            return None, f"Distribution History execution failed: {run_err}"
        
        return _to_section(rows or [], intent, sql=sql), None

    except Exception as exc:
        logger.error("Distribution History failed: %s", exc, exc_info=True)
        return None, str(exc)


def execute_distribution_query(intent: Dict[str, Any]) -> Tuple[Any, Optional[str]]:
    """
    Dispatch Distribution queries based on operation.
    """
    operation = str(intent.get("operation") or intent.get("subcategory") or "Latest").lower()
    
    if operation in ("latest", "latestdistribution", "latest_distribution"):
        return _execute_distribution_latest(intent)
    elif operation in ("byunit", "by_unit", "inunit", "in_unit"):
        return _execute_distribution_by_unit(intent)
    elif operation == "unassigned":
        return _execute_distribution_unassigned(intent)
    elif operation in ("topunit", "top_unit"):
        return _execute_distribution_top_unit(intent)
    elif operation in ("history", "records"):
        return _execute_distribution_history(intent)
    else:
        # Default to Latest
        return _execute_distribution_latest(intent)







def _resolve_attendance_range(intent: Dict) -> Tuple[Optional[str], Optional[str]]:
    """Returns (range_start, range_end) as YYYY-MM-DD date strings.
    Handles Daily, Weekly, Monthly, Summary operations cleanly.

    Delegates the actual phrase parsing to date_resolver.resolve_date_range
    so relative phrases in intent["date"]/"from_date"/"to_date" ("today",
    "last 7 days", "last week", "first week of July", ...) are understood
    here too — this used to just slice the raw string to 10 characters
    (str(single_date)[:10]), which only worked for values that were already
    plain ISO dates; a phrase like "last 7 days" silently truncated to the
    nonsense string "last 7 d" instead of being resolved to an actual range.
    """
    from intent_engine.date_resolver import resolve_date_range as _resolve_dates

    operation = intent.get("operation")
    # resolve_date_range() only has built-in operation defaults for
    # Daily/Weekly/Monthly — Summary means the same as Monthly here.
    _resolve_op = "Monthly" if operation == "Summary" else operation
    single_date = intent.get("date")
    from_date = intent.get("from_date")
    to_date = intent.get("to_date")

    resolved_date, resolved_from, resolved_to = _resolve_dates(
        operation=_resolve_op,
        date=single_date,
        from_date=from_date,
        to_date=to_date,
    )
    if resolved_date:
        return resolved_date[:10], resolved_date[:10]
    if resolved_from or resolved_to:
        start = resolved_from[:10] if resolved_from else None
        end = resolved_to[:10] if resolved_to else start
        start = start or end
        return start, end

    if operation == "Daily":
        day = datetime.date.today().isoformat()
        return day, day
    return None, None



# ── Safety validator ───────────────────────────────────────────────────────
def validate_sql(sql: str) -> Optional[str]:
    """Return an error string if the SQL is unsafe, else None.
    Delegates to the canonical sql_validator.validate_sql to remove duplicate logic.
    """
    from sql_validator import sql_validator

    is_valid, err = sql_validator.validate_sql(sql)
    if not is_valid:
        return err or "SQL validation failed."
    return None


# ── SQL generation (LLM) ───────────────────────────────────────────────────
# Active Tier 2 LLM-based SQL generation fallback for handling complex queries
# or filters that the golden path and query_planner_v2's AST path cannot express.
def generate_sql(
    question: str, intent: Optional[Dict] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Generate a single SELECT for `question`. Returns (sql, error)."""
    try:
        from config import DEFAULT_MODEL, ollama_session
        from ollama_cpu_chat import chat_with_fallback
        from sql_schema_guard import generate_dynamic_schema_card
        from business_rules import LLM_HARD_RULES
    except Exception as exc:  # pragma: no cover
        return None, f"LLM unavailable: {exc}"

    hint = ""
    if intent:
        hint = f"\nCLASSIFIER HINT (may be partial): {intent}\n"

    project_overview = _build_project_overview_context()
    dynamic_schema = generate_dynamic_schema_card()

    if SQL_SERVER_2008_COMPAT:
        dialect_hint = "\nDIALECT: SQL Server 2008 target. DO NOT use STRING_AGG, OFFSET/FETCH, IIF, CONCAT, or LAG/LEAD. Use TOP (n), STUFF+FOR XML PATH for aggregation, ROW_NUMBER() for paging.\n"
    else:
        dialect_hint = ""

    user = (
        f"{project_overview}\n\n"
        f"{dynamic_schema}\n\n"
        f"{LLM_HARD_RULES}\n"
        f"{dialect_hint}\n"
        f"{hint}\n"
        f"QUESTION: {question}\n"
        "SQL:"
    )
    try:
        result = chat_with_fallback(
            ollama_session,
            DEFAULT_MODEL,
            [
                {"role": "system", "content": _GENERATION_SYSTEM},
                {"role": "user", "content": user},
            ],
            stream_tokens=False,
        )
        text = str(getattr(result, "text", "") or "")
    except Exception as exc:
        return None, f"SQL generation failed: {exc}"

    sql = _extract_sql(text)
    if not sql or sql.strip().upper() == "CANNOT_ANSWER":
        return None, "CANNOT_ANSWER"
    return sql, None


def _extract_sql(text: str) -> str:
    """Strip markdown fences / stray prose, keep the SQL body."""
    t = text.strip()
    t = re.sub(r"```(?:sql)?", "", t, flags=re.IGNORECASE).strip()
    # Keep from the first SELECT/WITH onward.
    m = re.search(r"\b(with|select)\b", t, re.IGNORECASE)
    sql = t[m.start() :].strip() if m else t
    # Fix common LLM hallucination where it thinks AgniveerMaster has AgniveerId
    sql = re.sub(r"\ba\.AgniveerId\b", "a.Id", sql, flags=re.IGNORECASE)
    return sql


def _to_camel_case(name: str) -> str:
    """PascalCase SQL column name -> camelCase, matching System.Text.Json's
    default camelCase naming policy — the same simple "lowercase the first
    character only" rule .NET's JSON serialization uses, and the convention
    universal_normalizer.py / cross_filter_engine.py / compare_engine.py
    (and every test fixture for them) are built against (e.g. "agniveerNo",
    "fullName"). SQL Server's cursor.description returns column names
    verbatim ("AgniveerNo", "FullName"); without this conversion those keys
    never match _ID_FIELD_PRIORITY in either module, so no SQL-backend row
    is recognized as an Agniveer record and cross-filter intersection
    silently finds nothing to match on.

    Column names that are already all-uppercase acronym runs at the start
    (e.g. "OMRInputTotal") need more than "lowercase the first character":
    naively lowering only the first character produces "oMRInputTotal",
    which does NOT match utils.py's _SCORE_FIELDS entry "omrInputTotal" —
    the score silently fails to be found for any row shaped like this.
    System.Text.Json's camelCase policy lowercases the *entire* leading
    acronym run up to (but not including) the next capital that starts a
    new word, e.g. "OMRInputTotal" -> "omrInputTotal", "ID" -> "id".
    """
    if not name:
        return name
    if len(name) == 1:
        return name.lower()
    # Find the leading run of uppercase letters.
    upper_run_end = 0
    while upper_run_end < len(name) and name[upper_run_end].isupper():
        upper_run_end += 1
    if upper_run_end <= 1:
        # Plain PascalCase start ("FullName") — lowercase just the first char.
        return name[0].lower() + name[1:]
    if upper_run_end == len(name):
        # Entire name is uppercase (e.g. an acronym-only column) — lowercase all of it.
        return name.lower()
    # Leading acronym run followed by a new word (e.g. "OMRInputTotal"):
    # lowercase the whole run except its last character, which belongs to
    # the next word ("OMR" + "Input..." -> "omr" + "Input..." -> "omrInput...").
    return name[: upper_run_end - 1].lower() + name[upper_run_end - 1 :]


def _camel_case_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {_to_camel_case(k): _jsonable(v) for k, v in row.items()}


def _to_section(
    rows: List[Dict[str, Any]], intent: Optional[Dict] = None, sql: Optional[str] = None
) -> Dict[str, Any]:
    """Wrap flat rows in the same envelope shape the .NET path produces, so
    `universal_normalizer.normalize_response()` / `result_combiner._extract_records()`
    resolve the rows directly instead of falling back to raw-row scanning.

    Column names are camelCased here (see `_to_camel_case`) so the row shape
    actually matches that .NET envelope's key casing, not just its nesting.
    """
    camel_rows = [_camel_case_row(r) for r in rows]
    res = {
        "success": True,
        "records": camel_rows,
        "data": camel_rows,
        "count": len(camel_rows),
    }
    if sql:
        res["sql"] = sql
    return res


# ── Read-only execution ────────────────────────────────────────────────────
def run_readonly(
    sql: str, params: Optional[List[Any]] = None, max_rows: Optional[int] = None
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Execute a validated SELECT against the READ-ONLY login. (rows, error).

    `max_rows` is the belt-and-suspenders ROWCOUNT cap for THIS query — pass
    the same limit (or None) used to build its TOP clause so the two agree.
    Omitting it (None) falls back to SQL_MAX_ROWS, the safety net for callers
    that don't compute their own explicit limit (e.g. the LLM text2sql
    fallback, where "cap unbounded/hallucinated SQL" is exactly the point).
    Pass 0 explicitly to mean "no cap" (SQL Server's SET ROWCOUNT 0).
    """
    if not SQL_READONLY_CONN:
        return None, "SQL_READONLY_CONN is not configured."
    try:
        import pyodbc  # imported lazily so the rest of AgniAI runs without it

        pyodbc.pooling = True
    except Exception as exc:  # pragma: no cover
        return None, f"pyodbc not installed: {exc}"

    conn = None
    try:
        # Enforce transaction timeout limit between 1 and 30 seconds to avoid thread starvation
        timeout_limit = max(1, min(SQL_COMMAND_TIMEOUT_S, 30))
        conn = pyodbc.connect(SQL_READONLY_CONN, timeout=timeout_limit, autocommit=True)
        conn.timeout = timeout_limit
        cur = conn.cursor()
        # Belt-and-suspenders row cap regardless of query shape (works on 2008).
        effective_cap = SQL_MAX_ROWS if max_rows is None else max_rows
        cur.execute(f"SET ROWCOUNT {effective_cap}")
        if params:
            cur.execute(sql, tuple(params))
        else:
            cur.execute(sql)
        cols = [d[0] for d in cur.description]

        rows = []
        while True:
            chunk = cur.fetchmany(100)
            if not chunk:
                break
            for r in chunk:
                rows.append(dict(zip(cols, r)))

        cur.execute("SET ROWCOUNT 0")
        return rows, None
    except Exception as exc:
        # Never leak the raw SQL / connection details to the caller.
        logger.warning(
            "SQL execution error: %s | %s\nQuery: %s", type(exc).__name__, str(exc), sql
        )
        return None, "The generated query could not be executed against the database."
    finally:
        # Always release the connection back to the pool, even if execution
        # raised partway through (bad syntax, timeout, table not found,
        # etc.) — previously conn.close() sat at the end of the try block
        # and was skipped on any exception, leaking a live connection every
        # time a query failed and eventually exhausting the DB pool.
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def get_batch_ids_for_agniveers(
    agniveer_nos: List[str],
) -> Dict[str, Optional[int]]:
    """Ground-truth AgniveerNo -> BatchId lookup straight from AgniveerMaster.

    Per-builder SQL applies a BatchId filter inconsistently (some aggregate
    builders and the text2sql fallback have no code-level guarantee), so
    admin_pipeline.enforce_batch_scope uses this after the fact to verify
    every agniveer a query leg returned really belongs to the batch the
    frontend passed, dropping any that don't.
    """
    unique_nos = sorted({str(no) for no in agniveer_nos if no})
    if not unique_nos:
        return {}
    result: Dict[str, Optional[int]] = {}
    # Chunked to stay well under SQL Server's ~2100 parameter limit.
    chunk_size = 900
    for start in range(0, len(unique_nos), chunk_size):
        chunk = unique_nos[start : start + chunk_size]
        placeholders = ", ".join("?" for _ in chunk)
        sql = f"SELECT AgniveerNo, BatchId FROM AgniveerMaster WHERE AgniveerNo IN ({placeholders})"
        # max_rows=0 (no cap): a chunk can carry up to 900 AgniveerNos, more
        # than the default SQL_MAX_ROWS (500) ROWCOUNT safety net, which
        # would otherwise silently drop the tail of every large chunk.
        rows, err = run_readonly(sql, chunk, max_rows=0)
        if err or not rows:
            continue
        for row in rows:
            no = row.get("AgniveerNo")
            if no is not None:
                result[str(no)] = row.get("BatchId")
    return result


def _org_scope_sql(
    alias: str,
    intent: Dict[str, Any],
) -> Tuple[str, List[Any]]:
    """Build the ' AND ...' fragment scoping `alias` (an AgniveerMaster
    alias) to whichever of batch/platoon/company the request carries.

    Several raw-SQL fast paths below only wired up a subset of these three
    (e.g. Verification wired none, Equipment/PersonalDetails wired batch but
    not company) — each one independently re-implementing this by hand is
    exactly how those gaps happened. Centralising it here means a query
    scoped to a specific batch/platoon/company (from the frontend's batchId,
    or from a resolved company/platoon NAME upstream in
    admin_entity_resolver.py) is honoured by every fast path, not just the
    ones someone remembered to wire it into.
    """
    batch_id = intent.get("batch_id") or intent.get("batchId")
    platoon_id = intent.get("platoon_id") or intent.get("platoonId")
    company_id = intent.get("company_id") or intent.get("companyId")

    clauses: List[str] = []
    params: List[Any] = []
    if batch_id is not None:
        clauses.append(f"{alias}.BatchId = ?")
        params.append(int(batch_id))
    if platoon_id is not None:
        clauses.append(f"{alias}.PlatoonId = ?")
        params.append(int(platoon_id))
    if company_id is not None:
        clauses.append(
            f"EXISTS (SELECT 1 FROM PlatoonMaster p WHERE p.Id = {alias}.PlatoonId AND p.CompanyId = ?)"
        )
        params.append(int(company_id))

    if not clauses:
        return "", []
    return " AND " + " AND ".join(clauses), params


# ── Public entrypoint — AST Pipeline ──────
def execute_sql_query(
    payload: Optional[Dict] = None,
    *,
    question: str = "",
    intent: Optional[Dict] = None,
    trace_id: Optional[str] = None,
    **_ignored: Any,
) -> Tuple[Any, Optional[str]]:
    """
    Executes a query via the AST semantic compilation pipeline.
    """
    logger.debug(f"[DEBUG SQL EXECUTOR] question: {question!r}")
    if not intent:
        return None, "No intent provided to query planner."

    _raw_q = (question or intent.get("raw_query") or "").lower()

    if intent.get("query_type") == "text2sql":
        try:
            sql, gen_err = generate_sql(question, intent)
            if gen_err:
                metrics_hook("cannot_answer")
                return None, f"Fallback SQL generation failed: {gen_err}"

            is_sql_valid, sql_err = sql_validator.validate_sql(sql)
            if not is_sql_valid:
                metrics_hook("validator_rejected")
                return None, f"Fallback SQL validation failed: {sql_err}"

            rows, run_err = run_readonly(sql, [])
            if run_err:
                metrics_hook("exec_error")
                return None, f"Fallback SQL execution failed: {run_err}"

            metrics_hook("generated")
            metrics_hook("llm_fallback")
            res = _to_section(rows or [], intent, sql=sql)
            return res, None
        except Exception as exc:
            logger.error("Explicit text2sql fallback failed: %s", exc, exc_info=True)
            return None, f"Fallback LLM pipeline failed: {exc}"
        
    if intent.get("category") == "Performance":
        _cutoff_raw_q = (question or intent.get("raw_query") or "").lower()
        # "What's the cutoff for X" asks for the cutoff VALUE itself
        # (ScoreSubItemMaster.Cutoff — a real stored column, no per-Agniveer
        # data involved) — distinct from "who scored below/above the
        # cutoff", a per-Agniveer comparison query that must NOT be hijacked
        # by this value lookup, so it's excluded via the guard words below.
        if "cutoff" in _cutoff_raw_q and not re.search(
            r"\b(who|scored|below|above|passed|failed)\b", _cutoff_raw_q
        ):
            _co_section = str(intent.get("section") or "").strip()
            _co_sub_item = str(
                intent.get("sub_section") or intent.get("item_name") or ""
            ).strip()
            _co_clauses = []
            _co_params: List[Any] = []
            if _co_section:
                _co_clauses.append("LOWER(sec.SectionName) LIKE '%' + LOWER(?) + '%'")
                _co_params.append(_co_section)
            if _co_sub_item:
                _co_clauses.append("LOWER(si.Name) LIKE '%' + LOWER(?) + '%'")
                _co_params.append(_co_sub_item)
            _co_where = ("WHERE " + " AND ".join(_co_clauses)) if _co_clauses else ""
            _co_sql = f"""
SELECT sec.SectionName, si.Name AS SubItemName, si.MaxMarks, si.Cutoff
FROM ScoreSubItemMaster si
INNER JOIN ScoreSectionMaster sec ON sec.Id = si.SectionId
{_co_where}
ORDER BY sec.SectionName ASC, si.DisplayOrder ASC
"""
            _co_is_valid, _co_err = sql_validator.validate_sql(_co_sql)
            if not _co_is_valid:
                return None, f"Cutoff SQL validation failed: {_co_err}"
            _co_rows, _co_run_err = run_readonly(_co_sql, _co_params)
            if _co_run_err:
                return None, f"Cutoff execution failed: {_co_run_err}"
            return _to_section(_co_rows or [], intent, sql=_co_sql), None

    if intent.get("category") in ("Performance", "Overall"):
        from performance_executor import execute_performance_query
        return execute_performance_query(intent)

    if intent.get("category") == "Strength":
        sql = _build_strength_breakdown_sql()
        is_sql_valid, sql_err = sql_validator.validate_sql(sql)
        if not is_sql_valid:
            return None, f"Strength breakdown SQL validation failed: {sql_err}"
        rows, run_err = run_readonly(sql, [])
        if run_err:
            return None, f"Strength breakdown execution failed: {run_err}"
        res = _to_section(rows or [], intent, sql=sql)
        return res, None

    if intent.get("category") == "Verification":
        # Raw SQL fast-path: the AST planner cannot express OR filters or
        # LEFT JOIN + IS NULL checks needed for Pending / NotResponded.
        # We build the SQL directly here for all 5 verification statuses.
        _v_status_raw = (
            intent.get("filters", {}).get("operation")
            or intent.get("operation")
            or ""
        )
        _v_status = _v_status_raw.lower() if _v_status_raw else ""

        _limit = _get_top_n(intent)

        _base_cols = (
            "m.AgniveerNo, m.FullName, pv.Status, pv.SentDate, "
            "pv.PoliceStation, pv.ReceivedDate, pv.Remarks"
        )
        _args: List[Any] = []

        # An agniveerNo in the intent means the user asked about ONE person
        # (e.g. "police verification status of A0701749H") — every branch
        # below must scope to them, or the query silently returns the whole
        # roster's verification status instead.
        _v_agniveer_no = intent.get("agniveer_no") or intent.get("agniveerNo")
        _agniveer_filter = ""
        if _v_agniveer_no:
            _agniveer_filter = "AND m.AgniveerNo = ?"
            _args.append(_v_agniveer_no)

        # Batch/platoon/company scope (e.g. the frontend's batchId, or a
        # resolved company/platoon name) — was previously ignored entirely
        # by every Verification branch.
        _org_filter, _org_params = _org_scope_sql("m", intent)
        _agniveer_filter = f"{_agniveer_filter} {_org_filter}".strip()
        _args.extend(_org_params)

        if _v_status == "pending":
            # Pending = Rejected status OR no record in PoliceVerificationMaster at all
            _sql = f"""
SELECT {_top_clause(_limit)} {_base_cols}
FROM AgniveerMaster m
LEFT JOIN PoliceVerificationMaster pv ON pv.AgniveerId = m.Id
WHERE ISNULL(m.IsDisqualified,0) = 0
  AND (pv.Status = 'Rejected' OR pv.AgniveerId IS NULL)
  {_agniveer_filter}
ORDER BY m.AgniveerNo ASC
"""
        elif _v_status == "notresponded":
            # NotResponded = Status is 'Sent' AND ReturnDate/ReceivedDate IS NULL
            _sql = f"""
SELECT {_top_clause(_limit)} {_base_cols}
FROM AgniveerMaster m
INNER JOIN PoliceVerificationMaster pv ON pv.AgniveerId = m.Id
WHERE ISNULL(m.IsDisqualified,0) = 0
  AND pv.Status = 'Sent'
  AND pv.ReceivedDate IS NULL
  {_agniveer_filter}
ORDER BY pv.SentDate ASC
"""
        elif _v_status in ("verified", "completed"):
            _sql = f"""
SELECT {_top_clause(_limit)} {_base_cols}
FROM AgniveerMaster m
INNER JOIN PoliceVerificationMaster pv ON pv.AgniveerId = m.Id
WHERE ISNULL(m.IsDisqualified,0) = 0
  AND pv.Status IN ('Verified', 'Completed')
  {_agniveer_filter}
ORDER BY pv.ReceivedDate DESC
"""
        elif _v_status == "rejected":
            _sql = f"""
SELECT {_top_clause(_limit)} {_base_cols}
FROM AgniveerMaster m
INNER JOIN PoliceVerificationMaster pv ON pv.AgniveerId = m.Id
WHERE ISNULL(m.IsDisqualified,0) = 0
  AND pv.Status = 'Rejected'
  {_agniveer_filter}
ORDER BY m.AgniveerNo ASC
"""
        elif _v_status == "sent":
            _sql = f"""
SELECT {_top_clause(_limit)} {_base_cols}
FROM AgniveerMaster m
INNER JOIN PoliceVerificationMaster pv ON pv.AgniveerId = m.Id
WHERE ISNULL(m.IsDisqualified,0) = 0
  AND pv.Status = 'Sent'
  {_agniveer_filter}
ORDER BY pv.SentDate DESC
"""
        else:
            # Generic — show all with their current status
            _sql = f"""
SELECT {_top_clause(_limit)} {_base_cols}
FROM AgniveerMaster m
LEFT JOIN PoliceVerificationMaster pv ON pv.AgniveerId = m.Id
WHERE ISNULL(m.IsDisqualified,0) = 0
  {_agniveer_filter}
ORDER BY m.AgniveerNo ASC
"""

        _is_sql_valid, _sql_err = sql_validator.validate_sql(_sql)
        if not _is_sql_valid:
            return None, f"Verification SQL validation failed: {_sql_err}"
        _rows, _run_err = run_readonly(_sql, _args, max_rows=_row_cap(_limit))
        if _run_err:
            return None, f"Verification execution failed: {_run_err}"
        return _to_section(_rows or [], intent, sql=_sql), None


    if str(intent.get("category") or "").lower() in ("leave", "agniveerleave"):
        return execute_leave_query(intent)


    # ── Attendance Fast-Path ─────────────────────────────────────────────────
    if str(intent.get("category") or "").lower() in ("attendance", "attendancetracking", "present"):
        return execute_attendance_query(intent)
    # ── Medical Fast-Path ──────────────────────────────────────────────────
    if str(intent.get("category") or "").lower() in ("medical", "health", "bmi"):
        return execute_medical_query(intent)

    # ── Distribution Fast-Path ───────────────────────────────────────────────
    if str(intent.get("category") or "").lower() in ("distribution", "unitdistribution", "unit_distribution"):
        return execute_distribution_query(intent)

    # ── Equipment Fast-Path ────────────────────────────────────────────────
    if intent.get("category") == "Equipment":
        _eq_type = intent.get("equipment_type") or intent.get("item_name") or intent.get("equipmentType")
        _eq_op = intent.get("operation") or intent.get("subcategory")
        _eq_agniveer_no = intent.get("agniveer_no") or intent.get("agniveerNo")
        _limit = _get_top_n(intent)

        from intent_engine.intent_schema import (
            ISSUED_EQUIPMENT_ITEMS,
            PROCURED_EQUIPMENT_ITEMS,
        )

        # "Show equipment by category" — the two possible equipment TYPES
        # are "Issued" and "Procured"; the user wants only the item NAMES
        # that fall under each, not per-agniveer assignment rows. This is a
        # static master-list lookup (AgniveerEquipment.Type stores the item
        # NAME per row, e.g. "Kit Bag" — there's no separate item-catalog
        # table), so it's answered straight from the same catalog the
        # classifier itself uses to recognise item names, not a DB query.
        # Checked via raw query text because the classifier currently buckets
        # the phrase "by category" under the "ByName" (single-item lookup)
        # operation, which is the wrong shape for this request.
        if "by category" in _raw_q or "equipment categories" in _raw_q:
            _cat_rows = [
                {"EquipmentType": "Issued", "EquipmentName": n}
                for n in ISSUED_EQUIPMENT_ITEMS
            ] + [
                {"EquipmentType": "Procured", "EquipmentName": n}
                for n in PROCURED_EQUIPMENT_ITEMS
            ]
            return _to_section(_cat_rows, intent, sql=None), None

        # 1. Stats / Summary
        if _eq_op in ("Stats", "EquipmentSummary") or "summary" in _raw_q:
            _stats_org_filter, _stats_org_params = _org_scope_sql("m", intent)
            _stats_sql = f"""
WITH Categorized AS (
    SELECT eq.ReturnDateTime, eq.Type AS ItemCategory
    FROM AgniveerEquipment eq
    INNER JOIN AgniveerMaster m ON m.Id = eq.AgniveerId
    WHERE ISNULL(m.IsDisqualified,0) = 0
        AND eq.Type IS NOT NULL
        AND eq.Type IN ('Issued', 'Procured')
        {_stats_org_filter}
)
SELECT
    COUNT(*) AS TotalAssignedEquipments,
    SUM(CASE WHEN ItemCategory = 'Issued' THEN 1 ELSE 0 END) AS IssuedTotal,
    SUM(CASE WHEN ItemCategory = 'Issued' AND ReturnDateTime IS NULL THEN 1 ELSE 0 END) AS IssuedCurrentlyWithAgniveer,
    SUM(CASE WHEN ItemCategory = 'Issued' AND ReturnDateTime IS NOT NULL THEN 1 ELSE 0 END) AS IssuedReturned,
    SUM(CASE WHEN ItemCategory = 'Procured' THEN 1 ELSE 0 END) AS ProcuredTotal,
    SUM(CASE WHEN ItemCategory = 'Procured' AND ReturnDateTime IS NULL THEN 1 ELSE 0 END) AS ProcuredCurrentlyWithAgniveer,
    SUM(CASE WHEN ItemCategory = 'Procured' AND ReturnDateTime IS NOT NULL THEN 1 ELSE 0 END) AS ProcuredReturned
FROM Categorized
"""
            _stats_params = list(_stats_org_params)
            _stats_rows, _stats_run_err = run_readonly(_stats_sql, _stats_params)
            if not _stats_run_err:
                _r = (_stats_rows or [{}])[0]
                _tot_assigned = _r.get("TotalAssignedEquipments") or _r.get("TotalAssigned") or 0
                _issued_tot = _r.get("IssuedTotal") or 0
                _issued_curr = _r.get("IssuedCurrentlyWithAgniveer") or 0
                _issued_ret = _r.get("IssuedReturned") or 0
                _procured_tot = _r.get("ProcuredTotal") or 0
                _procured_curr = _r.get("ProcuredCurrentlyWithAgniveer") or 0
                _procured_ret = _r.get("ProcuredReturned") or 0

                _summary_row = {
                    "totalAssignedEquipments": _tot_assigned,
                    "issuedEquipments": {
                        "totalEquipments": _issued_tot,
                        "currentlyWithAgniveer": _issued_curr,
                        "returned": _issued_ret,
                    },
                    "procuredEquipments": {
                        "total": _procured_tot,
                        "currentlyWithAgniveer": _procured_curr,
                        "returned": _procured_ret,
                    },
                    "TotalAssignedEquipments": _tot_assigned,
                    "IssuedTotal": _issued_tot,
                    "IssuedCurrentlyWithAgniveer": _issued_curr,
                    "IssuedReturned": _issued_ret,
                    "ProcuredTotal": _procured_tot,
                    "ProcuredCurrentlyWithAgniveer": _procured_curr,
                    "ProcuredReturned": _procured_ret,
                }
                return _to_section([_summary_row], intent, sql=_stats_sql), None

        # 2. AgniveerWise (Specific Agniveer Equipment Lookup)
        if _eq_op == "AgniveerWise" or (_eq_agniveer_no and _eq_op not in ("Holding", "Returned", "HoldingEquipment", "ReturnedEquipment")):
            _sql = f"""
SELECT {_top_clause(_limit)}
    eq.Id AS AssignmentId,
    em.Name AS EquipmentName,
    em.Category AS EquipmentCategory,
    eq.Type,
    eq.GivenDateTime,
    eq.ReturnDateTime,
    eq.GivenCondition,
    eq.ReturnCondition,
    eq.Remarks,
    CASE WHEN eq.ReturnDateTime IS NOT NULL THEN 1 ELSE 0 END AS IsReturned
FROM AgniveerEquipment eq
INNER JOIN AgniveerMaster a ON a.Id = eq.AgniveerId
INNER JOIN EquipmentMaster em ON em.Id = eq.EquipmentId
WHERE LOWER(a.AgniveerNo) = LOWER(?)
ORDER BY eq.GivenDateTime DESC
"""
            _rows, _run_err = run_readonly(_sql, [str(_eq_agniveer_no)], max_rows=_row_cap(_limit))
            if not _run_err:
                return _to_section(_rows or [], intent, sql=_sql), None

        # 3. Returned Equipment
        if _eq_op in ("Returned", "ReturnedEquipment") or _eq_type == "Returned":
            clauses = [
                "eq.ReturnDateTime IS NOT NULL",
                "(a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)",
                "a.IsActive = 1",
            ]
            params = []
            if _eq_agniveer_no:
                clauses.append("LOWER(a.AgniveerNo) = LOWER(?)")
                params.append(str(_eq_agniveer_no))
            if _eq_type and _eq_type not in ("Issued", "Returned", "Holding"):
                clauses.append("(LOWER(eq.Type) LIKE '%' + LOWER(?) + '%' OR LOWER(em.Name) LIKE '%' + LOWER(?) + '%')")
                params.extend([str(_eq_type), str(_eq_type)])

            _org_filter, _org_params = _org_scope_sql("a", intent)
            where_str = "WHERE " + " AND ".join(clauses) + _org_filter
            params.extend(_org_params)
            _sql = f"""
SELECT {_top_clause(_limit)}
    eq.Id AS AssignmentId,
    a.AgniveerNo,
    a.FullName,
    a.PhotoPath,
    p.Name AS PlatoonName,
    em.Name AS EquipmentName,
    em.Category AS EquipmentCategory,
    eq.Type,
    eq.GivenDateTime,
    eq.ReturnDateTime,
    eq.GivenCondition,
    eq.ReturnCondition
FROM AgniveerEquipment eq
INNER JOIN AgniveerMaster a ON a.Id = eq.AgniveerId
LEFT JOIN EquipmentMaster em ON em.Id = eq.EquipmentId
LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
{where_str}
ORDER BY eq.ReturnDateTime DESC
"""
            _rows, _run_err = run_readonly(_sql, params, max_rows=_row_cap(_limit))
            if not _run_err:
                return _to_section(_rows or [], intent, sql=_sql), None

        # 4. Holding Equipment (Default for Equipment listing / overdue / holding)
        clauses = [
            "eq.ReturnDateTime IS NULL",
            "(a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)",
            "a.IsActive = 1",
        ]
        params = []
        if _eq_agniveer_no:
            clauses.append("LOWER(a.AgniveerNo) = LOWER(?)")
            params.append(str(_eq_agniveer_no))
        if _eq_type and _eq_type not in ("Issued", "Returned", "Holding"):
            clauses.append("(LOWER(eq.Type) LIKE '%' + LOWER(?) + '%' OR LOWER(em.Name) LIKE '%' + LOWER(?) + '%')")
            params.extend([str(_eq_type), str(_eq_type)])

        _org_filter, _org_params = _org_scope_sql("a", intent)
        where_str = "WHERE " + " AND ".join(clauses) + _org_filter
        params.extend(_org_params)
        _sql = f"""
SELECT {_top_clause(_limit)}
    eq.Id AS AssignmentId,
    a.AgniveerNo,
    a.FullName,
    a.PhotoPath,
    p.Name AS PlatoonName,
    em.Name AS EquipmentName,
    em.Category AS EquipmentCategory,
    eq.Type,
    eq.GivenDateTime,
    eq.GivenCondition
FROM AgniveerEquipment eq
INNER JOIN AgniveerMaster a ON a.Id = eq.AgniveerId
LEFT JOIN EquipmentMaster em ON em.Id = eq.EquipmentId
LEFT JOIN PlatoonMaster p ON p.Id = a.PlatoonId
{where_str}
ORDER BY eq.GivenDateTime DESC
"""
        _rows, _run_err = run_readonly(_sql, params, max_rows=_row_cap(_limit))
        if not _run_err:
            return _to_section(_rows or [], intent, sql=_sql), None

    # ── Schedule Fast-Path ───────────────────────────────────────────────────
    if intent.get("category") == "Schedule":
        return _execute_schedule_query(intent)

    # ── Disqualified Fast-Path ───────────────────────────────────────────────
    if str(intent.get("category") or "").lower() in ("disqualified", "disqualification"):
        return execute_disqualified_query(intent)

    # ── Verification Fast-Path ───────────────────────────────────────────────
    if str(intent.get("category") or "").lower() in ("verification", "policeverification"):
        return execute_verification_query(intent)

    # ── Medical Fast-Path ────────────────────────────────────────────────────
    if str(intent.get("category") or "").lower() in ("medical", "health", "bmi"):
        return execute_medical_query(intent)

    # ── PersonalDetails & Skills Fast-Path ─────────────────────────────────
    _cat_lower = str(intent.get("category") or "").lower()
    if _cat_lower in ("personaldetails", "personaldetail", "personal_details", "skills", "skill") or (not intent.get("category") and (intent.get("agniveer_no") or intent.get("agniveerNo"))):

        _p_agniveer_no = intent.get("agniveer_no") or intent.get("agniveerNo")
        _p_class = intent.get("class") or intent.get("class_")
        _p_state = intent.get("state") or intent.get("district")
        _p_sport = (
            intent.get("sport")
            or intent.get("value")
            or intent.get("filters", {}).get("sport")
            or intent.get("filters", {}).get("value")
        )
        if not _p_sport:
            for s_token in ("volleyball", "cricket", "football", "soccer", "hockey", "basketball", "kabaddi", "badminton", "tennis", "swimming", "athletics", "boxing", "wrestling", "handball", "squash"):
                if s_token in _raw_q:
                    _p_sport = s_token.capitalize()
                    break
        _p_blood_group = (
            intent.get("blood_group")
            or intent.get("bloodGroup")
            or intent.get("filters", {}).get("bloodGroup")
            or intent.get("filters", {}).get("blood_group")
        )
        if not _p_blood_group:
            for bg_token in ("o+", "b+", "a+", "ab+", "o-", "b-", "a-", "ab-"):
                if bg_token in _raw_q:
                    _p_blood_group = bg_token.upper()
                    break

        _p_op = intent.get("operation") or ""
        _p_metric = intent.get("metric") or ""
        _limit = _get_top_n(intent)

        _p_org_filter, _p_org_params = _org_scope_sql("m", intent)

        # Active / inactive status (AgniveerMaster.IsActive — see
        # personal_details_parser.py's ActiveStatusCount/ActiveStatusList).
        # is_active is None for a plain total headcount ("how many Agniveers
        # are there in total?") — no IsActive filter at all in that case.
        if _p_op in ("ActiveStatusCount", "ActiveStatusList"):
            _p_is_active = intent.get("is_active")
            _p_active_where = (
                f"ISNULL(m.IsActive,0) = {1 if _p_is_active else 0}"
                if _p_is_active is not None
                else "1 = 1"
            )
            # "Is A0701948W active?" — one specific Agniveer, not the whole
            # roster.
            _p_active_agn_filter = ""
            _p_active_params = list(_p_org_params)
            if _p_agniveer_no:
                _p_active_agn_filter = "AND LOWER(m.AgniveerNo) = LOWER(?)"
                _p_active_params = [str(_p_agniveer_no)] + _p_active_params
            if _p_op == "ActiveStatusCount":
                _sql = f"""
SELECT COUNT(*) AS AgniveerCount
FROM AgniveerMaster m
WHERE {_p_active_where}
  {_p_active_agn_filter}
  {_p_org_filter}
"""
            else:
                _sql = f"""
SELECT {_top_clause(_limit)} m.AgniveerNo, m.FullName, m.IsActive
FROM AgniveerMaster m
WHERE {_p_active_where}
  {_p_active_agn_filter}
  {_p_org_filter}
ORDER BY m.AgniveerNo ASC
"""
            _rows, _run_err = run_readonly(_sql, _p_active_params, max_rows=_row_cap(_limit))
            if not _run_err:
                return _to_section(_rows or [], intent, sql=_sql), None

        # BloodGroup summary breakdown query (e.g. "Show blood group details")
        if ("blood group" in _raw_q or _p_metric == "BloodGroup" or _p_op in ("BloodGroup", "BloodGroupDetails")) and not _p_blood_group and not _p_agniveer_no:
            _sql = f"""
SELECT {_top_clause(_limit)} m.BloodGroup, COUNT(*) AS AgniveerCount
FROM AgniveerMaster m
WHERE ISNULL(m.IsDisqualified,0) = 0 AND m.BloodGroup IS NOT NULL AND TRIM(m.BloodGroup) <> ''
  {_p_org_filter}
GROUP BY m.BloodGroup
ORDER BY AgniveerCount DESC
"""
            _rows, _run_err = run_readonly(_sql, _p_org_params, max_rows=_row_cap(_limit))
            if not _run_err:
                return _to_section(_rows or [], intent, sql=_sql), None

        clauses = ["ISNULL(m.IsDisqualified,0) = 0"]
        params = []
        joins = []

        _p_company_name = intent.get("company_name") or intent.get("companyName")
        _p_company_id = intent.get("company_id") or intent.get("companyId")
        _p_platoon_name = intent.get("platoon_name") or intent.get("platoonName")
        _p_platoon_id = intent.get("platoon_id") or intent.get("platoonId")
        _p_batch_id = intent.get("batch_id") or intent.get("batchId")

        if _p_company_name or _p_company_id or _p_platoon_name or _p_platoon_id:
            joins.append("INNER JOIN PlatoonMaster p ON p.Id = m.PlatoonId")
        if _p_company_name or _p_company_id:
            joins.append("INNER JOIN CompanyMaster c ON c.Id = p.CompanyId")

        if _p_company_name:
            clauses.append("LOWER(c.Name) LIKE '%' + LOWER(?) + '%'")
            params.append(str(_p_company_name))
        elif _p_company_id:
            clauses.append("c.Id = ?")
            params.append(int(_p_company_id))

        if _p_platoon_name:
            clauses.append("LOWER(p.Name) LIKE '%' + LOWER(?) + '%'")
            params.append(str(_p_platoon_name))
        elif _p_platoon_id:
            clauses.append("p.Id = ?")
            params.append(int(_p_platoon_id))

        if _p_batch_id:
            clauses.append("m.BatchId = ?")
            params.append(int(_p_batch_id))

        if _p_agniveer_no:
            clauses.append("LOWER(m.AgniveerNo) = LOWER(?)")
            params.append(str(_p_agniveer_no))
        if _p_class:
            clauses.append("LOWER(m.Class) LIKE '%' + LOWER(?) + '%'")
            params.append(str(_p_class))
        if _p_state:
            clauses.append("(LOWER(m.State) LIKE '%' + LOWER(?) + '%' OR LOWER(m.District) LIKE '%' + LOWER(?) + '%' OR LOWER(m.Address) LIKE '%' + LOWER(?) + '%')")
            params.extend([str(_p_state)] * 3)
        if _p_sport:
            clauses.append("(LOWER(m.Sports) LIKE '%' + LOWER(?) + '%' OR LOWER(m.Skill) LIKE '%' + LOWER(?) + '%' OR LOWER(m.Hobby) LIKE '%' + LOWER(?) + '%')")
            params.extend([str(_p_sport)] * 3)
        if _p_blood_group:
            clauses.append("UPPER(REPLACE(m.BloodGroup, ' ', '')) = UPPER(REPLACE(?, ' ', ''))")
            params.append(str(_p_blood_group))
        _p_height_filter = intent.get("height_filter")
        if isinstance(_p_height_filter, dict) and _p_height_filter.get("operator") in (">", "<", ">=", "<="):
            clauses.append(f"m.Height {_p_height_filter['operator']} ?")
            params.append(float(_p_height_filter["value"]))
        _p_join_year = intent.get("join_year")
        if _p_join_year:
            clauses.append("YEAR(m.DateOfJoining) = ?")
            params.append(int(_p_join_year))

        join_str = (" " + " ".join(joins)) if joins else ""
        where_str = f"FROM AgniveerMaster m{join_str}\nWHERE " + " AND ".join(clauses) + _p_org_filter
        params.extend(_p_org_params)

        # One or more specific fields were asked about (e.g. "height and
        # weight of all agniveers" -> metrics=["Height","Weight"], set by
        # personal_details_parser.py) — return just those fields plus the
        # identifying columns, not the entire 14-column profile. A query
        # naming N fields must get all N back, not just the first — this
        # used to read only the singular `metric`, so a multi-field ask
        # silently dropped every field but the first one requested.
        # Validated against the same column whitelist personal_details_parser.py
        # extracts metrics from, so this can never become a column-name
        # injection point.
        from intent_engine.personal_details_parser import AGNIVEER_PERSONAL_COLUMNS, COL_MAP

        raw_metrics = intent.get("metrics")
        if not raw_metrics and intent.get("metric"):
            raw_metrics = [intent.get("metric")]
        if isinstance(raw_metrics, str):
            raw_metrics = [raw_metrics]

        _p_metrics: List[str] = []
        for m in (raw_metrics or []):
            m_str = str(m).strip()
            canonical = COL_MAP.get(m_str.lower())
            if not canonical:
                for col in AGNIVEER_PERSONAL_COLUMNS:
                    if col.lower() == m_str.lower():
                        canonical = col
                        break
            if canonical and canonical not in ("AgniveerNo", "FullName") and canonical not in _p_metrics:
                _p_metrics.append(canonical)
            elif m_str in AGNIVEER_PERSONAL_COLUMNS and m_str not in ("AgniveerNo", "FullName") and m_str not in _p_metrics:
                _p_metrics.append(m_str)

        # Fallback: If _p_metrics is still empty, scan raw_query for specific column mentions in COL_MAP
        if not _p_metrics:
            _raw_q = str(intent.get("raw_query") or "").lower()
            if _raw_q:
                for col_alias in sorted(COL_MAP.keys(), key=len, reverse=True):
                    if re.search(rf'\b{re.escape(col_alias)}\b', _raw_q):
                        canonical = COL_MAP[col_alias]
                        if canonical not in ("AgniveerNo", "FullName") and canonical not in _p_metrics:
                            _p_metrics.append(canonical)

        if _p_metrics:
            _select_cols = "m.AgniveerNo, m.FullName, " + ", ".join(
                f"m.{m}" for m in dict.fromkeys(_p_metrics)
            )
        else:
            _select_cols = (
                "m.AgniveerNo, m.FullName, m.Class, m.State, m.District, m.Qualification, "
                "m.Sports, m.Skill, m.Hobby, m.BloodGroup, m.DateOfBirth, m.Height, m.Weight, "
                "m.MobileNo, m.Email, m.Address"
            )

        _sql = f"""
SELECT {_top_clause(_limit)} {_select_cols}
{where_str}
ORDER BY m.AgniveerNo ASC
"""
        _rows, _run_err = run_readonly(_sql, params, max_rows=_row_cap(_limit))
        if not _run_err:
            return _to_section(_rows or [], intent, sql=_sql), None

    # ── Users & Roles Fast-Path ─────────────────────────────────────────────
    if intent.get("category") == "UsersRoles":
        _u_op = intent.get("operation")
        _limit = _get_top_n(intent)
        # Password is on the hard denylist (sql_validator.DENIED_COLUMNS) —
        # never select it here either, defense in depth.
        _u_cols = "u.Id, u.Username, u.FullName, u.Email, u.ContactNo, u.IsActive"
        _u_sql: Optional[str] = None
        _u_params: List[Any] = []

        if _u_op == "ByAgniveer" and intent.get("agniveer_no"):
            _u_sql = f"""
SELECT {_top_clause(_limit)} {_u_cols}
FROM UserMaster u
INNER JOIN AgniveerMaster a ON a.Id = u.AgniVeerId
WHERE LOWER(a.AgniveerNo) = LOWER(?)
"""
            _u_params = [str(intent.get("agniveer_no"))]
        elif _u_op == "ActiveList":
            _u_is_active = 1 if intent.get("is_active") else 0
            _u_sql = f"""
SELECT {_top_clause(_limit)} {_u_cols}
FROM UserMaster u
WHERE ISNULL(u.IsActive,0) = {_u_is_active}
ORDER BY u.Username ASC
"""
        elif _u_op == "ByRole":
            _u_role = str(intent.get("role") or "").strip()
            _u_sql = f"""
SELECT {_top_clause(_limit)} {_u_cols}, r.Role
FROM UserMaster u
INNER JOIN UserRole ur ON ur.UserId = u.Id
INNER JOIN RoleMaster r ON r.Id = ur.RoleId
WHERE LOWER(r.Role) LIKE '%' + LOWER(?) + '%'
ORDER BY u.Username ASC
"""
            _u_params = [_u_role]

        if _u_sql is not None:
            _u_is_valid, _u_err = sql_validator.validate_sql(_u_sql)
            if not _u_is_valid:
                return None, f"Users/Roles SQL validation failed: {_u_err}"
            _u_rows, _u_run_err = run_readonly(_u_sql, _u_params, max_rows=_row_cap(_limit))
            if _u_run_err:
                return None, f"Users/Roles execution failed: {_u_run_err}"
            return _to_section(_u_rows or [], intent, sql=_u_sql), None
        return None, "Unsupported Users/Roles question — logged-in-session and login-token questions are not exposed by this system."

    # ── Organizational Hierarchy Fast-Path ──────────────────────────────────
    if intent.get("category") == "OrgHierarchy":
        _oh_op = intent.get("operation")
        _oh_company_id = intent.get("company_id") or intent.get("companyId")
        _oh_platoon_id = intent.get("platoon_id") or intent.get("platoonId")
        _limit = _get_top_n(intent)
        _oh_sql: Optional[str] = None
        _oh_params: List[Any] = []

        # CommanderId / CommandingOfficerId / PlatoonCommanderId reference
        # UserMaster.Id — there is no separate "officer" table in the
        # schema, and commanders are administrative accounts, not
        # Agniveers, so UserMaster is the only sensible join target.
        if _oh_op == "CurrentCommander":
            if intent.get("target") == "Platoon":
                _oh_sql = """
SELECT p.Name AS PlatoonName, p.PlatoonNo, u.FullName AS CommanderName, u.Username
FROM PlatoonMaster p
LEFT JOIN UserMaster u ON u.Id = p.PlatoonCommanderId
WHERE p.Id = ?
"""
                _oh_params = [int(_oh_platoon_id)] if _oh_platoon_id else []
            else:
                _oh_sql = """
SELECT c.Name AS CompanyName, cmd.FullName AS CommanderName, cmd.Username AS CommanderUsername,
       co.FullName AS CommandingOfficerName, co.Username AS CommandingOfficerUsername
FROM CompanyMaster c
LEFT JOIN UserMaster cmd ON cmd.Id = c.CompanyCommanderId
LEFT JOIN UserMaster co ON co.Id = c.CommandingOfficerId
WHERE c.Id = ?
"""
                _oh_params = [int(_oh_company_id)] if _oh_company_id else []

        elif _oh_op == "HistoricalOfficer":
            # "last year" / "previously" — the most recent tenure that has
            # already ENDED (EndDate NOT NULL), i.e. the officer before the
            # current one. A specific past date isn't extracted (none of
            # the example questions gave one), so this returns the full
            # history ordered most-recent-first rather than guess a cutoff.
            if intent.get("target") == "Platoon":
                _oh_sql = """
SELECT p.Name AS PlatoonName, u.FullName AS CommanderName, h.StartDate, h.EndDate
FROM PlatoonCommanderHistory h
INNER JOIN PlatoonMaster p ON p.Id = h.PlatoonId
LEFT JOIN UserMaster u ON u.Id = h.CommanderId
WHERE h.PlatoonId = ? AND h.EndDate IS NOT NULL
ORDER BY h.EndDate DESC
"""
                _oh_params = [int(_oh_platoon_id)] if _oh_platoon_id else []
            else:
                _oh_sql = """
SELECT c.Name AS CompanyName, u.FullName AS CommandingOfficerName, h.StartDate, h.EndDate
FROM CompanyCommandingOfficerHistory h
INNER JOIN CompanyMaster c ON c.Id = h.CompanyId
LEFT JOIN UserMaster u ON u.Id = h.CommandingOfficerId
WHERE h.CompanyId = ? AND h.EndDate IS NOT NULL
ORDER BY h.EndDate DESC
"""
                _oh_params = [int(_oh_company_id)] if _oh_company_id else []

        elif _oh_op == "PredecessorCommander":
            if intent.get("target") == "Platoon":
                _oh_sql = """
SELECT TOP (1) p.Name AS PlatoonName, u.FullName AS CommanderName, h.StartDate, h.EndDate
FROM PlatoonCommanderHistory h
INNER JOIN PlatoonMaster p ON p.Id = h.PlatoonId
LEFT JOIN UserMaster u ON u.Id = h.CommanderId
WHERE h.PlatoonId = ? AND h.EndDate IS NOT NULL
ORDER BY h.EndDate DESC
"""
                _oh_params = [int(_oh_platoon_id)] if _oh_platoon_id else []
            else:
                _oh_sql = """
SELECT TOP (1) c.Name AS CompanyName, u.FullName AS CommanderName, h.StartDate, h.EndDate
FROM CompanyCommanderHistory h
INNER JOIN CompanyMaster c ON c.Id = h.CompanyId
LEFT JOIN UserMaster u ON u.Id = h.CommanderId
WHERE h.CompanyId = ? AND h.EndDate IS NOT NULL
ORDER BY h.EndDate DESC
"""
                _oh_params = [int(_oh_company_id)] if _oh_company_id else []

        elif _oh_op == "PlatoonsUnderCompany":
            _oh_sql = f"""
SELECT {_top_clause(_limit)} p.Name AS PlatoonName, p.PlatoonNo, p.IsActive
FROM PlatoonMaster p
WHERE p.CompanyId = ?
ORDER BY p.Name ASC
"""
            _oh_params = [int(_oh_company_id)] if _oh_company_id else []

        elif _oh_op == "HeadcountByCompany":
            _oh_sql = """
SELECT c.Name AS CompanyName, COUNT(a.Id) AS AgniveerCount
FROM CompanyMaster c
LEFT JOIN PlatoonMaster p ON p.CompanyId = c.Id
LEFT JOIN AgniveerMaster a ON a.PlatoonId = p.Id AND ISNULL(a.IsDisqualified,0) = 0
GROUP BY c.Name
ORDER BY AgniveerCount DESC
"""

        elif _oh_op == "TopCompanyByHeadcount":
            _oh_order = "DESC" if intent.get("descending", True) else "ASC"
            _oh_sql = f"""
SELECT TOP (1) c.Name AS CompanyName, COUNT(a.Id) AS AgniveerCount
FROM CompanyMaster c
LEFT JOIN PlatoonMaster p ON p.CompanyId = c.Id
LEFT JOIN AgniveerMaster a ON a.PlatoonId = p.Id AND ISNULL(a.IsDisqualified,0) = 0
GROUP BY c.Name
ORDER BY AgniveerCount {_oh_order}
"""

        elif _oh_op == "WhichCompanyForAgniveer" and intent.get("agniveer_no"):
            _oh_sql = """
SELECT m.AgniveerNo, m.FullName, p.Name AS PlatoonName, c.Name AS CompanyName
FROM AgniveerMaster m
LEFT JOIN PlatoonMaster p ON p.Id = m.PlatoonId
LEFT JOIN CompanyMaster c ON c.Id = p.CompanyId
WHERE LOWER(m.AgniveerNo) = LOWER(?)
"""
            _oh_params = [str(intent.get("agniveer_no"))]

        if _oh_sql is not None and (_oh_params or _oh_op in ("HeadcountByCompany", "TopCompanyByHeadcount")):
            _oh_is_valid, _oh_err = sql_validator.validate_sql(_oh_sql)
            if not _oh_is_valid:
                return None, f"Organizational hierarchy SQL validation failed: {_oh_err}"
            _oh_rows, _oh_run_err = run_readonly(_oh_sql, _oh_params, max_rows=_row_cap(_limit))
            if _oh_run_err:
                return None, f"Organizational hierarchy execution failed: {_oh_run_err}"
            return _to_section(_oh_rows or [], intent, sql=_oh_sql), None
        return None, "Could not resolve the company/platoon for this organizational question."

    if intent.get("category") == "Schedule":

        _s_company_id = intent.get("company_id") or intent.get("companyId")
        _s_company_name = (
            intent.get("company_name")
            or intent.get("companyName")
            or intent.get("Company")
            or intent.get("company")
        )
        _s_platoon_id = intent.get("platoon_id") or intent.get("platoonId")
        _s_platoon_name = intent.get("platoon_name") or intent.get("platoonName")
        _s_agniveer_no = intent.get("agniveer_no") or intent.get("agniveerNo")
        _s_date = intent.get("date")
        _s_from_date = intent.get("from_date") or intent.get("fromDate")
        _s_to_date = intent.get("to_date") or intent.get("toDate")
        _s_top_n = _get_top_n(intent)


        _resolved_company_id: Optional[int] = None
        _lookup_sql: Optional[str] = None
        _lookup_params: Optional[List[Any]] = None

        if _s_company_id is not None:
            _resolved_company_id = int(_s_company_id)
        elif _s_agniveer_no:
            _lookup_sql = _COMPANY_ID_BY_AGNIVEER_NO_SQL
            _lookup_params = [str(_s_agniveer_no)]
        elif _s_platoon_id is not None:
            _lookup_sql = _COMPANY_ID_BY_PLATOON_ID_SQL
            _lookup_params = [int(_s_platoon_id)]
        elif _s_platoon_name:
            _lookup_sql = _COMPANY_ID_BY_PLATOON_NAME_SQL
            _lookup_params = [str(_s_platoon_name)]
        elif _s_company_name:
            _lookup_sql = _COMPANY_ID_BY_NAME_SQL
            _lookup_params = [str(_s_company_name)]

        if _resolved_company_id is None and _lookup_sql is not None:
            _is_lookup_valid, _lookup_err = sql_validator.validate_sql(_lookup_sql)
            if not _is_lookup_valid:
                return None, f"Schedule company lookup SQL validation failed: {_lookup_err}"
            _lookup_rows, _lookup_run_err = run_readonly(_lookup_sql, _lookup_params)
            if _lookup_run_err:
                return None, f"Schedule company lookup failed: {_lookup_run_err}"
            _resolved_company_id = _lookup_rows[0].get("CompanyId") if _lookup_rows else None
            if _resolved_company_id is None:
                return _to_section([], intent), None


        sql, params = _build_company_schedule_sql(
            company_id=_resolved_company_id,
            date=_s_date,
            from_date=_s_from_date,
            to_date=_s_to_date,
            top_n=_s_top_n,
        )
        is_sql_valid, sql_err = sql_validator.validate_sql(sql)
        if not is_sql_valid:
            return None, f"Schedule SQL validation failed: {sql_err}"
        rows, run_err = run_readonly(sql, params, max_rows=_row_cap(_s_top_n))
        if run_err:
            return None, f"Schedule execution failed: {run_err}"
        return _to_section(rows or [], intent, sql=sql), None

    from query_planner_v2 import query_planner_v2
    from sql_builder import sql_builder

    import time
    from explainability_engine import explainability_engine

    # Convert legacy intent to v2 intent format
    def _pick_legacy_value(*keys: str) -> Any:
        for key in keys:
            value = intent.get(key)
            if value not in (None, "", [], {}):
                return value
        return None

    filters: Dict[str, Any] = {}
    if isinstance(intent.get("filters"), dict):
        filters.update(
            {
                key: value
                for key, value in intent["filters"].items()
                if value not in (None, "", [], {})
            }
        )

    company_id = _pick_legacy_value("companyId", "company_id")
    platoon_id = _pick_legacy_value("platoonId", "platoon_id")
    batch_id = _pick_legacy_value("batchId", "batch_id")
    agniveer_no = _pick_legacy_value("agniveerNo", "agniveer_no")
    medical_status = _pick_legacy_value("medicalStatus", "medical_status")
    company_name = _pick_legacy_value("Company", "company", "companyName", "company_name")
    platoon_name = _pick_legacy_value("platoonName", "platoon_name")
    class_ = _pick_legacy_value("class", "class_")
    blood_group = _pick_legacy_value("bloodGroup", "blood_group")
    sport = _pick_legacy_value("sport")
    diagnose = _pick_legacy_value("diagnose")
    hospital_name = _pick_legacy_value("hospitalName", "hospital_name")
    medical_date = _pick_legacy_value("date")
    unit_name = _pick_legacy_value("unitName", "unit_name")
    from_date = _pick_legacy_value("fromDate", "from_date")
    to_date = _pick_legacy_value("toDate", "to_date")
    leave_status = _pick_legacy_value("leaveStatus", "leave_status", "leaveType", "leave_type")

    if company_id is not None:
        filters.setdefault("Company.Id", company_id)
    elif company_name is not None:
        filters.setdefault("Company.Name", company_name)

    if platoon_name is not None and platoon_id is None:
        filters.setdefault("Platoon.Name", platoon_name)

    if platoon_id is not None:
        filters.setdefault("Agniveer.PlatoonId", platoon_id)
    if batch_id is not None:
        filters.setdefault("Agniveer.BatchId", batch_id)
    if agniveer_no is not None:
        filters.setdefault("Agniveer.AgniveerNo", agniveer_no)
    if medical_status is not None:
        filters.setdefault("Medical.Status", medical_status)
        # "admitted to hospital THIS MONTH" carries from_date/to_date on the
        # intent, but nothing below this point ever reads them for a generic
        # Medical/status query (only the "disqualified" and "Leave" branches
        # apply their own date filters) — without this, the date range is
        # silently dropped and the query returns every Admitted record ever,
        # not just this month's.
        if from_date is not None and to_date is not None:
            filters.setdefault(
                "AND",
                [
                    {"Medical.VisitDate": {"operator": ">=", "value": from_date}},
                    {"Medical.VisitDate": {"operator": "<=", "value": to_date}},
                ],
            )
        elif from_date is not None:
            filters.setdefault("Medical.VisitDate", {"operator": ">=", "value": from_date})
        elif to_date is not None:
            filters.setdefault("Medical.VisitDate", {"operator": "<=", "value": to_date})
    if class_ is not None:
        filters.setdefault("Agniveer.Class", class_)
    if blood_group is not None:
        filters.setdefault("Agniveer.BloodGroup", blood_group)
    if sport is not None:
        filters.setdefault(
            "Agniveer.Sports", {"operator": "LIKE", "value": f"%{sport}%"}
        )
    if diagnose is not None:
        filters.setdefault(
            "Medical.Diagnosis", {"operator": "LIKE", "value": f"%{diagnose}%"}
        )
        # "who has fever right now" scopes the match to today's VisitDate;
        # "who has suffered with fever" (no current-hint, medical_date is
        # None) stays unscoped — i.e. "ever diagnosed".
        if medical_date is not None:
            filters.setdefault("Medical.VisitDate", medical_date)
    if hospital_name is not None:
        filters.setdefault(
            "Medical.HospitalNameLocation",
            {"operator": "LIKE", "value": f"%{hospital_name}%"},
        )
    if unit_name is not None:
        filters.setdefault(
            "Distribution.Name", {"operator": "LIKE", "value": f"%{unit_name}%"}
        )

    category = intent.get("category")
    operation = intent.get("operation")

    # 1. Bucket (ii) Category/Concept mappings
    base_concept = category if category else "Agniveer"

    if category == "disqualified":
        base_concept = "Agniveer"
        filters.setdefault("Agniveer.IsDisqualified", 1)

        # "who got disqualified today" / "disqualified between X and Y" ->
        # filter on DisqualifiedDate. "when did X get disqualified" needs no
        # filter here — DisqualifiedDate is already in the auto-selected
        # AgniveerMaster columns, so the answer just shows up as a column.
        if from_date is not None and to_date is not None:
            filters.setdefault(
                "AND",
                [
                    {"Agniveer.DisqualifiedDate": {"operator": ">=", "value": from_date}},
                    {"Agniveer.DisqualifiedDate": {"operator": "<=", "value": to_date}},
                ],
            )
        elif from_date is not None:
            filters.setdefault("Agniveer.DisqualifiedDate", {"operator": ">=", "value": from_date})
        elif to_date is not None:
            filters.setdefault("Agniveer.DisqualifiedDate", {"operator": "<=", "value": to_date})
        else:
            disqualified_date = _pick_legacy_value("date")
            if disqualified_date is not None:
                filters.setdefault("Agniveer.DisqualifiedDate", {"operator": "=", "value": disqualified_date})
    elif category == "personaldetail":
        base_concept = "Agniveer"
        extra_aggregates = None
        extra_order_by = None
        extra_group_by = None
        metric = intent.get("metric")
        if metric:
            col_ref = f"Agniveer.{metric}"
            if operation == "average":
                extra_aggregates = [{"function": "AVG", "concept": None, "column": metric, "alias": f"Average{metric}"}]
            elif operation == "max":
                filters.setdefault(col_ref, {"operator": "=", "value": {"__raw_sql": f"(SELECT MAX({metric}) FROM AgniveerMaster)"}})
            elif operation == "min":
                filters.setdefault(col_ref, {"operator": "=", "value": {"__raw_sql": f"(SELECT MIN({metric}) FROM AgniveerMaster)"}})
            elif operation == "above_average":
                filters.setdefault(col_ref, {"operator": ">", "value": {"__raw_sql": f"(SELECT AVG({metric}) FROM AgniveerMaster)"}})
            elif operation == "below_average":
                filters.setdefault(col_ref, {"operator": "<", "value": {"__raw_sql": f"(SELECT AVG({metric}) FROM AgniveerMaster)"}})
            elif operation == "match":
                val = intent.get("value")
                if val:
                    filters.setdefault(col_ref, {"operator": "LIKE", "value": f"%{val}%"})
    elif category == "Skills":
        base_concept = "Agniveer"
        if operation == "BySport":
            filters.setdefault(
                "AND",
                [
                    {"Agniveer.Sports": {"operator": "!=", "value": None}},
                    {"Agniveer.Sports": {"operator": "!=", "value": ""}},
                ],
            )
        elif operation == "ByClass":
            filters.setdefault(
                "AND",
                [
                    {"Agniveer.Class": {"operator": "!=", "value": None}},
                    {"Agniveer.Class": {"operator": "!=", "value": ""}},
                ],
            )
    elif category == "Leave":
        base_concept = "Leave"
        
        extra_aggregates = None
        extra_order_by = None
        extra_group_by = None
        
        # --- Type of Leave filters ---
        # "these are the types of leaves exist in our system so if user has asked any types of these leaves then you have to check these columns in agniveer leave master tables if its true"
        leave_col_mapping = {
            "Sick": "OnSickLeave",
            "Hospitalized": "IsHospitalized",
            "Medical": "OnMedicalLeave",
            "Absconded": "IsAbscondedLeave",
            "ATTNC": "OnATTN'C'",
            "ExPPG": "OnEX PPG",
            "Annual": "OnAnnualLeave",
        }
        if leave_status in leave_col_mapping:
            col_name = leave_col_mapping[leave_status]
            filters.setdefault(f"Leave.{col_name}", 1)
        
        # --- Operation specific filters ---
        if operation == "Current" and from_date is None and to_date is None:
            import datetime
            today = datetime.date.today().isoformat()
            filters.setdefault("Leave.FromDate", {"operator": "<=", "value": today})
            filters.setdefault("Leave.ToDate", {"operator": ">=", "value": today})
        elif operation in ("Most", "Least"):
            # We want to sum the datediff of leaves per Agniveer
            extra_aggregates = [
                {
                    "function": "SUM",
                    "concept": None,
                    "column": "DATEDIFF(day, FromDate, ToDate) + 1",
                    "alias": "TotalLeaveDays",
                }
            ]
            descending = False if operation == "Least" else True
            extra_order_by = [
                {"concept": None, "column": "TotalLeaveDays", "descending": descending},
                {"concept": "Agniveer", "column": "AgniveerNo", "descending": False},
            ]
            extra_group_by = [
                {"concept": "Agniveer", "column": "AgniveerNo"},
                {"concept": "Agniveer", "column": "FullName"},
            ]
        elif operation == "Threshold":
            threshold_sql = (
                "(SELECT AgniveerId FROM AgniveerLeaveMaster "
                "GROUP BY AgniveerId "
                "HAVING SUM(DATEDIFF(day, FromDate, ToDate) + 1) >= 55 "
                "OR MAX(DATEDIFF(day, FromDate, ToDate) + 1) >= 40)"
            )
            # The AST needs Agniveer.Id IN (...) so we filter the base_concept (which is Leave) 
            # Wait, base_concept for Leave is "Leave"!
            # So if base_concept is "Leave", the AST generates SELECT ... FROM AgniveerLeaveMaster JOIN AgniveerMaster
            # So filtering Agniveer.Id is totally fine and supported!
            filters.setdefault("Agniveer.Id", {"operator": "IN", "value": {"__raw_sql": threshold_sql}})
        else:
            if from_date is not None:
                filters.setdefault("Leave.ToDate", {"operator": ">=", "value": from_date})
            if to_date is not None:
                filters.setdefault("Leave.FromDate", {"operator": "<=", "value": to_date})
    elif category == "Verification":
        base_concept = "Agniveer"
        verification_status = _pick_legacy_value("verificationStatus", "verification_status")
        if verification_status is not None:
            v_status_lower = verification_status.lower()
            if v_status_lower == "pending":
                filters.setdefault("OR", [
                    {"Verification.Status": "Rejected"},
                    {"Verification.AgniveerId": {"operator": "=", "value": None}}
                ])
            elif v_status_lower == "not responded":
                filters.setdefault("Verification.Status", "Sent")
                filters.setdefault("Verification.ReturnDate", {"operator": "=", "value": None})
            elif v_status_lower in ("verified", "completed"):
                filters.setdefault("OR", [
                    {"Verification.Status": "Verified"},
                    {"Verification.Status": "Completed"}
                ])
            else:
                filters.setdefault("Verification.Status", verification_status.capitalize())
    elif category == "Equipment":
        base_concept = "Equipment"

        # These arrive as flat camelCase keys inside intent["filters"] (set by
        # admin_intent.py), which query_planner_v2._parse_filters silently
        # drops since they have no "Concept.Column" dot. Pull them out and
        # re-express them with the concept-qualified keys the AST planner
        # actually understands.
        equipment_name = _pick_legacy_value("equipmentName", "equipment_name", "item_name")
        equipment_type = _pick_legacy_value("equipmentType", "equipment_type")
        given_condition = _pick_legacy_value("givenCondition", "given_condition")
        return_condition = _pick_legacy_value("returnCondition", "return_condition")
        for legacy_key in ("equipmentName", "equipmentType", "givenCondition", "returnCondition"):
            filters.pop(legacy_key, None)

        # Equipment name -> join AgniveerEquipment.EquipmentId to
        # EquipmentMaster.Id and match on the master item name.
        if equipment_name is not None:
            filters.setdefault(
                "EquipmentMaster.Name", {"operator": "LIKE", "value": f"%{equipment_name}%"}
            )
        # "Issued" / "Procured" type of equipment -> AgniveerEquipment.Type.
        if equipment_type is not None:
            filters.setdefault("Equipment.Type", equipment_type)
        if given_condition is not None:
            filters.setdefault("Equipment.GivenCondition", given_condition)
        if return_condition is not None:
            filters.setdefault("Equipment.ReturnCondition", return_condition)

        # "Holding" = who currently has equipment issued to them (borrowed it
        # and hasn't returned it) -> ReturnDateTime IS NULL.
        # "Returned" = who has returned their equipment -> ReturnDateTime IS NOT NULL.
        if operation == "Holding":
            filters.setdefault(
                "Equipment.ReturnDateTime", {"operator": "IS NULL", "value": None}
            )
        elif operation == "Returned":
            filters.setdefault(
                "Equipment.ReturnDateTime", {"operator": "IS NOT NULL", "value": None}
            )

    v2_limit = intent.get("number") or intent.get("top_n")
    v2_intent = {
        "base_concept": base_concept,
        "filters": filters,
        "limit": v2_limit,
    }
    
    if category in ("Leave", "personaldetail"):
        if extra_aggregates:
            v2_intent["aggregates"] = extra_aggregates
        if extra_order_by:
            v2_intent["order_by"] = extra_order_by
        if extra_group_by:
            v2_intent["group_by"] = extra_group_by

    ast = None
    sql = ""
    params: List[Any] = []
    rows = None
    planned_ok = False

    planning_duration_ms = 0
    compilation_duration_ms = 0
    execution_duration_ms = 0

    # TIER 1: AST Query Planner Path
    try:
        t_plan_start = time.time()

        # Intercept and translate/route aggregation & ranking operations
        section = str(intent.get("section") or intent.get("sub_section") or "").strip()

        # 1. Supported ranking/aggregation queries (Mapped to AST / performance_executor)
        # NOTE: All Performance/Overall intents are dispatched to performance_executor.py
        # at the top of execute_sql_query (line 460) BEFORE reaching this block.
        # This guard here is only a safety fallback for non-performance categories.
        if (category == "Performance" and operation in ("Top", "Bottom")) or (
            category == "Overall" and operation == "OverallPerformance"
        ):
            # performance_executor.py handles section filters natively — no CapabilityGapError needed

            base_concept = "Agniveer"
            v2_intent["base_concept"] = "Agniveer"

            alias_name = (
                "OverallMarks" if operation == "OverallPerformance" else "TotalMarks"
            )
            v2_intent["aggregates"] = [
                {
                    "function": "SUM",
                    "concept": "Performance",
                    "column": "MarksObtained",
                    "alias": alias_name,
                }
            ]

            # Must scope MarksObtained to best attempt as per R7 business rules
            filters.setdefault("Performance.IsBestAttempt", 1)

            descending = False if operation == "Bottom" else True
            v2_intent["order_by"] = [
                {"concept": None, "column": alias_name, "descending": descending},
                {"concept": "Agniveer", "column": "AgniveerNo", "descending": False},
            ]

        elif category == "Medical" and operation == "BMI":
            bmi_category = str(
                intent.get("bmiCategory") or intent.get("bmi_category") or ""
            ).strip()
            _bmi_top_n = int(intent["number"]) if intent.get("number") else None
            sql, params = _build_medical_bmi_sql(
                top_n=_bmi_top_n,
                bmi_category=bmi_category,
                batch_id=batch_id,
                platoon_id=platoon_id,
                company_id=company_id,
                company_name=company_name,
                agniveer_no=agniveer_no,
                agniveer_class=class_,
            )
            is_sql_valid, sql_err = sql_validator.validate_sql(sql)
            if not is_sql_valid:
                return None, f"Medical BMI SQL validation failed: {sql_err}"
            rows, run_err = run_readonly(sql, params, max_rows=_row_cap(_bmi_top_n))
            if run_err:
                return None, f"Medical BMI execution failed: {run_err}"
            metrics_hook("generated")
            return _to_section(rows or [], intent, sql=sql), None

        elif category == "Medical" and operation == "BloodGroup":
            blood_group_value = str(
                intent.get("bloodGroup") or intent.get("blood_group") or ""
            ).strip()
            report_mode = bool(
                re.search(
                    r"\b(report|summary|count|counts|distribution|breakdown|how many)\b",
                    question,
                    re.IGNORECASE,
                )
            )
            _bg_top_n = int(intent["number"]) if intent.get("number") else None
            sql, params = _build_medical_blood_group_sql(
                top_n=_bg_top_n,
                report_mode=report_mode and not bool(
                    intent.get("agniveer_no")
                    or company_id
                    or platoon_id
                    or company_name
                    or platoon_name
                ),
                blood_group=blood_group_value,
                batch_id=batch_id,
                platoon_id=platoon_id,
                company_id=company_id,
                company_name=company_name,
                platoon_name=platoon_name,
                agniveer_no=agniveer_no,
                agniveer_class=class_,
            )
            is_sql_valid, sql_err = sql_validator.validate_sql(sql)
            if not is_sql_valid:
                return None, f"Medical blood group SQL validation failed: {sql_err}"
            rows, run_err = run_readonly(sql, params, max_rows=_row_cap(_bg_top_n))
            if run_err:
                return None, f"Medical blood group execution failed: {run_err}"
            metrics_hook("generated")
            return _to_section(rows or [], intent, sql=sql), None

        # 2. Gaps/Unsupported aggregation operations (routed to Tier 2 Fallback)
        elif (
            (
                category == "Performance"
                and operation
                in (
                    "Average",
                    "BestAttempt",
                    "Grading",
                    "GradingSummary",
                    "Improvement",
                    "Drop",
                    "AttemptWise",
                    "Trend",
                )
            )
            or (False and category == "Medical")
            # "Disease" only needs the LLM/CTE path for genuine aggregates
            # ("top diseases", "most common disease" — no specific disease
            # named, so `diagnose` is None). A named-disease lookup ("who has
            # fever") is just a Medical.Diagnosis LIKE filter, already wired
            # up above, and goes through the deterministic AST path.
            or (category == "Medical" and operation == "Disease" and diagnose is None)
            or (
                category == "Attendance"
                and operation in ("Monthly", "Weekly", "Summary")
            )
            or (category == "Equipment" and operation == "AgniveerWise")
            or (category == "Distribution" and operation in ("Latest", "TopUnit"))
            or (category == "Strength")
        ):
            raise CapabilityGapError(
                f"Operation '{category}/{operation}' is a known capability gap (subquery/CTE/conditional-sum) and is routed to LLM."
            )

        ast = query_planner_v2.plan_query(v2_intent)

        if category == "Verification" and verification_status and verification_status.lower() == "pending":
            for j in ast.joins:
                if j.right_table == "PoliceVerificationMaster":
                    j.join_type = "LEFT"

        is_ast_valid, ast_err = sql_validator.validate_ast(ast)
        if not is_ast_valid:
            raise ValidatorRejectionError(f"AST validation failed: {ast_err}")

        t_comp_start = time.time()
        sql, params = sql_builder.build(ast)

        is_sql_valid, sql_err = sql_validator.validate_sql(sql)
        if not is_sql_valid:
            raise ValidatorRejectionError(f"Compiled SQL validation failed: {sql_err}")

        t_exec_start = time.time()
        rows, run_err = run_readonly(sql, params, max_rows=_row_cap(v2_limit))
        if run_err:
            raise DatabaseExecutionError(f"Database query execution failed: {run_err}")

        t_now = time.time()
        planning_duration_ms = int((t_comp_start - t_plan_start) * 1000)
        compilation_duration_ms = int((t_exec_start - t_comp_start) * 1000)
        execution_duration_ms = int((t_now - t_exec_start) * 1000)

        planned_ok = True
        metrics_hook("generated")
    except CapabilityGapError as exc:
        logger.warning(
            f"AST pipeline capability gap (concept={base_concept}, operation={operation}): {exc}. "
            f"Falling back to LLM-based SQL generation."
        )
        metrics_hook("capability_gap_fallback")
    except ValidatorRejectionError as exc:
        logger.warning(
            f"AST pipeline validator rejection (concept={base_concept}, operation={operation}): {exc}. "
            f"Falling back to LLM-based SQL generation."
        )
        metrics_hook("structural_reject_fallback")
    except DatabaseExecutionError as exc:
        logger.warning(
            f"AST pipeline database execution error (concept={base_concept}, operation={operation}): {exc}. "
            f"Bubbling database error without fallback."
        )
        metrics_hook("exec_error")
        return None, f"Database query execution failed: {exc}"
    except Exception as exc:
        logger.error(
            f"AST pipeline unexpected error (concept={base_concept}, operation={operation}): {exc}. "
            f"Falling back to LLM-based SQL generation.",
            exc_info=True,
        )
        metrics_hook("unexpected_ast_error")

    # TIER 2: LLM Fallback Path
    if not planned_ok:
        try:
            t_fallback_start = time.time()
            sql, gen_err = generate_sql(question, intent)
            if gen_err:
                metrics_hook("cannot_answer")
                return None, f"Fallback SQL generation failed: {gen_err}"

            is_sql_valid, sql_err = sql_validator.validate_sql(sql)
            if not is_sql_valid:
                metrics_hook("validator_rejected")
                return None, f"Fallback SQL validation failed: {sql_err}"

            t_exec_start = time.time()
            rows, run_err = run_readonly(sql, [])
            if run_err:
                metrics_hook("exec_error")
                return None, f"Fallback SQL execution failed: {run_err}"

            t_now = time.time()
            planning_duration_ms = int((t_exec_start - t_fallback_start) * 1000)
            compilation_duration_ms = 0
            execution_duration_ms = int((t_now - t_exec_start) * 1000)

            ast = None
            metrics_hook("generated")
            metrics_hook("llm_fallback")
        except Exception as exc:
            logger.error(f"Fallback LLM pipeline failed: {exc}")
            return None, f"Fallback LLM pipeline failed: {exc}"

    # Explainability & Metadata
    explanation = (
        explainability_engine.explain(ast)
        if ast is not None
        else {
            "intent": "Database Query",
            "base_table": base_concept,
            "joins": [],
            "filters": [],
            "groupings": [],
            "having": [],
            "aggregations": [],
            "sorting": [],
            "limit": v2_limit,
        }
    )

    execution_metadata = {
        "planning_duration_ms": planning_duration_ms,
        "compilation_duration_ms": compilation_duration_ms,
        "execution_duration_ms": execution_duration_ms,
        "rows_returned": len(rows) if rows else 0,
        "explanation": explanation,
        "sql": sql,
    }

    res = _to_section(rows or [], intent, sql=sql)
    res["execution_metadata"] = execution_metadata
    return res, None


def _get_top_n(intent: Dict) -> Optional[int]:
    """Row limit the user actually asked for, or None for "every matching row".

    A number named in the question (any phrasing, not just explicit
    top/bottom/rank) is honored exactly. No number means no cap at all —
    callers must not substitute an arbitrary default like SQL_MAX_ROWS here.
    """
    num = intent.get("number")
    return int(num) if num is not None else None


def _top_clause(limit: Optional[int]) -> str:
    """SQL Server TOP clause for `limit`, or "" (every row) when limit is None."""
    return f"TOP ({int(limit)})" if limit is not None else ""


def _row_cap(limit: Optional[int]) -> int:
    """ROWCOUNT value matching `limit` — 0 is SQL Server's "no limit" sentinel."""
    return int(limit) if limit is not None else 0


def metrics_hook(event: str) -> None:
    """Best-effort metrics increment — never lets an observability failure
    affect the query result. Imported lazily to avoid a hard dependency."""
    try:
        from metrics import metrics_collector

        {
            "generated": metrics_collector.inc_sql_generated,
            "validator_rejected": metrics_collector.inc_sql_validator_rejected,
            "cannot_answer": metrics_collector.inc_sql_cannot_answer,
            "exec_error": metrics_collector.inc_sql_exec_error,
            "llm_fallback": metrics_collector.inc_sql_llm_fallback,
            "capability_gap_fallback": metrics_collector.inc_sql_capability_gap_fallback,
            "structural_reject_fallback": metrics_collector.inc_sql_structural_reject_fallback,
        }[event]()
    except Exception:
        pass
