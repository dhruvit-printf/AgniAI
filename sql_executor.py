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
    top_n: int,
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
            f"SELECT TOP ({top_n}) "
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
            f"SELECT TOP ({top_n}) "
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
    top_n: int,
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
        SELECT TOP ({top_n})
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


def _build_company_schedule_sql(
    *,
    company_id: Optional[int] = None,
    date: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    top_n: int = 500,
) -> Tuple[str, List[Any]]:
    """Build the deterministic CompanySchedule query for a resolved company or all companies.

    No date scope (bycompany with nothing else specified) returns the whole
    schedule; a single `date` scopes to that day; a `from_date`/`to_date` pair
    scopes to that range.
    """
    clauses = []
    params: List[Any] = []

    if company_id is not None:
        clauses.append("s.CompanyId = ?")
        params.append(int(company_id))

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

    where_str = f"WHERE {' AND '.join(clauses)}" if clauses else ""

    sql = f"""
    SELECT TOP ({top_n})
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
    {where_str}
    ORDER BY s.ScheduleDate ASC, s.Pd ASC
    """
    return sql, params



def _resolve_attendance_range(intent: Dict) -> Tuple[Optional[str], Optional[str]]:
    """Returns (range_start, range_end) as YYYY-MM-DD date strings.
    Handles Daily, Weekly, Monthly, Summary operations cleanly."""
    operation = intent.get("operation")
    single_date = intent.get("date")
    from_date = intent.get("from_date")
    to_date = intent.get("to_date")
    if operation == "Daily":
        day = single_date or from_date or to_date
        if not day:
            day = datetime.date.today().isoformat()
        return str(day)[:10], str(day)[:10]
    if operation in ("Weekly", "Monthly", "Summary"):
        if from_date and to_date:
            return str(from_date)[:10], str(to_date)[:10]
        if operation == "Weekly":
            today = datetime.date.today()
            monday = today - datetime.timedelta(days=today.weekday())
            sunday = monday + datetime.timedelta(days=6)
            return monday.isoformat()[:10], sunday.isoformat()[:10]
        if operation in ("Monthly", "Summary"):
            today = datetime.date.today()
            first = today.replace(day=1)
            if first.month == 12:
                next_month = first.replace(year=first.year + 1, month=1)
            else:
                next_month = first.replace(month=first.month + 1)
            last = next_month - datetime.timedelta(days=1)
            return first.isoformat()[:10], last.isoformat()[:10]
    range_start = from_date or single_date
    range_end = to_date or single_date
    if range_start:
        range_start = str(range_start)[:10]
    if range_end:
        range_end = str(range_end)[:10]
    return range_start, range_end



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
    sql: str, params: Optional[List[Any]] = None
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Execute a validated SELECT against the READ-ONLY login. (rows, error)."""
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
        cur.execute(f"SET ROWCOUNT {SQL_MAX_ROWS}")
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
        rows, err = run_readonly(sql, chunk)
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
        
    print(f"[DEBUG INTERCEPT] intent: {intent}", flush=True)

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
SELECT TOP ({_limit}) {_base_cols}
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
SELECT TOP ({_limit}) {_base_cols}
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
SELECT TOP ({_limit}) {_base_cols}
FROM AgniveerMaster m
INNER JOIN PoliceVerificationMaster pv ON pv.AgniveerId = m.Id
WHERE ISNULL(m.IsDisqualified,0) = 0
  AND pv.Status IN ('Verified', 'Completed')
  {_agniveer_filter}
ORDER BY pv.ReceivedDate DESC
"""
        elif _v_status == "rejected":
            _sql = f"""
SELECT TOP ({_limit}) {_base_cols}
FROM AgniveerMaster m
INNER JOIN PoliceVerificationMaster pv ON pv.AgniveerId = m.Id
WHERE ISNULL(m.IsDisqualified,0) = 0
  AND pv.Status = 'Rejected'
  {_agniveer_filter}
ORDER BY m.AgniveerNo ASC
"""
        elif _v_status == "sent":
            _sql = f"""
SELECT TOP ({_limit}) {_base_cols}
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
SELECT TOP ({_limit}) {_base_cols}
FROM AgniveerMaster m
LEFT JOIN PoliceVerificationMaster pv ON pv.AgniveerId = m.Id
WHERE ISNULL(m.IsDisqualified,0) = 0
  {_agniveer_filter}
ORDER BY m.AgniveerNo ASC
"""

        _is_sql_valid, _sql_err = sql_validator.validate_sql(_sql)
        if not _is_sql_valid:
            return None, f"Verification SQL validation failed: {_sql_err}"
        _rows, _run_err = run_readonly(_sql, _args)
        if _run_err:
            return None, f"Verification execution failed: {_run_err}"
        return _to_section(_rows or [], intent, sql=_sql), None


    if intent.get("category") == "Leave":
        _op = intent.get("operation", "")
        _limit = _get_top_n(intent)

        _leave_type = (
            intent.get("filters", {}).get("leaveType")
            or intent.get("leave_type")
            or ""
        )

        # Map leave type name to actual DB column name
        _leave_col_map = {
            "Sick": "OnSickLeave",
            "Hospitalized": "IsHospitalized",
            "Medical": "OnMedicalLeave",
            "Absconded": "IsAbscondedLeave",
            "Annual": "OnAnnualLeave",
            "ExPPG": "OnEX PPG",
            "ATTNC": "OnATTN'C'",
        }
        _leave_col_filter = ""
        if _leave_type in _leave_col_map:
            _leave_col_filter = f"AND lm.[{_leave_col_map[_leave_type]}] = 1"

        # Agniveer / batch / platoon / company scope — none of the three
        # Leave branches below applied any of these before.
        _l_agniveer_no = intent.get("agniveer_no") or intent.get("agniveerNo")
        _l_agniveer_filter = ""
        _l_args: List[Any] = []
        if _l_agniveer_no:
            _l_agniveer_filter = "AND m.AgniveerNo = ?"
            _l_args.append(_l_agniveer_no)
        _l_org_filter, _l_org_params = _org_scope_sql("m", intent)
        _leave_scope_filter = f"{_l_agniveer_filter} {_l_org_filter}".strip()
        _l_args.extend(_l_org_params)

        # All 7 real leave-type flags on AgniveerLeaveMaster — ATTNC and
        # ExPPG were missing here even though the WHERE-clause filter above
        # already scopes correctly to them: a leave record matched by
        # `leave_type=ATTNC` came back with every OTHER flag shown as 0 and
        # no ATTNC/ExPPG column at all, so the result looked unclassified.
        _base_select = (
            "lm.FromDate, lm.ToDate, lm.Remarks, lm.MarkedBy, "
            "lm.OnSickLeave, lm.IsHospitalized, lm.OnMedicalLeave, "
            "lm.IsAbscondedLeave, lm.OnAnnualLeave, "
            "lm.[OnATTN'C'] AS OnATTNC, lm.[OnEX PPG] AS OnEXPPG, "
            "m.AgniveerNo, m.FullName"
        )

        # When intent engine sends leaveType=Threshold (misfired from intent),
        # treat it as the Threshold operation explicitly
        _effective_op = _op
        if _leave_type and _leave_type.lower() == "threshold":
            _effective_op = "Threshold"
            _leave_col_filter = ""  # No per-type filter for threshold

        if _effective_op == "Current":
            import datetime as _dt
            _today = _dt.date.today().isoformat()
            _sql = f"""
SELECT TOP ({_limit}) {_base_select}
FROM AgniveerLeaveMaster lm
INNER JOIN AgniveerMaster m ON m.Id = lm.AgniveerId
WHERE ISNULL(m.IsDisqualified,0) = 0
  AND lm.FromDate <= '{_today}'
  AND lm.ToDate >= '{_today}'
  {_leave_col_filter}
  {_leave_scope_filter}
ORDER BY lm.FromDate ASC
"""

            _is_sql_valid, _sql_err = sql_validator.validate_sql(_sql)
            if not _is_sql_valid:
                return None, f"Leave Current SQL validation failed: {_sql_err}"
            _rows, _run_err = run_readonly(_sql, _l_args)
            if _run_err:
                return None, f"Leave Current execution failed: {_run_err}"
            return _to_section(_rows or [], intent, sql=_sql), None

        elif _effective_op in ("Most", "Least"):
            _order = "DESC" if _effective_op == "Most" else "ASC"
            _sql = f"""
SELECT TOP ({_limit}) m.AgniveerNo, m.FullName,
    SUM(DATEDIFF(day, lm.FromDate, lm.ToDate) + 1) AS TotalLeaveDays
FROM AgniveerLeaveMaster lm
INNER JOIN AgniveerMaster m ON m.Id = lm.AgniveerId
WHERE ISNULL(m.IsDisqualified,0) = 0
  {_leave_col_filter}
  {_leave_scope_filter}
GROUP BY m.AgniveerNo, m.FullName
ORDER BY TotalLeaveDays {_order}, m.AgniveerNo ASC
"""
            _is_sql_valid, _sql_err = sql_validator.validate_sql(_sql)
            if not _is_sql_valid:
                return None, f"Leave {_effective_op} SQL validation failed: {_sql_err}"
            _rows, _run_err = run_readonly(_sql, _l_args)
            if _run_err:
                return None, f"Leave {_effective_op} execution failed: {_run_err}"
            return _to_section(_rows or [], intent, sql=_sql), None

        elif _effective_op == "Threshold":
            _sql = f"""
SELECT TOP ({_limit}) m.AgniveerNo, m.FullName,
    SUM(DATEDIFF(day, lm.FromDate, lm.ToDate) + 1) AS TotalLeaveDays,
    MAX(DATEDIFF(day, lm.FromDate, lm.ToDate) + 1) AS MaxContinuousLeave
FROM AgniveerLeaveMaster lm
INNER JOIN AgniveerMaster m ON m.Id = lm.AgniveerId
WHERE ISNULL(m.IsDisqualified,0) = 0
  {_leave_scope_filter}
GROUP BY m.AgniveerNo, m.FullName
HAVING SUM(DATEDIFF(day, lm.FromDate, lm.ToDate) + 1) >= 55
    OR MAX(DATEDIFF(day, lm.FromDate, lm.ToDate) + 1) >= 40
ORDER BY TotalLeaveDays DESC
"""
            _is_sql_valid, _sql_err = sql_validator.validate_sql(_sql)
            if not _is_sql_valid:
                return None, f"Leave Threshold SQL validation failed: {_sql_err}"
            _rows, _run_err = run_readonly(_sql, _l_args)
            if _run_err:
                return None, f"Leave Threshold execution failed: {_run_err}"
            return _to_section(_rows or [], intent, sql=_sql), None
        # For other leave operations, fall through to AST pipeline


    if intent.get("category") == "Attendance" and intent.get("operation") in (
        "Monthly",
        "Weekly",
        "Summary",
        "Daily",
        "Present",
    ):
        agniveer_no = intent.get("agniveer_no") or intent.get("agniveerNo")
        # A platoon/company/batch scope with no agniveer_no is a unit-level
        # aggregate (e.g. "attendance of Platoon 1" for a compare query), not
        # an individual lookup — this fast path only builds a single person's
        # day-by-day calendar, so let those fall through to the AST/LLM
        # capability-gap pipeline below instead of hard-erroring here.
        _has_unit_scope = any(
            intent.get(k)
            for k in (
                "platoon_id",
                "platoonId",
                "company_id",
                "companyId",
                "batch_id",
                "batchId",
            )
        )
        if (
            not agniveer_no
            and not _has_unit_scope
            and intent.get("operation") in ("Monthly", "Weekly", "Summary", "Daily")
        ):
            return None, "Please provide an Agniveer number for the attendance query."

        if agniveer_no:
            is_lookup_valid, lookup_err = sql_validator.validate_sql(_AGNIVEER_LOOKUP_SQL)
            if not is_lookup_valid:
                return None, f"Attendance lookup SQL validation failed: {lookup_err}"
            agniveer_rows, lookup_run_err = run_readonly(_AGNIVEER_LOOKUP_SQL, [agniveer_no])
            if lookup_run_err:
                return None, f"Attendance lookup failed: {lookup_run_err}"
            if not agniveer_rows:
                return _to_section([], intent), None

            agniveer_row = agniveer_rows[0]
            range_start, range_end = _resolve_attendance_range(intent)
            if not range_start or not range_end:
                return None, "Could not resolve a date range for the attendance query."


            sql = _build_attendance_calendar_sql()
            is_sql_valid, sql_err = sql_validator.validate_sql(sql)
            if not is_sql_valid:
                return None, f"Attendance calendar SQL validation failed: {sql_err}"
            rows, run_err = run_readonly(
                sql, [range_start, range_end, agniveer_row["Id"]]
            )
            if run_err:
                return None, f"Attendance calendar execution failed: {run_err}"
            for row in rows or []:
                row["AgniveerNo"] = agniveer_row["AgniveerNo"]
                row["FullName"] = agniveer_row["FullName"]
            return _to_section(rows or [], intent, sql=sql), None
    # ── Medical Fast-Path ──────────────────────────────────────────────────
    _raw_q = (question or intent.get("raw_query") or "").lower()
    if intent.get("category") == "Medical" or "bmi" in _raw_q:
        _med_op = intent.get("operation") or intent.get("subcategory")
        _m_agniveer_no = intent.get("agniveer_no") or intent.get("agniveerNo")
        _m_diagnosis = intent.get("diagnose") or intent.get("diagnosis")
        _m_hospital = intent.get("hospital_name") or intent.get("hospitalName")
        _m_blood_group = (
            intent.get("blood_group")
            or intent.get("bloodGroup")
            or intent.get("filters", {}).get("bloodGroup")
            or intent.get("filters", {}).get("blood_group")
        )
        _limit = _get_top_n(intent)


        if _med_op in ("BMI", "BMIAnalysis") or "bmi" in _raw_q:
            bmi_clause = "BmiValue IS NOT NULL"
            if "above 25" in _raw_q or "overweight" in _raw_q or "greater than 25" in _raw_q or "> 25" in _raw_q or ">25" in _raw_q:
                bmi_clause = "BmiValue > 25.0"
            elif "below 18.5" in _raw_q or "underweight" in _raw_q or "< 18.5" in _raw_q or "<18.5" in _raw_q:
                bmi_clause = "BmiValue < 18.5"

            bg_clause = ""
            bg_param: List[Any] = []
            if _m_blood_group or "o+" in _raw_q or "b+" in _raw_q or "a+" in _raw_q or "ab+" in _raw_q:
                bg_val = _m_blood_group
                if not bg_val:
                    for bg_token in ("o+", "b+", "a+", "ab+", "o-", "b-", "a-", "ab-"):
                        if bg_token in _raw_q:
                            bg_val = bg_token.upper()
                            break
                if bg_val:
                    bg_clause = " AND UPPER(REPLACE(BloodGroup, ' ', '')) = UPPER(REPLACE(?, ' ', ''))"
                    bg_param.append(str(bg_val))

            _bmi_agniveer_filter = ""
            _bmi_agniveer_params: List[Any] = []
            if _m_agniveer_no:
                _bmi_agniveer_filter = "AND LOWER(a.AgniveerNo) = LOWER(?)"
                _bmi_agniveer_params.append(str(_m_agniveer_no))
            _bmi_org_filter, _bmi_org_params = _org_scope_sql("a", intent)

            # Params must be positional in the SAME order their "?" appear in
            # the compiled SQL text below: the Vitals CTE (agniveer/org scope)
            # comes before the final SELECT's bg_clause.
            params = [*_bmi_agniveer_params, *_bmi_org_params, *bg_param]

            _sql = f"""
WITH LatestMedical AS (
    SELECT mr.AgniveerId, mr.Height, mr.Weight,
           ROW_NUMBER() OVER (PARTITION BY mr.AgniveerId ORDER BY mr.VisitDate DESC, mr.Id DESC) AS rn
    FROM MedicalRecordMaster mr
    WHERE mr.Height IS NOT NULL AND mr.Weight IS NOT NULL
),
Vitals AS (
    SELECT a.Id AS AgniveerId, a.AgniveerNo, a.FullName, a.BatchId, a.Class, a.BloodGroup,
           COALESCE(lm.Height, a.Height) AS EffHeight,
           COALESCE(lm.Weight, a.Weight) AS EffWeight
    FROM AgniveerMaster a
    LEFT JOIN LatestMedical lm ON lm.AgniveerId = a.Id AND lm.rn = 1
    WHERE ISNULL(a.IsDisqualified,0) = 0
      {_bmi_agniveer_filter}
      {_bmi_org_filter}
),
Scored AS (
    SELECT AgniveerNo, FullName, BloodGroup, EffHeight AS Height, EffWeight AS Weight,
           CASE WHEN EffHeight IS NULL OR EffWeight IS NULL OR EffHeight <= 0 THEN NULL
                ELSE CAST(EffWeight / POWER(EffHeight / 100.0, 2) AS DECIMAL(10, 2))
           END AS BmiValue
    FROM Vitals
)
SELECT TOP ({_limit}) AgniveerNo, FullName, BloodGroup, Height, Weight, BmiValue,
       CASE WHEN BmiValue IS NULL THEN NULL
            WHEN BmiValue < 18.5 THEN 'Underweight'
            WHEN BmiValue < 25.0 THEN 'Normal'
            WHEN BmiValue < 30.0 THEN 'Overweight'
            ELSE 'Obese' END AS BmiCategory
FROM Scored
WHERE {bmi_clause}{bg_clause}
ORDER BY BmiValue DESC
"""
            _rows, _run_err = run_readonly(_sql, params)
            if not _run_err:
                return _to_section(_rows or [], intent, sql=_sql), None

        elif _med_op in ("Disease", "DiseaseStatistics", "Diagnosed", "Individual", "IndividualMedical") or _m_diagnosis or _m_agniveer_no:
            clauses = ["ISNULL(m.IsDisqualified,0) = 0"]
            params = []
            if _m_agniveer_no:
                clauses.append("LOWER(m.AgniveerNo) = LOWER(?)")
                params.append(str(_m_agniveer_no))
            if _m_diagnosis:
                clauses.append("LOWER(mr.Diagnosis) LIKE '%' + LOWER(?) + '%'")
                params.append(str(_m_diagnosis))
            if _m_hospital:
                clauses.append("LOWER(mr.HospitalNameLocation) LIKE '%' + LOWER(?) + '%'")
                params.append(str(_m_hospital))

            _org_filter, _org_params = _org_scope_sql("m", intent)
            where_str = "WHERE " + " AND ".join(clauses) + _org_filter
            params.extend(_org_params)
            _sql = f"""
SELECT TOP ({_limit}) m.AgniveerNo, m.FullName, mr.VisitDate, mr.Diagnosis, mr.HospitalNameLocation,
       mr.Height, mr.Weight, mr.EyeSight, mr.BloodPressure, mr.HeartRate, mr.Status, mr.Prescriptions, mr.Remarks
FROM MedicalRecordMaster mr
INNER JOIN AgniveerMaster m ON m.Id = mr.AgniveerId
{where_str}
ORDER BY mr.VisitDate DESC, m.AgniveerNo ASC
"""
            _rows, _run_err = run_readonly(_sql, params)
            if not _run_err:
                return _to_section(_rows or [], intent, sql=_sql), None

    # ── Equipment Fast-Path ────────────────────────────────────────────────
    if intent.get("category") == "Equipment":
        _eq_type = intent.get("equipment_type") or intent.get("item_name") or intent.get("equipmentType")
        _eq_op = intent.get("operation") or intent.get("subcategory")
        _eq_agniveer_no = intent.get("agniveer_no") or intent.get("agniveerNo")
        _limit = _get_top_n(intent)


        clauses = ["ISNULL(m.IsDisqualified,0) = 0"]
        params = []

        if _eq_op == "Holding" or _eq_type in ("Issued", "Holding"):
            # "Holding" is the classifier's actual operation value for both
            # "who's currently holding X" and "overdue equipment" queries
            # (there's no due-date column in AgniveerEquipment, so "overdue"
            # is intentionally treated as "still issued, not yet returned" —
            # see intent_classifier.py / tests/test_equipment_items.py).
            # "Issued" is not a real classifier operation value (Equipment's
            # only operations are ByName/AgniveerWise/Holding/Stats/Returned)
            # so this clause never fired and every Holding/overdue query
            # silently returned ALL equipment records, issued or returned.
            clauses.append("eq.ReturnDateTime IS NULL")
        elif _eq_op == "Returned" or _eq_type == "Returned":
            clauses.append("eq.ReturnDateTime IS NOT NULL")

        if _eq_agniveer_no:
            clauses.append("LOWER(m.AgniveerNo) = LOWER(?)")
            params.append(str(_eq_agniveer_no))
        if _eq_type and _eq_type not in ("Issued", "Returned", "Holding"):
            clauses.append("LOWER(eq.Type) LIKE '%' + LOWER(?) + '%'")
            params.append(str(_eq_type))

        _org_filter, _org_params = _org_scope_sql("m", intent)
        where_str = "WHERE " + " AND ".join(clauses) + _org_filter
        params.extend(_org_params)
        _sql = f"""
SELECT TOP ({_limit}) m.AgniveerNo, m.FullName, eq.Type, eq.GivenCondition, eq.GivenDateTime, eq.ReturnCondition, eq.ReturnDateTime, eq.Remarks
FROM AgniveerEquipment eq
INNER JOIN AgniveerMaster m ON m.Id = eq.AgniveerId
{where_str}
ORDER BY eq.GivenDateTime DESC, m.AgniveerNo ASC
"""
        _rows, _run_err = run_readonly(_sql, params)
        if not _run_err:
            return _to_section(_rows or [], intent, sql=_sql), None

    # ── PersonalDetails & Skills Fast-Path ─────────────────────────────────
    _cat_lower = str(intent.get("category") or "").lower()
    if _cat_lower in ("personaldetails", "personaldetail", "personal_details", "skills", "skill") or (not intent.get("category") and (intent.get("agniveer_no") or intent.get("agniveerNo"))):

        _p_agniveer_no = intent.get("agniveer_no") or intent.get("agniveerNo")
        _p_class = intent.get("class") or intent.get("class_")
        _p_state = intent.get("state") or intent.get("district")
        _p_sport = intent.get("sport")
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

        # BloodGroup summary breakdown query (e.g. "Show blood group details")
        if ("blood group" in _raw_q or _p_metric == "BloodGroup" or _p_op in ("BloodGroup", "BloodGroupDetails")) and not _p_blood_group and not _p_agniveer_no:
            _sql = f"""
SELECT TOP ({_limit}) m.BloodGroup, COUNT(*) AS AgniveerCount
FROM AgniveerMaster m
WHERE ISNULL(m.IsDisqualified,0) = 0 AND m.BloodGroup IS NOT NULL AND TRIM(m.BloodGroup) <> ''
  {_p_org_filter}
GROUP BY m.BloodGroup
ORDER BY AgniveerCount DESC
"""
            _rows, _run_err = run_readonly(_sql, _p_org_params)
            if not _run_err:
                return _to_section(_rows or [], intent, sql=_sql), None

        clauses = ["ISNULL(m.IsDisqualified,0) = 0"]
        params = []

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

        where_str = "WHERE " + " AND ".join(clauses) + _p_org_filter
        params.extend(_p_org_params)

        # A specific field was asked about (e.g. "what is the height of
        # X" -> metric="Height", set by personal_details_parser.py) — return
        # just that field plus the identifying columns, not the entire
        # 14-column profile. Validated against the same column whitelist
        # personal_details_parser.py extracts `metric` from, so this can
        # never become a column-name injection point.
        from intent_engine.personal_details_parser import AGNIVEER_PERSONAL_COLUMNS

        _p_metric = intent.get("metric")
        if _p_metric and _p_metric in AGNIVEER_PERSONAL_COLUMNS and _p_metric not in ("AgniveerNo", "FullName"):
            _select_cols = f"m.AgniveerNo, m.FullName, m.{_p_metric}"
        else:
            _select_cols = (
                "m.AgniveerNo, m.FullName, m.Class, m.State, m.District, m.Qualification, "
                "m.Sports, m.Skill, m.Hobby, m.BloodGroup, m.DateOfBirth, m.Height, m.Weight, "
                "m.MobileNo, m.Email, m.Address"
            )

        _sql = f"""
SELECT TOP ({_limit}) {_select_cols}
FROM AgniveerMaster m
{where_str}
ORDER BY m.AgniveerNo ASC
"""
        _rows, _run_err = run_readonly(_sql, params)
        if not _run_err:
            return _to_section(_rows or [], intent, sql=_sql), None




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
        rows, run_err = run_readonly(sql, params)
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

    v2_intent = {
        "base_concept": base_concept,
        "filters": filters,
        "limit": intent.get("number") or intent.get("top_n") or SQL_MAX_ROWS,
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
            sql, params = _build_medical_bmi_sql(
                top_n=int(intent.get("number") or SQL_MAX_ROWS),
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
            rows, run_err = run_readonly(sql, params)
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
            sql, params = _build_medical_blood_group_sql(
                top_n=int(intent.get("number") or SQL_MAX_ROWS),
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
            rows, run_err = run_readonly(sql, params)
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
        rows, run_err = run_readonly(sql, params)
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
            "limit": intent.get("number") or intent.get("top_n") or SQL_MAX_ROWS,
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


def _get_top_n(intent: Dict, default: int = 500) -> int:
    num = intent.get("number")
    if num is None:
        return default
    op = str(intent.get("operation") or intent.get("subcategory") or "").lower()
    raw_q = str(intent.get("raw_query") or "").lower()
    if op in ("top", "bottom", "rank", "topperformers", "lowestperformers") or "top" in raw_q or "bottom" in raw_q or "rank" in raw_q:
        return int(num)
    return default


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
