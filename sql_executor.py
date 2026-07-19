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

import logging
import os
import re
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
    "SQL: SELECT a.AgniveerNo, a.FullName, SUM(sa.MarksObtained) AS BestTotal FROM AgniveerMaster a INNER JOIN AgniveerScoreAttempt sa ON a.Id = sa.AgniveerId WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL) AND sa.IsBestAttempt = 1 AND a.Sports = 'Volleyball' GROUP BY a.AgniveerNo, a.FullName ORDER BY BestTotal DESC"
)


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

    dynamic_schema = generate_dynamic_schema_card()
    
    if SQL_SERVER_2008_COMPAT:
        dialect_hint = "\nDIALECT: SQL Server 2008 target. DO NOT use STRING_AGG, OFFSET/FETCH, IIF, CONCAT, or LAG/LEAD. Use TOP (n), STUFF+FOR XML PATH for aggregation, ROW_NUMBER() for paging.\n"
    else:
        dialect_hint = ""

    user = f"{dynamic_schema}\n{LLM_HARD_RULES}\n{dialect_hint}\n{hint}\nQUESTION: {question}\nSQL:"
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
    sql = re.sub(r'\ba\.AgniveerId\b', 'a.Id', sql, flags=re.IGNORECASE)
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
    silently finds nothing to match on."""
    if not name:
        return name
    return name[0].lower() + name[1:]


def _camel_case_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {_to_camel_case(k): v for k, v in row.items()}


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
def run_readonly(sql: str, params: Optional[List[Any]] = None) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Execute a validated SELECT against the READ-ONLY login. (rows, error)."""
    if not SQL_READONLY_CONN:
        return None, "SQL_READONLY_CONN is not configured."
    try:
        import pyodbc  # imported lazily so the rest of AgniAI runs without it
        pyodbc.pooling = True
    except Exception as exc:  # pragma: no cover
        return None, f"pyodbc not installed: {exc}"

    try:
        # Enforce transaction timeout limit between 1 and 30 seconds to avoid thread starvation
        timeout_limit = max(1, min(SQL_COMMAND_TIMEOUT_S, 30))
        conn = pyodbc.connect(
            SQL_READONLY_CONN, timeout=timeout_limit, autocommit=True
        )
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
        conn.close()
        return rows, None
    except Exception as exc:
        # Never leak the raw SQL / connection details to the caller.
        logger.warning("SQL execution error: %s | %s\nQuery: %s", type(exc).__name__, str(exc), sql)
        return None, "The generated query could not be executed against the database."


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
    company_name = _pick_legacy_value("Company", "company")
    class_ = _pick_legacy_value("class", "class_")
    blood_group = _pick_legacy_value("bloodGroup", "blood_group")
    sport = _pick_legacy_value("sport")
    diagnose = _pick_legacy_value("diagnose")
    unit_name = _pick_legacy_value("unitName", "unit_name")

    if company_id is not None:
        filters.setdefault("Company.Id", company_id)
    elif company_name is not None:
        filters.setdefault("Company.Name", company_name)

    if platoon_id is not None:
        filters.setdefault("Agniveer.PlatoonId", platoon_id)
    if batch_id is not None:
        filters.setdefault("Agniveer.BatchId", batch_id)
    if agniveer_no is not None:
        filters.setdefault("Agniveer.AgniveerNo", agniveer_no)
    if medical_status is not None:
        filters.setdefault("Medical.Status", medical_status)
    if class_ is not None:
        filters.setdefault("Agniveer.Class", class_)
    if blood_group is not None:
        filters.setdefault("Agniveer.BloodGroup", blood_group)
    if sport is not None:
        filters.setdefault("Agniveer.Sports", {"operator": "LIKE", "value": f"%{sport}%"})
    if diagnose is not None:
        filters.setdefault("Medical.Diagnosis", {"operator": "LIKE", "value": f"%{diagnose}%"})
    if unit_name is not None:
        filters.setdefault("Distribution.Name", {"operator": "LIKE", "value": f"%{unit_name}%"})

    category = intent.get("category")
    operation = intent.get("operation")
    
    # 1. Bucket (ii) Category/Concept mappings
    base_concept = category if category else "Agniveer"
    
    if category == "disqualified":
        base_concept = "Agniveer"
        filters.setdefault("Agniveer.IsDisqualified", 1)
    elif category == "personaldetail":
        base_concept = "Agniveer"
    elif category == "Skills":
        base_concept = "Agniveer"
        if operation == "BySport":
            filters.setdefault("AND", [
                {"Agniveer.Sports": {"operator": "!=", "value": None}},
                {"Agniveer.Sports": {"operator": "!=", "value": ""}}
            ])
        elif operation == "ByClass":
            filters.setdefault("AND", [
                {"Agniveer.Class": {"operator": "!=", "value": None}},
                {"Agniveer.Class": {"operator": "!=", "value": ""}}
            ])

    v2_intent = {
        "base_concept": base_concept,
        "filters": filters,
        "limit": intent.get("number") or intent.get("top_n") or SQL_MAX_ROWS
    }

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

        # 1. Supported ranking/aggregation queries (Mapped to AST)
        if (category == "Performance" and operation in ("Top", "Bottom")) or (category == "Overall" and operation == "OverallPerformance"):
            if section:
                # Section filters on Performance require a chain join through ScoreSubItemMaster -> ScoreSectionMaster,
                # which is a capability gap for the current shortest-path planner (since base_concept is Agniveer).
                raise CapabilityGapError(f"Performance ranking queries with section filter '{section}' are not supported in AST.")
            
            # Overall performance or ranking queries (no section filter)
            base_concept = "Agniveer"
            v2_intent["base_concept"] = "Agniveer"
            
            alias_name = "OverallMarks" if operation == "OverallPerformance" else "TotalMarks"
            v2_intent["aggregates"] = [
                {
                    "function": "SUM",
                    "concept": "Performance",
                    "column": "MarksObtained",
                    "alias": alias_name
                }
            ]
            
            # Must scope MarksObtained to best attempt as per R7 business rules
            filters.setdefault("Performance.IsBestAttempt", 1)
            
            descending = False if operation == "Bottom" else True
            v2_intent["order_by"] = [
                {"concept": None, "column": alias_name, "descending": descending},
                {"concept": "Agniveer", "column": "AgniveerNo", "descending": False}
            ]

        # 2. Gaps/Unsupported aggregation operations (routed to Tier 2 Fallback)
        elif (
            (category == "Performance" and operation in ("Average", "BestAttempt", "Grading", "GradingSummary", "Improvement", "Drop", "AttemptWise", "Trend")) or
            (category == "Leave" and operation in ("Most", "Least")) or
            (category == "Medical" and operation in ("BMI", "Disease", "BloodGroup")) or
            (category == "Attendance" and operation in ("Monthly", "Weekly", "Summary")) or
            (category == "Equipment" and operation == "AgniveerWise") or
            (category == "Distribution" and operation in ("Latest", "TopUnit")) or
            (category == "Strength")
        ):
            raise CapabilityGapError(f"Operation '{category}/{operation}' is a known capability gap (subquery/CTE/conditional-sum) and is routed to LLM.")

        ast = query_planner_v2.plan_query(v2_intent)
        
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
            exc_info=True
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
    explanation = explainability_engine.explain(ast) if ast is not None else {
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
    
    execution_metadata = {
        "planning_duration_ms": planning_duration_ms,
        "compilation_duration_ms": compilation_duration_ms,
        "execution_duration_ms": execution_duration_ms,
        "rows_returned": len(rows) if rows else 0,
        "explanation": explanation,
        "sql": sql
    }
    
    res = _to_section(rows or [], intent, sql=sql)
    res["execution_metadata"] = execution_metadata
    return res, None


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
