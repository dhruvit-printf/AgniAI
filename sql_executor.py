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

Flow:  intent/question -> [golden fast-path?] -> generate SQL (LLM)
       -> validate (hard safety gate) -> run read-only -> rows

SAFETY MODEL (do not weaken any of these):
  1. Connect with a READ-ONLY SQL login (db_datareader only, DENY on
     UserMaster.Password / LoginToken). The login is the real wall; the
     validator below is defense-in-depth, not the primary control.
  2. Only a single SELECT / WITH...SELECT statement is ever executed.
  3. Column/table allowlist derived from the schema card. Sensitive columns
     are hard-denied even if a grant slips.
  4. Row cap (SET ROWCOUNT) + command timeout on every execution.
  5. Every number in the final narrative is grounded downstream by
     grounding_utils.ground_and_sanitize against these rows.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

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

# ─────────────────────────────────────────────────────────────────────────────
#  SCHEMA CARD — curated, NOT a raw INFORMATION_SCHEMA dump.
#  This is the single source the generator sees. The three landmines that an
#  LLM will otherwise get wrong are stated as hard rules, not left to inference.
# ─────────────────────────────────────────────────────────────────────────────
SCHEMA_CARD = r"""
DATABASE: DB_Agni (Microsoft SQL Server)

CORE ENTITY
  AgniveerMaster a  (PK a.Id bigint)
    a.AgniveerNo, a.FullName, a.BatchId, a.PlatoonId, a.MainCategory, a.Class,
    a.DateOfBirth, a.DateOfJoining, a.Height, a.Weight, a.BloodGroup,
    a.Skill, a.Sports, a.District, a.State, a.IsActive, a.IsDisqualified

ORG HIERARCHY (join map)
  Agniveer -> Platoon -> Company :  a.PlatoonId = p.Id ,  p.CompanyId = c.Id
  Agniveer -> Batch (independent axis) :  a.BatchId = b.Id
    PlatoonMaster  p (p.Id, p.Name, p.PlatoonNo, p.CompanyId)
    CompanyMaster  c (c.Id, c.Name)
    BatchMaster    b (b.Id, b.BatchName)

DOMAIN TABLES (all join to AgniveerMaster on <table>.AgniveerId = a.Id)
  AgniveerAttendanceMaster (AgniveerId, AttendanceDateTime, IsPresent bit, MarkedBy)
  AgniveerLeaveMaster      (AgniveerId, FromDate, ToDate, OnAnnualLeave bit,
                            OnMedicalLeave bit, OnSickLeave bit,
                            [OnATTN'C'] bit, [OnEX PPG] bit, IsHospitalized bit,
                            IsAbscondedLeave bit)
  MedicalRecordMaster m    (AgniveerId, Type, Status, VisitDate, Diagnosis,
                            TreatmentGiven, LeaveType, BloodPressure, HeartRate)
  PoliceVerificationMaster (AgniveerId, PoliceStation, SentDate, ReceivedDate, Status)
  ScoreSectionMaster   s   (Id, SectionName, IsExceptional bit)
  ScoreSubItemMaster       (Id, SectionId, Name, MaxMarks, Cutoff)
  AgniveerScoreAttempt     (AgniveerId, SubItemId, AttemptNo, MarksObtained,
                            IsBestAttempt bit)
  AgniveerSectionResult    (AgniveerId, SectionId, AttemptNo, OmrInputTotal,
                            SubItemTotalMarks, ExceptionalMarks, Grading)
  EquipmentMaster e        (Id, Name, Category, Type)
  AgniveerEquipment        (AgniveerId, EquipmentId, Type, GivenDateTime,
                            ReturnDateTime, GivenCondition, ReturnCondition)
  DistributionMaster d     (Id, Name)
  AgniveerRelationMaster   (AgniveerId, DistributionId)

HARD RULES — these are NON-NEGOTIABLE. Break one and the query is rejected.
  R1. ACTIVE FILTER: every query touching AgniveerMaster MUST include
        (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
      Exception: an explicit "disqualified agniveers" request inverts it to
        (a.IsDisqualified = 1).
  R2. SPECIAL COLUMNS: these two AgniveerLeaveMaster columns contain a
      space / apostrophe and MUST be written bracket-quoted EXACTLY:
        [OnATTN'C']   and   [OnEX PPG]
      Never rename, alias-away, or unquote them.
  R3. EXCEPTIONAL MARKS: when a section may be exceptional, use
        CASE WHEN s.IsExceptional = 1
             THEN COALESCE(r.ExceptionalMarks, 0.0)
             ELSE r.SubItemTotalMarks END
  R4. CROSS-FILTER shape: each independent condition becomes a CTE that
      outputs DISTINCT AgniveerId; the final SELECT INNER JOINs every CTE back
      to AgniveerMaster on a.Id (intersection), displaying a.AgniveerNo.
  R5. NEVER select UserMaster.Password, LoginToken.*, or DefaultLog.
"""

if SQL_SERVER_2008_COMPAT:
    SCHEMA_CARD += (
        "\nDIALECT: SQL Server 2008 target. DO NOT use STRING_AGG, "
        "OFFSET/FETCH, IIF, CONCAT, or LAG/LEAD. Use TOP (n), "
        "STUFF+FOR XML PATH for aggregation, ROW_NUMBER() for paging.\n"
    )

# Golden fast-path: (category, subcategory, operation) -> parameterized SQL.
# Fill this by porting AiCommandService.cs handlers over time. Anything found
# here SKIPS the LLM entirely (deterministic, cheap, testable).
#
# Templates may contain the single placeholder {top_n} — the only thing ever
# substituted into a golden query (see _render_golden_query below), and only
# with a validated integer, so this can never introduce SQL injection.
GOLDEN_QUERIES: Dict[Tuple[str, str, str], str] = {
    (
        "Performance",
        "TopPerformers",
        "Top",
    ): """
SELECT TOP ({top_n}) a.AgniveerNo, a.FullName, SUM(sr.SubItemTotalMarks) AS TotalMarks
FROM AgniveerMaster a
INNER JOIN AgniveerSectionResult sr ON sr.AgniveerId = a.Id
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
GROUP BY a.AgniveerNo, a.FullName
ORDER BY SUM(sr.SubItemTotalMarks) DESC
""".strip(),
    (
        "Performance",
        "LowestPerformers",
        "Bottom",
    ): """
SELECT TOP ({top_n}) a.AgniveerNo, a.FullName, SUM(sr.SubItemTotalMarks) AS TotalMarks
FROM AgniveerMaster a
INNER JOIN AgniveerSectionResult sr ON sr.AgniveerId = a.Id
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
GROUP BY a.AgniveerNo, a.FullName
ORDER BY SUM(sr.SubItemTotalMarks) ASC
""".strip(),
    (
        "Performance",
        "AverageScore",
        "Average",
    ): """
SELECT s.SectionName, AVG(sr.SubItemTotalMarks) AS AverageMarks,
       COUNT(DISTINCT sr.AgniveerId) AS AgniveerCount
FROM AgniveerSectionResult sr
INNER JOIN ScoreSectionMaster s ON s.Id = sr.SectionId
INNER JOIN AgniveerMaster a ON a.Id = sr.AgniveerId
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
GROUP BY s.SectionName
ORDER BY AVG(sr.SubItemTotalMarks) DESC
""".strip(),
    (
        "Attendance",
        "AttendanceSummary",
        "Summary",
    ): """
SELECT a.AgniveerNo, a.FullName,
       SUM(CASE WHEN att.IsPresent = 1 THEN 1 ELSE 0 END) AS PresentDays,
       COUNT(att.Id) AS TotalDays
FROM AgniveerMaster a
INNER JOIN AgniveerAttendanceMaster att ON att.AgniveerId = a.Id
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
GROUP BY a.AgniveerNo, a.FullName
ORDER BY a.AgniveerNo
""".strip(),
    (
        "Leave",
        "CurrentLeaveStatus",
        "Current",
    ): """
SELECT TOP ({top_n}) a.AgniveerNo, a.FullName, l.FromDate, l.ToDate,
       l.OnAnnualLeave, l.OnMedicalLeave, l.OnSickLeave,
       l.[OnATTN'C'], l.[OnEX PPG], l.IsHospitalized
FROM AgniveerMaster a
INNER JOIN AgniveerLeaveMaster l ON l.AgniveerId = a.Id
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND GETDATE() BETWEEN l.FromDate AND l.ToDate
ORDER BY l.FromDate DESC
""".strip(),
    (
        "Medical",
        "DiseaseStatistics",
        "Disease",
    ): """
SELECT TOP ({top_n}) m.Diagnosis, COUNT(DISTINCT m.AgniveerId) AS AffectedCount
FROM MedicalRecordMaster m
INNER JOIN AgniveerMaster a ON a.Id = m.AgniveerId
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND m.Diagnosis IS NOT NULL
GROUP BY m.Diagnosis
ORDER BY COUNT(DISTINCT m.AgniveerId) DESC
""".strip(),
    (
        "Verification",
        "PendingVerification",
        "Pending",
    ): """
SELECT a.AgniveerNo, a.FullName, v.PoliceStation, v.SentDate, v.Status
FROM AgniveerMaster a
INNER JOIN PoliceVerificationMaster v ON v.AgniveerId = a.Id
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND v.Status = 'Pending'
ORDER BY v.SentDate ASC
""".strip(),
    (
        "Equipment",
        "HoldingEquipment",
        "Holding",
    ): """
SELECT a.AgniveerNo, a.FullName, e.Name AS EquipmentName, ae.GivenDateTime, ae.GivenCondition
FROM AgniveerMaster a
INNER JOIN AgniveerEquipment ae ON ae.AgniveerId = a.Id
INNER JOIN EquipmentMaster e ON e.Id = ae.EquipmentId
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND ae.ReturnDateTime IS NULL
ORDER BY ae.GivenDateTime DESC
""".strip(),
    (
        "Attendance",
        "PresentToday",
        "Present",
    ): """
SELECT a.AgniveerNo, a.FullName, p.Name AS Platoon, c.Name AS Company
FROM AgniveerMaster a
INNER JOIN AgniveerAttendanceMaster att ON att.AgniveerId = a.Id
LEFT JOIN PlatoonMaster p ON a.PlatoonId = p.Id
LEFT JOIN CompanyMaster c ON p.CompanyId = c.Id
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND att.IsPresent = 1
  AND CAST(att.AttendanceDateTime AS DATE) = CAST(GETDATE() AS DATE)
ORDER BY a.AgniveerNo
""".strip(),
    (
        "Distribution",
        "LatestDistribution",
        "Latest",
    ): """
SELECT d.Name AS DistributionName, COUNT(DISTINCT r.AgniveerId) AS AgniveerCount
FROM DistributionMaster d
INNER JOIN AgniveerRelationMaster r ON r.DistributionId = d.Id
INNER JOIN AgniveerMaster a ON a.Id = r.AgniveerId
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
GROUP BY d.Name
ORDER BY COUNT(DISTINCT r.AgniveerId) DESC
""".strip(),
}


def _render_golden_query(template: str, intent: Optional[Dict]) -> str:
    """Fill the {top_n} placeholder a golden template may contain from the
    classifier intent. The only substitution ever performed on a golden
    query is this validated integer — never raw user text — so this path
    can never introduce SQL injection."""
    intent = intent or {}
    raw_n = intent.get("number") or intent.get("top_n") or 10
    try:
        top_n = int(raw_n)
    except (TypeError, ValueError):
        top_n = 10
    top_n = max(1, min(top_n, SQL_MAX_ROWS))
    try:
        return template.format(top_n=top_n)
    except (KeyError, IndexError):
        return template


_GENERATION_SYSTEM = (
    "You are a T-SQL generator for a read-only reporting API. "
    "Given the schema and a question, output ONE SQL Server SELECT statement "
    "and NOTHING else — no prose, no markdown, no comments, no semicolons. "
    "Obey every HARD RULE. If the question cannot be answered from the schema, "
    "output exactly: CANNOT_ANSWER"
)

# ── Safety validator ───────────────────────────────────────────────────────
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|merge|drop|alter|truncate|create|grant|revoke|"
    r"exec|execute|sp_|xp_|openrowset|openquery|bulk|shutdown|reconfigure|"
    r"waitfor)\b",
    re.IGNORECASE,
)
_MULTI_STATEMENT = re.compile(r";\s*\S")  # a ';' followed by more content
_COMMENT = re.compile(r"(--|/\*|\*/)")


def validate_sql(sql: str) -> Optional[str]:
    """Return an error string if the SQL is unsafe, else None."""
    if not sql or not sql.strip():
        return "Empty SQL."
    s = sql.strip().rstrip(";").strip()

    low = s.lower()
    if not (low.startswith("select") or low.startswith("with")):
        return "Only single SELECT / WITH...SELECT statements are allowed."
    if _MULTI_STATEMENT.search(s):
        return "Multiple statements are not allowed."
    if _COMMENT.search(s):
        return "Comments are not allowed."
    if _FORBIDDEN.search(low):
        return "Statement contains a forbidden keyword."

    for denied in DENIED_TABLES:
        if re.search(rf"\b{denied}\b", low):
            return f"Access to table '{denied}' is denied."
    for denied in DENIED_COLUMNS:
        tbl, col = denied.split(".")
        if col in low and tbl in low:
            return f"Access to column '{denied}' is denied."
    return None


# ── SQL generation (LLM) ───────────────────────────────────────────────────
def generate_sql(
    question: str, intent: Optional[Dict] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Generate a single SELECT for `question`. Returns (sql, error)."""
    try:
        from config import DEFAULT_MODEL, ollama_session
        from ollama_cpu_chat import chat_with_fallback
    except Exception as exc:  # pragma: no cover
        return None, f"LLM unavailable: {exc}"

    hint = ""
    if intent:
        hint = f"\nCLASSIFIER HINT (may be partial): {intent}\n"

    user = f"{SCHEMA_CARD}\n{hint}\nQUESTION: {question}\nSQL:"
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
    return t[m.start() :].strip() if m else t


def _to_section(
    rows: List[Dict[str, Any]], intent: Optional[Dict] = None
) -> Dict[str, Any]:
    """Wrap flat rows in the same envelope shape the .NET path produces, so
    `universal_normalizer.normalize_response()` / `result_combiner._extract_records()`
    resolve the rows directly instead of falling back to raw-row scanning."""
    return {
        "success": True,
        "records": rows,
        "data": rows,
        "count": len(rows),
    }


# ── Read-only execution ────────────────────────────────────────────────────
def run_readonly(sql: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Execute a validated SELECT against the READ-ONLY login. (rows, error)."""
    if not SQL_READONLY_CONN:
        return None, "SQL_READONLY_CONN is not configured."
    try:
        import pyodbc  # imported lazily so the rest of AgniAI runs without it
    except Exception as exc:  # pragma: no cover
        return None, f"pyodbc not installed: {exc}"

    try:
        conn = pyodbc.connect(
            SQL_READONLY_CONN, timeout=SQL_COMMAND_TIMEOUT_S, autocommit=True
        )
        conn.timeout = SQL_COMMAND_TIMEOUT_S
        cur = conn.cursor()
        # Belt-and-suspenders row cap regardless of query shape (works on 2008).
        cur.execute(f"SET ROWCOUNT {SQL_MAX_ROWS}")
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        cur.execute("SET ROWCOUNT 0")
        conn.close()
        return rows, None
    except Exception as exc:
        # Never leak the raw SQL / connection details to the caller.
        logger.warning("SQL execution error: %s", type(exc).__name__)
        return None, "The generated query could not be executed against the database."


# ── Public entrypoint — mirrors dotnet_executor._call_dotnet contract ──────
def execute_sql_query(
    payload: Optional[Dict] = None,
    *,
    question: str = "",
    intent: Optional[Dict] = None,
    trace_id: Optional[str] = None,
    **_ignored: Any,
) -> Tuple[Any, Optional[str]]:
    """
    Drop-in alternative to _call_dotnet(...). Returns (section, None) on
    success — `section` is the `_to_section()` envelope
    ({"success", "records", "data", "count"}) so downstream normalization
    resolves it exactly like a .NET response — or (None, error) on failure.
    `question` is the raw NL query; `intent` is the classifier output (used
    for the golden fast-path and as a hint).
    """
    # 1) Golden fast-path — deterministic, no LLM.
    if intent:
        key = (
            str(intent.get("category") or ""),
            str(intent.get("subcategory") or ""),
            str(intent.get("operation") or ""),
        )
        golden = GOLDEN_QUERIES.get(key)
        if golden:
            rendered = _render_golden_query(golden, intent)
            err = validate_sql(rendered)
            if err:
                return None, f"Golden query rejected: {err}"
            rows, run_err = run_readonly(rendered)
            if run_err:
                return None, run_err
            metrics_hook("golden_hit")
            return _to_section(rows or [], intent), None

    # 2) Generate.
    sql, gen_err = generate_sql(question, intent)
    if gen_err or sql is None:
        if gen_err == "CANNOT_ANSWER":
            metrics_hook("cannot_answer")
        return None, gen_err or "CANNOT_ANSWER"
    metrics_hook("generated")

    # 3) Validate (hard gate).
    val_err = validate_sql(sql)
    if val_err:
        logger.info("Generated SQL rejected by validator: %s", val_err)
        metrics_hook("validator_rejected")
        return None, val_err

    # 4) Execute read-only.
    rows, run_err = run_readonly(sql)
    if run_err:
        metrics_hook("exec_error")
        return None, run_err
    return _to_section(rows or [], intent), None


def metrics_hook(event: str) -> None:
    """Best-effort metrics increment — never lets an observability failure
    affect the query result. Imported lazily to avoid a hard dependency."""
    try:
        from metrics import metrics_collector

        {
            "generated": metrics_collector.inc_sql_generated,
            "golden_hit": metrics_collector.inc_sql_golden_hit,
            "validator_rejected": metrics_collector.inc_sql_validator_rejected,
            "cannot_answer": metrics_collector.inc_sql_cannot_answer,
            "exec_error": metrics_collector.inc_sql_exec_error,
        }[event]()
    except Exception:
        pass
