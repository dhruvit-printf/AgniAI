"""
sql_schema_guard.py
====================
Builds a dynamic table/column allowlist from the live INFORMATION_SCHEMA and
diffs it against sql_executor.SCHEMA_CARD's curated table list, warning
(never raising) on drift so hallucinated columns are rejected against
reality instead of a hand-maintained list going stale.

The hard DENIED_TABLES / DENIED_COLUMNS denylist in sql_executor.py is an
override that always wins — a denied table/column is excluded from the
allowlist even if it is present (and grantable) in the live schema.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

from sql_executor import (
    DENIED_COLUMNS,
    DENIED_TABLES,
    SQL_COMMAND_TIMEOUT_S,
    SQL_READONLY_CONN,
)

logger = logging.getLogger(__name__)

# The table names sql_executor.SCHEMA_CARD documents. Kept here as a plain
# set (rather than parsed out of the free-text schema card) so the drift
# diff is exact and doesn't depend on prose formatting.
CURATED_TABLES: frozenset = frozenset(
    {
        "AgniveerMaster",
        "BatchMaster",
        "CompanyMaster",
        "PlatoonMaster",
        "AgniveerAttendanceMaster",
        "AgniveerLeaveMaster",
        "MedicalRecordMaster",
        "PoliceVerificationMaster",
        "ScoreSectionMaster",
        "ScoreSubItemMaster",
        "AgniveerScoreAttempt",
        "AgniveerSectionResult",
        "EquipmentMaster",
        "AgniveerEquipment",
        "DistributionMaster",
        "AgniveerRelationMaster",
        "CompanySchedule",
        "AgniveerPlatoonHistory",
        "DistributionHistoryMaster",
        "PlatoonCompanyHistory",
        "CompanyCommanderHistory",
        "PlatoonCommanderHistory",
        "CompanyCommandingOfficerHistory",
        "UserMaster",
        "RoleMaster",
        "UserRole",
    }
)


def fetch_live_schema() -> Optional[Dict[str, Set[str]]]:
    """Read {table: {column names}} from INFORMATION_SCHEMA.COLUMNS via the
    read-only connection. Returns None — never raises — if the DB is
    unreachable or pyodbc isn't installed; callers must treat that as "skip
    the check", not as a fatal startup error."""
    if not SQL_READONLY_CONN:
        return None
    try:
        import pyodbc
    except Exception:
        return None

    try:
        conn = pyodbc.connect(
            SQL_READONLY_CONN, timeout=SQL_COMMAND_TIMEOUT_S, autocommit=True
        )
        cur = conn.cursor()
        cur.execute("SELECT TABLE_NAME, COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS")
        schema: Dict[str, Set[str]] = {}
        for table_name, column_name in cur.fetchall():
            schema.setdefault(table_name, set()).add(column_name)
        conn.close()
        return schema
    except Exception as exc:
        logger.warning(
            "sql_schema_guard: could not read live schema: %s", type(exc).__name__
        )
        return None


def build_allowlist(live_schema: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """Build the table/column allowlist from the live schema. The hard
    denylist always wins: a denied table is dropped entirely, a denied
    column is dropped from its table's allowed-columns set — even if the
    live grants would otherwise permit it."""
    allowlist: Dict[str, Set[str]] = {}
    for table, columns in live_schema.items():
        if table.lower() in DENIED_TABLES:
            continue
        allowed_columns = {
            column
            for column in columns
            if f"{table.lower()}.{column.lower()}" not in DENIED_COLUMNS
        }
        allowlist[table] = allowed_columns
    return allowlist


def check_schema_drift(live_schema: Dict[str, Set[str]]) -> List[str]:
    """Diff live table names against the curated SCHEMA_CARD table list.
    Returns human-readable warning strings; never raises."""
    warnings: List[str] = []
    live_tables = set(live_schema.keys())
    live_tables_lower = {t.lower() for t in live_tables}
    curated_lower = {t.lower() for t in CURATED_TABLES}

    missing = {t for t in CURATED_TABLES if t.lower() not in live_tables_lower}
    extra = {t for t in live_tables if t.lower() not in curated_lower}

    if missing:
        warnings.append(
            f"Schema drift: curated tables missing from live DB: {sorted(missing)}"
        )
    if extra:
        warnings.append(
            f"Schema drift: live DB has tables not in the curated schema card: {sorted(extra)}"
        )
    return warnings


def run_schema_guard() -> Optional[Dict[str, Set[str]]]:
    """Entry point for startup / a CI check: fetch the live schema, log
    (never raise) any drift, and return the dynamic allowlist. Returns None
    if the live schema could not be read (e.g. DB unreachable, SQL backend
    disabled) — callers should treat that as "nothing to enforce yet", not
    as a startup failure."""
    live_schema = fetch_live_schema()
    if live_schema is None:
        logger.info("sql_schema_guard: live schema unavailable — skipping drift check.")
        return None

    for warning in check_schema_drift(live_schema):
        logger.warning(warning)

    return build_allowlist(live_schema)


def _generate_schema_card_from_engine() -> str:
    try:
        from schema_engine import schema_engine

        schema = {}
        for table in schema_engine.get_tables():
            if table.lower() in DENIED_TABLES:
                continue
            cols = []
            for col in schema_engine.get_columns(table):
                if f"{table.lower()}.{col.lower()}" not in DENIED_COLUMNS:
                    cols.append(col)
            if cols:
                schema[table] = cols

        fks = []
        for table in schema.keys():
            for fk in schema_engine.get_foreign_keys(table):
                p_col = fk["column"]
                r_tab = fk["referenced_table"]
                r_col = fk["referenced_column"]
                if r_tab in schema:
                    fks.append(f"{table}.{p_col} = {r_tab}.{r_col}")

        lines = ["DATABASE: DB_Agni (Microsoft SQL Server)", "", "TABLES AND VIEWS:"]
        for table, cols in schema.items():
            lines.append(f"  {table} ({', '.join(cols)})")

        lines.append("")
        lines.append("RELATIONSHIPS (Foreign Keys):")
        for fk in fks:
            lines.append(f"  {fk}")

        return "\n".join(lines)
    except Exception as exc:
        logger.error(f"Failed to generate dynamic schema from engine: {exc}")
        return "SCHEMA GENERATION FAILED"


def generate_dynamic_schema_card() -> str:
    """Generate the dynamic SCHEMA_CARD using INFORMATION_SCHEMA and sys.foreign_keys."""
    if not SQL_READONLY_CONN:
        return _generate_schema_card_from_engine()

    try:
        import pyodbc

        conn = pyodbc.connect(
            SQL_READONLY_CONN, timeout=SQL_COMMAND_TIMEOUT_S, autocommit=True
        )
        cur = conn.cursor()

        # 1. Fetch tables and views
        cur.execute("SELECT TABLE_NAME, COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS")
        schema = {}
        for table, col in cur.fetchall():
            if (
                table.lower() not in DENIED_TABLES
                and f"{table.lower()}.{col.lower()}" not in DENIED_COLUMNS
            ):
                schema.setdefault(table, []).append(col)

        # 2. Fetch foreign keys
        cur.execute("""
            SELECT 
                OBJECT_NAME(f.parent_object_id) AS TableName,
                COL_NAME(fc.parent_object_id, fc.parent_column_id) AS ColumnName,
                OBJECT_NAME(f.referenced_object_id) AS ReferenceTableName,
                COL_NAME(fc.referenced_object_id, fc.referenced_column_id) AS ReferenceColumnName
            FROM sys.foreign_keys AS f
            INNER JOIN sys.foreign_key_columns AS fc 
                ON f.object_id = fc.constraint_object_id
        """)
        fks = []
        for p_tab, p_col, r_tab, r_col in cur.fetchall():
            if p_tab in schema and r_tab in schema:
                fks.append(f"{p_tab}.{p_col} = {r_tab}.{r_col}")

        conn.close()

        lines = ["DATABASE: DB_Agni (Microsoft SQL Server)", "", "TABLES AND VIEWS:"]
        for table, cols in schema.items():
            lines.append(f"  {table} ({', '.join(cols)})")

        lines.append("")
        lines.append("RELATIONSHIPS (Foreign Keys):")
        for fk in fks:
            lines.append(f"  {fk}")

        return "\n".join(lines)
    except Exception as exc:
        logger.error(
            f"Failed to generate dynamic schema from DB, falling back to schema engine: {exc}"
        )
        return _generate_schema_card_from_engine()
