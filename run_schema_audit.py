"""
run_schema_audit.py
===================
Developer utility to audit the database schema columns against the code definitions.
Detects any schema drift and generates a report.
"""

import json
import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

# Load environment
from dotenv import load_dotenv

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.WARNING)

# Ensure workspace is in sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import sql_executor
import sql_schema_guard
from intent_engine.query_planner import plan_query
from schema_engine import schema_engine

# Questions to verify in Step 6
VERIFY_QUESTIONS = [
    (2, "List agniveers playing volleyball in Jaswant company"),
    (7, "List cricket players in Krishna company"),
    (8, "Show disqualified agniveers in Mahadev company"),
    (13, "Agniveers in Jaswant company who play volleyball and are on medical leave"),
    (
        15,
        "Show agniveers in Krishna company who are disqualified and have pending equipment",
    ),
    (21, "Compare Lakhwinder company vs Jaswant company performance"),
    (22, "Compare Krishna company vs Mahadev company performance"),
]

# Set up tracing to capture executed queries & errors
execution_trace = {"sql": None, "params": None, "error": None, "db_error": None}

# Keep original references
original_run_readonly = sql_executor.run_readonly


def patched_run_readonly(sql: str, params: Optional[List[Any]] = None):
    execution_trace["sql"] = sql
    execution_trace["params"] = params
    return original_run_readonly(sql, params)


sql_executor.run_readonly = patched_run_readonly


class TraceLogger:
    def __init__(self, original_logger):
        self.original_logger = original_logger

    def warning(self, msg, *args, **kwargs):
        formatted_msg = msg % args if args else msg
        if (
            "SQL execution error" in formatted_msg
            or "Database query execution failed" in formatted_msg
        ):
            execution_trace["db_error"] = formatted_msg
        self.original_logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        formatted_msg = msg % args if args else msg
        if (
            "SQL execution error" in formatted_msg
            or "Database query execution failed" in formatted_msg
        ):
            execution_trace["db_error"] = formatted_msg
        self.original_logger.error(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.original_logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.original_logger.info(msg, *args, **kwargs)


sql_executor.logger = TraceLogger(sql_executor.logger)


def run_pipeline_for_question(q_num: int, question: str) -> Dict[str, Any]:
    plan = plan_query(question)
    query_type = plan.query_type.value
    ops = plan.operations

    legs = []

    if not ops:
        execution_trace["sql"] = None
        execution_trace["params"] = None
        execution_trace["error"] = None
        execution_trace["db_error"] = None

        try:
            res, err = sql_executor.execute_sql_query(question=question, intent=None)
            if err:
                execution_trace["error"] = err
        except Exception as e:
            execution_trace["error"] = str(e)

        legs.append(
            {
                "leg_idx": 1,
                "sql": execution_trace["sql"],
                "params": execution_trace["params"],
                "error": execution_trace["error"],
                "db_error": execution_trace["db_error"],
            }
        )
    else:
        for idx, op in enumerate(ops):
            execution_trace["sql"] = None
            execution_trace["params"] = None
            execution_trace["error"] = None
            execution_trace["db_error"] = None

            leg_q = op.raw_fragment or question

            try:
                res, err = sql_executor.execute_sql_query(
                    question=leg_q, intent=op.intent_result
                )
                if err:
                    execution_trace["error"] = err
            except Exception as e:
                execution_trace["error"] = str(e)

            legs.append(
                {
                    "leg_idx": idx + 1,
                    "sql": execution_trace["sql"],
                    "params": execution_trace["params"],
                    "error": execution_trace["error"],
                    "db_error": execution_trace["db_error"],
                }
            )

    return {
        "q_num": q_num,
        "question": question,
        "query_type": query_type,
        "legs": legs,
    }


def clear_schema_caches():
    schema_engine.get_columns.cache_clear()
    schema_engine.get_table_for_concept.cache_clear()
    schema_engine.get_concept_for_table.cache_clear()
    schema_engine.get_foreign_keys.cache_clear()
    schema_engine.get_reverse_relationships.cache_clear()
    schema_engine.get_display_columns.cache_clear()
    schema_engine.get_searchable_columns.cache_clear()
    schema_engine.get_aggregation_columns.cache_clear()
    schema_engine.get_ranking_columns.cache_clear()
    schema_engine.get_status_columns.cache_clear()
    schema_engine.get_date_columns.cache_clear()


def main():
    print("=====================================================================")
    print("AGNIAI SCHEMA-DRIFT AUDIT REPORT")
    print("=====================================================================")

    # 1. Fetch live schema
    print("\n[STEP 1] Fetching Live Schema from Database...")
    live_schema = sql_schema_guard.fetch_live_schema()
    if live_schema is None:
        print("ERROR: sql_schema_guard.fetch_live_schema() returned None.")
        print("Could not connect to the database or pyodbc is missing.")
        sys.exit(1)
    print("SUCCESS: Live schema fetched successfully.")
    print(f"Total tables found in live database: {len(live_schema)}")

    # 2. Check schema drift warnings (table-level)
    print("\n[STEP 2] Running check_schema_drift warnings...")
    warnings = sql_schema_guard.check_schema_drift(live_schema)
    if not warnings:
        print("No table-level schema drift warnings.")
    else:
        for warning in warnings:
            print(f"WARNING: {warning}")

    # 3. Column-level diff
    print("\n[STEP 3] Column-Level Diff (actual_schema.json vs Live DB)...")
    json_path = schema_engine.schema_path
    with open(json_path, "r", encoding="utf-8") as f:
        json_schema = json.load(f)

    print(
        f"{'Table Name':<35} | {'Missing From DB (Drifted)':<35} | {'Missing From JSON (New)':<35}"
    )
    print("-" * 111)

    drifted_columns_by_table = {}

    for table_name in sorted(json_schema.keys()):
        if table_name == "__EFMigrationsHistory":
            continue
        json_cols = set(json_schema[table_name].keys())
        live_cols = live_schema.get(table_name, set())

        missing_from_db = json_cols - live_cols
        missing_from_json = live_cols - json_cols

        if missing_from_db:
            drifted_columns_by_table[table_name] = missing_from_db

        missing_from_db_str = (
            ", ".join(sorted(missing_from_db)) if missing_from_db else "None"
        )
        missing_from_json_str = (
            ", ".join(sorted(missing_from_json)) if missing_from_json else "None"
        )

        print(
            f"{table_name:<35} | {missing_from_db_str:<35} | {missing_from_json_str:<35}"
        )

    # 4. Cross-check sql_executor.py
    print(
        "\n[STEP 4] Cross-checking Prompt & Hardcoded References in sql_executor.py..."
    )
    live_tables_map = {t.lower(): t for t in live_schema.keys()}

    sql_lines = []
    for line in sql_executor._GENERATION_SYSTEM.split("\n"):
        if line.startswith("SQL: "):
            sql_lines.append(line[5:])

    for idx, sql_text in enumerate(sql_lines):
        print(f"\nChecking Example {idx+1} SQL:")
        print(f"  {sql_text[:120]}...")

        ctes = set(re.findall(r"(\b\w+)\s+AS\s*\(", sql_text, re.IGNORECASE))
        words = re.findall(r"\b[a-zA-Z_]\w*\b", sql_text)
        tables_found = set()
        for w in words:
            if w.lower() in live_tables_map and w not in ctes:
                tables_found.add(live_tables_map[w.lower()])

        # Check tables
        for t in sorted(tables_found):
            if t not in live_schema:
                print(f"  [ALERT] Table '{t}' referenced but missing in live DB!")
            else:
                print(f"  - Table '{t}' verified")

        # Check columns of type alias.column or table.column
        alias_matches = re.finditer(
            r"\b([a-zA-Z_]\w*)\s+(?:AS\s+)?([a-z])\b", sql_text, re.IGNORECASE
        )
        alias_map = {}
        for m in alias_matches:
            tbl, alias = m.groups()
            if tbl.lower() in live_tables_map:
                alias_map[alias.lower()] = live_tables_map[tbl.lower()]

        col_refs = re.findall(
            r"\b([a-z0-9_]+)\.([a-zA-Z_]\w*)\b", sql_text, re.IGNORECASE
        )
        for alias, col in col_refs:
            if alias.lower() in alias_map:
                resolved_table = alias_map[alias.lower()]
                if resolved_table in live_schema:
                    live_cols = live_schema[resolved_table]
                    live_cols_lower = {c.lower() for c in live_cols}
                    if col.lower() not in live_cols_lower:
                        print(
                            f"  [ALERT] Column '{resolved_table}.{col}' referenced but missing in live DB!"
                        )
            elif alias.lower() in live_tables_map:
                resolved_table = live_tables_map[alias.lower()]
                live_cols = live_schema[resolved_table]
                live_cols_lower = {c.lower() for c in live_cols}
                if col.lower() not in live_cols_lower:
                    print(
                        f"  [ALERT] Column '{resolved_table}.{col}' referenced but missing in live DB!"
                    )

    # 5. Verify DisqualifiedDate
    print(
        "\n[STEP 5] Verifying DisqualifiedDate Column declared instances vs live database..."
    )
    declared_tables_for_dq = []

    # Read extracted_schema.sql
    try:
        with open("extracted_schema.sql", "r", encoding="utf-8") as f:
            lines = f.readlines()
        current_table = None
        for line in lines:
            m_table = re.search(r"CREATE TABLE \[dbo\]\.\[(\w+)\]", line)
            if m_table:
                current_table = m_table.group(1)
            if "DisqualifiedDate" in line and current_table:
                declared_tables_for_dq.append(current_table)
    except Exception as e:
        print(f"Could not read extracted_schema.sql: {e}")
        # fallback
        declared_tables_for_dq = ["AgniveerMaster"]

    print(
        f"extracted_schema.sql declares 'DisqualifiedDate' in tables: {declared_tables_for_dq}"
    )
    for tbl in sorted(set(declared_tables_for_dq)):
        if tbl not in live_schema:
            print(f"  Table '{tbl}': Does not exist in live DB.")
        else:
            has_col = any(c.lower() == "disqualifieddate" for c in live_schema[tbl])
            print(f"  Table '{tbl}': Genuine live column? {has_col}")

    # 6. Run Pipeline 7 questions (Before/After patch simulation)
    print(
        "\n[STEP 6] Running Pipeline Verification (Before / After Patch Simulation)..."
    )

    print("\n--- PHASE 1: Running BEFORE Patch ---")
    before_results = {}
    for q_num, question in VERIFY_QUESTIONS:
        print(f"Running Q{q_num}...")
        res = run_pipeline_for_question(q_num, question)
        before_results[q_num] = res

    # Apply schema patch in-memory
    print("\nApplying schema patch in-memory (removing drifted columns)...")
    original_schema_backup = json.loads(json.dumps(schema_engine.schema))

    for tbl, cols in drifted_columns_by_table.items():
        for col in cols:
            if col in schema_engine.schema.get(tbl, {}):
                print(f"  - Removing '{col}' from '{tbl}' schema definition in-memory.")
                del schema_engine.schema[tbl][col]

    clear_schema_caches()

    print("\n--- PHASE 2: Running AFTER Patch ---")
    after_results = {}
    for q_num, question in VERIFY_QUESTIONS:
        print(f"Running Q{q_num}...")
        res = run_pipeline_for_question(q_num, question)
        after_results[q_num] = res

    # Restore original schema
    schema_engine.schema = original_schema_backup
    clear_schema_caches()

    # Report before/after comparison
    print("\n=====================================================================")
    print("BEFORE / AFTER COMPARISON FOR THE 7 SPECIFIED QUESTIONS")
    print("=====================================================================")
    for q_num, question in VERIFY_QUESTIONS:
        b_res = before_results[q_num]
        a_res = after_results[q_num]

        print(f'\nQ{q_num}: "{question}"')
        for idx in range(len(b_res["legs"])):
            b_leg = b_res["legs"][idx]
            a_leg = a_res["legs"][idx]

            print(f"  Leg {idx+1}:")
            # Before status
            if b_leg["error"]:
                b_status = f"FAILED: {b_leg['error']}"
                if b_leg["db_error"]:
                    b_status += f" (Detail: {b_leg['db_error'].strip()})"
            else:
                b_status = "SUCCESS"

            # After status
            if a_leg["error"]:
                a_status = f"FAILED: {a_leg['error']}"
                if a_leg["db_error"]:
                    a_status += f" (Detail: {a_leg['db_error'].strip()})"
            else:
                a_status = "SUCCESS"

            print(f"    - BEFORE: {b_status}")
            print(f"    - AFTER:  {a_status}")
            if not a_leg["error"] and a_leg["sql"]:
                print(f"    - Compiled SQL after patch: {a_leg['sql']}")
                print(f"    - Parameters: {a_leg['params']}")


if __name__ == "__main__":
    main()
