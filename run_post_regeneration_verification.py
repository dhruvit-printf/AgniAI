"""
run_post_regeneration_verification.py
=====================================
Developer utility to verify compile and execution behavior of the database query 
pipeline after a schema regeneration.
"""

import os
import sys
import json
import logging
import glob
import re
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
from schema_engine import schema_engine
from intent_engine.query_planner import plan_query

# 7 verification questions (Step 7)
VERIFY_QUESTIONS = [
    (2, "List agniveers playing volleyball in Jaswant company"),
    (7, "List cricket players in Krishna company"),
    (8, "Show disqualified agniveers in Mahadev company"),
    (13, "Agniveers in Jaswant company who play volleyball and are on medical leave"),
    (15, "Show agniveers in Krishna company who are disqualified and have pending equipment"),
    (21, "Compare Lakhwinder company vs Jaswant company performance"),
    (22, "Compare Krishna company vs Mahadev company performance")
]

# All 40 questions (Step 8)
ALL_QUESTIONS = [
    (1, "Show top 10 BPET performers in Lakhwinder company"),
    (2, "List agniveers playing volleyball in Jaswant company"),
    (3, "Show medical leave records for Krishna company"),
    (4, "Show bottom 5 PPT performers in Mahadev company"),
    (5, "Show top performers in Lakhwinder company"),
    (6, "Show firing scores for Jaswant company"),
    (7, "List cricket players in Krishna company"),
    (8, "Show disqualified agniveers in Mahadev company"),
    (9, "Show top performers in Jaswant company"),
    (10, "Show equipment issued in Lakhwinder company"),
    (11, "Top performers in Krishna company who are on medical leave"),
    (12, "Show top 10 BPET performers in Lakhwinder company whose police verification is rejected"),
    (13, "Agniveers in Jaswant company who play volleyball and are on medical leave"),
    (14, "Top performers in Mahadev company who have blood group O positive"),
    (15, "Show agniveers in Krishna company who are disqualified and have pending equipment"),
    (16, "Agniveers in Lakhwinder company who failed firing and still have issued equipment"),
    (17, "Top BPET scorers in Jaswant company who are on medical leave"),
    (18, "Top drill performers in Mahadev company whose medical status is hospitalized"),
    (19, "Agniveers in Krishna company whose police verification is pending and are absent"),
    (20, "Top performers in Lakhwinder company who have blood group B positive"),
    (21, "Compare Lakhwinder company vs Jaswant company performance"),
    (22, "Compare Krishna company vs Mahadev company performance"),
    (23, "Compare BPET scores of Lakhwinder vs Jaswant company"),
    (24, "Compare firing scores between Krishna and Mahadev company"),
    (25, "Compare attendance of Lakhwinder company vs Krishna company"),
    (26, "Compare leave counts of Jaswant company vs Mahadev company"),
    (27, "Compare medical fitness of Lakhwinder vs Mahadev company"),
    (28, "Compare drill performance of Jaswant vs Krishna company"),
    (29, "Compare equipment holding of Lakhwinder company vs Mahadev company"),
    (30, "Compare verification status of Jaswant company vs Krishna company"),
    (31, "Give me attendance summary and performance summary for Lakhwinder company"),
    (32, "Attendance summary for Jaswant company, then performance summary"),
    (33, "Show leave summary and equipment summary for Krishna company"),
    (34, "Performance summary for Mahadev company, then medical summary"),
    (35, "Give me verification summary and attendance summary for Lakhwinder company"),
    (36, "Show equipment summary for Jaswant company, then leave summary"),
    (37, "Medical summary and performance summary for Krishna company"),
    (38, "Give me distribution summary and attendance summary for Mahadev company"),
    (39, "Attendance summary and equipment summary for Lakhwinder company"),
    (40, "Show verification summary for Jaswant company, then medical summary")
]

# Set up tracing to capture executed queries & errors
execution_trace = {
    "tier": "AST",
    "reason": None,
    "sql": None,
    "params": None,
    "error": None,
    "db_error": None
}

# Original references and patched functions to capture trace
original_run_readonly = sql_executor.run_readonly
original_generate_sql = sql_executor.generate_sql

def patched_run_readonly(sql: str, params: Optional[List[Any]] = None):
    execution_trace["sql"] = sql
    execution_trace["params"] = params
    return original_run_readonly(sql, params)

def patched_generate_sql(question: str, intent: Optional[Dict] = None):
    execution_trace["tier"] = "LLM fallback"
    return original_generate_sql(question, intent)

sql_executor.run_readonly = patched_run_readonly
sql_executor.generate_sql = patched_generate_sql

class TraceLogger:
    def __init__(self, original_logger):
        self.original_logger = original_logger

    def warning(self, msg, *args, **kwargs):
        formatted_msg = msg % args if args else msg
        if "SQL execution error" in formatted_msg or "Database query execution failed" in formatted_msg:
            execution_trace["db_error"] = formatted_msg
        else:
            execution_trace["reason"] = formatted_msg
        self.original_logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        formatted_msg = msg % args if args else msg
        if "SQL execution error" in formatted_msg or "Database query execution failed" in formatted_msg:
            execution_trace["db_error"] = formatted_msg
        else:
            execution_trace["reason"] = formatted_msg
        self.original_logger.error(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.original_logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.original_logger.info(msg, *args, **kwargs)

sql_executor.logger = TraceLogger(sql_executor.logger)


def run_question(q_num: int, question: str) -> Dict[str, Any]:
    plan = plan_query(question)
    query_type = plan.query_type.value
    ops = plan.operations
    
    legs = []
    any_tier2 = False
    
    if not ops:
        execution_trace["tier"] = "AST"
        execution_trace["reason"] = None
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
            
        if execution_trace["tier"] == "LLM fallback":
            any_tier2 = True
            
        legs.append({
            "leg_idx": 1,
            "tier": execution_trace["tier"],
            "reason": execution_trace["reason"],
            "sql": execution_trace["sql"],
            "params": execution_trace["params"],
            "error": execution_trace["error"],
            "db_error": execution_trace["db_error"]
        })
    else:
        for idx, op in enumerate(ops):
            execution_trace["tier"] = "AST"
            execution_trace["reason"] = None
            execution_trace["sql"] = None
            execution_trace["params"] = None
            execution_trace["error"] = None
            execution_trace["db_error"] = None
            
            leg_q = op.raw_fragment or question
            
            try:
                res, err = sql_executor.execute_sql_query(question=leg_q, intent=op.intent_result)
                if err:
                    execution_trace["error"] = err
            except Exception as e:
                execution_trace["error"] = str(e)
                
            if execution_trace["tier"] == "LLM fallback":
                any_tier2 = True
                
            legs.append({
                "leg_idx": idx + 1,
                "tier": execution_trace["tier"],
                "reason": execution_trace["reason"],
                "sql": execution_trace["sql"],
                "params": execution_trace["params"],
                "error": execution_trace["error"],
                "db_error": execution_trace["db_error"]
            })
            
    return {
        "q_num": q_num,
        "question": question,
        "query_type": query_type,
        "legs": legs,
        "any_tier2": any_tier2
    }


def main():
    print("=====================================================================")
    print("POST-REGENERATION PIPELINE VERIFICATION")
    print("=====================================================================")
    
    # 1. Step 3 Sanity Check (New schema vs Backup counts)
    print("\n[STEP 3] Comparing counts: actual_schema.json vs Backup...")
    backups = sorted(glob.glob("actual_schema.json.*.bak"))
    if not backups:
        print("ERROR: No backup actual_schema.json.*.bak files found.")
        sys.exit(1)
        
    latest_backup = backups[-1]
    print(f"Latest backup file identified: {latest_backup}")
    
    with open("actual_schema.json", "r", encoding="utf-8") as f:
        new_schema = json.load(f)
    with open(latest_backup, "r", encoding="utf-8") as f:
        old_schema = json.load(f)
        
    old_tables = len(old_schema)
    new_tables = len(new_schema)
    
    old_cols = sum(len(old_schema[t]) for t in old_schema)
    new_cols = sum(len(new_schema[t]) for t in new_schema)
    
    print(f"Old Backup: {old_tables} tables, {old_cols} total columns.")
    print(f"New Schema: {new_tables} tables, {new_cols} total columns.")
    print(f"Difference: {new_tables - old_tables} tables, {new_cols - old_cols} columns.")
    
    # 2. Step 4 Reload Verification
    print("\n[STEP 4] Verifying SchemaEngine Reload...")
    engine_tables = len(schema_engine.get_tables())
    engine_cols = sum(len(schema_engine.get_columns(t)) for t in schema_engine.get_tables())
    print(f"SchemaEngine current state: {engine_tables} active tables, {engine_cols} total columns.")
    if engine_tables == new_tables:
        print("SUCCESS: SchemaEngine has successfully reloaded the new actual_schema.json.")
    else:
        print("WARNING: SchemaEngine table count does not match actual_schema.json. Caches might need clearing.")
        
    # 3. Step 5 check_schema_drift
    print("\n[STEP 5] Re-running check_schema_drift()...")
    live_schema = sql_schema_guard.fetch_live_schema()
    if live_schema is None:
        print("ERROR: Fetching live schema failed.")
        sys.exit(1)
        
    warnings = sql_schema_guard.check_schema_drift(live_schema)
    print(f"check_schema_drift() returned warnings: {warnings}")
    
    # 4. Step 6 Column-Level Diff
    print("\n[STEP 6] Re-running Column-Level Diff...")
    print(f"{'Table Name':<35} | {'Missing From DB':<35} | {'Missing From JSON':<35}")
    print("-" * 111)
    
    drift_found = False
    for table_name in sorted(new_schema.keys()):
        if table_name == "__EFMigrationsHistory":
            continue
        json_cols = set(new_schema[table_name].keys())
        live_cols = live_schema.get(table_name, set())
        
        missing_from_db = json_cols - live_cols
        missing_from_json = live_cols - json_cols
        
        if missing_from_db or missing_from_json:
            drift_found = True
            
        missing_from_db_str = ", ".join(sorted(missing_from_db)) if missing_from_db else "None"
        missing_from_json_str = ", ".join(sorted(missing_from_json)) if missing_from_json else "None"
        print(f"{table_name:<35} | {missing_from_db_str:<35} | {missing_from_json_str:<35}")
        
    print(f"Total column drift detected: {drift_found}")

    # 5. Step 7 Verify 7 Questions
    print("\n[STEP 7] Verifying the 7 target questions...")
    for q_num, question in VERIFY_QUESTIONS:
        print(f"\nQ{q_num}: \"{question}\"")
        res = run_question(q_num, question)
        
        for leg in res["legs"]:
            print(f"  Leg {leg['leg_idx']}:")
            print(f"    - Tier: {leg['tier']}")
            if leg["error"]:
                print(f"    - ERROR: {leg['error']}")
                if leg["db_error"]:
                    print(f"    - Detail: {leg['db_error'].strip()}")
            else:
                print(f"    - Compiled SQL: {leg['sql']}")
                print(f"    - Parameters: {leg['params']}")

    # 6. Step 8 Full 40-Question Batch Run
    print("\n[STEP 8] Running Full 40-Question Batch Verification...")
    
    batch_results = []
    for q_num, question in ALL_QUESTIONS:
        sys.stdout.write(f"\rRunning Q{q_num}/40...")
        sys.stdout.flush()
        res = run_question(q_num, question)
        batch_results.append(res)
    print("\rBatch execution completed. Generating summary...")

    print("\nBATCH SUMMARY TABLE:")
    print(f"{'Question':<8} | {'Query Type':<20} | {'Legs':<5} | {'Tier 2 Fallback?':<16} | {'Status/Error':<60}")
    print("-" * 120)
    
    for r in batch_results:
        q_label = f"Q{r['q_num']}"
        q_type = r["query_type"]
        legs_count = len(r["legs"])
        tier2 = "Yes" if r["any_tier2"] else "No"
        
        # Determine overall success / error status
        errors = [l["error"] for l in r["legs"] if l["error"]]
        if not errors:
            status = "SUCCESS"
        else:
            status = f"FAILED: {errors[0]}"
            
        print(f"{q_label:<8} | {q_type:<20} | {legs_count:<5} | {tier2:<16} | {status:<60}")

if __name__ == "__main__":
    main()
