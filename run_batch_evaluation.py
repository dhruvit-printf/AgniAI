"""
run_batch_evaluation.py
=======================
Developer utility to run batch evaluations of the AgniAI query pipeline.
Executes a pre-defined set of test questions (Q1–Q40) and saves the 
compiled query execution results and performance metrics to batch_evaluation_output.txt.
"""

import os
import sys
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Setup basic logging to stdout so we see imports and internal logs if needed
logging.basicConfig(level=logging.WARNING)

# Ensure workspace is in sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Now import the modules
import sql_executor
import query_planner_v2
import sql_builder
import sql_validator
from intent_engine.query_planner import plan_query

# Global trace structure
execution_trace = {
    "tier": "AST",
    "reason": None,
    "sql": None,
    "params": None,
    "error": None
}

# Keep original references
original_plan_query = query_planner_v2.QueryPlannerV2.plan_query
original_build = sql_builder.SqlBuilder.build
original_validate_ast = sql_validator.SqlValidator.validate_ast
original_validate_sql = sql_validator.SqlValidator.validate_sql
original_generate_sql = sql_executor.generate_sql
original_run_readonly = sql_executor.run_readonly


def patched_plan_query(self, intent: Dict[str, Any]):
    try:
        res = original_plan_query(self, intent)
        return res
    except Exception as e:
        execution_trace["reason"] = f"AST planner error: {type(e).__name__}: {str(e)}"
        raise


def patched_build(self, ast):
    try:
        res = original_build(self, ast)
        return res
    except Exception as e:
        execution_trace["reason"] = f"AST builder error: {type(e).__name__}: {str(e)}"
        raise


def patched_validate_ast(self, ast):
    res, err = original_validate_ast(self, ast)
    if not res:
        execution_trace["reason"] = f"AST Validation rejected: {err}"
    return res, err


def patched_validate_sql(self, sql):
    res, err = original_validate_sql(self, sql)
    if not res:
        execution_trace["reason"] = f"SQL Validation rejected: {err}"
    return res, err


def patched_generate_sql(question: str, intent: Optional[Dict] = None):
    execution_trace["tier"] = "LLM fallback"
    # If a reason wasn't set by capability gap, set it here
    if not execution_trace["reason"]:
        execution_trace["reason"] = "LLM fallback triggered"
    return original_generate_sql(question, intent)


def patched_run_readonly(sql: str, params: Optional[List[Any]] = None):
    execution_trace["sql"] = sql
    execution_trace["params"] = params
    return original_run_readonly(sql, params)

# Apply class-level patches
query_planner_v2.QueryPlannerV2.plan_query = patched_plan_query
sql_builder.SqlBuilder.build = patched_build
sql_validator.SqlValidator.validate_ast = patched_validate_ast
sql_validator.SqlValidator.validate_sql = patched_validate_sql
sql_executor.generate_sql = patched_generate_sql
sql_executor.run_readonly = patched_run_readonly

# We will also wrap the logger in sql_executor to capture capability gap / validation warnings
class TraceLogger:
    def __init__(self, original_logger):
        self.original_logger = original_logger

    def warning(self, msg, *args, **kwargs):
        formatted_msg = msg % args if args else msg
        if "SQL execution error" in formatted_msg:
            execution_trace["db_error"] = formatted_msg
        else:
            execution_trace["reason"] = formatted_msg
        self.original_logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        formatted_msg = msg % args if args else msg
        if "SQL execution error" in formatted_msg:
            execution_trace["db_error"] = formatted_msg
        else:
            execution_trace["reason"] = formatted_msg
        self.original_logger.error(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self.original_logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.original_logger.info(msg, *args, **kwargs)

sql_executor.logger = TraceLogger(sql_executor.logger)

QUESTIONS = [
    # SIMPLE
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

    # CROSS_FILTER
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

    # COMPARE
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

    # MULTI_INDEPENDENT
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

results_output = []
summary_table_rows = []

for q_num, question in QUESTIONS:
    # 1. Run through query planner to get query_type and operations
    plan = None
    try:
        plan = plan_query(question)
        query_type = plan.query_type.value
        ops = plan.operations
    except Exception as e:
        query_type = "unknown"
        ops = []
        err_msg = f"Query planner failed: {str(e)}"
        
    results_output.append(f"### Q{q_num}: \"{question}\"")
    results_output.append(f"- query_type: {query_type}")
    
    leg_details = []
    any_tier2 = False
    any_error = False
    error_list = []
    
    if not ops:
        # Fallback to single leg with question text
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
            
        tier_val = execution_trace["tier"]
        reason_val = execution_trace["reason"] or "N/A"
        sql_val = execution_trace["sql"] or "N/A"
        params_val = str(execution_trace["params"]) if execution_trace["params"] is not None else "None"
        err_val = execution_trace["error"]
        db_err_val = execution_trace.get("db_error")
        
        if tier_val == "LLM fallback":
            any_tier2 = True
        if err_val:
            any_error = True
            error_list.append(err_val)
            
        results_output.append(f"- Leg 1: category=None operation=None section=None")
        results_output.append(f"  - tier: {tier_val} ({reason_val})")
        if sql_val and sql_val != "N/A":
            results_output.append(f"  - SQL: {sql_val}")
            results_output.append(f"  - params: {params_val}")
        if err_val:
            if db_err_val:
                results_output.append(f"  - error: {err_val} (Detail: {db_err_val.strip()})")
            else:
                results_output.append(f"  - error: {err_val}")
    else:
        for idx, op in enumerate(ops):
            execution_trace["tier"] = "AST"
            execution_trace["reason"] = None
            execution_trace["sql"] = None
            execution_trace["params"] = None
            execution_trace["error"] = None
            execution_trace["db_error"] = None
            
            category = op.intent_result.get("category")
            operation = op.intent_result.get("operation")
            section = op.intent_result.get("section") or op.intent_result.get("sub_section")
            
            leg_q = op.raw_fragment or question
            
            try:
                res, err = sql_executor.execute_sql_query(question=leg_q, intent=op.intent_result)
                if err:
                    execution_trace["error"] = err
            except Exception as e:
                execution_trace["error"] = str(e)
                
            tier_val = execution_trace["tier"]
            reason_val = execution_trace["reason"] or "N/A"
            sql_val = execution_trace["sql"] or "N/A"
            params_val = str(execution_trace["params"]) if execution_trace["params"] is not None else "None"
            err_val = execution_trace["error"]
            db_err_val = execution_trace.get("db_error")
            
            if tier_val == "LLM fallback":
                any_tier2 = True
            if err_val:
                any_error = True
                error_list.append(err_val)
                
            results_output.append(f"- Leg {idx + 1}: category={category} operation={operation} section={section}")
            results_output.append(f"  - tier: {tier_val} ({reason_val})")
            if sql_val and sql_val != "N/A":
                results_output.append(f"  - SQL: {sql_val}")
                results_output.append(f"  - params: {params_val}")
            if err_val:
                if db_err_val:
                    results_output.append(f"  - error: {err_val} (Detail: {db_err_val.strip()})")
                else:
                    results_output.append(f"  - error: {err_val}")
                
    results_output.append("") # blank line between questions
    
    # Save for summary table
    summary_table_rows.append({
        "question": f"Q{q_num}",
        "query_type": query_type,
        "legs": len(ops) if ops else 1,
        "tier2": "Yes" if any_tier2 else "No",
        "errors": ", ".join(error_list) if any_error else "None"
    })

# Write standard output and save to file
output_text = "\n".join(results_output)
print(output_text)

# Build summary table markdown
summary_table = []
summary_table.append("| Question | Query Type | Legs | Any Tier-2 Fallbacks | Any Errors |")
summary_table.append("| --- | --- | --- | --- | --- |")
for row in summary_table_rows:
    summary_table.append(f"| {row['question']} | {row['query_type']} | {row['legs']} | {row['tier2']} | {row['errors']} |")

summary_text = "\n".join(summary_table)
print("\n" + summary_text)

# Write to e:\AgniAI\batch_evaluation_output.txt
with open(os.path.join(os.path.dirname(__file__), "batch_evaluation_output.txt"), "w", encoding="utf-8") as f:
    f.write(output_text)
    f.write("\n\n" + summary_text)
