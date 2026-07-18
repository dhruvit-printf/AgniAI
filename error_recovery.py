import logging
from typing import Dict, Any, Tuple, Optional
from ast_models import ASTNode
from schema_engine import schema_engine
from query_planner_v2 import query_planner_v2
from sql_builder import sql_builder
from sql_validator import sql_validator
from sql_executor import run_readonly, metrics_hook

logger = logging.getLogger(__name__)

class ErrorRecoveryEngine:
    """
    Self-healing engine for the AST compilation pipeline.
    Catches SQL execution exceptions and attempts to repair the AST.
    """
    def __init__(self):
        self.engine = schema_engine

    def execute_with_recovery(self, intent: Dict[str, Any]) -> Tuple[Optional[list], Optional[str]]:
        ast = None
        sql = None
        
        # Initial attempt
        try:
            ast = query_planner_v2.plan_query(intent)
            
            # 3-Retry Loop
            retries = 0
            MAX_RETRIES = 3
            is_valid = False
            
            while retries <= MAX_RETRIES:
                is_valid, err = sql_validator.validate_ast(ast)
                
                if is_valid:
                    break
                    
                logger.warning(f"AST Validation failed: {err}. Attempting recovery (Attempt {retries+1}/{MAX_RETRIES})...")
                recovered_ast = self._attempt_repair(ast, err)
                if recovered_ast:
                    ast = recovered_ast
                    retries += 1
                else:
                    return None, f"AST Validation failed and recovery aborted: {err}"
            
            if not is_valid:
                return None, "AST Recovery exhausted all retries."
                
            sql, params = sql_builder.build(ast)
            is_sql_valid, sql_err = sql_validator.validate_sql(sql)
            if not is_sql_valid:
                return None, f"SQL Validation failed: {sql_err}"
                
            rows, run_err = run_readonly(sql, params)
            if not run_err:
                return rows, None
                
            return None, f"Execution failed: {run_err}"
                
        except Exception as e:
            return None, f"Pipeline exception: {e}"

    def _attempt_repair(self, ast: ASTNode, error_msg: str) -> Optional[ASTNode]:
        """
        Intelligent repair: if a specific column is invalid, use fuzzy string matching
        to find the correct column in the schema instead of dropping the condition.
        """
        import re
        import difflib
        from ast_models import WhereNode, ConditionGroupNode
        
        # Detect validator output: "Column 'XYZ' does not exist in table 'ABC'."
        match = re.search(r"Column '([^']+)' does not exist in table '([^']+)'", error_msg)
        if match:
            invalid_col = match.group(1)
            tbl = match.group(2)
            
            valid_cols = self.engine.get_columns(tbl)
            
            # STRICT MATCHING: Minimum 0.85 confidence, and exactly ONE unambiguous match.
            matches = difflib.get_close_matches(invalid_col, valid_cols, n=2, cutoff=0.85)
            
            if len(matches) == 1:
                correct_col = f"{tbl}.{matches[0]}"
                incorrect_col = f"{tbl}.{invalid_col}"
                
                def repair_conditions(conditions) -> bool:
                    repaired_any = False
                    for w in conditions:
                        if isinstance(w, WhereNode) and w.column == incorrect_col:
                            w.column = correct_col
                            repaired_any = True
                        elif isinstance(w, ConditionGroupNode):
                            if repair_conditions(w.conditions):
                                repaired_any = True
                    return repaired_any
                
                repaired = False
                
                # 1. Repair WHERE
                if repair_conditions(ast.where): repaired = True
                
                # 2. Repair HAVING
                if repair_conditions(getattr(ast, "having", [])): repaired = True
                
                # 3. Repair GROUP BY
                if hasattr(ast, "group_by"):
                    for i, g in enumerate(ast.group_by):
                        if g == incorrect_col:
                            ast.group_by[i] = correct_col
                            repaired = True
                            
                # 4. Repair ORDER BY
                if hasattr(ast, "order_by"):
                    for o in ast.order_by:
                        if o.column == incorrect_col:
                            o.column = correct_col
                            repaired = True
                            
                # 5. Repair AGGREGATES
                if hasattr(ast, "aggregates"):
                    for agg in ast.aggregates:
                        if agg.column == incorrect_col:
                            agg.column = correct_col
                            repaired = True
                            
                # 6. Repair COLUMNS
                if hasattr(ast, "columns"):
                    for i, c in enumerate(ast.columns):
                        if c == incorrect_col:
                            ast.columns[i] = correct_col
                            repaired = True
                
                if repaired:
                    logger.info(f"Fuzzy repaired column '{invalid_col}' to '{matches[0]}' in table '{tbl}'")
                    return ast
                
        return None

error_recovery_engine = ErrorRecoveryEngine()
