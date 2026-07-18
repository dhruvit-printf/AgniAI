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
            is_valid, err = sql_validator.validate_ast(ast)
            if not is_valid:
                return None, f"AST Validation failed: {err}"
                
            sql, params = sql_builder.build(ast)
            is_sql_valid, sql_err = sql_validator.validate_sql(sql)
            if not is_sql_valid:
                return None, f"SQL Validation failed: {sql_err}"
                
            rows, run_err = run_readonly(sql, params)
            if not run_err:
                return rows, None
                
            logger.warning(f"Initial SQL execution failed: {run_err}. Attempting recovery...")
            
            # Recovery Attempt 1: Drop failing conditions
            recovered_ast = self._attempt_repair(ast, run_err)
            if recovered_ast:
                logger.info("AST successfully repaired. Re-compiling...")
                sql, params = sql_builder.build(recovered_ast)
                rows, run_err2 = run_readonly(sql, params)
                if not run_err2:
                    logger.info("Recovery successful.")
                    return rows, None
                else:
                    return None, f"Recovery failed: {run_err2}"
            else:
                return None, f"Execution failed, and no recovery was possible: {run_err}"
                
        except Exception as e:
            return None, f"Pipeline exception: {e}"

    def _attempt_repair(self, ast: ASTNode, error_msg: str) -> Optional[ASTNode]:
        """
        Intelligent repair: if a specific column is invalid, use fuzzy string matching
        to find the correct column in the schema instead of dropping the condition.
        """
        import re
        import difflib
        from ast_models import WhereNode
        
        # Detect "Invalid column name 'XYZ'"
        match = re.search(r"Invalid column name '([^']+)'", error_msg)
        if match:
            invalid_col = match.group(1)
            repaired = False
            
            # Since ast.where now uses ConditionNode, we need a recursive replacer.
            # For simplicity in this implementation, we will iterate over where conditions
            # and repair WhereNodes directly if they are at the top level.
            for i, w in enumerate(ast.where):
                if isinstance(w, WhereNode) and invalid_col in w.column:
                    parts = w.column.split(".")
                    if len(parts) == 2:
                        tbl, col = parts
                        valid_cols = self.engine.get_columns(tbl)
                        matches = difflib.get_close_matches(invalid_col, valid_cols, n=1, cutoff=0.6)
                        if matches:
                            w.column = f"{tbl}.{matches[0]}"
                            repaired = True
                            logger.info(f"Fuzzy repaired column '{invalid_col}' to '{matches[0]}' in table '{tbl}'")
            
            if repaired:
                return ast
                
        return None

error_recovery_engine = ErrorRecoveryEngine()
