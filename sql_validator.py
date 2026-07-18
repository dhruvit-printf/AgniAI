import logging
from typing import List, Optional, Tuple, Set
from ast_models import ASTNode, ConditionNode, WhereNode, ConditionGroupNode
from schema_engine import schema_engine

logger = logging.getLogger(__name__)

class SqlValidator:
    """
    Validates ASTs and compiled SQL before execution to ensure schema compliance and safety.
    """
    def __init__(self):
        self.engine = schema_engine

    def validate_ast(self, ast: ASTNode) -> Tuple[bool, Optional[str]]:
        """
        Validates an AST against the schema.
        Returns (is_valid, error_message).
        """
        # Validate base table
        tables = self.engine.get_tables()
        if ast.base_table not in tables:
            return False, f"Base table '{ast.base_table}' does not exist in schema."

        valid_tables = {ast.base_table}
        
        # Validate joins
        for join in ast.joins:
            if join.left_table not in tables:
                return False, f"Join table '{join.left_table}' does not exist."
            if join.right_table not in tables:
                return False, f"Join table '{join.right_table}' does not exist."
            
            left_cols = self.engine.get_columns(join.left_table)
            right_cols = self.engine.get_columns(join.right_table)
            
            if join.left_column not in left_cols:
                return False, f"Join column '{join.left_column}' not in '{join.left_table}'."
            if join.right_column not in right_cols:
                return False, f"Join column '{join.right_column}' not in '{join.right_table}'."
                
            valid_tables.add(join.right_table)
            valid_tables.add(join.left_table)

        # Validate conditions recursively
        for condition in ast.where:
            is_valid, err = self._validate_condition(condition, valid_tables)
            if not is_valid:
                return False, err

        # Validate aggregates
        for agg in ast.aggregates:
            parts = agg.column.split(".")
            if len(parts) == 2:
                tbl, col = parts
                if tbl not in valid_tables:
                    return False, f"Aggregate references unjoined table '{tbl}'."
                cols = self.engine.get_columns(tbl)
                if col not in cols:
                    return False, f"Column '{col}' does not exist in table '{tbl}'."

        return True, None

    def _validate_condition(self, node: ConditionNode, valid_tables: Set[str]) -> Tuple[bool, Optional[str]]:
        if isinstance(node, WhereNode):
            parts = node.column.split(".")
            if len(parts) == 2:
                tbl, col = parts
                if tbl not in valid_tables:
                    return False, f"Condition references unjoined table '{tbl}'."
                cols = self.engine.get_columns(tbl)
                if col not in cols:
                    return False, f"Column '{col}' does not exist in table '{tbl}'."
                    
                # Type checking
                col_type = self.engine.get_column_type(tbl, col)
                val_type = type(node.value)
                if col_type == "integer" and not isinstance(node.value, int) and str(node.value).isdigit() == False:
                    return False, f"Type mismatch: '{node.column}' expects integer, got {val_type.__name__}."
                if col_type == "boolean" and not isinstance(node.value, bool) and str(node.value) not in ["0", "1", "true", "false"]:
                    return False, f"Type mismatch: '{node.column}' expects boolean."
                    
            return True, None
            
        elif isinstance(node, ConditionGroupNode):
            for c in node.conditions:
                is_valid, err = self._validate_condition(c, valid_tables)
                if not is_valid:
                    return False, err
            return True, None
            
        return False, "Unknown ConditionNode type."

    def validate_sql(self, sql: str) -> Tuple[bool, Optional[str]]:
        """
        Final safety net to ensure generated SQL has no forbidden commands.
        (This replaces the regexes from the old sql_executor.py)
        """
        import re
        s = sql.lower()
        forbidden = r"\b(insert|update|delete|merge|drop|alter|truncate|create|grant|revoke|exec|execute|sp_|xp_)\b"
        if re.search(forbidden, s):
            return False, "Generated SQL contains forbidden keywords."
            
        if ";" in s:
            return False, "Multiple statements are not allowed."
            
        return True, None

sql_validator = SqlValidator()
