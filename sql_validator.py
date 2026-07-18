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
        seen_joins = set()
        
        # Validate joins (and Cartesian check)
        for join in ast.joins:
            if join.left_table not in tables:
                return False, f"Join table '{join.left_table}' does not exist."
            if join.right_table not in tables:
                return False, f"Join table '{join.right_table}' does not exist."
            if join.join_type.upper() not in ["INNER", "LEFT", "RIGHT", "FULL"]:
                return False, f"Invalid join type '{join.join_type}'."
                
            join_sig = (join.left_table, join.right_table)
            if join_sig in seen_joins:
                return False, f"Duplicate join detected between '{join.left_table}' and '{join.right_table}'."
            seen_joins.add(join_sig)
            
            # Cartesian Check: Left table must already be in the valid graph
            if join.left_table not in valid_tables:
                return False, f"Cartesian join detected: '{join.left_table}' is disconnected from the base graph."
            
            left_cols = self.engine.get_columns(join.left_table)
            right_cols = self.engine.get_columns(join.right_table)
            
            if join.left_column not in left_cols:
                return False, f"Join column '{join.left_column}' not in '{join.left_table}'."
            if join.right_column not in right_cols:
                return False, f"Join column '{join.right_column}' not in '{join.right_table}'."
                
            valid_tables.add(join.right_table)

        seen_aliases = set()
        
        # Validate aggregates and collect aliases
        for agg in ast.aggregates:
            parts = agg.column.split(".")
            if len(parts) == 2:
                tbl, col = parts
                if tbl not in valid_tables:
                    return False, f"Aggregate references unjoined table '{tbl}'."
                cols = self.engine.get_columns(tbl)
                if col not in cols:
                    return False, f"Column '{col}' does not exist in table '{tbl}'."
            
            if agg.alias:
                if agg.alias in seen_aliases:
                    return False, f"Duplicate aggregate alias '{agg.alias}'."
                seen_aliases.add(agg.alias)

        # Validate explicitly requested columns
        for c in ast.columns:
            if c != "*" and not c.endswith(".*"):
                parts = c.split(".")
                if len(parts) == 2:
                    tbl, col = parts
                    if tbl not in valid_tables:
                        return False, f"Select column references unjoined table '{tbl}'."
                    if col not in self.engine.get_columns(tbl):
                        return False, f"Column '{col}' does not exist in table '{tbl}'."
                else:
                    return False, f"Select column '{c}' is missing a table qualifier or is an invalid alias."
                        
        # Validate Group By
        for c in getattr(ast, "group_by", []):
            parts = c.split(".")
            if len(parts) == 2:
                tbl, col = parts
                if tbl not in valid_tables:
                    return False, f"Group By references unjoined table '{tbl}'."
                if col not in self.engine.get_columns(tbl):
                    return False, f"Column '{col}' does not exist in table '{tbl}'."
            elif c not in seen_aliases:
                return False, f"Group By references unknown column or alias '{c}'."

        # Validate Order By
        for o in getattr(ast, "order_by", []):
            parts = o.column.split(".")
            if len(parts) == 2:
                tbl, col = parts
                if tbl not in valid_tables:
                    return False, f"Order By references unjoined table '{tbl}'."
                if col not in self.engine.get_columns(tbl):
                    return False, f"Column '{col}' does not exist in table '{tbl}'."
            elif o.column not in seen_aliases:
                return False, f"Order By references unknown column or alias '{o.column}'."

        # Validate conditions recursively
        for condition in ast.where:
            is_valid, err = self._validate_condition(condition, valid_tables, seen_aliases)
            if not is_valid:
                return False, err
                
        # Validate having
        for condition in getattr(ast, "having", []):
            is_valid, err = self._validate_condition(condition, valid_tables, seen_aliases)
            if not is_valid:
                return False, err

        return True, None

    def _validate_condition(self, node: ConditionNode, valid_tables: Set[str], seen_aliases: Set[str] = None) -> Tuple[bool, Optional[str]]:
        seen_aliases = seen_aliases or set()
        
        if isinstance(node, WhereNode):
            # SQL Injection check on operator
            allowed_ops = {"=", "!=", ">", "<", ">=", "<=", "LIKE", "IN", "IS NULL", "IS NOT NULL"}
            if node.operator.upper() not in allowed_ops:
                return False, f"Unsafe or unknown operator: '{node.operator}'."
                
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
                if node.value is not None:
                    if col_type == "integer" and not isinstance(node.value, int) and str(node.value).isdigit() == False:
                        return False, f"Type mismatch: '{node.column}' expects integer, got {val_type.__name__}."
                    if col_type == "boolean" and not isinstance(node.value, bool) and str(node.value).lower() not in ["0", "1", "true", "false"]:
                        return False, f"Type mismatch: '{node.column}' expects boolean."
                elif node.operator.upper() not in ("IS NULL", "IS NOT NULL", "=", "!="):
                    return False, f"Missing parameter value for operator '{node.operator}'."
            elif node.column not in seen_aliases:
                return False, f"Condition references unknown alias or column '{node.column}'."
                
            return True, None
            
        elif isinstance(node, ConditionGroupNode):
            for c in node.conditions:
                is_valid, err = self._validate_condition(c, valid_tables, seen_aliases)
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
