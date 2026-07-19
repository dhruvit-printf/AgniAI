import logging
from typing import Optional, Tuple, Set
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

            # frozenset (not a plain tuple) so a join declared as A->B and one
            # later declared as B->A are recognized as the SAME join. A plain
            # (left, right) tuple treats those as two different signatures —
            # (A, B) != (B, A) — so a reversed duplicate join slipped past
            # this check entirely and would only ever fail later, at the
            # pyodbc driver level, as an opaque execution error.
            join_sig = frozenset((join.left_table, join.right_table))
            if join_sig in seen_joins:
                return (
                    False,
                    f"Duplicate join detected between '{join.left_table}' and '{join.right_table}'.",
                )
            seen_joins.add(join_sig)

            # Cartesian Check: Left table must already be in the valid graph
            if join.left_table not in valid_tables:
                return (
                    False,
                    f"Cartesian join detected: '{join.left_table}' is disconnected from the base graph.",
                )

            left_cols = self.engine.get_columns(join.left_table)
            right_cols = self.engine.get_columns(join.right_table)

            if join.left_column not in left_cols:
                return (
                    False,
                    f"Join column '{join.left_column}' not in '{join.left_table}'.",
                )
            if join.right_column not in right_cols:
                return (
                    False,
                    f"Join column '{join.right_column}' not in '{join.right_table}'.",
                )

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
                        return (
                            False,
                            f"Select column references unjoined table '{tbl}'.",
                        )
                    if col not in self.engine.get_columns(tbl):
                        return False, f"Column '{col}' does not exist in table '{tbl}'."
                else:
                    return (
                        False,
                        f"Select column '{c}' is missing a table qualifier or is an invalid alias.",
                    )

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
                return (
                    False,
                    f"Order By references unknown column or alias '{o.column}'.",
                )

        # Validate conditions recursively
        for condition in ast.where:
            is_valid, err = self._validate_condition(
                condition, valid_tables, seen_aliases
            )
            if not is_valid:
                return False, err

        # Validate having
        for condition in getattr(ast, "having", []):
            is_valid, err = self._validate_condition(
                condition, valid_tables, seen_aliases
            )
            if not is_valid:
                return False, err

        return True, None

    def _validate_condition(
        self,
        node: ConditionNode,
        valid_tables: Set[str],
        seen_aliases: Set[str] = None,
        depth: int = 0,
    ) -> Tuple[bool, Optional[str]]:
        if depth > 20:
            return False, "Query condition depth limit exceeded."
        seen_aliases = seen_aliases or set()

        if isinstance(node, WhereNode):
            # SQL Injection check on operator
            allowed_ops = {
                "=",
                "!=",
                ">",
                "<",
                ">=",
                "<=",
                "LIKE",
                "IN",
                "IS NULL",
                "IS NOT NULL",
            }
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
                    if (
                        col_type == "integer"
                        and not isinstance(node.value, int)
                        and not str(node.value).isdigit()
                    ):
                        return (
                            False,
                            f"Type mismatch: '{node.column}' expects integer, got {val_type.__name__}.",
                        )
                    if (
                        col_type == "boolean"
                        and not isinstance(node.value, bool)
                        and str(node.value).lower() not in ["0", "1", "true", "false"]
                    ):
                        return False, f"Type mismatch: '{node.column}' expects boolean."
                elif node.operator.upper() not in ("IS NULL", "IS NOT NULL", "=", "!="):
                    return (
                        False,
                        f"Missing parameter value for operator '{node.operator}'.",
                    )
            elif node.column not in seen_aliases:
                return (
                    False,
                    f"Condition references unknown alias or column '{node.column}'.",
                )

            return True, None

        elif isinstance(node, ConditionGroupNode):
            for c in node.conditions:
                is_valid, err = self._validate_condition(
                    c, valid_tables, seen_aliases, depth + 1
                )
                if not is_valid:
                    return False, err
            return True, None

        return False, "Unknown ConditionNode type."

    def validate_sql(self, sql: str) -> Tuple[bool, Optional[str]]:
        """
        Final safety net to ensure generated SQL has no forbidden commands.

        This is the validator `execute_sql_query` actually calls (via
        `from sql_validator import sql_validator`) — `sql_executor.py` also
        defines a module-level `validate_sql()` with a fuller rule set
        (R7 best-attempt scoping, DENIED_TABLES/DENIED_COLUMNS, comment
        stripping), but nothing in the live request path ever calls that
        one; only tests do. That silently meant R7, the hard column/table
        denylist, and the comment check were not actually enforced against
        any query this system executes. Ported here so this being the
        validator that's really on the request path.
        """
        import re

        if not sql or not sql.strip():
            return False, "Empty SQL."

        s = sql.strip().rstrip(";").strip().lower()

        if not (s.startswith("select") or s.startswith("with")):
            return False, "Only single SELECT / WITH...SELECT statements are allowed."

        forbidden = r"\b(insert|update|delete|merge|drop|alter|truncate|create|grant|revoke|exec|execute|sp_|xp_|openrowset|openquery|bulk|shutdown|reconfigure|waitfor)\b"
        if re.search(forbidden, s):
            return False, "Generated SQL contains forbidden keywords."

        if re.search(r";\s*\S", sql.strip().rstrip(";")):
            return False, "Multiple statements are not allowed."

        if re.search(r"(--|/\*|\*/)", sql):
            return False, "Comments are not allowed."

        # R7 (business_rules.LLM_HARD_RULES): any query touching
        # AgniveerScoreAttempt.MarksObtained must scope to a single attempt
        # per Agniveer/sub-item (IsBestAttempt = 1 or a specific AttemptNo) —
        # otherwise summing MarksObtained double/triple-counts every retake.
        if (
            re.search(r"\bagniveerscoreattempt\b", s)
            and re.search(r"\bmarksobtained\b", s)
            and not re.search(r"\bisbestattempt\b", s)
            and not re.search(r"\battemptno\b", s)
        ):
            return False, (
                "Query uses AgniveerScoreAttempt.MarksObtained without scoping to a "
                "single attempt (IsBestAttempt = 1 or AttemptNo) — would double-count "
                "marks across retakes (R7)."
            )

        # NOTE on R1 (exclude disqualified agniveers): deliberately NOT
        # enforced here as a blanket "AgniveerMaster present -> must contain
        # IsDisqualified" text check. Several golden queries legitimately
        # omit it by design — a single-Agniveer lookup by AgniveerNo
        # (personaldetail/info, Medical/Individual) or a Verification-status
        # query — matching the .NET reference's own behavior for those
        # operations (see AiCommandService.Cmd_AgniveerPersonalDetails /
        # Cmd_IndividualMedicalReport / Cmd_VerificationByStatus, none of
        # which filter by IsDisqualified). A regex over raw SQL text can't
        # distinguish "this query type doesn't need the filter" from "this
        # forgot the filter", so it would reject correct queries as often as
        # it caught bugs. R1 is instead enforced structurally: via
        # business_ontology.json's Agniveer.implicit_filters (query_planner_v2
        # AST path) and directly in each GOLDEN_QUERIES template that needs it.

        from sql_executor import DENIED_COLUMNS, DENIED_TABLES

        for denied in DENIED_TABLES:
            if re.search(rf"\b{denied}\b", s):
                return False, f"Access to table '{denied}' is denied."
        for denied in DENIED_COLUMNS:
            tbl, col = denied.split(".")
            if col in s and tbl in s:
                return False, f"Access to column '{denied}' is denied."

        return True, None


sql_validator = SqlValidator()
