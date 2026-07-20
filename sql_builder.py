from typing import Tuple, Dict, Any, List
from ast_models import ASTNode, ConditionNode, WhereNode, ConditionGroupNode
from schema_engine import schema_engine


class SqlBuilder:
    """
    Compiles an ASTNode into parameterized T-SQL.
    """

    def __init__(self):
        self.parameters: List[Any] = []

    def _next_param(self, value: Any) -> str:
        self.parameters.append(value)
        return "?"

    def build(self, ast: ASTNode) -> Tuple[str, List[Any]]:
        self.parameters = []
        self.aliases = {}

        # Build aliases
        self._get_alias(ast.base_table)
        for j in ast.joins:
            self._get_alias(j.left_table)
            self._get_alias(j.right_table)

        sql = self._build_select(ast)
        return sql, self.parameters

    def _get_alias(self, table: str) -> str:
        if table not in self.aliases:
            self.aliases[table] = f"t{len(self.aliases)}"
        return self.aliases[table]

    def _quote_identifier(self, name: str) -> str:
        if not name:
            return name
        if name.startswith("[") and name.endswith("]"):
            return name
        if "(" in name or ")" in name:
            return name
        if name.isidentifier():
            return name
        return f"[{name.replace(']', ']]')}]"

    def _quote_table(self, table: str) -> str:
        return self._quote_identifier(table)

    def _apply_alias(self, column: str) -> str:
        parts = column.split(".")
        if len(parts) == 2:
            table, col = parts
            alias = self.aliases.get(table, table)
            return f"{alias}.{self._quote_identifier(col)}"
        return column

    def _build_select(self, ast: ASTNode) -> str:
        select_clause = self._build_select_clause(ast)
        from_clause = self._build_from_clause(ast)
        where_clause = self._build_where_clause(ast)
        group_by_clause = self._build_group_by_clause(ast)
        having_clause = self._build_having_clause(ast)
        order_by_clause = self._build_order_by_clause(ast)

        query = f"{select_clause} {from_clause}"
        if where_clause:
            query += f" {where_clause}"
        if group_by_clause:
            query += f" {group_by_clause}"
        if having_clause:
            query += f" {having_clause}"
        if order_by_clause:
            query += f" {order_by_clause}"

        return query

    def _build_select_clause(self, ast: ASTNode) -> str:
        parts = []
        if ast.limit:
            if ast.is_distinct:
                parts.append(f"SELECT DISTINCT TOP ({ast.limit})")
            else:
                parts.append(f"SELECT TOP ({ast.limit})")
        else:
            if ast.is_distinct:
                parts.append("SELECT DISTINCT")
            else:
                parts.append("SELECT")

        columns = list(ast.columns)
        # Apply alias safely to normal columns ONLY
        aliased_cols = []
        for c in columns:
            if c == "*":
                table_cols = schema_engine.get_columns(ast.base_table)
                alias = self.aliases.get(ast.base_table, ast.base_table)
                aliased_cols.extend(
                    [f"{alias}.{self._quote_identifier(col)}" for col in table_cols]
                )
            elif c.endswith(".*"):
                table = c.split(".")[0]
                table_cols = schema_engine.get_columns(table)
                alias = self.aliases.get(table, table)
                aliased_cols.extend(
                    [f"{alias}.{self._quote_identifier(col)}" for col in table_cols]
                )
            else:
                aliased_cols.append(self._apply_alias(c) if "." in c else c)

        for agg in ast.aggregates:
            aliased_col = self._apply_alias(agg.column)
            col_str = f"{agg.function}({aliased_col})"
            if agg.alias:
                col_str += f" AS {agg.alias}"
            aliased_cols.append(col_str)

        if not aliased_cols:
            # Fallback to expanding base table to avoid SELECT *
            table_cols = schema_engine.get_columns(ast.base_table)
            alias = self.aliases.get(ast.base_table, ast.base_table)
            aliased_cols.extend(
                [f"{alias}.{self._quote_identifier(col)}" for col in table_cols]
            )

        parts.append(", ".join(aliased_cols))
        return " ".join(parts)

    def _build_from_clause(self, ast: ASTNode) -> str:
        base_alias = self._get_alias(ast.base_table)
        parts = [f"FROM {self._quote_table(ast.base_table)} {base_alias}"]
        for join in ast.joins:
            left_alias = self._get_alias(join.left_table)
            right_alias = self._get_alias(join.right_table)
            parts.append(
                f"{join.join_type} JOIN {self._quote_table(join.right_table)} {right_alias} "
                f"ON {left_alias}.{self._quote_identifier(join.left_column)} = "
                f"{right_alias}.{self._quote_identifier(join.right_column)}"
            )
        return " ".join(parts)

    def _build_where_clause(self, ast: ASTNode) -> str:
        if not ast.where:
            return ""
        conditions = [self._build_condition(w) for w in ast.where]
        return "WHERE " + " AND ".join(conditions)

    def _build_condition(self, node: ConditionNode) -> str:
        if isinstance(node, WhereNode):
            aliased_col = self._apply_alias(node.column)
            if node.value is None:
                op_upper = node.operator.upper()
                if op_upper in ("=", "IS NULL"):
                    return f"{aliased_col} IS NULL"
                elif op_upper in ("!=", "IS NOT NULL"):
                    return f"{aliased_col} IS NOT NULL"
                # No other operator can be compiled without a value — the
                # validator only lets =, !=, IS NULL, IS NOT NULL pair with
                # value=None (see sql_validator._validate_condition), so
                # reaching here means an operator/value combination the
                # validator should have rejected slipped through.
                raise ValueError(
                    f"Cannot compile operator '{node.operator}' on '{node.column}' "
                    f"with a NULL value — no parameter to bind."
                )

            if isinstance(node.value, dict) and "__raw_sql" in node.value:
                return f"{aliased_col} {node.operator} {node.value['__raw_sql']}"

            if node.operator.upper() == "IN" and isinstance(node.value, (list, tuple)):
                params = [self._next_param(v) for v in node.value]
                return f"{aliased_col} IN ({', '.join(params)})"

            p = self._next_param(node.value)
            return f"{aliased_col} {node.operator} {p}"
        elif isinstance(node, ConditionGroupNode):
            parts = [self._build_condition(c) for c in node.conditions]
            op = f" {node.operator} "
            return f"({op.join(parts)})"
        return ""

    def _build_group_by_clause(self, ast: ASTNode) -> str:
        if not ast.group_by:
            return ""
        aliased = [self._apply_alias(c) for c in ast.group_by]
        return "GROUP BY " + ", ".join(aliased)

    def _build_having_clause(self, ast: ASTNode) -> str:
        if not getattr(ast, "having", None):
            return ""
        conditions = [self._build_condition(h) for h in ast.having]
        return "HAVING " + " AND ".join(conditions)

    def _build_order_by_clause(self, ast: ASTNode) -> str:
        if not ast.order_by:
            return ""
        parts = []
        for ob in ast.order_by:
            direction = "DESC" if ob.descending else "ASC"
            aliased_col = self._apply_alias(ob.column)
            parts.append(f"{aliased_col} {direction}")
        return "ORDER BY " + ", ".join(parts)


sql_builder = SqlBuilder()
