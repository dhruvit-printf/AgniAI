from typing import Tuple, Dict, Any, List
from ast_models import ASTNode, ConditionNode, WhereNode, ConditionGroupNode

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

    def _apply_alias(self, column: str) -> str:
        parts = column.split(".")
        if len(parts) == 2:
            table, col = parts
            alias = self.aliases.get(table, table)
            return f"{alias}.{col}"
        return column

    def _build_select(self, ast: ASTNode) -> str:
        select_clause = self._build_select_clause(ast)
        from_clause = self._build_from_clause(ast)
        where_clause = self._build_where_clause(ast)
        group_by_clause = self._build_group_by_clause(ast)
        order_by_clause = self._build_order_by_clause(ast)
        
        query = f"{select_clause} {from_clause}"
        if where_clause:
            query += f" {where_clause}"
        if group_by_clause:
            query += f" {group_by_clause}"
        if order_by_clause:
            query += f" {order_by_clause}"
            
        return query

    def _build_select_clause(self, ast: ASTNode) -> str:
        parts = []
        if ast.limit:
            parts.append(f"SELECT TOP ({ast.limit})")
        else:
            parts.append("SELECT")
            
        columns = list(ast.columns)
        for agg in ast.aggregates:
            aliased_col = self._apply_alias(agg.column)
            col_str = f"{agg.function}({aliased_col})"
            if agg.alias:
                col_str += f" AS {agg.alias}"
            columns.append(col_str)
            
        aliased_cols = [self._apply_alias(c) if "." in c else c for c in columns]
        if not aliased_cols:
            aliased_cols = ["*"]
            
        parts.append(", ".join(aliased_cols))
        return " ".join(parts)

    def _build_from_clause(self, ast: ASTNode) -> str:
        base_alias = self._get_alias(ast.base_table)
        parts = [f"FROM {ast.base_table} {base_alias}"]
        for join in ast.joins:
            left_alias = self._get_alias(join.left_table)
            right_alias = self._get_alias(join.right_table)
            parts.append(
                f"{join.join_type} JOIN {join.right_table} {right_alias} "
                f"ON {left_alias}.{join.left_column} = {right_alias}.{join.right_column}"
            )
        return " ".join(parts)

    def _build_where_clause(self, ast: ASTNode) -> str:
        if not ast.where:
            return ""
        conditions = [self._build_condition(w) for w in ast.where]
        return "WHERE " + " AND ".join(conditions)

    def _build_condition(self, node: ConditionNode) -> str:
        if isinstance(node, WhereNode):
            p = self._next_param(node.value)
            aliased_col = self._apply_alias(node.column)
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
