import logging
from typing import Any, Dict, List

from ast_models import (
    AggregateNode,
    ASTNode,
    ConditionGroupNode,
    ConditionNode,
    JoinNode,
    OrderByNode,
    WhereNode,
)
from relationship_graph import relationship_graph
from schema_engine import schema_engine

logger = logging.getLogger(__name__)


class QueryPlannerV2:
    """
    Translates a high-level Semantic Intent into a deterministic AST.
    """

    def __init__(self):
        self.engine = schema_engine
        self.graph = relationship_graph

    def plan_query(self, intent: Dict[str, Any]) -> ASTNode:
        """
        Expects intent dict with:
        - base_concept: "Agniveer", "Attendance", etc.
        - filters: {"Class": "Dogra"}
        - limit: int
        - aggregates: list of dicts {"function": "SUM", "concept": "Performance"}
        """
        ast = ASTNode()

        # 1. Source Selection
        base_concept = intent.get("base_concept", "Agniveer")
        base_table = self.engine.get_table_for_concept(base_concept)
        if not base_table:
            raise ValueError(f"Unknown base concept: {base_concept}")
        ast.base_table = base_table

        all_cols = self.engine.get_columns(base_table)
        selected_cols = []
        for c in all_cols:
            cl = c.lower()
            if (
                cl == "id"
                or cl.endswith("id")
                or cl in ("insertedby", "inserteddate", "updatedby", "updateddate")
            ):
                continue
            selected_cols.append(f"{base_table}.{c}")

        if "AgniveerId" in all_cols and base_table != "AgniveerMaster":
            self._add_joins(ast, base_table, "AgniveerMaster")
            if "AgniveerMaster.AgniveerNo" not in selected_cols:
                selected_cols.append("AgniveerMaster.AgniveerNo")
            if "AgniveerMaster.FullName" not in selected_cols:
                selected_cols.append("AgniveerMaster.FullName")
        elif base_table == "AgniveerMaster":
            if "AgniveerMaster.AgniveerNo" not in selected_cols:
                selected_cols.append("AgniveerMaster.AgniveerNo")
            if "AgniveerMaster.FullName" not in selected_cols:
                selected_cols.append("AgniveerMaster.FullName")

        if not selected_cols:
            selected_cols = [f"{base_table}.*"]

        ast.columns = selected_cols

        # 2. Filter Injection & Join Resolution
        ast.where.extend(
            self._parse_filters(intent.get("filters", {}), ast, base_table)
        )

        # Add Implicit Filters for the base concept (only if not already filtered)
        implicit = self.engine.get_implicit_filters(base_concept)
        for col, val in implicit.items():
            column_ref = f"{base_table}.{col}"
            if not self._has_column_filter(ast.where, column_ref):
                ast.where.append(WhereNode(column=column_ref, operator="=", value=val))

        # 3. Aggregation Setup
        aggregates = intent.get("aggregates", [])
        if aggregates:
            # Clear the columns for aggregates, but keep AgniveerNo to ensure
            # cross-filter logic has a unique human-readable identity to intersect on.
            ast.columns = []
            if "AgniveerMaster.AgniveerNo" in selected_cols:
                ast.columns.append("AgniveerMaster.AgniveerNo")
                ast.columns.append("AgniveerMaster.FullName")
                ast.group_by.append("AgniveerMaster.AgniveerNo")
                ast.group_by.append("AgniveerMaster.FullName")
            elif f"{base_table}.AgniveerNo" in selected_cols:
                ast.columns.append(f"{base_table}.AgniveerNo")
                ast.columns.append(f"{base_table}.FullName")
                ast.group_by.append(f"{base_table}.AgniveerNo")
                ast.group_by.append(f"{base_table}.FullName")
            for agg in aggregates:
                func = agg.get("function")
                concept = agg.get("concept")
                col = agg.get("column")
                target_table = self.engine.get_table_for_concept(concept)

                if target_table and target_table != base_table:
                    self._add_joins(ast, base_table, target_table)

                col_name = f"{target_table}.{col}" if target_table else col
                ast.aggregates.append(
                    AggregateNode(
                        column=col_name,
                        function=func,
                        alias=agg.get("alias"),
                    )
                )

        # 4. Grouping Setup
        groups = intent.get("group_by", [])
        if groups:
            if not ast.aggregates:
                ast.columns = []
            for g in groups:
                target_table = self.engine.get_table_for_concept(g.get("concept"))
                if target_table:
                    if target_table != base_table:
                        self._add_joins(ast, base_table, target_table)
                    col_ref = f"{target_table}.{g.get('column')}"
                    ast.group_by.append(col_ref)
                    if col_ref not in ast.columns:
                        ast.columns.append(col_ref)

        ast.limit = intent.get("limit")

        # 5. Sorting Setup
        orders = intent.get("order_by", [])
        if orders:
            for o in orders:
                concept = o.get("concept")
                target_table = (
                    self.engine.get_table_for_concept(concept) if concept else None
                )
                if target_table and target_table != base_table:
                    self._add_joins(ast, base_table, target_table)

                col_name = (
                    f"{target_table}.{o.get('column')}"
                    if target_table
                    else o.get("column")
                )
                ast.order_by.append(
                    OrderByNode(
                        column=col_name,
                        descending=o.get("descending", True),
                    )
                )
        elif ast.limit and not ast.group_by and not ast.aggregates:
            # Add an order by if limiting, typically by PK
            pk = self.engine.get_primary_key(base_table)
            if pk:
                ast.order_by.append(OrderByNode(column=f"{base_table}.{pk}"))

        # 6. Metadata Injection
        ast.metadata = {
            "intent_version": "v2",
            "base_concept": base_concept,
            "has_aggregations": bool(ast.aggregates),
            "has_grouping": bool(ast.group_by),
        }

        return ast

    def _parse_filters(
        self, filters: Dict[str, Any], ast: ASTNode, base_table: str
    ) -> List[ConditionNode]:
        conditions: List[ConditionNode] = []

        for key, value in filters.items():
            if key in ["AND", "OR"] and isinstance(value, list):
                # Nested logic tree. Recurses against the SAME `ast` (not a
                # fresh ASTNode()) so join bookkeeping — _add_joins' dedup
                # via ast.joins/ast.base_table — is shared with the parent
                # scope instead of restarting from an empty, base_table-less
                # node. Previously a target table already joined by a flat
                # sibling condition could get joined again inside this
                # group, which the validator's duplicate-join check then
                # rejected as a false-positive Cartesian join on legitimate
                # nested/cross-filter questions.
                group_conditions: List[ConditionNode] = []
                for sub_filter in value:
                    group_conditions.extend(
                        self._parse_filters(sub_filter, ast, base_table)
                    )
                conditions.append(
                    ConditionGroupNode(operator=key, conditions=group_conditions)
                )
            else:
                parts = key.split(".")
                if len(parts) == 2:
                    concept_name, col_name = parts
                    target_table = self.engine.get_table_for_concept(concept_name)
                    if target_table:
                        if target_table != base_table:
                            self._add_joins(ast, base_table, target_table)

                        # Infer operator from value if it's a dict e.g. {"value": 10, "operator": ">"}
                        if (
                            isinstance(value, dict)
                            and "operator" in value
                            and "value" in value
                        ):
                            op = value["operator"]
                            val = value["value"]
                        else:
                            op = "="
                            val = value

                        conditions.append(
                            WhereNode(
                                column=f"{target_table}.{col_name}",
                                operator=op,
                                value=val,
                            )
                        )
        return conditions

    def _has_column_filter(
        self, conditions: List[Any], column_ref: str, depth: int = 0
    ) -> bool:
        if depth > 20:
            return False
        for cond in conditions:
            if isinstance(cond, WhereNode):
                if cond.column == column_ref:
                    return True
            elif isinstance(cond, ConditionGroupNode):
                if self._has_column_filter(cond.conditions, column_ref, depth + 1):
                    return True
        return False

    def _add_joins(self, ast: ASTNode, start_table: str, end_table: str):
        # Check if already joined
        existing_tables = set(
            [ast.base_table]
            + [j.right_table for j in ast.joins]
            + [j.left_table for j in ast.joins]
        )
        if end_table in existing_tables:
            return

        path = self.graph.find_shortest_path(start_table, end_table)
        if not path:
            logger.warning(f"No join path from {start_table} to {end_table}")
            return

        for step in path:
            # Avoid adding duplicate joins
            if step["right"] not in existing_tables:
                ast.joins.append(
                    JoinNode(
                        left_table=step["left"],
                        right_table=step["right"],
                        left_column=step["left_col"],
                        right_column=step["right_col"],
                    )
                )
                existing_tables.add(step["right"])


query_planner_v2 = QueryPlannerV2()
