import logging
from typing import Dict, Any, List

from schema_engine import schema_engine
from relationship_graph import relationship_graph
from ast_models import ASTNode, JoinNode, WhereNode, AggregateNode, OrderByNode

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
        ast.columns = [f"{base_table}.*"]

        # 2. Filter Injection & Join Resolution
        self._parse_filters(intent.get("filters", {}), ast, base_table)

        # Add Implicit Filters for the base concept
        implicit = self.engine.get_implicit_filters(base_concept)
        for col, val in implicit.items():
            ast.where.append(WhereNode(
                column=f"{base_table}.{col}",
                operator="=",
                value=val
            ))

        # 3. Aggregation Setup
        aggregates = intent.get("aggregates", [])
        if aggregates:
            ast.columns = [] # clear * if aggregating
            for agg in aggregates:
                func = agg.get("function")
                concept = agg.get("concept")
                col = agg.get("column")
                target_table = self.engine.get_table_for_concept(concept)
                
                if target_table != base_table:
                    self._add_joins(ast, base_table, target_table)
                
                ast.aggregates.append(AggregateNode(
                    column=f"{target_table}.{col}",
                    function=func,
                    alias=agg.get("alias")
                ))

        # 4. Grouping Setup
        groups = intent.get("group_by", [])
        if groups:
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
                target_table = self.engine.get_table_for_concept(o.get("concept"))
                if target_table:
                    if target_table != base_table:
                        self._add_joins(ast, base_table, target_table)
                    ast.order_by.append(OrderByNode(
                        column=f"{target_table}.{o.get('column')}",
                        descending=o.get("descending", True)
                    ))
        elif ast.limit:
            # Add an order by if limiting, typically by PK
            pk = self.engine.get_primary_key(base_table)
            if pk:
                ast.order_by.append(OrderByNode(column=f"{base_table}.{pk}"))

        # 6. Metadata Injection
        ast.metadata = {
            "intent_version": "v2",
            "base_concept": base_concept,
            "has_aggregations": bool(ast.aggregates),
            "has_grouping": bool(ast.group_by)
        }

        return ast

    def _parse_filters(self, filters: Dict[str, Any], ast: ASTNode, base_table: str):
        from ast_models import ConditionGroupNode, WhereNode
        
        # Handle dicts as flat AND conditions
        for key, value in filters.items():
            if key in ["AND", "OR"] and isinstance(value, list):
                # Nested logic tree
                group_node = ConditionGroupNode(operator=key, conditions=[])
                for sub_filter in value:
                    sub_ast = ASTNode()
                    self._parse_filters(sub_filter, sub_ast, base_table)
                    group_node.conditions.extend(sub_ast.where)
                    ast.joins.extend(sub_ast.joins) # Bubble up required joins
                ast.where.append(group_node)
            else:
                parts = key.split(".")
                if len(parts) == 2:
                    concept_name, col_name = parts
                    target_table = self.engine.get_table_for_concept(concept_name)
                    if target_table:
                        if target_table != base_table:
                            self._add_joins(ast, base_table, target_table)
                        
                        # Infer operator from value if it's a dict e.g. {"value": 10, "operator": ">"}
                        if isinstance(value, dict) and "operator" in value and "value" in value:
                            op = value["operator"]
                            val = value["value"]
                        else:
                            op = "="
                            val = value

                        ast.where.append(WhereNode(
                            column=f"{target_table}.{col_name}",
                            operator=op,
                            value=val
                        ))

    def _add_joins(self, ast: ASTNode, start_table: str, end_table: str):
        # Check if already joined
        existing_tables = set([ast.base_table] + [j.right_table for j in ast.joins] + [j.left_table for j in ast.joins])
        if end_table in existing_tables:
            return
            
        path = self.graph.find_shortest_path(start_table, end_table)
        if not path:
            logger.warning(f"No join path from {start_table} to {end_table}")
            return
            
        for step in path:
            # Avoid adding duplicate joins
            if step['right'] not in existing_tables:
                ast.joins.append(JoinNode(
                    left_table=step['left'],
                    right_table=step['right'],
                    left_column=step['left_col'],
                    right_column=step['right_col']
                ))
                existing_tables.add(step['right'])

query_planner_v2 = QueryPlannerV2()
