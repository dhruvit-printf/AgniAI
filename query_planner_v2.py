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
        filters = intent.get("filters", {})
        for key, value in filters.items():
            # For simplicity, assuming filter keys are concept_name.ColumnName
            # e.g. Agniveer.Class
            parts = key.split(".")
            if len(parts) == 2:
                concept_name, col_name = parts
                target_table = self.engine.get_table_for_concept(concept_name)
                if target_table:
                    # Add Joins if target is not base_table
                    if target_table != base_table:
                        self._add_joins(ast, base_table, target_table)
                    
                    ast.where.append(WhereNode(
                        column=f"{target_table}.{col_name}",
                        operator="=",
                        value=value
                    ))

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

        ast.limit = intent.get("limit")
        
        if ast.limit:
            # Add an order by if limiting, typically by PK
            pk = self.engine.get_primary_key(base_table)
            if pk:
                ast.order_by.append(OrderByNode(column=f"{base_table}.{pk}"))

        return ast

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
