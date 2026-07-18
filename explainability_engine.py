import logging
from typing import Dict, Any, List
from ast_models import ASTNode, WhereNode, ConditionGroupNode

logger = logging.getLogger(__name__)

class ExplainabilityEngine:
    """
    Translates an ASTNode into plain English for the user interface,
    so they understand how the system reached its answer.
    """
    
    def explain(self, ast: ASTNode) -> Dict[str, Any]:
        explanation = {
            "intent": "Database Query",
            "base_table": ast.base_table,
            "joins": [],
            "filters": [],
            "aggregations": [],
            "sorting": [],
            "limit": ast.limit
        }
        
        # Explain joins
        for join in ast.joins:
            explanation["joins"].append(f"Linked {join.left_table} to {join.right_table} via {join.left_column}")
            
        # Explain filters
        for cond in ast.where:
            explanation["filters"].append(self._explain_condition(cond))
            
        # Explain aggregations
        for agg in ast.aggregates:
            desc = f"Calculated {agg.function} of {agg.column}"
            if agg.alias:
                desc += f" (as {agg.alias})"
            explanation["aggregations"].append(desc)
            
        # Explain sorting
        for ob in ast.order_by:
            dir_str = "descending" if ob.descending else "ascending"
            explanation["sorting"].append(f"Sorted by {ob.column} ({dir_str})")
            
        return explanation

    def _explain_condition(self, node: Any) -> str:
        if isinstance(node, WhereNode):
            return f"Filtered where {node.column} {node.operator} '{node.value}'"
        elif isinstance(node, ConditionGroupNode):
            parts = [self._explain_condition(c) for c in node.conditions]
            op = f" {node.operator} "
            return f"({op.join(parts)})"
        return "Unknown filter"

explainability_engine = ExplainabilityEngine()
