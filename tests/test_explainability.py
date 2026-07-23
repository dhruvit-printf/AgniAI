import pytest

from ast_models import AggregateNode, ASTNode, OrderByNode, WhereNode
from explainability_engine import explainability_engine


def test_explainability_advanced_features():
    ast = ASTNode(base_table="AgniveerMaster", is_distinct=True, limit=5)
    ast.group_by = ["AgniveerMaster.Class"]
    ast.aggregates = [
        AggregateNode(column="AgniveerMaster.Id", function="COUNT", alias="Total")
    ]
    ast.having = [WhereNode(column="Total", operator=">", value=10)]
    ast.order_by = [OrderByNode(column="Total", descending=True)]

    explanation = explainability_engine.explain(ast)

    assert explanation["intent"] == "Distinct Database Query"
    assert "Grouped by AgniveerMaster.Class" in explanation["groupings"]
    assert "Filtered where Total > '10'" in explanation["having"]
    assert (
        "Calculated COUNT of AgniveerMaster.Id (as Total)"
        in explanation["aggregations"]
    )
    assert "Sorted by Total (descending)" in explanation["sorting"]
    assert explanation["limit"] == 5
