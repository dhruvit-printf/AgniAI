import pytest
from ast_models import ASTNode, WhereNode, ConditionGroupNode, JoinNode, AggregateNode, OrderByNode

def test_ast_serialization_with_nested_filters():
    ast = ASTNode(base_table="AgniveerMaster", is_distinct=True, limit=5)
    ast.columns = ["AgniveerMaster.Id", "AgniveerMaster.FullName"]
    ast.joins.append(JoinNode(left_table="AgniveerMaster", right_table="PlatoonMaster", left_column="PlatoonId", right_column="Id"))
    ast.aggregates.append(AggregateNode(column="AgniveerMaster.Id", function="COUNT", alias="Total"))
    ast.group_by.append("PlatoonMaster.Id")
    ast.order_by.append(OrderByNode(column="Total", descending=True))
    
    # Nested filters
    nested = ConditionGroupNode(
        operator="OR",
        conditions=[
            WhereNode(column="AgniveerMaster.Class", operator="=", value="Dogra"),
            WhereNode(column="AgniveerMaster.Class", operator="=", value="Rajput")
        ]
    )
    ast.where.append(nested)
    ast.having.append(WhereNode(column="Total", operator=">", value=10))
    
    ast.metadata["test"] = "123"
    
    data = ast.to_dict()
    
    assert data["type"] == "Select"
    assert data["base_table"] == "AgniveerMaster"
    assert data["limit"] == 5
    assert data["is_distinct"] is True
    assert len(data["columns"]) == 2
    assert len(data["joins"]) == 1
    assert data["joins"][0]["left_table"] == "AgniveerMaster"
    
    assert len(data["aggregates"]) == 1
    assert data["aggregates"][0]["alias"] == "Total"
    
    assert len(data["group_by"]) == 1
    assert len(data["order_by"]) == 1
    
    assert len(data["where"]) == 1
    assert data["where"][0]["group_operator"] == "OR"
    assert len(data["where"][0]["conditions"]) == 2
    
    assert len(data["having"]) == 1
    assert data["metadata"]["test"] == "123"
