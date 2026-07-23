import pytest
from ast_models import ASTNode, WhereNode, ConditionGroupNode, AggregateNode
from sql_builder import sql_builder


def test_sql_builder_distinct_and_top():
    ast = ASTNode(base_table="AgniveerMaster", is_distinct=True, limit=10)
    ast.columns = ["AgniveerMaster.Id"]
    sql, params = sql_builder.build(ast)
    assert sql == "SELECT DISTINCT TOP (10) t0.Id FROM AgniveerMaster t0"
    assert len(params) == 0


def test_sql_builder_is_null_interceptor():
    ast = ASTNode(base_table="AgniveerMaster")
    ast.columns = ["AgniveerMaster.Id"]
    ast.where.append(
        WhereNode(column="AgniveerMaster.Remarks", operator="=", value=None)
    )

    sql, params = sql_builder.build(ast)
    assert sql == "SELECT t0.Id FROM AgniveerMaster t0 WHERE t0.Remarks IS NULL"
    assert len(params) == 0

    ast.where = [WhereNode(column="AgniveerMaster.Remarks", operator="!=", value=None)]
    sql, params = sql_builder.build(ast)
    assert sql == "SELECT t0.Id FROM AgniveerMaster t0 WHERE t0.Remarks IS NOT NULL"
    assert len(params) == 0


def test_sql_builder_having_clause():
    ast = ASTNode(base_table="AgniveerMaster")
    ast.aggregates = [
        AggregateNode(column="AgniveerMaster.Id", function="COUNT", alias="TotalCount")
    ]
    ast.having = [
        WhereNode(column="TotalCount", operator=">", value=5)
    ]  # Having filters on alias usually or aggregate

    sql, params = sql_builder.build(ast)
    expected = (
        "SELECT COUNT(t0.Id) AS TotalCount FROM AgniveerMaster t0 HAVING TotalCount > ?"
    )
    assert sql == expected
    assert params == [5]


def test_sql_builder_safe_aggregate_aliasing():
    ast = ASTNode(base_table="CompanyMaster")
    ast.columns = ["CompanyMaster.Name"]
    ast.aggregates = [
        AggregateNode(column="CompanyMaster.Id", function="COUNT", alias="Total")
    ]

    sql, params = sql_builder.build(ast)
    expected = "SELECT t0.Name, COUNT(t0.Id) AS Total FROM CompanyMaster t0"
    assert sql == expected


def test_sql_builder_quotes_special_column_names():
    ast = ASTNode(base_table="AgniveerLeaveMaster")
    ast.columns = [
        "AgniveerLeaveMaster.OnATTN'C'",
        "AgniveerLeaveMaster.OnEX PPG",
    ]

    sql, params = sql_builder.build(ast)
    expected = "SELECT t0.[OnATTN'C'], t0.[OnEX PPG] FROM AgniveerLeaveMaster t0"
    assert sql == expected
    assert params == []


def test_sql_builder_group_by_does_not_select_star():
    from query_planner_v2 import query_planner_v2

    intent = {
        "base_concept": "Agniveer",
        "filters": {},
        "group_by": [{"concept": "Agniveer", "column": "Class"}],
        "limit": 10,
    }
    ast = query_planner_v2.plan_query(intent)
    sql, params = sql_builder.build(ast)

    assert "AgniveerMaster.*" not in sql
    assert "GROUP BY t0.Class" in sql
    assert "ORDER BY t0.Id" not in sql
