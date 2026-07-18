import pytest
from ast_models import ASTNode, WhereNode, ConditionGroupNode, JoinNode, AggregateNode, OrderByNode
from sql_validator import sql_validator

def test_operator_sql_injection_is_blocked():
    ast = ASTNode(base_table="AgniveerMaster")
    # Operator injection
    ast.where.append(WhereNode(column="AgniveerMaster.Id", operator="= 1; DROP TABLE Users --", value=1))
    
    is_valid, err = sql_validator.validate_ast(ast)
    assert is_valid is False
    assert "Unsafe or unknown operator" in err

def test_cartesian_join_is_blocked():
    ast = ASTNode(base_table="AgniveerMaster")
    # A valid join
    ast.joins.append(JoinNode(
        left_table="AgniveerMaster", right_table="PlatoonMaster",
        left_column="PlatoonId", right_column="Id"
    ))
    # A disconnected Cartesian join (CompanyMaster and DistributionMaster aren't connected to Agniveer or Platoon yet)
    ast.joins.append(JoinNode(
        left_table="CompanyMaster", right_table="DistributionMaster",
        left_column="Id", right_column="CompanyId"
    ))
    
    is_valid, err = sql_validator.validate_ast(ast)
    assert is_valid is False
    assert "Cartesian join detected: 'CompanyMaster' is disconnected" in err

def test_duplicate_join_is_blocked():
    ast = ASTNode(base_table="AgniveerMaster")
    join1 = JoinNode(left_table="AgniveerMaster", right_table="PlatoonMaster", left_column="PlatoonId", right_column="Id")
    ast.joins = [join1, join1]
    
    is_valid, err = sql_validator.validate_ast(ast)
    assert is_valid is False
    assert "Duplicate join detected" in err

def test_duplicate_aggregate_alias_is_blocked():
    ast = ASTNode(base_table="AgniveerMaster")
    ast.aggregates = [
        AggregateNode(column="AgniveerMaster.Id", function="COUNT", alias="Total"),
        AggregateNode(column="AgniveerMaster.PlatoonId", function="COUNT", alias="Total")
    ]
    
    is_valid, err = sql_validator.validate_ast(ast)
    assert is_valid is False
    assert "Duplicate aggregate alias 'Total'" in err

def test_having_references_alias_is_allowed():
    ast = ASTNode(base_table="AgniveerMaster")
    ast.aggregates = [AggregateNode(column="AgniveerMaster.Id", function="COUNT", alias="TotalCount")]
    # Having node references the alias
    ast.having = [WhereNode(column="TotalCount", operator=">", value=10)]
    
    is_valid, err = sql_validator.validate_ast(ast)
    assert is_valid is True
    assert err is None

def test_invalid_group_by_is_blocked():
    ast = ASTNode(base_table="AgniveerMaster")
    ast.group_by = ["AgniveerMaster.FakeColumn"]
    
    is_valid, err = sql_validator.validate_ast(ast)
    assert is_valid is False
    assert "FakeColumn" in err
