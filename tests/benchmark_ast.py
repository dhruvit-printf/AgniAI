import pytest
from query_planner_v2 import query_planner_v2
from sql_builder import sql_builder
from sql_validator import sql_validator
from ast_models import ASTNode

# Test cases: (SemanticIntent, ExpectedBaseTable)
BENCHMARKS = [
    (
        {"base_concept": "Agniveer", "filters": {"Class": "Dogra"}},
        "AgniveerMaster"
    ),
    (
        {"base_concept": "Attendance", "filters": {"Company.Name": "Alpha"}},
        "AgniveerAttendanceMaster"
    ),
    (
        {"base_concept": "Performance", "filters": {"IsBestAttempt": 1}},
        "AgniveerScoreAttempt"
    )
]

@pytest.mark.parametrize("intent, expected_base", BENCHMARKS)
def test_ast_generation(intent, expected_base):
    # 1. Plan AST
    ast = query_planner_v2.plan_query(intent)
    assert isinstance(ast, ASTNode)
    assert ast.base_table == expected_base
    
    # 2. Validate AST
    is_valid, err = sql_validator.validate_ast(ast)
    assert is_valid is True, f"AST Validation failed: {err}"
    
    # 3. Build SQL
    sql, params = sql_builder.build(ast)
    assert "SELECT" in sql.upper()
    assert "FROM" in sql.upper()
    
    # 4. Validate SQL
    is_sql_valid, sql_err = sql_validator.validate_sql(sql)
    assert is_sql_valid is True, f"SQL Validation failed: {sql_err}"
    
def test_cartesian_product_prevention():
    # Construct an invalid AST manually
    ast = ASTNode(
        base_table="AgniveerMaster",
        where=[
            # This references a table not in joins
            __import__('ast_models').WhereNode(column="CompanyMaster.Name", operator="=", value="Alpha")
        ]
    )
    is_valid, err = sql_validator.validate_ast(ast)
    assert is_valid is False
    assert "unjoined table" in err
