import pytest
from ast_models import ASTNode, WhereNode, ConditionGroupNode
from error_recovery import error_recovery_engine

def test_strict_confidence_aborts_on_ambiguity():
    ast = ASTNode(base_table="AgniveerMaster")
    # "Idd" could be "Id". But if we use "I", it might match "Id". 
    # Let's use a very wrong string to ensure 0.85 cutoff blocks it.
    ast.where.append(WhereNode(column="AgniveerMaster.TotallyFakeColumnName", operator="=", value=1))
    
    error_msg = "Column 'TotallyFakeColumnName' does not exist in table 'AgniveerMaster'."
    repaired_ast = error_recovery_engine._attempt_repair(ast, error_msg)
    
    # Must abort and return None
    assert repaired_ast is None

def test_recursive_fuzzy_repair_with_high_confidence():
    ast = ASTNode(base_table="AgniveerMaster")
    # Intentional typo: FullNam instead of FullName. This is highly similar and should pass 0.85.
    ast.where.append(WhereNode(column="AgniveerMaster.FullNam", operator="=", value="John"))
    
    error_msg = "Column 'FullNam' does not exist in table 'AgniveerMaster'."
    repaired_ast = error_recovery_engine._attempt_repair(ast, error_msg)
    
    assert repaired_ast is not None
    assert repaired_ast.where[0].column == "AgniveerMaster.FullName"

def test_repair_applies_to_group_by():
    ast = ASTNode(base_table="AgniveerMaster")
    # Typo in group_by
    ast.group_by = ["AgniveerMaster.Clas"] 
    
    error_msg = "Column 'Clas' does not exist in table 'AgniveerMaster'."
    repaired_ast = error_recovery_engine._attempt_repair(ast, error_msg)
    
    assert repaired_ast is not None
    assert repaired_ast.group_by[0] == "AgniveerMaster.Class"
