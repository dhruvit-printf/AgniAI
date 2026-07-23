import pytest
from schema_engine import schema_engine


def test_schema_loads_metadata():
    tables = schema_engine.get_tables()
    assert "AgniveerMaster" in tables

    columns = schema_engine.get_columns("AgniveerMaster")
    assert "Id" in columns
    assert "FullName" in columns

    col_type = schema_engine.get_column_type("AgniveerMaster", "Id")
    assert col_type == "integer"


def test_schema_rejects_unknown_concept():
    table = schema_engine.get_table_for_concept("AlienSpaceship")
    assert table is None


def test_schema_get_implicit_filters():
    filters = schema_engine.get_implicit_filters("Performance")
    # Will be empty if not in ontology
    assert isinstance(filters, dict)


def test_schema_get_ranking_columns():
    cols = schema_engine.get_ranking_columns("AgniveerScoreAttempt")
    assert isinstance(cols, list)


def test_schema_treats_agniveer_numbers_as_strings():
    assert schema_engine.get_column_type("AgniveerMaster", "AgniveerNo") == "string"
    assert schema_engine.get_column_type("PlatoonMaster", "PlatoonNo") == "string"
