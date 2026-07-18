from __future__ import annotations

from relationship_graph import relationship_graph


def test_attendance_to_company_join_path_is_discoverable():
    path = relationship_graph.find_shortest_path(
        "AgniveerAttendanceMaster", "CompanyMaster"
    )

    assert path is not None
    assert len(path) >= 2
    assert path[0]["left"] == "AgniveerAttendanceMaster"
    assert path[-1]["right"] == "CompanyMaster"


def test_medical_to_company_join_path():
    path = relationship_graph.find_shortest_path("MedicalRecordMaster", "CompanyMaster")
    assert path is not None
    assert len(path) == 3
    assert path[0]["left"] == "MedicalRecordMaster"
    assert path[-1]["right"] == "CompanyMaster"


def test_performance_to_company_join_path():
    path = relationship_graph.find_shortest_path("AgniveerScoreAttempt", "CompanyMaster")
    assert path is not None
    assert len(path) == 3
    assert path[0]["left"] == "AgniveerScoreAttempt"
    assert path[-1]["right"] == "CompanyMaster"


def test_equipment_to_company_join_path():
    path = relationship_graph.find_shortest_path("AgniveerEquipment", "CompanyMaster")
    assert path is not None
    assert len(path) == 3
    assert path[0]["left"] == "AgniveerEquipment"
    assert path[-1]["right"] == "CompanyMaster"


def test_distribution_to_company_join_path():
    # Distribution joins via the AgniveerRelationMaster bridge table
    path = relationship_graph.find_shortest_path("DistributionMaster", "CompanyMaster")
    assert path is not None
    assert len(path) == 4
    assert path[0]["left"] == "DistributionMaster"
    assert path[-1]["right"] == "CompanyMaster"


def test_reverse_lookup_join_path():
    # Start backwards from CompanyMaster, look for AgniveerMaster
    path = relationship_graph.find_shortest_path("CompanyMaster", "AgniveerMaster")
    assert path is not None
    assert len(path) > 0
    assert path[0]["left"] == "CompanyMaster"
    assert path[-1]["right"] == "AgniveerMaster"

def test_graph_returns_none_for_disjoint_tables():
    # If a table is truly disjoint, find_shortest_path should return None.
    # Note: the current schema might be fully connected. Let's ask it for an AlienSpaceship.
    path = relationship_graph.find_shortest_path("AgniveerMaster", "AlienSpaceship")
    assert path is None

def test_graph_handles_cycles_without_infinite_loops(monkeypatch):
    original_get_tables = relationship_graph.engine.get_tables
    original_get_columns = relationship_graph.engine.get_columns
    
    def mock_get_tables():
        tables = original_get_tables()
        tables.append("FakeMaster")
        return tables
        
    def mock_get_columns(table):
        if table == "AgniveerMaster":
            return ["Id", "FakeId"]
        if table == "FakeMaster":
            return ["Id", "AgniveerId"]
        return original_get_columns(table)
        
    monkeypatch.setattr(relationship_graph.engine, "get_tables", mock_get_tables)
    monkeypatch.setattr(relationship_graph.engine, "get_columns", mock_get_columns)
    
    relationship_graph.build_graph()
    
    path = relationship_graph.find_shortest_path("AgniveerMaster", "FakeMaster")
    assert path is not None
    assert len(path) == 1

def test_graph_ignores_duplicate_relationships(monkeypatch):
    original_get_columns = relationship_graph.engine.get_columns
    
    def mock_get_columns(table):
        cols = original_get_columns(table)
        if table == "AgniveerMaster":
            # Add duplicate PlatoonId column
            cols.append("PlatoonId")
        return cols
        
    monkeypatch.setattr(relationship_graph.engine, "get_columns", mock_get_columns)
    relationship_graph.build_graph()
    
    path = relationship_graph.find_shortest_path("AgniveerMaster", "PlatoonMaster")
    assert path is not None
    assert len(path) == 1
