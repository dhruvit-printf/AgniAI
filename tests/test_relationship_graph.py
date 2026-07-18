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
