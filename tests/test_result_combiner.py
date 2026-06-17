"""
tests/test_result_combiner.py
==============================
Unit tests for the AgniAI Result Combiner.

Tests the three combination strategies:
  - intersect_results() — CROSS_FILTER by agniveerId
  - merge_results()     — MULTI_INDEPENDENT concatenation
  - compare_results()   — COMPARISON side-by-side

CRITICAL: intersection must ONLY use agniveerId, NEVER names.
"""

import pytest
from result_combiner import (
    intersect_results,
    merge_results,
    compare_results,
    _extract_records,
    _extract_agniveer_ids,
)


# =============================================================================
# SAMPLE DATA — mirrors real .NET response shapes
# =============================================================================

PERFORMANCE_RECORDS = [
    {"agniveerId": 101, "fullName": "AMIT KUMAR", "bestTotal": 450, "rank": 1},
    {"agniveerId": 102, "fullName": "RAJ SINGH", "bestTotal": 430, "rank": 2},
    {"agniveerId": 103, "fullName": "SUNIL VERMA", "bestTotal": 420, "rank": 3},
    {"agniveerId": 104, "fullName": "DEEPAK RAO", "bestTotal": 410, "rank": 4},
    {"agniveerId": 105, "fullName": "VIKRAM JHA", "bestTotal": 400, "rank": 5},
]

SKILLS_RECORDS = [
    {"agniveerId": 101, "fullName": "AMIT KUMAR", "sports": "Cricket"},
    {"agniveerId": 103, "fullName": "SUNIL VERMA", "sports": "Cricket, Football"},
    {"agniveerId": 106, "fullName": "MOHAN DAS", "sports": "Cricket"},
]

ATTENDANCE_WRAPPED = {
    "success": True,
    "commandLabel": "Present Today",
    "data": {
        "totalAgniveers": 610,
        "activeCount": 608,
        "presentToday": 550,
    },
}

EQUIPMENT_RECORDS = [
    {"agniveerId": 201, "fullName": "PAWAN SINGH", "itemName": "DMS Boot", "dueDate": "2026-06-01"},
    {"agniveerId": 202, "fullName": "RAVI KUMAR", "itemName": "Combat Coat", "dueDate": "2026-05-15"},
]

LEAVE_RECORDS = [
    {"agniveerId": 301, "fullName": "GOPAL RAO", "fromDate": "2026-06-10", "toDate": None},
    {"agniveerId": 302, "fullName": "TARUN KUMAR", "fromDate": "2026-06-12", "toDate": None},
]

MEDICAL_RECORDS = [
    {"agniveerId": 103, "fullName": "SUNIL VERMA", "disease": "Fever"},
    {"agniveerId": 106, "fullName": "MOHAN DAS", "disease": "Fracture"},
    {"agniveerId": 107, "fullName": "KARAN THAPA", "disease": "Cold"},
]

DISTRIBUTION_WITH_TEAMS = {
    "success": True,
    "data": {
        "distributionId": 5,
        "teams": [
            {
                "teamId": 4,
                "teamName": "14 Punjab",
                "memberCount": 2,
                "members": [
                    {"agniveerId": 101, "fullName": "AMIT KUMAR", "rank": 1},
                    {"agniveerId": 102, "fullName": "RAJ SINGH", "rank": 2},
                ],
            },
        ],
    },
}


# =============================================================================
# RECORD EXTRACTION
# =============================================================================

class TestExtractRecords:
    """Test _extract_records with various .NET response shapes."""

    def test_bare_list(self):
        records = _extract_records(PERFORMANCE_RECORDS)
        assert len(records) == 5

    def test_wrapped_dict_with_data_list(self):
        wrapped = {"success": True, "data": PERFORMANCE_RECORDS}
        records = _extract_records(wrapped)
        assert len(records) == 5

    def test_nested_dict(self):
        """Strength breakdown shape: { data: { totalAgniveers: 610, ... } }"""
        records = _extract_records(ATTENDANCE_WRAPPED)
        # The inner data is a dict, not a list, so no records extracted
        assert isinstance(records, list)

    def test_teams_members(self):
        records = _extract_records(DISTRIBUTION_WITH_TEAMS)
        assert len(records) == 2
        assert records[0]["agniveerId"] == 101

    def test_empty_data(self):
        assert _extract_records(None) == []
        assert _extract_records({}) == []
        assert _extract_records([]) == []


class TestExtractAgniveerIds:
    """Test ID extraction from records."""

    def test_standard_records(self):
        ids = _extract_agniveer_ids(PERFORMANCE_RECORDS)
        assert ids == {101, 102, 103, 104, 105}

    def test_empty_records(self):
        ids = _extract_agniveer_ids([])
        assert ids == set()

    def test_missing_id_field(self):
        records = [{"fullName": "Test", "score": 100}]
        ids = _extract_agniveer_ids(records)
        assert ids == set()


# =============================================================================
# INTERSECT (CROSS_FILTER)
# =============================================================================

class TestIntersectResults:
    """Test cross-filter intersection by Agniveer ID."""

    def test_basic_intersection(self):
        """Performance ∩ Skills → only IDs in both sets."""
        result = intersect_results([PERFORMANCE_RECORDS, SKILLS_RECORDS])
        assert result["queryType"] == "cross_filter"
        assert result["matchCount"] == 2  # IDs 101 and 103
        record_ids = {r["agniveerId"] for r in result["records"]}
        assert record_ids == {101, 103}

    def test_intersection_preserves_primary_data(self):
        """Records come from primary set (index 0) with full details."""
        result = intersect_results([PERFORMANCE_RECORDS, SKILLS_RECORDS])
        for record in result["records"]:
            assert "bestTotal" in record  # From performance set
            assert "rank" in record

    def test_no_overlap(self):
        """No common IDs → empty result."""
        result = intersect_results([EQUIPMENT_RECORDS, LEAVE_RECORDS])
        assert result["matchCount"] == 0
        assert result["records"] == []

    def test_empty_result_sets(self):
        result = intersect_results([])
        assert result["matchCount"] == 0
        assert result["records"] == []

    def test_single_result_set(self):
        """With only one set, all records match (self-intersection)."""
        result = intersect_results([PERFORMANCE_RECORDS])
        assert result["matchCount"] == 5

    def test_total_before_filter(self):
        result = intersect_results([PERFORMANCE_RECORDS, SKILLS_RECORDS])
        assert result["totalBeforeFilter"] == 5  # Primary set had 5 records

    def test_never_matches_by_name(self):
        """Even if names match, only ID intersection counts."""
        set_a = [{"agniveerId": 1, "fullName": "AMIT KUMAR"}]
        set_b = [{"agniveerId": 999, "fullName": "AMIT KUMAR"}]  # Same name, different ID
        result = intersect_results([set_a, set_b])
        assert result["matchCount"] == 0

    def test_medical_cross_skills(self):
        """Medical records ∩ Skills records."""
        result = intersect_results([MEDICAL_RECORDS, SKILLS_RECORDS])
        # Common IDs: 103 (SUNIL VERMA), 106 (MOHAN DAS)
        assert result["matchCount"] == 2


# =============================================================================
# MERGE (MULTI_INDEPENDENT)
# =============================================================================

class TestMergeResults:
    """Test multi-independent result merging."""

    def test_basic_merge(self):
        result = merge_results([
            ("Attendance", ATTENDANCE_WRAPPED),
            ("Equipment", EQUIPMENT_RECORDS),
        ])
        assert result["queryType"] == "multi_independent"
        assert result["sectionCount"] == 2
        assert result["sections"][0]["label"] == "Attendance"
        assert result["sections"][1]["label"] == "Equipment"

    def test_merge_preserves_raw_data(self):
        result = merge_results([
            ("Leave", LEAVE_RECORDS),
            ("Medical", MEDICAL_RECORDS),
        ])
        assert result["sections"][0]["data"] is LEAVE_RECORDS
        assert result["sections"][1]["data"] is MEDICAL_RECORDS

    def test_merge_record_counts(self):
        result = merge_results([
            ("Leave", LEAVE_RECORDS),
            ("Medical", MEDICAL_RECORDS),
        ])
        assert result["sections"][0]["recordCount"] == 2
        assert result["sections"][1]["recordCount"] == 3

    def test_empty_merge(self):
        result = merge_results([])
        assert result["sectionCount"] == 0
        assert result["sections"] == []


# =============================================================================
# COMPARE (COMPARISON)
# =============================================================================

class TestCompareResults:
    """Test comparison result building."""

    def test_basic_comparison(self):
        result = compare_results([
            ("PPT Performance", PERFORMANCE_RECORDS),
            ("BPET Performance", PERFORMANCE_RECORDS[:3]),
        ])
        assert result["queryType"] == "comparison"
        assert len(result["sides"]) == 2
        assert result["sides"][0]["label"] == "PPT Performance"
        assert result["sides"][1]["label"] == "BPET Performance"

    def test_comparison_extracts_metrics(self):
        result = compare_results([
            ("Attendance", ATTENDANCE_WRAPPED),
            ("Leave", {"totalRecords": 5, "data": LEAVE_RECORDS}),
        ])
        assert len(result["sides"]) == 2
        # Attendance has numeric metrics in its data dict
        assert "comparedMetrics" in result

    def test_comparison_preserves_raw_data(self):
        result = compare_results([
            ("Side A", PERFORMANCE_RECORDS),
            ("Side B", MEDICAL_RECORDS),
        ])
        assert result["sides"][0]["data"] is PERFORMANCE_RECORDS
        assert result["sides"][1]["data"] is MEDICAL_RECORDS

    def test_empty_comparison(self):
        result = compare_results([])
        assert result["queryType"] == "comparison"
        assert result["sides"] == []
