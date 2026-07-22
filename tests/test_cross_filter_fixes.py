from __future__ import annotations

from unittest.mock import patch
import pytest

from intent_engine.personal_details_parser import parse_personal_details
from intent_engine.query_planner import plan_query, QueryType
from cross_filter_engine import cross_filter_datasets


def test_multi_domain_query_bypasses_personal_details_parser():
    queries = [
        "agniveer with excellent grade in ppt whose verification is pending",
        "Show overweight Agniveers who scored Good in BPET",
        "Show Agniveers with O+ blood group and agniveer who has issued equipment",
        "Show Agniveers with O+ blood group and who has issued equipment",
    ]
    for q in queries:
        # Multi-domain queries should NOT be intercepted by personal_details_parser as a simple lookup
        assert parse_personal_details(q) is None

        # Multi-domain queries should compile to CROSS_FILTER via QueryPlanner
        plan = plan_query(q)
        assert plan.query_type == QueryType.CROSS_FILTER
        assert len(plan.operations) >= 2


def test_cross_filter_dataset_intersection_with_pascalcase_agniveerno():
    dataset1 = [
        {
            "AgniveerNo": "A0701882L",
            "FullName": "HARMAN SINGH",
            "BmiCategory": "Overweight",
        }
    ]
    dataset2 = [
        {
            "AgniveerNo": "A0701882L",
            "FullName": "HARMAN SINGH",
            "SectionName": "BPET",
            "Grade": "Good",
        }
    ]

    res = cross_filter_datasets([dataset1, dataset2])
    assert res.get("status") is True
    assert res.get("matchCount") == 1
    assert len(res.get("records", [])) == 1
    rec = res["records"][0]
    assert rec["AgniveerNo"] == "A0701882L"
    assert rec["BmiCategory"] == "Overweight"
    assert rec["Grade"] == "Good"


def test_performance_query_does_not_extract_spurious_return_condition():
    q = "Show overweight Agniveers who scored Good in BPET"
    plan = plan_query(q)
    for op in plan.operations:
        # 'returnCondition' should not be present in performance or medical filters
        assert "returnCondition" not in op.intent_result.get("filters", {})
