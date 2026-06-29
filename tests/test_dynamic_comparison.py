"""
Tests for the dynamic comparison visualization engine.
"""

import pytest
from intent_engine.query_planner import plan_query, QueryType
from visualization_intent import build_visualization_intent
from compare_engine import compare_datasets, select_visualization_type
from response_builder import build_response


def test_semantic_comparison_detection():
    # 2-way comparison
    plan = plan_query("compare PPT vs BPET results for Alpha", session_id="test")
    assert plan.query_type == QueryType.COMPARISON
    assert len(plan.comparison_execution_plan) == 2
    assert plan.comparison_execution_plan[0]["label"] == "PPT"
    assert plan.comparison_execution_plan[1]["label"] == "BPET"

    # N-way comparison
    plan = plan_query("compare PPT vs BPET vs Medical results for Alpha", session_id="test")
    assert plan.query_type == QueryType.COMPARISON
    assert len(plan.comparison_execution_plan) == 3
    assert plan.comparison_execution_plan[0]["label"] == "PPT"
    assert plan.comparison_execution_plan[1]["label"] == "BPET"
    assert plan.comparison_execution_plan[2]["label"] == "Medical"


def test_dynamic_visualization_selection():
    # Table shape (contain names)
    sides_table = [
        {"data": [{"fullName": "John Doe", "score": 90}]},
        {"data": [{"fullName": "Jane Doe", "score": 95}]}
    ]
    assert select_visualization_type(sides_table) == "COMPARE_TABLE"

    # Line chart shape (contain dates/attempts)
    sides_line = [
        {"data": [{"date": "2026-01-01", "score": 80}]},
        {"data": [{"date": "2026-01-02", "score": 85}]}
    ]
    assert select_visualization_type(sides_line) == "COMPARE_CHART_LINE"

    # Pie chart shape (contain leave/medical category)
    sides_pie = [
        {"data": [{"leaveType": "Sick", "count": 2}]},
        {"data": [{"leaveType": "Sick", "count": 3}]}
    ]
    assert select_visualization_type(sides_pie) == "COMPARE_CHART_PIE"

    # Bar chart shape (contain platoon/company)
    sides_bar = [
        {"data": [{"platoon": "Platoon 1", "score": 75}]},
        {"data": [{"platoon": "Platoon 2", "score": 80}]}
    ]
    assert select_visualization_type(sides_bar) == "COMPARE_CHART_BAR"

    # Card shape (1 record, 2 keys)
    sides_card = [
        {"data": [{"score": 90}]},
        {"data": [{"score": 95}]}
    ]
    assert select_visualization_type(sides_card) == "COMPARE_CARD"


def test_compare_datasets_metrics():
    labeled_results = [
        ("PPT", {"data": [{"score": 100}, {"score": 90}], "recordCount": 2, "averageScore": 95.0}),
        ("BPET", {"data": [{"score": 80}, {"score": 70}], "recordCount": 2, "averageScore": 75.0}),
        ("Medical", {"data": [{"score": 90}, {"score": 80}], "recordCount": 2, "averageScore": 85.0})
    ]
    
    result = compare_datasets(labeled_results)
    
    assert "comparisonMetrics" in result
    metrics = result["comparisonMetrics"]
    assert metrics["difference"]["averageScore"] == 20.0  # 95.0 - 75.0
    assert metrics["highest"]["averageScore"]["label"] == "PPT"
    assert metrics["lowest"]["averageScore"]["label"] == "BPET"
    assert metrics["variance"]["averageScore"] == pytest.approx(66.67, 0.1)


def test_response_builder_format():
    metadata = {
        "queryType": "COMPARISON",
        "comparisonMetrics": {
            "recordCount": {"difference": 0.0},
            "highest": {},
            "lowest": {},
            "difference": {},
            "percentageDifference": {},
            "variance": {}
        }
    }
    formatted_data = {
        "type": "COMPARE_CARD",
        "title": "Comparison Result",
        "data": {"metrics": []}
    }
    dotnet_payload = [
        {"id": "dataset_1", "label": "PPT"}
    ]
    
    response = build_response(
        message="Comparison generated",
        formatted_data=formatted_data,
        metadata=metadata,
        session_id="test_session",
        dotnet_payload=dotnet_payload
    )
    
    assert response["status"] is True
    assert isinstance(response["formattedData"], dict)
    assert response["formattedData"]["type"] == "COMPARE_CARD"
    assert "comparisonMetrics" in response
    assert response["dotnetPayload"] == dotnet_payload
