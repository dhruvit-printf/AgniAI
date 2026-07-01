"""
Tests for the dynamic comparison visualization engine.
"""

import pytest

from compare_engine import compare_datasets, select_visualization_type
from intent_engine.query_planner import QueryType, plan_query
from response_builder import build_response
from visualization_intent import build_visualization_intent


def test_semantic_comparison_detection():
    # 2-way comparison
    plan = plan_query("compare PPT vs BPET results for Alpha")
    assert plan.query_type == QueryType.COMPARISON
    assert len(plan.comparison_execution_plan) == 2
    assert plan.comparison_execution_plan[0]["label"] == "PPT"
    assert plan.comparison_execution_plan[1]["label"] == "BPET"

    # N-way comparison
    plan = plan_query("compare PPT vs BPET vs Medical results for Alpha")
    assert plan.query_type == QueryType.COMPARISON
    assert len(plan.comparison_execution_plan) == 3
    assert plan.comparison_execution_plan[0]["label"] == "PPT"
    assert plan.comparison_execution_plan[1]["label"] == "BPET"
    assert plan.comparison_execution_plan[2]["label"] == "Medical"


def test_dynamic_visualization_selection():
    # Table shape (contain names)
    sides_table = [
        {"data": [{"fullName": "John Doe", "score": 90}]},
        {"data": [{"fullName": "Jane Doe", "score": 95}]},
    ]
    assert select_visualization_type(sides_table) == "COMPARE_TABLE"

    # Line chart shape (contain dates/attempts)
    sides_line = [
        {"data": [{"date": "2026-01-01", "score": 80}]},
        {"data": [{"date": "2026-01-02", "score": 85}]},
    ]
    assert select_visualization_type(sides_line) == "COMPARE_LINE_CHART"

    # Pie chart shape (contain leave/medical category)
    sides_pie = [
        {"data": [{"leaveType": "Sick", "count": 2}]},
        {"data": [{"leaveType": "Sick", "count": 3}]},
    ]
    assert select_visualization_type(sides_pie) == "COMPARE_PIE_CHART"

    # Bar chart shape (contain platoon/company)
    sides_bar = [
        {"data": [{"platoon": "Platoon 1", "score": 75}]},
        {"data": [{"platoon": "Platoon 2", "score": 80}]},
    ]
    assert select_visualization_type(sides_bar) == "COMPARE_BAR_CHART"

    # Card shape (1 record, 2 keys)
    sides_card = [{"data": [{"score": 90}]}, {"data": [{"score": 95}]}]
    assert select_visualization_type(sides_card) == "COMPARE_CARD"

    # Mixed shapes should fall back to table
    sides_mixed = [
        {"data": [{"platoon": "Platoon 1", "score": 75}]},
        {"data": [{"date": "2026-01-02", "score": 85}]},
    ]
    assert select_visualization_type(sides_mixed) == "COMPARE_TABLE"


def test_compare_datasets_metrics():
    labeled_results = [
        (
            "PPT",
            {
                "data": [{"score": 100}, {"score": 90}],
                "recordCount": 2,
                "averageScore": 95.0,
            },
        ),
        (
            "BPET",
            {
                "data": [{"score": 80}, {"score": 70}],
                "recordCount": 2,
                "averageScore": 75.0,
            },
        ),
        (
            "Medical",
            {
                "data": [{"score": 90}, {"score": 80}],
                "recordCount": 2,
                "averageScore": 85.0,
            },
        ),
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
            "variance": {},
        },
    }
    formatted_data = {
        "type": "COMPARE_CARD",
        "title": "Comparison Result",
        "data": {"metrics": []},
    }
    dotnet_payload = [{"id": "dataset_1", "label": "PPT"}]

    response = build_response(
        message="Comparison generated",
        formatted_data=formatted_data,
        metadata=metadata,
        session_id="test_session",
        dotnet_payload=dotnet_payload,
    )

    assert response["status"] is True
    assert isinstance(response["formattedData"], list)
    assert response["formattedData"][0]["type"] == "COMPARE_CARD"
    assert "comparisonMetrics" in response["metadata"]
    assert response["dotnetPayload"] == dotnet_payload


def test_compare_datasets_auto_resolve_record_count():
    labeled_results = [
        (
            "Pending",
            {
                "data": {
                    "totalAgniveers": 605,
                    "pendingCount": 545,
                    "sentCount": 30,
                    "notRespondedCount": 30,
                    "verifiedCount": 21,
                    "rejectedCount": 9
                }
            }
        ),
        (
            "Completed",
            {
                "data": {
                    "totalAgniveers": 605,
                    "pendingCount": 545,
                    "sentCount": 30,
                    "notRespondedCount": 30,
                    "verifiedCount": 21,
                    "rejectedCount": 9
                }
            }
        )
    ]

    comparison_context = {
        "datasets": [
            {
                "id": "dataset_1",
                "label": "Pending",
                "intent": {"category": "Verification", "operation": "Pending", "subcategory": "PendingVerification"},
                "rawData": labeled_results[0][1],
            },
            {
                "id": "dataset_2",
                "label": "Completed",
                "intent": {"category": "Verification", "operation": "Completed", "subcategory": "CompletedVerification"},
                "rawData": labeled_results[1][1],
            }
        ]
    }

    result = compare_datasets(labeled_results, comparison_context=comparison_context)
    assert result["left"]["metrics"]["recordCount"] == 545
    assert result["right"]["metrics"]["recordCount"] == 21


# ── Tests for comparison chart override via visualization_intent ──────────


def test_visualization_intent_comparison_chart_override_line():
    """When user says 'compare X vs Y in line chart', chart_type should be preserved."""
    intent = build_visualization_intent(
        "compare PPT vs BPET in line chart",
        {"category": "Performance", "operation": "Compare"},
    )
    assert intent["comparison"] is True
    assert intent["comparison_chart_override"] == "line"
    assert intent["chart_type"] == "line"
    assert intent["frontend_override"] is True


def test_visualization_intent_comparison_chart_override_pie():
    """When user says 'compare ... in pie chart', chart_type should be preserved."""
    intent = build_visualization_intent(
        "compare leave types in pie chart",
        {"category": "Leave", "operation": "Compare"},
    )
    assert intent["comparison"] is True
    assert intent["comparison_chart_override"] == "pie"
    assert intent["chart_type"] == "pie"
    assert intent["frontend_override"] is True


def test_visualization_intent_comparison_chart_override_bar():
    """When user says 'compare ... in bar chart', chart_type should be preserved."""
    intent = build_visualization_intent(
        "compare platoon scores in bar chart",
        {"category": "Performance", "operation": "Compare"},
    )
    assert intent["comparison"] is True
    assert intent["comparison_chart_override"] == "bar"
    assert intent["chart_type"] == "bar"
    assert intent["frontend_override"] is True


def test_visualization_intent_comparison_without_chart_request():
    """Plain comparison without explicit chart request should null presentation/chart_type."""
    intent = build_visualization_intent(
        "compare PPT vs BPET",
        {"category": "Performance", "operation": "Compare"},
    )
    assert intent["comparison"] is True
    assert intent.get("comparison_chart_override") is None
    assert intent["presentation"] is None
    assert intent["chart_type"] is None


def test_widget_selector_comparison_chart_override_line():
    """WidgetSelector._comparison should honor comparison_chart_override='line'."""
    from widget_selector import WidgetSelector

    combined_result = {
        "left": {"label": "PPT", "data": [{"score": 80}]},
        "right": {"label": "BPET", "data": [{"score": 85}]},
    }
    selector = WidgetSelector()
    specs = selector.select(
        query_type="compare",
        intent={"category": "Performance"},
        combined_result=combined_result,
        primary_widget_type="AREA_CHART",
        comparison_chart_override="line",
    )
    types = [s.widget_type for s in specs]
    assert "COMPARE_LINE_CHART" in types


def test_widget_selector_comparison_chart_override_pie():
    """WidgetSelector._comparison should honor comparison_chart_override='pie'."""
    from widget_selector import WidgetSelector

    combined_result = {
        "left": {"label": "Sick", "data": [{"count": 5}]},
        "right": {"label": "Casual", "data": [{"count": 3}]},
    }
    selector = WidgetSelector()
    specs = selector.select(
        query_type="compare",
        intent={"category": "Leave"},
        combined_result=combined_result,
        primary_widget_type="AREA_CHART",
        comparison_chart_override="pie",
    )
    types = [s.widget_type for s in specs]
    assert "COMPARE_PIE_CHART" in types


def test_build_comparison_widgets_with_override():
    """build_comparison_widgets should produce COMPARE_LINE_CHART when override is set."""
    from widget_engine import build_comparison_widgets

    combined_result = {
        "left": {"label": "PPT", "data": [{"score": 80, "date": "2026-01"}]},
        "right": {"label": "BPET", "data": [{"score": 85, "date": "2026-01"}]},
        "visualizationType": "COMPARE_CARD",  # default auto-detected
    }
    widgets = build_comparison_widgets(
        combined_result,
        "Performance Compare",
        visualization_intent={"comparison_chart_override": "line"},
    )
    types = [w["type"] for w in widgets]
    assert "COMPARE_LINE_CHART" in types


def test_build_comparison_widgets_without_override_uses_autodetected():
    """Without override, build_comparison_widgets uses the auto-detected visualizationType."""
    from widget_engine import build_comparison_widgets

    combined_result = {
        "left": {"label": "A", "data": [{"score": 80}]},
        "right": {"label": "B", "data": [{"score": 85}]},
        "visualizationType": "COMPARE_BAR_CHART",
    }
    widgets = build_comparison_widgets(combined_result, "Test Compare")
    types = [w["type"] for w in widgets]
    assert "COMPARE_BAR_CHART" in types


def test_build_comparison_widgets_maps_compare_card_to_table():
    """COMPARE_CARD should render as a table, not as a summary card widget."""
    from widget_engine import build_comparison_widgets

    combined_result = {
        "left": {"label": "A", "data": [{"score": 80}]},
        "right": {"label": "B", "data": [{"score": 85}]},
        "visualizationType": "COMPARE_CARD",
    }
    widgets = build_comparison_widgets(combined_result, "Test Compare")
    types = [w["type"] for w in widgets]
    assert "COMPARE_CARD" not in types
    assert "COMPARE_TABLE" in types
