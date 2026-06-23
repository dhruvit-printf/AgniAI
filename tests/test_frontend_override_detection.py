from __future__ import annotations

from admin_pipeline import _extract_frontend_visualization_intent
from visualization_intent import build_visualization_intent


def test_natural_language_chart_request_sets_frontend_override():
    intent = build_visualization_intent(
        "Who are the top 10 performers in BPET in pie chart?",
        {"category": "Performance", "operation": "Top"},
        {"sections": [{"label": "Result", "data": [{"id": 1}, {"id": 2}]}]},
    )

    assert intent["frontend_override"] is True
    assert intent["presentation"] == "chart"
    assert intent["chart_type"] == "pie"


def test_group_by_does_not_auto_override():
    body = {"intent": {"group_by": "platoon"}}
    visual = _extract_frontend_visualization_intent(body)

    assert visual.get("group_by") == "platoon"
    assert visual.get("frontend_override") is None


def test_explicit_display_fields_set_override():
    body = {"intent": {"presentation": "chart", "chart_type": "bar"}}
    visual = _extract_frontend_visualization_intent(body)

    assert visual["frontend_override"] is True
    assert visual["presentation"] == "chart"
    assert visual["chart_type"] == "bar"
