"""
tests/test_query_planner.py
============================
Unit tests for the AgniAI Query Planning Layer.

Tests the classification of admin queries into SIMPLE, CROSS_FILTER,
COMPARISON, and MULTI_INDEPENDENT types, with emphasis on:
  - Correct type detection
  - False-split prevention (same-category compounds)
  - Confidence scoring
  - Backward compatibility (simple queries stay simple)
"""

import pytest
from query_planner import plan_query, QueryType


# =============================================================================
# SIMPLE QUERIES — must remain SIMPLE (backward compatibility)
# =============================================================================

class TestSimpleQueries:
    """Simple queries must pass through unchanged."""

    def test_top_performers_basic(self):
        plan = plan_query("Show top 10 performers in PPT")
        assert plan.query_type == QueryType.SIMPLE
        assert len(plan.operations) == 1
        assert plan.operations[0].intent_result["category"] == "Performance"

    def test_bottom_performers(self):
        plan = plan_query("Who are the worst 5 performers in BPET?")
        assert plan.query_type == QueryType.SIMPLE

    def test_current_leave(self):
        plan = plan_query("Who is on leave today?")
        assert plan.query_type == QueryType.SIMPLE

    def test_equipment_overdue(self):
        plan = plan_query("Show overdue equipment")
        assert plan.query_type == QueryType.SIMPLE

    def test_active_medical_cases(self):
        plan = plan_query("How many active medical cases are there?")
        assert plan.query_type == QueryType.SIMPLE

    def test_strength_breakdown(self):
        plan = plan_query("Give me the strength breakdown")
        assert plan.query_type == QueryType.SIMPLE

    def test_verification_pending(self):
        plan = plan_query("Show pending verifications")
        assert plan.query_type == QueryType.SIMPLE

    def test_distribution_latest(self):
        plan = plan_query("Show latest distribution")
        assert plan.query_type == QueryType.SIMPLE

    def test_skills_by_sport(self):
        plan = plan_query("Show roster by sport")
        assert plan.query_type == QueryType.SIMPLE

    def test_average_score_ppt(self):
        plan = plan_query("What is the average score in PPT?")
        assert plan.query_type == QueryType.SIMPLE

    def test_empty_query(self):
        plan = plan_query("")
        assert plan.query_type == QueryType.SIMPLE
        assert plan.confidence == 0.0

    def test_unknown_query(self):
        """Truly unknown queries should still be SIMPLE.
        Note: 'weather today' gets matched to Attendance by the existing
        intent classifier because of the 'today' keyword — this is correct
        existing behavior that we preserve."""
        plan = plan_query("What is the weather today?")
        assert plan.query_type == QueryType.SIMPLE


# =============================================================================
# FALSE SPLIT PREVENTION — same-category compounds must NOT be split
# =============================================================================

class TestFalseSplitPrevention:
    """Queries with 'and' connecting items within the same category must stay SIMPLE."""

    def test_approved_and_pending_leave(self):
        plan = plan_query("approved and pending leave")
        assert plan.query_type == QueryType.SIMPLE

    def test_top_and_bottom_performers(self):
        plan = plan_query("top and bottom performers in PPT")
        assert plan.query_type == QueryType.SIMPLE

    def test_pass_and_fail_percentage(self):
        plan = plan_query("show pass and fail percentage")
        assert plan.query_type == QueryType.SIMPLE

    def test_improvement_and_drop(self):
        plan = plan_query("show improvement and drop in scores")
        assert plan.query_type == QueryType.SIMPLE

    def test_issued_and_procured_items(self):
        plan = plan_query("show issued and procured items")
        assert plan.query_type == QueryType.SIMPLE

    def test_pending_and_completed_verification(self):
        plan = plan_query("show pending and completed verification")
        assert plan.query_type == QueryType.SIMPLE

    def test_best_and_worst_performers(self):
        plan = plan_query("best and worst performers")
        assert plan.query_type == QueryType.SIMPLE

    def test_annual_and_medical_leave(self):
        plan = plan_query("annual and medical leave taken")
        assert plan.query_type == QueryType.SIMPLE


# =============================================================================
# CROSS_FILTER — queries bridging two categories with a filter keyword
# =============================================================================

class TestCrossFilter:
    """Cross-filter queries must produce 2+ operations from different categories."""

    def test_top_performer_who_plays_cricket(self):
        plan = plan_query("Show top performer in PPT who plays cricket")
        assert plan.query_type == QueryType.CROSS_FILTER
        assert len(plan.operations) >= 2
        categories = {op.intent_result["category"] for op in plan.operations}
        assert "Performance" in categories
        assert "Skills" in categories
        assert plan.confidence >= 0.5

    def test_medical_cases_among_football_players(self):
        plan = plan_query("Show medical cases among football players")
        assert plan.query_type == QueryType.CROSS_FILTER
        assert len(plan.operations) >= 2
        categories = {op.intent_result["category"] for op in plan.operations}
        assert "Medical" in categories
        assert "Skills" in categories

    def test_top_performers_who_play_hockey(self):
        plan = plan_query("Show top performers who play hockey")
        assert plan.query_type == QueryType.CROSS_FILTER
        assert len(plan.operations) >= 2

    def test_leave_records_among_cricket_players(self):
        plan = plan_query("Show leave records among cricket players")
        assert plan.query_type == QueryType.CROSS_FILTER
        assert len(plan.operations) >= 2


# =============================================================================
# COMPARISON — queries asking to compare two things
# =============================================================================

class TestComparison:
    """Comparison queries between different categories."""

    def test_compare_attendance_and_leave(self):
        plan = plan_query("Compare attendance and leave records")
        assert plan.query_type == QueryType.COMPARISON
        assert len(plan.operations) >= 2
        categories = {op.intent_result["category"] for op in plan.operations}
        assert len(categories) >= 2

    def test_intra_performance_comparison_stays_simple(self):
        """PPT vs BPET within Performance — let .NET handle natively."""
        plan = plan_query("Compare PPT and BPET performance")
        # This should stay SIMPLE because the .NET API handles it
        assert plan.query_type == QueryType.SIMPLE


# =============================================================================
# MULTI_INDEPENDENT — separate queries joined by connectors
# =============================================================================

class TestMultiIndependent:
    """Multi-independent queries for different categories."""

    def test_attendance_and_equipment_overdue(self):
        plan = plan_query("Show attendance and equipment overdue records")
        assert plan.query_type == QueryType.MULTI_INDEPENDENT
        assert len(plan.operations) >= 2
        categories = {op.intent_result["category"] for op in plan.operations}
        assert "Attendance" in categories
        assert "Equipment" in categories

    def test_leave_along_with_medical(self):
        plan = plan_query("Show leave status along with medical cases")
        assert plan.query_type == QueryType.MULTI_INDEPENDENT
        assert len(plan.operations) >= 2

    def test_verification_as_well_as_distribution(self):
        plan = plan_query("Show pending verification as well as latest distribution")
        assert plan.query_type == QueryType.MULTI_INDEPENDENT
        assert len(plan.operations) >= 2


# =============================================================================
# CONFIDENCE SCORING
# =============================================================================

class TestConfidence:
    """Verify confidence scores are in expected ranges."""

    def test_simple_high_confidence(self):
        plan = plan_query("Show top 10 performers in PPT")
        assert plan.confidence >= 0.9

    def test_cross_filter_medium_confidence(self):
        plan = plan_query("Show top performer in PPT who plays cricket")
        assert 0.5 <= plan.confidence <= 1.0

    def test_truly_unknown_low_confidence(self):
        """A query with zero admin category matches should have low confidence."""
        plan = plan_query("Tell me a joke about elephants")
        assert plan.confidence < 0.5


# =============================================================================
# QUERY PLAN SERIALISATION
# =============================================================================

class TestQueryPlanDict:
    """Verify to_dict() produces the expected shape."""

    def test_to_dict_has_required_fields(self):
        plan = plan_query("Show top 10 performers in PPT")
        d = plan.to_dict()
        assert "queryType" in d
        assert "confidence" in d
        assert "operationCount" in d
        assert "reasoning" in d
        assert "operations" in d
        assert d["queryType"] == "simple"
        assert d["operationCount"] == 1

    def test_to_dict_operations_have_fields(self):
        plan = plan_query("Show top 10 performers in PPT")
        d = plan.to_dict()
        op = d["operations"][0]
        assert "rawFragment" in op
        assert "intentResult" in op
        assert "dotnetPayload" in op
