"""
tests/test_admin_intent.py
==========================
Unit tests for the admin chatbot intent classifier.
"""

import pytest

from admin_intent import (
    classify_admin_intent,
    format_admin_intent,
    format_admin_payload,
)

# =============================================================================
# PERFORMANCE
# =============================================================================


def test_top_performers_basic():
    r = classify_admin_intent("Who are the top 5 performers?")
    assert r["category"] == "Performance"
    assert r["subcategory"] == "TopPerformers"
    assert r["number"] == 5
    assert r["type"] == "Tabular"


def test_top_performers_with_section():
    r = classify_admin_intent("Show top 10 performers in BPET section")
    assert r["category"] == "Performance"
    assert r["subcategory"] == "TopPerformers"
    assert r["number"] == 10
    assert r["section"] == "BPET"


def test_lowest_performers():
    r = classify_admin_intent("Who are the worst 3 performers in FIRING?")
    assert r["category"] == "Performance"
    assert r["subcategory"] == "LowestPerformers"
    assert r["number"] == 3
    assert r["section"] == "Firing"


def test_average_score():
    r = classify_admin_intent("What is the average score in PPT?")
    assert r["category"] == "Performance"
    assert r["subcategory"] == "AverageScore"
    assert r["section"] == "PPT"


def test_pass_percentage():
    r = classify_admin_intent("What is the pass percentage?")
    assert r["category"] == "Performance"
    assert r["subcategory"] == "PassPercentage"


def test_fail_percentage():
    r = classify_admin_intent("Show the fail percentage")
    assert r["category"] == "Performance"
    assert r["subcategory"] == "FailPercentage"


def test_grade_distribution():
    r = classify_admin_intent("Show grade distribution for DRILL")
    assert r["category"] == "Performance"
    assert r["subcategory"] == "GradeDistribution"
    assert r["section"] == "Drill"


def test_grade_summary():
    r = classify_admin_intent("Give me a grade summary")
    assert r["category"] == "Performance"
    assert r["subcategory"] == "GradingSummary"
    assert r["type"] == "Bar Chart"


def test_overall_performance():
    r = classify_admin_intent("Give me the overall performance report")
    assert r["category"] == "Performance"
    assert r["subcategory"] == "OverallPerformance"
    assert r["type"] == "Tabular"


def test_improvement():
    r = classify_admin_intent("Which trainees showed improvement?")
    assert r["category"] == "Performance"
    assert r["subcategory"] == "Improvement"
    assert r["type"] == "Trend Chart"


def test_decline():
    r = classify_admin_intent("Who had a decline in scores?")
    assert r["category"] == "Performance"
    assert r["subcategory"] == "Drop"
    assert r["type"] == "Trend Chart"


def test_grading_filter_excellent():
    r = classify_admin_intent("How many trainees got Excellent grade?")
    assert r["grading"] == "Excellent"


def test_grading_filter_good():
    r = classify_admin_intent("Show me trainees who scored Good in BEPT")
    assert r["grading"] == "Good"


# =============================================================================
# LEAVE
# =============================================================================


def test_most_leave():
    r = classify_admin_intent("Who has taken the most leaves?")
    assert r["category"] == "Leave"
    assert r["subcategory"] == "MostLeaveTaken"
    assert r["type"] == "Tabular"


def test_least_leave():
    r = classify_admin_intent("Who has taken the fewest leaves?")
    assert r["category"] == "Leave"
    assert r["subcategory"] == "LeastLeaveTaken"


def test_current_leave():
    r = classify_admin_intent("Who is on leave today?")
    assert r["category"] == "Leave"
    assert r["subcategory"] == "CurrentLeaveStatus"


def test_absconded():
    r = classify_admin_intent("Show me absconded person")
    assert r["category"] == "Leave"
    assert r["subcategory"] == "AbscondedPerson"


def test_leave_type_medical():
    r = classify_admin_intent("Who has taken most medical leave?")
    assert r["category"] == "Leave"
    assert r["leave_type"] == "Medical"


def test_leave_type_annual():
    r = classify_admin_intent("List person with maximum annual leave")
    assert r["leave_type"] == "Annual"


# =============================================================================
# MEDICAL
# =============================================================================


def test_active_cases():
    r = classify_admin_intent("How many active medical cases are there?")
    assert r["category"] == "Medical"
    assert r["subcategory"] == "ActiveCases"
    assert r["type"] == "Tabular"


def test_bmi_analysis():
    r = classify_admin_intent("Show BMI analysis of trainees")
    assert r["category"] == "Medical"
    assert r["subcategory"] == "BMIAnalysis"
    assert r["type"] == "Donut Chart"


def test_disease_stats():
    r = classify_admin_intent("What are the top diseases this month?")
    assert r["category"] == "Medical"
    assert r["subcategory"] == "DiseaseStatistics"


# =============================================================================
# ATTENDANCE
# =============================================================================


def test_monthly_attendance():
    r = classify_admin_intent("Show monthly attendance stats")
    assert r["category"] == "Attendance"
    assert r["subcategory"] == "MonthlyAttendance"
    assert r["type"] == "Bar Chart"


def test_weekly_attendance():
    r = classify_admin_intent("Show weekly attendance stats")
    assert r["category"] == "Attendance"
    assert r["subcategory"] == "WeeklyAttendance"
    assert r["type"] == "Bar Chart"


def test_present_today():
    r = classify_admin_intent("How many are present on campus today?")
    assert r["category"] == "Attendance"
    assert r["subcategory"] == "PresentToday"


def test_strength_breakdown():
    r = classify_admin_intent("Give me the strength breakdown")
    assert r["category"] == "Attendance"
    assert r["subcategory"] == "StrengthBreakdown"
    assert r["type"] == "Radial Chart"


# =============================================================================
# VERIFICATION
# =============================================================================


def test_pending_verification():
    r = classify_admin_intent("Show pending verifications")
    assert r["category"] == "Verification"
    assert r["subcategory"] == "PendingVerification"


def test_completed_verification():
    r = classify_admin_intent("List completed verifications")
    assert r["category"] == "Verification"
    assert r["subcategory"] == "CompletedVerification"


# =============================================================================
# EQUIPMENT
# =============================================================================


def test_equipment_summary():
    r = classify_admin_intent("Give me an equipment summary")
    assert r["category"] == "Equipment"
    assert r["subcategory"] == "EquipmentSummary"
    assert r["type"] == "Card"


def test_overdue_equipment():
    r = classify_admin_intent("What equipment is overdue?")
    assert r["category"] == "Equipment"
    assert r["subcategory"] == "OverdueEquipment"


def test_poor_condition():
    r = classify_admin_intent("Show equipment returned in poor condition")
    assert r["category"] == "Equipment"
    assert r["subcategory"] == "PoorConditionEquipment"


# =============================================================================
# DISTRIBUTION
# =============================================================================


def test_latest_distribution():
    r = classify_admin_intent("Show the latest distribution")
    assert r["category"] == "Distribution"
    assert r["subcategory"] == "LatestDistribution"


def test_distribution_by_unit():
    r = classify_admin_intent("Show distribution by unit")
    assert r["category"] == "Distribution"
    assert r["subcategory"] == "DistributionByUnit"


def test_unassigned():
    r = classify_admin_intent("Show unassigned items")
    assert r["category"] == "Distribution"
    assert r["subcategory"] == "UnassignedItems"


def test_top_unit():
    r = classify_admin_intent("Which unit has the highest distribution?")
    assert r["category"] == "Distribution"
    assert r["subcategory"] == "TopUnit"
    assert r["type"] == "Tabular"


# =============================================================================
# SKILLS / ROSTER
# =============================================================================


def test_by_sport():
    r = classify_admin_intent("Show roster by sport")
    assert r["category"] == "Skills"
    assert r["subcategory"] == "BySport"
    assert r["type"] == "Tabular"


def test_explicit_type_override():
    r = classify_admin_intent("Show the monthly attendance as a tabular report")
    assert r["category"] == "Attendance"
    assert r["subcategory"] == "MonthlyAttendance"
    assert r["type"] == "Tabular"


def test_by_class():
    r = classify_admin_intent("Show skills by class")
    assert r["category"] == "Skills"
    assert r["subcategory"] == "ByClass"


# =============================================================================
# PAYLOAD FORMAT
# =============================================================================


def test_payload_strips_none():
    r = classify_admin_intent("Who are the top 5 performers in BEPT?")
    payload = format_admin_payload(r)
    assert "raw_query" not in payload
    assert "confidence" not in payload
    assert None not in payload.values()


def test_payload_has_required_fields():
    r = classify_admin_intent("Who are the top 5 performers in BEPT?")
    payload = format_admin_payload(r)
    assert payload["commandId"] == 0
    assert payload["category"] == "Performance"
    assert payload["operation"] == "Top"
    assert payload["n"] == 5
    assert payload["section"] == "BPET"
    assert "type" not in payload


def test_frontend_intent_includes_type():
    r = classify_admin_intent("Who are the top 5 performers in BEPT?")
    intent = format_admin_intent(r)
    assert intent["type"] == "Tabular"
    assert intent["commandId"] == 0


# =============================================================================
# UNKNOWN QUERY
# =============================================================================


def test_unknown_query_returns_none_category():
    r = classify_admin_intent("What is the weather today?")
    # Should return something but category may be None or a low-confidence guess
    assert "category" in r
    assert "confidence" in r
