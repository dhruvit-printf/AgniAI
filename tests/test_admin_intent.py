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
from conversation_detector import is_conversational_query

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
    assert r["category"] == "Overall"
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


def test_month_year_attendance_defaults_to_monthly():
    r = classify_admin_intent("Give June 2026 attendance of Agniveer A0701558N.")
    assert r["category"] == "Attendance"
    assert r["subcategory"] == "MonthlyAttendance"
    assert r["type"] == "Bar Chart"
    assert r["date"] == "June 2026"


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
    assert r["category"] == "Strength"
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
    assert r["category"] == "Roster"
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


def test_disclaimer_banner_is_not_verification_intent():
    text = "AgniAI may make mistakes. Verify important information."
    assert is_conversational_query(text) is True

    r = classify_admin_intent(text)
    assert r["category"] is None
    assert r["subcategory"] is None
    assert r["operation"] is None


# =============================================================================
# AUDIT & SYNONYMS
# =============================================================================


def test_new_operations():
    # Sent verification
    r1 = classify_admin_intent("Show verifications sent today")
    assert r1["category"] == "Verification"
    assert r1["subcategory"] == "SentVerification"
    p1 = format_admin_payload(r1)
    assert p1["operation"] == "Sent"

    # Holding equipment
    r2 = classify_admin_intent("Who is holding equipment?")
    assert r2["category"] == "Equipment"
    assert r2["subcategory"] == "HoldingEquipment"
    p2 = format_admin_payload(r2)
    assert p2["operation"] == "Holding"

    # AgniveerWise equipment
    r3 = classify_admin_intent("Show agniveer wise equipment report")
    assert r3["category"] == "Equipment"
    assert r3["subcategory"] == "AgniveerWiseEquipment"
    p3 = format_admin_payload(r3)
    assert p3["operation"] == "AgniveerWise"

    # Individual medical
    r4 = classify_admin_intent("Show medical details of agniveer 12345")
    assert r4["category"] == "Medical"
    assert r4["subcategory"] == "IndividualMedical"
    p4 = format_admin_payload(r4)
    assert p4["operation"] == "Individual"
    assert p4["agniveerNo"] == "12345"

    # Yearly attendance
    r5 = classify_admin_intent("Show yearly attendance stats")
    assert r5["category"] == "Attendance"
    assert r5["subcategory"] == "YearlyAttendance"
    p5 = format_admin_payload(r5)
    assert p5["operation"] == "Yearly"

    # Attendance summary
    r6 = classify_admin_intent("Show attendance summary")
    assert r6["category"] == "Attendance"
    assert r6["subcategory"] == "AttendanceSummary"
    p6 = format_admin_payload(r6)
    assert p6["operation"] == "Summary"


def test_synonyms():
    # highest -> Top, best -> Top
    r1 = classify_admin_intent("Who got the highest score in BPET?")
    assert r1["category"] == "Performance"
    assert r1["subcategory"] == "TopPerformers"
    p1 = format_admin_payload(r1)
    assert p1["operation"] == "Top"

    r2 = classify_admin_intent("Who is the best performer in PPT?")
    assert r2["category"] == "Performance"
    assert r2["subcategory"] == "TopPerformers"
    p2 = format_admin_payload(r2)
    assert p2["operation"] == "Top"

    # lowest -> Bottom
    r3 = classify_admin_intent("Who got the lowest marks in DRILL?")
    assert r3["category"] == "Performance"
    assert r3["subcategory"] == "LowestPerformers"
    p3 = format_admin_payload(r3)
    assert p3["operation"] == "Bottom"

    # compare & vs -> Compare
    r4 = classify_admin_intent("BPET vs PPT")
    assert r4["category"] == "Performance"
    assert r4["subcategory"] == "Comparison"
    p4 = format_admin_payload(r4)
    assert p4["operation"] == "Compare"

    # annual attendance -> Yearly
    r5 = classify_admin_intent("Show annual attendance")
    assert r5["category"] == "Attendance"
    assert r5["subcategory"] == "YearlyAttendance"
    p5 = format_admin_payload(r5)
    assert p5["operation"] == "Yearly"

    # football -> BySport
    r6 = classify_admin_intent("Show roster for football")
    assert r6["category"] == "Roster"
    assert r6["subcategory"] == "BySport"
    assert r6["sport"] == "Football"
    p6 = format_admin_payload(r6)
    assert p6["operation"] == "BySport"
    assert p6["sport"] == "Football"

    # obese -> BMI
    r7 = classify_admin_intent("Show obese trainees")
    assert r7["category"] == "Medical"
    assert r7["subcategory"] == "BMIAnalysis"
    assert r7["bmi_category"] == "Obese"
    p7 = format_admin_payload(r7)
    assert p7["operation"] == "BMI"
    assert p7["bmiCategory"] == "Obese"

    # pending police verification -> Pending
    r8 = classify_admin_intent("Show pending police verification")
    assert r8["category"] == "Verification"
    assert r8["subcategory"] == "PendingVerification"
    p8 = format_admin_payload(r8)
    assert p8["operation"] == "Pending"

    # unit alpha -> ByUnit
    r9 = classify_admin_intent("Show distribution for unit alpha")
    assert r9["category"] == "Distribution"
    assert r9["subcategory"] == "DistributionByUnit"
    assert r9["unit_name"] == "Alpha Unit"
    p9 = format_admin_payload(r9)
    assert p9["operation"] == "ByUnit"
    assert p9["unitName"] == "Alpha Unit"


def test_parameter_propagation():
    # blood_group
    r1 = classify_admin_intent("Show medical stats for blood group O+")
    assert r1["category"] == "Medical"
    assert r1["blood_group"] == "O+"
    p1 = format_admin_payload(r1)
    assert p1["bloodGroup"] == "O+"

    # item_name
    r2 = classify_admin_intent("Who was issued pt shoes brown?")
    assert r2["category"] == "Equipment"
    assert r2["item_name"] == "Pt Shoes Brown"
    p2 = format_admin_payload(r2)
    assert p2["equipmentName"] == "Pt Shoes Brown"

    # unit_name
    r3 = classify_admin_intent("Show attendance for bravo unit")
    assert r3["category"] == "Attendance"
    assert r3["unit_name"] == "Bravo Unit"
    p3 = format_admin_payload(r3)
    assert p3["unitName"] == "Bravo Unit"

    # medical_status
    r4 = classify_admin_intent("Show active medical cases")
    assert r4["category"] == "Medical"
    assert r4["medical_status"] == "Active"
    p4 = format_admin_payload(r4)
    assert p4["medicalStatus"] == "Active"


def test_auto_detect_agniveer_no():
    # Test that the Agniveer ID pattern A0701515Y is automatically detected without the prefix 'agniveer no'
    r1 = classify_admin_intent("Show details for A0701515Y")
    assert r1["agniveer_no"] == "A0701515Y"
    p1 = format_admin_payload(r1)
    assert p1["agniveerNo"] == "A0701515Y"

    r2 = classify_admin_intent("Is B1234567Z on leave?")
    assert r2["agniveer_no"] == "B1234567Z"
    p2 = format_admin_payload(r2)
    assert p2["agniveerNo"] == "B1234567Z"


def test_id_vs_count_exclusion():
    # "show top performers in section 123" -> number is None, not 123
    r1 = classify_admin_intent("show top performers in section 123")
    assert r1["number"] is None

    # "show agniveer 12345 performance" -> agniveer_no is "12345", number is None
    r2 = classify_admin_intent("show agniveer 12345 performance")
    assert r2["agniveer_no"] == "12345"
    assert r2["number"] is None

    # "show top 5 agniveers in section 123" -> number is 5
    r3 = classify_admin_intent("show top 5 agniveers in section 123")
    assert r3["number"] == 5


def test_subcategory_override_canonical_names():
    # Test subcategory override mapping for poor condition equipment items
    r = classify_admin_intent("show pt shoes in poor condition")
    assert r["subcategory"] == "PoorConditionEquipment"
    assert r["item_name"] == "Pt Shoes"
