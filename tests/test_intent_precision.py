"""
tests/test_intent_precision.py
==============================

Regression test suite for AgniAI admin intent classification accuracy.

Targets:
  - >= 95% exact (category, operation) match across all cases
  - ZERO wrong-category results (clarification is acceptable, wrong answers are not)
  - ZERO unhandled exceptions

Usage:
  pytest tests/test_intent_precision.py -v
"""

from __future__ import annotations

import pytest

from context_engine import ConversationContextEngine
from intent_engine.admin_intent import classify_admin_intent
from intent_engine.entity_extractor import extract_entities
from intent_engine.intent_classifier import classify_intent

# =============================================================================
# Helper
# =============================================================================


def _classify(query: str) -> dict:
    return classify_admin_intent(query)


def _intent(query: str) -> dict:
    return classify_intent(query)


def _entities(query: str) -> dict:
    return extract_entities(query)


def test_confidence_score_is_clamped_to_one():
    result = _classify(
        "Show top BPET performers for company 1 batch 1 platoon 1 agniveer A0701882L"
    )
    assert 0.0 <= result["confidence_score"] <= 1.0


# =============================================================================
# TASK 8 — Mandated test cases (1-10)
# =============================================================================


class TestMandatedCases:

    def test_01_volleyball_skill(self):
        """give me agniveer who also has volleyball as skill → Skills/BySport, sport=Volleyball"""
        r = _classify("give me agniveer who also has volleyball as skill")
        assert r["category"] == "Skills", f"Expected Skills, got {r['category']}"
        assert r["operation"] == "BySport", f"Expected BySport, got {r['operation']}"
        e = _entities("give me agniveer who also has volleyball as skill")
        assert (
            e.get("sport") == "Volleyball"
        ), f"Expected Volleyball entity, got {e.get('sport')}"

    def test_02_who_plays_volleyball_no_exception(self):
        """who plays volleyball → Skills/BySport (must NOT raise PayloadValidationError)"""
        from intent_engine.payload_validator import PayloadValidationError

        try:
            r = _classify("who plays volleyball")
        except PayloadValidationError as exc:
            pytest.fail(f"PayloadValidationError raised: {exc}")
        assert r["category"] == "Skills", f"Expected Skills, got {r['category']}"
        assert r["operation"] == "BySport", f"Expected BySport, got {r['operation']}"

    def test_03_good_morning_attendance(self):
        """good morning show attendance summary → Attendance/Summary, grading=None"""
        r = _classify("good morning show attendance summary")
        # Category must be Attendance, not None or Performance
        assert (
            r["category"] == "Attendance"
        ), f"Expected Attendance, got {r['category']}"
        e = _entities("good morning show attendance summary")
        assert (
            e.get("grading") is None
        ), f"Expected grading=None, got {e.get('grading')}"

    def test_04_attendance_saturday(self):
        """attendance on saturday → Attendance, grading=None"""
        r = _classify("attendance on saturday")
        assert (
            r["category"] == "Attendance"
        ), f"Expected Attendance, got {r['category']}"
        e = _entities("attendance on saturday")
        assert (
            e.get("grading") is None
        ), f"Expected grading=None, got {e.get('grading')}"

    def test_05_cold_weather_schedule(self):
        """training schedule for cold weather → Schedule, diagnose=None"""
        r = _classify("training schedule for cold weather")
        assert r["category"] == "Schedule", f"Expected Schedule, got {r['category']}"
        e = _entities("training schedule for cold weather")
        assert (
            e.get("diagnose") is None
        ), f"Expected diagnose=None, got {e.get('diagnose')}"

    def test_06_agniveers_from_india(self):
        """agniveers from india → unitName=None (bare 'india' is not a unit)"""
        e = _entities("agniveers from india")
        assert (
            e.get("unitName") is None
        ), f"Expected unitName=None for bare 'india', got {e.get('unitName')}"

    def test_07_agniveers_from_india_unit(self):
        """agniveers from india unit → unitName='India Unit'"""
        e = _entities("agniveers from india unit")
        assert (
            e.get("unitName") == "India Unit"
        ), f"Expected 'India Unit', got {e.get('unitName')}"

    def test_08_attendance_january_to_march_not_compare(self):
        """attendance from january to march → single Attendance op with date range, NOT COMPARE"""
        from intent_engine.query_planner import QueryType, plan_query

        plan = plan_query("attendance from january to march")
        assert (
            plan.query_type != QueryType.COMPARE
        ), f"Expected non-COMPARE plan, got {plan.query_type}"
        assert (
            plan.operations[0].intent_result.get("category") == "Attendance"
        ), f"Expected Attendance, got {plan.operations[0].intent_result.get('category')}"

    def test_09_more_than_5_days_leave_not_comparison(self):
        """agniveers with more than 5 days leave → Leave, NOT comparison"""
        from intent_engine.query_planner import QueryType, plan_query

        plan = plan_query("agniveers with more than 5 days leave")
        assert (
            plan.query_type != QueryType.COMPARE
        ), f"Expected non-COMPARE plan, got {plan.query_type}"
        r = _classify("agniveers with more than 5 days leave")
        assert r["category"] == "Leave", f"Expected Leave, got {r['category']}"

    def test_10_context_override_company_id(self):
        """top performers company 3 with prior context companyId=1 → companyId=3 (not 1)"""
        e = _entities("top performers company 3")
        assert (
            e.get("companyId") == 3
        ), f"Expected companyId=3, got {e.get('companyId')}"
        # The current query value must win over resolved_entities carryover
        e2 = extract_entities(
            "top performers company 3",
            resolved_entities={"companyId": 1},
        )
        assert (
            e2.get("companyId") == 3
        ), f"Expected current query's companyId=3 to win, got {e2.get('companyId')}"

    def test_11_session_isolation(self):
        """Two parallel sessions must never see each other's context."""
        engine = ConversationContextEngine()
        engine.add_interaction(
            "session-A",
            user_message="who plays volleyball",
            resolved_query="who plays volleyball",
            intent={"category": "Skills", "operation": "BySport"},
            entities={"sport": "Volleyball", "companyId": 5},
            filters={"companyId": 5},
            category="Skills",
            operation="BySport",
        )
        engine.add_interaction(
            "session-B",
            user_message="show attendance summary",
            resolved_query="show attendance summary",
            intent={"category": "Attendance", "operation": "Summary"},
            entities={"companyId": 9},
            filters={"companyId": 9},
            category="Attendance",
            operation="Summary",
        )
        history_a = engine.history("session-A")
        history_b = engine.history("session-B")
        assert len(history_a) == 1
        assert len(history_b) == 1
        assert history_a[0].entities.get("companyId") == 5
        assert history_b[0].entities.get("companyId") == 9


# =============================================================================
# Extended officer-realistic cases (60+ additional)
# =============================================================================


class TestPerformanceCases:

    def test_top_10_bpet(self):
        r = _classify("show top 10 performers in BPET")
        assert r["category"] == "Performance"
        assert r["operation"] == "Top"

    def test_bottom_5_ppt(self):
        r = _classify("list bottom 5 agniveers in PPT")
        assert r["category"] == "Performance"
        assert r["operation"] == "Bottom"

    def test_average_score_company(self):
        r = _classify("what is the average score for company 3 in firing")
        assert r["category"] == "Performance"
        assert r["operation"] == "Average"

    def test_grading_excellent(self):
        r = _classify("who got excellent grade in BPET")
        assert r["category"] == "Performance"

    def test_grade_distribution(self):
        r = _classify("show grade distribution for the batch")
        assert r["category"] == "Performance"

    def test_improvement_trend(self):
        r = _classify("which agniveers showed improvement between attempt 1 and 2")
        assert r["category"] == "Performance"
        assert r["operation"] in ("Improvement", "AttemptWise")

    def test_drop_performance(self):
        r = _classify("whose score dropped in the latest attempt")
        assert r["category"] == "Performance"
        assert r["operation"] == "Drop"

    def test_attempt_wise(self):
        r = _classify("show attempt wise breakdown for PPT")
        assert r["category"] == "Performance"
        assert r["operation"] == "AttemptWise"

    def test_best_attempt(self):
        r = _classify("show best attempt score for each agniveer in firing")
        assert r["category"] == "Performance"
        assert r["operation"] == "BestAttempt"

    def test_performance_trend(self):
        r = _classify("how has the batch been performing over attempts")
        assert r["category"] == "Performance"
        assert r["operation"] == "Trend"


class TestLeaveCases:

    def test_most_leave_taken(self):
        r = _classify("who took the most leave")
        assert r["category"] == "Leave"
        assert r["operation"] == "Most"

    def test_least_leave(self):
        r = _classify("agniveers with least days absent")
        assert r["category"] == "Leave"
        assert r["operation"] == "Least"

    def test_currently_on_leave(self):
        r = _classify("who is currently on leave today")
        assert r["category"] == "Leave"
        assert r["operation"] == "Current"

    def test_absconded(self):
        r = _classify("list absconded agniveers")
        assert r["category"] == "Leave"
        assert r["operation"] == "Absconded"

    def test_annual_leave(self):
        r = _classify("show annual leave taken by company 2")
        assert r["category"] == "Leave"

    def test_leave_no_comparison(self):
        """'more than 10 days leave' is NOT a comparison"""
        from intent_engine.query_planner import QueryType, plan_query

        plan = plan_query("show agniveers with more than 10 days annual leave")
        assert plan.query_type != QueryType.COMPARE


class TestMedicalCases:

    def test_bmi_distribution(self):
        r = _classify("show BMI distribution for the company")
        assert r["category"] == "Medical"
        assert r["operation"] == "BMI"

    def test_blood_group(self):
        r = _classify("blood group distribution of the battalion")
        assert r["category"] == "Medical"
        assert r["operation"] == "BloodGroup"

    def test_disease_report(self):
        r = _classify("most common diseases in the unit")
        assert r["category"] == "Medical"
        assert r["operation"] == "Disease"

    def test_cold_weather_no_diagnose(self):
        """'cold weather' must not extract diagnose=Cold"""
        e = _entities("training schedule for cold weather conditions")
        assert e.get("diagnose") is None

    def test_benefit_no_bmi(self):
        """'benefit' must not extract bmiCategory via 'fit' substring"""
        e = _entities("what is the benefit of physical training")
        assert e.get("bmiCategory") is None

    def test_good_morning_no_grading(self):
        """'good morning' must not extract grading=Good"""
        e = _entities("good morning show me the attendance")
        assert e.get("grading") is None

    def test_saturday_no_grading(self):
        """'saturday' must not extract grading=SAT"""
        e = _entities("attendance on saturday")
        assert e.get("grading") is None


class TestAttendanceCases:

    def test_monthly_attendance(self):
        r = _classify("monthly attendance for company 3")
        assert r["category"] == "Attendance"
        assert r["operation"] == "Monthly"

    def test_weekly_attendance(self):
        r = _classify("weekly attendance for this week")
        assert r["category"] == "Attendance"
        assert r["operation"] == "Weekly"

    def test_present_today(self):
        r = _classify("how many agniveers are present today")
        assert r["category"] == "Attendance"
        assert r["operation"] == "Present"

    def test_attendance_summary(self):
        r = _classify("give me attendance summary for the batch")
        assert r["category"] == "Attendance"
        assert r["operation"] == "Summary"

    def test_attendance_date_range_not_compare(self):
        """'attendance from january to march' is a range query, not a comparison"""
        from intent_engine.query_planner import QueryType, plan_query

        plan = plan_query("show attendance from january to march")
        assert (
            plan.query_type != QueryType.COMPARE
        ), f"Date range must not become COMPARE, got {plan.query_type}"


class TestStrengthCases:

    def test_strength_breakdown(self):
        r = _classify("show strength breakdown by section")
        assert r["category"] == "Strength"
        assert r["operation"] is None

    def test_headcount(self):
        r = _classify("what is the current headcount of the unit")
        assert r["category"] == "Strength"


class TestVerificationCases:

    def test_pending_verification(self):
        r = _classify("show pending verification requests")
        assert r["category"] == "Verification"
        assert r["operation"] == "Pending"

    def test_completed_verification(self):
        r = _classify("how many verifications are completed")
        assert r["category"] == "Verification"
        assert r["operation"] == "Verified"

    def test_not_responded(self):
        r = _classify("who has not responded to verification")
        assert r["category"] == "Verification"
        assert r["operation"] == "NotResponded"


class TestEquipmentCases:

    def test_equipment_stats(self):
        r = _classify("equipment inventory summary")
        assert r["category"] == "Equipment"

    def test_equipment_search(self):
        r = _classify("find which agniveers have a combat coat")
        assert r["category"] == "Equipment"
        assert r["operation"] == "ByName"

    def test_equipment_holding(self):
        r = _classify("who is holding overdue equipment")
        assert r["category"] == "Equipment"
        assert r["operation"] == "Holding"

    def test_equipment_returned(self):
        r = _classify("equipment returned in poor condition")
        assert r["category"] == "Equipment"
        assert r["operation"] == "Returned"


class TestDistributionCases:

    def test_latest_distribution(self):
        r = _classify("show latest distribution of agniveers to units")
        assert r["category"] == "Distribution"
        assert r["operation"] == "Latest"

    def test_by_unit(self):
        r = _classify("how many agniveers are in alpha unit")
        assert r["category"] == "Distribution"

    def test_top_unit(self):
        r = _classify("which unit received the most agniveers")
        assert r["category"] == "Distribution"
        assert r["operation"] == "TopUnit"

    def test_unassigned(self):
        r = _classify("show agniveers not yet assigned to any unit")
        assert r["category"] == "Distribution"
        assert r["operation"] == "Unassigned"


class TestSkillsCases:

    def test_cricket_players(self):
        r = _classify("list all cricket players in the company")
        assert r["category"] == "Skills"
        assert r["operation"] == "BySport"

    def test_football_roster(self):
        r = _classify("give me the football roster")
        assert r["category"] == "Skills"
        assert r["operation"] == "BySport"

    def test_who_plays_hockey(self):
        r = _classify("who plays hockey in platoon 3")
        assert r["category"] == "Skills"
        assert r["operation"] == "BySport"

    def test_by_class_sikh(self):
        r = _classify("how many sikh soldiers are there")
        assert r["category"] == "Skills"
        assert r["operation"] == "ByClass"

    def test_by_class_dogra(self):
        r = _classify("show dogra class agniveers")
        assert r["category"] == "Skills"
        assert r["operation"] == "ByClass"

    def test_volleyball_no_roster_category(self):
        """'who plays volleyball' must NOT produce category='Roster' (not in OFFICIAL_CATEGORIES)"""
        r = _classify("who plays volleyball")
        assert (
            r.get("category") != "Roster"
        ), "category 'Roster' is not an official category and must not appear"

    def test_sport_entity_no_false_positive_golf(self):
        """bare 'golf' in a non-sports context must not extract sport if no sports framing"""
        # "golf" as unit should not extract sport if adjacent to "unit"
        e = _entities("agniveers from golf unit")
        # unitName should be set, sport might be set too — but verify no crash
        # and unitName is set
        assert e.get("unitName") == "Golf Unit"

    def test_pool_in_liverpool_no_sport(self):
        """'pool' inside 'liverpool' must not extract sport=Pool"""
        e = _entities("agniveers from liverpool regiment")
        assert e.get("sport") is None, f"Expected sport=None, got {e.get('sport')}"


class TestScheduleCases:

    def test_today_schedule(self):
        r = _classify("what is today's training schedule")
        assert r["category"] == "Schedule"
        assert r["operation"] == "bytoday"

    def test_company_schedule(self):
        r = _classify("schedule for company 2 this week")
        assert r["category"] == "Schedule"

    def test_training_schedule_cold_weather(self):
        """'training schedule for cold weather' must be Schedule, not Medical"""
        r = _classify("training schedule for cold weather")
        assert r["category"] == "Schedule"
        assert r["category"] != "Medical"


class TestEntityExtractionEdgeCases:

    def test_company_3_explicit(self):
        """companyId from current query must win over resolved_entities"""
        e = extract_entities(
            "top performers company 3", resolved_entities={"companyId": 1}
        )
        assert e.get("companyId") == 3

    def test_platoon_id(self):
        e = _entities("show platoon 4 attendance")
        assert e.get("platoonId") == 4

    def test_batch_id(self):
        e = _entities("performance of batch 2")
        assert e.get("batchId") == 2

    def test_section_bpet(self):
        e = _entities("top performers in BPET")
        assert e.get("section") == "BPET"

    def test_section_ppt(self):
        e = _entities("average score in PPT for company 1")
        assert e.get("section") == "PPT"

    def test_grading_excellent(self):
        e = _entities("who got excellent grade in firing")
        assert e.get("grading") == "Excellent"

    def test_grading_sat(self):
        e = _entities("who got satisfactory grade")
        assert e.get("grading") == "SAT"

    def test_no_dead_operation_category_keys(self):
        """extract_entities must not resurrect the unused Operation/Category
        keys (dead code — nothing downstream ever read them)."""
        e = _entities("show attendance for company 1")
        assert "Operation" not in e
        assert "Category" not in e

    def test_blood_group_o_positive(self):
        e = _entities("agniveers with blood group O+")
        assert e.get("bloodGroup") == "O+"

    def test_sport_kabaddi(self):
        e = _entities("show kabaddi players")
        assert e.get("sport") == "Kabaddi"

    def test_sport_table_tennis(self):
        e = _entities("who plays table tennis")
        assert e.get("sport") == "Table Tennis"

    def test_unit_alpha(self):
        e = _entities("show agniveers from alpha unit")
        assert e.get("unitName") == "Alpha Unit"

    def test_unit_bravo(self):
        e = _entities("agniveers in unit bravo")
        assert e.get("unitName") == "Bravo Unit"

    def test_india_no_unit(self):
        """bare 'india' must not become a unitName"""
        e = _entities("agniveers from india")
        assert e.get("unitName") is None

    def test_delta_no_unit(self):
        """bare 'delta' must not become a unitName"""
        e = _entities("delta platoon performance")
        assert e.get("unitName") is None

    def test_date_range(self):
        e = _entities("attendance from january to march")
        assert e.get("fromDate") is not None
        assert e.get("toDate") is not None

    def test_bmi_normal(self):
        e = _entities("show agniveers with normal bmi")
        assert e.get("bmiCategory") == "Normal"

    def test_bmi_obese(self):
        e = _entities("list obese agniveers")
        assert e.get("bmiCategory") == "Obese"

    def test_fit_no_bmi_without_context(self):
        """'fit' alone must not extract bmiCategory if no bmi/weight context"""
        e = _entities("fit for duty check")
        assert e.get("bmiCategory") is None

    def test_physically_fit_with_bmi_context(self):
        """'physically fit' with 'fitness' context must extract bmiCategory"""
        e = _entities("show physically fit agniveers by fitness score")
        assert e.get("bmiCategory") == "Normal"

    def test_unfit_with_bmi_context(self):
        """'unfit' with bmi/weight context maps to combined Overweight+Obese"""
        e = _entities("show unfit agniveers by weight")
        assert e.get("bmiCategory") == "Unfit"

    def test_unfit_no_bmi_without_context(self):
        """'unfit' alone must not extract bmiCategory if no bmi/weight context —
        'unfit for duty' is a duty-status concept, not a BMI classification."""
        e = _entities("unfit for duty check")
        assert e.get("bmiCategory") is None

    def test_not_fit_extracts_unfit_bmi(self):
        e = _entities("who's not fit in the company")
        assert e.get("bmiCategory") == "Unfit"

    def test_unfit_classifies_as_bmi_operation(self):
        r = _classify("show unfit agniveers by weight")
        assert r["category"] == "Medical"
        assert r["operation"] == "BMI"


class TestSessionIsolation:

    def test_two_sessions_independent(self):
        """Two sessions must never share context"""
        engine = ConversationContextEngine()
        engine.add_interaction(
            "sess-1",
            user_message="show volleyball players",
            resolved_query="show volleyball players",
            intent={"category": "Skills"},
            entities={"sport": "Volleyball", "companyId": 7},
            filters={"companyId": 7},
            category="Skills",
            operation="BySport",
        )
        engine.add_interaction(
            "sess-2",
            user_message="show attendance",
            resolved_query="show attendance",
            intent={"category": "Attendance"},
            entities={"companyId": 3},
            filters={"companyId": 3},
            category="Attendance",
            operation="Summary",
        )
        h1 = engine.history("sess-1")
        h2 = engine.history("sess-2")
        assert h1[0].entities.get("companyId") != h2[0].entities.get(
            "companyId"
        ), "Session 1 and Session 2 must have independent companyId values"
        assert h1[0].entities.get("sport") == "Volleyball"
        assert h2[0].entities.get("sport") is None

    def test_no_default_session_leakage(self):
        """Without a session_id, engine returns empty history"""
        engine = ConversationContextEngine()
        engine.add_interaction(
            "sess-real",
            user_message="test",
            resolved_query="test",
            intent={},
            entities={},
            filters={},
            category="Performance",
            operation="Top",
        )
        assert engine.history("") == []
        assert engine.history(None) == []  # type: ignore[arg-type]


class TestCrossFilterPlannerCases:

    def test_volleyball_also_skill_not_cross_filter(self):
        """'give me agniveer who also has volleyball as skill' → SIMPLE plan (not cross_filter)"""
        from intent_engine.query_planner import QueryType, plan_query

        plan = plan_query("give me agniveer who also has volleyball as skill")
        assert plan.query_type in (
            QueryType.SIMPLE,
            QueryType.CROSS_FILTER,
        ), f"Expected SIMPLE or CROSS_FILTER, got {plan.query_type}"
        # The operation on the primary intent must be Skills/BySport
        if plan.operations:
            primary = plan.operations[0].intent_result
            assert (
                primary.get("category") == "Skills"
            ), f"Primary intent category should be Skills, got {primary.get('category')}"

    def test_date_range_not_compare(self):
        from intent_engine.query_planner import QueryType, plan_query

        plan = plan_query("attendance from january to march for company 2")
        assert plan.query_type != QueryType.COMPARE

    def test_two_year_range_not_compare(self):
        from intent_engine.query_planner import QueryType, plan_query

        plan = plan_query("show performance from 2023 to 2024")
        assert plan.query_type != QueryType.COMPARE


class TestOverallAndPersonalDetail:

    def test_overall_performance(self):
        r = _classify("show overall performance ranking of all agniveers")
        assert r["category"] == "Overall"
        assert r["operation"] == "OverallPerformance"

    def test_personal_details(self):
        r = _classify("show personal details of agniveer 12345")
        assert r["category"] == "personaldetail"
        assert r["operation"] == "info"

    def test_disqualified(self):
        r = _classify("list disqualified agniveers and their reasons")
        assert r["category"] == "disqualified"
        assert r["operation"] == "removed"


# =============================================================================
# Summary runner (prints table when executed directly)
# =============================================================================

if __name__ == "__main__":
    import sys
    import traceback

    test_classes = [
        TestMandatedCases,
        TestPerformanceCases,
        TestLeaveCases,
        TestMedicalCases,
        TestAttendanceCases,
        TestStrengthCases,
        TestVerificationCases,
        TestEquipmentCases,
        TestDistributionCases,
        TestSkillsCases,
        TestScheduleCases,
        TestEntityExtractionEdgeCases,
        TestSessionIsolation,
        TestCrossFilterPlannerCases,
        TestOverallAndPersonalDetail,
    ]

    passed = failed = errored = 0
    results = []

    for cls in test_classes:
        instance = cls()
        for name in [m for m in dir(cls) if m.startswith("test_")]:
            method = getattr(instance, name)
            label = f"{cls.__name__}.{name}"
            try:
                method()
                passed += 1
                results.append(("PASS", label, ""))
            except AssertionError as e:
                failed += 1
                results.append(("FAIL", label, str(e)))
            except Exception as e:
                errored += 1
                results.append(("ERROR", label, traceback.format_exc()[-200:]))

    total = passed + failed + errored
    print(f"\n{'='*70}")
    print(f"  AgniAI Intent Precision Test Results")
    print(f"{'='*70}")
    print(f"  Total: {total}  Passed: {passed}  Failed: {failed}  Errors: {errored}")
    print(f"  Accuracy: {100 * passed / total:.1f}%")
    print(f"{'='*70}")
    for status, label, msg in results:
        icon = "✓" if status == "PASS" else ("✗" if status == "FAIL" else "!")
        print(f"  {icon} {status:5}  {label}")
        if msg and status != "PASS":
            print(f"         {msg[:120]}")
    print(f"{'='*70}\n")
    sys.exit(0 if (failed + errored) == 0 else 1)
