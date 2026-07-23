import unittest

import query_understanding_engine
from intent_engine import query_planner
from intent_engine.query_planner import QueryType, plan_query


class TestQueryPlanner(unittest.TestCase):

    def test_confidence_is_clamped_to_one(self):
        plan = plan_query(
            "Show top BPET performers",
            semantic={"query_type": "simple", "confidence": 1.7},
        )
        self.assertLessEqual(plan.confidence, 1.0)

    def test_comparison_markers_share_single_source(self):
        """Regression guard: query_planner._COMPARISON_KEYWORDS must stay an
        import of query_understanding_engine._COMPARISON_MARKERS, not an
        independently re-typed copy that can silently re-diverge."""
        self.assertIs(
            query_planner._COMPARISON_KEYWORDS,
            query_understanding_engine._COMPARISON_MARKERS,
        )

    def test_simple_query(self):
        plan = plan_query("Show top 10 performers in PPT")
        self.assertEqual(plan.query_type, QueryType.FILTER_QUERY)
        self.assertTrue(len(plan.operations) == 1)

    def test_cross_filter_query(self):
        plan = plan_query("Show top performer in PPT who plays cricket")
        self.assertEqual(plan.query_type, QueryType.CROSS_FILTER)
        self.assertEqual(len(plan.operations), 2)
        self.assertEqual(plan.operations[0].intent_result["category"], "Performance")
        self.assertEqual(plan.operations[1].intent_result["sport"], "Cricket")

    def test_comparison_query(self):
        plan = plan_query("Compare leave status and medical cases")
        self.assertEqual(plan.query_type, QueryType.COMPARISON)
        self.assertEqual(len(plan.operations), 2)

    def test_multi_independent_query(self):
        plan = plan_query("Show attendance stats as well as equipment overdue records")
        self.assertEqual(plan.query_type, QueryType.MULTI_OPERATION)
        self.assertEqual(len(plan.operations), 2)

    def test_independent_and_query_with_bmi_section(self):
        plan = plan_query("Show top performers and BMI analysis")
        self.assertEqual(plan.query_type, QueryType.MULTI_OPERATION)
        self.assertEqual(len(plan.operations), 2)
        self.assertEqual(plan.operations[0].intent_result["category"], "Performance")
        self.assertEqual(plan.operations[1].intent_result["category"], "Medical")

    def test_leave_and_strength_breakdown_is_multi_independent(self):
        plan = plan_query("Show current leave records and strength breakdown")
        self.assertEqual(plan.query_type, QueryType.MULTI_OPERATION)
        self.assertEqual(len(plan.operations), 2)
        self.assertEqual(plan.operations[0].intent_result["category"], "Leave")
        self.assertEqual(plan.operations[1].intent_result["category"], "Strength")

    def test_same_agniveer_multi_section_request_is_multi_independent(self):
        plan = plan_query("Show attendance and current leave records for agniveer 12345")
        self.assertEqual(plan.query_type, QueryType.MULTI_OPERATION)
        self.assertEqual(len(plan.operations), 2)
        self.assertEqual(plan.operations[0].intent_result["category"], "Attendance")
        self.assertEqual(plan.operations[1].intent_result["category"], "Leave")

    def test_leave_attendance_equipment_comma_separated_multi_independent(self):
        plan = plan_query(
            "Show current leave records, attendance status, and equipment overdue records."
        )
        self.assertEqual(plan.query_type, QueryType.MULTI_OPERATION)
        self.assertEqual(len(plan.operations), 3)
        self.assertEqual(plan.operations[0].intent_result["category"], "Leave")
        self.assertEqual(plan.operations[1].intent_result["category"], "Attendance")
        self.assertEqual(plan.operations[2].intent_result["category"], "Equipment")

    def test_no_split_phrase_guard(self):
        plan = plan_query("show approved and pending leave records")
        self.assertEqual(plan.query_type, QueryType.FILTER_QUERY)

    def test_disqualified_agniveer_simple_query(self):
        plan = plan_query("Show disqualified agniveers list")
        self.assertEqual(plan.query_type, QueryType.FILTER_QUERY)
        self.assertEqual(len(plan.operations), 1)
        self.assertEqual(plan.operations[0].intent_result["category"], "disqualified")

    def test_disqualified_and_personal_details_split(self):
        plan = plan_query("Show disqualified agniveers and personal details for A0701515Y")
        self.assertEqual(plan.query_type, QueryType.MULTI_OPERATION)
        self.assertEqual(len(plan.operations), 2)
        self.assertEqual(plan.operations[0].intent_result["category"], "disqualified")
        self.assertEqual(plan.operations[1].intent_result["category"], "personaldetail")


if __name__ == "__main__":
    unittest.main()
