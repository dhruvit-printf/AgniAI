import unittest
from query_planner import plan_query, QueryType


class TestQueryPlanner(unittest.TestCase):

    def test_simple_query(self):
        plan = plan_query("Show top 10 performers in PPT")
        self.assertEqual(plan.query_type, QueryType.SIMPLE)
        self.assertTrue(len(plan.operations) == 1)

    def test_cross_filter_query(self):
        plan = plan_query("Show top performer in PPT who plays cricket")
        self.assertEqual(plan.query_type, QueryType.CROSS_FILTER)
        self.assertEqual(len(plan.operations), 2)
        self.assertEqual(plan.operations[0].intent_result["category"], "Performance")
        self.assertEqual(plan.operations[1].intent_result["category"], "Skills")

    def test_comparison_query(self):
        plan = plan_query("Compare leave status and medical cases")
        self.assertEqual(plan.query_type, QueryType.COMPARISON)
        self.assertEqual(len(plan.operations), 2)

    def test_multi_independent_query(self):
        plan = plan_query("Show attendance stats as well as equipment overdue records")
        self.assertEqual(plan.query_type, QueryType.MULTI_INDEPENDENT)
        self.assertEqual(len(plan.operations), 2)

    def test_no_split_phrase_guard(self):
        plan = plan_query("show approved and pending leave records")
        self.assertEqual(plan.query_type, QueryType.SIMPLE)


if __name__ == "__main__":
    unittest.main()