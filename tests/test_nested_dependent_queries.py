import unittest

from intent_engine.query_planner import QueryType, plan_query
from query_understanding_engine import understand_query


class TestNestedDependentQueries(unittest.TestCase):
    """Dependent/nested queries — a clause introduced by a connector like
    "additionally"/"then"/"also" that refers back to the primary request's
    result set ("any of them", "check whether ...") must be intersected
    against that result set, not executed as an independent second request.
    """

    def test_backreference_clause_is_dependent_not_independent(self):
        semantic = understand_query(
            "Please provide the top 10 BPET performers. Additionally check "
            "whether any of them have rejected police verification."
        )
        self.assertTrue(semantic["dependent_intent"])
        self.assertEqual(semantic["query_type"], "cross_filter")

    def test_dependent_query_plans_as_cross_filter(self):
        plan = plan_query(
            "Please provide the top 10 BPET performers. Additionally check "
            "whether any of them have rejected police verification."
        )
        self.assertEqual(plan.query_type, QueryType.CROSS_FILTER)
        self.assertEqual(len(plan.operations), 2)
        categories = {op.intent_result["category"] for op in plan.operations}
        self.assertEqual(categories, {"Performance", "Verification"})
        performance_op = next(
            op
            for op in plan.operations
            if op.intent_result["category"] == "Performance"
        )
        self.assertEqual(performance_op.intent_result["operation"], "Top")
        self.assertEqual(performance_op.intent_result["number"], 10)
        verification_op = next(
            op
            for op in plan.operations
            if op.intent_result["category"] == "Verification"
        )
        self.assertEqual(verification_op.intent_result["operation"], "Rejected")

    def test_three_clause_dependent_chain_splits_into_three_legs(self):
        plan = plan_query(
            "Find the top 20 performers. Then tell me which of them are "
            "medically unfit. Also show their attendance."
        )
        self.assertEqual(plan.query_type, QueryType.CROSS_FILTER)
        self.assertEqual(len(plan.operations), 3)
        categories = {op.intent_result["category"] for op in plan.operations}
        self.assertEqual(categories, {"Performance", "Medical", "Attendance"})

    def test_independent_multi_request_still_unaffected(self):
        # No back-reference — genuinely independent legs must still classify
        # as multi_independent, not get swept into the new dependent path.
        plan = plan_query("Show attendance stats as well as equipment overdue records")
        self.assertEqual(plan.query_type, QueryType.MULTI_INDEPENDENT)
        self.assertEqual(len(plan.operations), 2)

    def test_ordinary_cross_filter_still_unaffected(self):
        semantic = understand_query("Show top performer in PPT who plays cricket")
        self.assertFalse(semantic["dependent_intent"])
        self.assertTrue(semantic["cross_filter_intent"])
        self.assertEqual(semantic["query_type"], "cross_filter")


if __name__ == "__main__":
    unittest.main()
