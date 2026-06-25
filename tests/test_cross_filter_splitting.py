import unittest

from intent_engine.query_planner import QueryType, plan_query


class TestCrossFilterSplitting(unittest.TestCase):

    def test_cross_filter_fever_splitting(self):
        plan = plan_query(
            "give me 10 agniveer who score highest in PPT and plays cricket and suffered from fever"
        )
        self.assertEqual(plan.query_type, QueryType.CROSS_FILTER)
        self.assertEqual(len(plan.operations), 3)
        categories = {op.intent_result["category"] for op in plan.operations}
        self.assertEqual(categories, {"Performance", "Skills", "Medical"})

        # Check that Skills op has sport == Cricket
        skills_op = next(
            op for op in plan.operations if op.intent_result["category"] == "Skills"
        )
        self.assertEqual(skills_op.intent_result.get("sport"), "Cricket")

    def test_cross_filter_injury_splitting(self):
        plan = plan_query(
            "show trainees with score above 80 in PPT and suffering from injury"
        )
        self.assertEqual(plan.query_type, QueryType.CROSS_FILTER)
        self.assertEqual(len(plan.operations), 2)
        categories = {op.intent_result["category"] for op in plan.operations}
        self.assertEqual(categories, {"Performance", "Medical"})

    def test_cross_filter_malaria_splitting(self):
        plan = plan_query(
            "show 5 agniveers who have high score in firing and who had malaria"
        )
        self.assertEqual(plan.query_type, QueryType.CROSS_FILTER)
        self.assertEqual(len(plan.operations), 2)
        categories = {op.intent_result["category"] for op in plan.operations}
        self.assertEqual(categories, {"Performance", "Medical"})


if __name__ == "__main__":
    unittest.main()
