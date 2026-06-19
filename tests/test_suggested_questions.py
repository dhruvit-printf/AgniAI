import unittest

from response_builder import build_response
from suggested_questions import generate_suggested_questions


class TestSuggestedQuestions(unittest.TestCase):

    def test_suggested_questions_by_category(self):
        # 1. Performance Category
        intent_perf = {"category": "Performance"}
        questions_perf = generate_suggested_questions("filter_query", intent_perf, {})
        self.assertEqual(len(questions_perf), 4)
        self.assertIn("Who are the lowest performers in this section?", questions_perf)

        # 2. Leave Category
        intent_leave = {"category": "Leave"}
        questions_leave = generate_suggested_questions("filter_query", intent_leave, {})
        self.assertEqual(len(questions_leave), 4)
        self.assertIn("Who took the most leaves this month?", questions_leave)

        # 3. Unrecognized Category
        intent_unknown = {"category": "Greeting"}
        questions_unknown = generate_suggested_questions("greeting", intent_unknown, {})
        self.assertEqual(questions_unknown, [])

        # 4. (category, subcategory) override
        intent_perf_top = {"category": "Performance", "subcategory": "TopPerformers"}
        questions_top = generate_suggested_questions(
            "filter_query", intent_perf_top, {}
        )
        self.assertEqual(len(questions_top), 4)
        self.assertIn(
            "Compare this section's top performers with another section.",
            questions_top,
        )

        # 5. query_type override
        questions_comp = generate_suggested_questions(
            "comparison", {"category": "Skills"}, {}
        )
        self.assertEqual(len(questions_comp), 4)
        self.assertIn("Show side-by-side metric distributions.", questions_comp)

    def test_build_response_payload_suggestions(self):
        # Verify that build_response correctly integrates suggestedQuestions
        intent = {
            "category": "Performance",
            "subcategory": "TopPerformers",
            "confidence": "high",
        }
        suggested = ["Q1", "Q2", "Q3"]
        payload = build_response(
            query_type="filter_query",
            intro_message="Intro",
            combined_result={},
            analysis={"summary": "Sum", "observations": [], "insights": []},
            conclusion={"summary": "Conc"},
            intent=intent,
            raw_results=[],
            confidence=0.9,
            operation_count=1,
            formatted_data="formatted",
            suggested_questions=suggested,
        )
        self.assertEqual(payload["suggestedQuestions"], suggested)

        # Default fallback to empty list
        payload_default = build_response(
            query_type="greeting",
            intro_message="Hi",
            combined_result={},
            analysis=None,
            conclusion=None,
            intent={"category": "greeting"},
            raw_results=[],
            confidence=1.0,
            operation_count=0,
            formatted_data="",
        )
        self.assertEqual(payload_default["suggestedQuestions"], [])


if __name__ == "__main__":
    unittest.main()
