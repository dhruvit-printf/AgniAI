import unittest

from response_builder import build_response
from suggested_question_engine import generate_suggested_questions

# Suggestions are only ever cross-filter style (multiple related conditions
# on the same record set) — simple / compare / multi_independent phrasing is
# no longer offered, regardless of the query_type that produced the answer.
_COMPARE_MARKERS = (" vs ", " versus ", "compare ")


class TestSuggestedQuestions(unittest.TestCase):

    def _assert_cross_filter_only(self, questions):
        self.assertEqual(len(questions), 4)
        self.assertEqual(len(set(q.lower() for q in questions)), 4)
        for q in questions:
            lowered = q.lower()
            self.assertTrue(q.strip())
            self.assertFalse(
                any(marker in lowered for marker in _COMPARE_MARKERS),
                f"suggestion reads like a compare question: {q!r}",
            )

    def test_suggested_questions_by_category(self):
        # 1. Performance Category
        intent_perf = {"category": "Performance"}
        questions_perf = generate_suggested_questions("filter_query", intent_perf, {})
        self._assert_cross_filter_only(questions_perf)

        # 2. Leave Category
        intent_leave = {"category": "Leave"}
        questions_leave = generate_suggested_questions("filter_query", intent_leave, {})
        self._assert_cross_filter_only(questions_leave)

        # 3. Unrecognized Category
        intent_unknown = {"category": "Greeting"}
        questions_unknown = generate_suggested_questions("greeting", intent_unknown, {})
        self.assertEqual(questions_unknown, [])

        # 4. (category, subcategory) override
        intent_perf_top = {"category": "Performance", "subcategory": "TopPerformers"}
        questions_top = generate_suggested_questions(
            "filter_query", intent_perf_top, {}
        )
        self._assert_cross_filter_only(questions_top)

        # 5. query_type no longer changes the suggestion flavor — compare
        # and multi_independent still resolve to cross-filter suggestions,
        # not compare/multi-section phrasing.
        questions_compare = generate_suggested_questions("compare", intent_perf, {})
        questions_multi = generate_suggested_questions(
            "multi_independent", intent_perf, {}
        )
        self._assert_cross_filter_only(questions_compare)
        self._assert_cross_filter_only(questions_multi)

    def test_build_response_payload_suggestions(self):
        # Verify that build_response correctly integrates suggestedQuestions
        suggested = ["Q1", "Q2", "Q3"]
        payload = build_response(
            message="Intro",
            formatted_data={"type": "TABLE", "data": {}},
            metadata={},
            session_id="session-123",
            suggested_questions=suggested,
        )
        self.assertEqual(payload["suggestedQuestions"], suggested)

        # Default fallback to empty list
        payload_default = build_response(
            message="Hi",
            formatted_data={},
            metadata={},
            session_id="session-123",
        )
        self.assertEqual(payload_default["suggestedQuestions"], [])


if __name__ == "__main__":
    unittest.main()
