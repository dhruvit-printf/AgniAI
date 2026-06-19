import unittest

from report_generator import _ground_and_sanitize, generate_rule_based_predictions


class TestPredictions(unittest.TestCase):

    def test_grounding_guard_strips_ungrounded_numbers(self):
        aggregate_text = (
            "Match Count: 10\nTotal Before Filter: 100\nMatch Percentage: 10%"
        )

        # Valid prediction (uses 10% which is grounded)
        valid_pred = (
            "If the current rate of 10% holds, future queries will return 10% match."
        )
        clean_valid = _ground_and_sanitize(valid_pred, aggregate_text)
        self.assertEqual(clean_valid, valid_pred)

        # Invalid prediction (uses 15% which is ungrounded)
        invalid_pred = "We expect a future match rate of 15%."
        clean_invalid = _ground_and_sanitize(invalid_pred, aggregate_text)
        self.assertEqual(clean_invalid, "")

        # Mixed sentences: valid sentence followed by invalid sentence
        mixed_pred = "If the rate of 10% holds. We expect a future match rate of 15%."
        clean_mixed = _ground_and_sanitize(mixed_pred, aggregate_text)
        self.assertEqual(clean_mixed, "If the rate of 10% holds.")

    def test_generate_rule_based_predictions_grounding(self):
        combined = {
            "matchCount": 5,
            "totalBeforeFilter": 10,
            "filterDepth": 2,
            "records": [{"agniveerNo": "1"}, {"agniveerNo": "2"}],
        }

        # 1. With PCT 50.0% in aggregate_text
        aggregate_text_grounded = (
            "Match Count: 5\nTotal Before Filter: 10\nMatch Percentage: 50.0%"
        )
        preds = generate_rule_based_predictions(
            combined,
            "cross_filter",
            {"category": "Performance"},
            aggregate_text_grounded,
        )
        self.assertTrue(len(preds) > 0)
        self.assertIn("50.0%", preds[0])

        # 2. Without PCT 50.0% in aggregate_text (should fallback to un-numbered prediction)
        aggregate_text_ungrounded = "Match Count: 5\nTotal Before Filter: 10"
        preds_fallback = generate_rule_based_predictions(
            combined,
            "cross_filter",
            {"category": "Performance"},
            aggregate_text_ungrounded,
        )
        self.assertTrue(len(preds_fallback) > 0)
        # Should not contain "50.0%" because it would be stripped
        for p in preds_fallback:
            self.assertNotIn("50.0", p)

    def test_analysis_is_none_path(self):
        # When analysis is None (LLM failure path), building the response should stay clean.
        from response_builder import build_response

        intent = {
            "category": "Performance",
            "subcategory": "TopPerformers",
            "confidence": "high",
        }
        resp = build_response(
            query_type="simple",
            intro_message="Intro",
            combined_result={},
            analysis=None,
            conclusion=None,
            intent=intent,
            raw_results=[],
            confidence=0.9,
            operation_count=1,
            formatted_data="formatted",
        )
        # Should not crash and analysis block should be None (or omitted) as today
        self.assertIsNone(resp.get("analysis"))


if __name__ == "__main__":
    unittest.main()
