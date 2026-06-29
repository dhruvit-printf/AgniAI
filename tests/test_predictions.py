import unittest

from prediction_engine import _ground_and_sanitize


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

    @unittest.skip(
        "generate_rule_based_predictions was removed — covered by generate_predictions"
    )
    def test_generate_rule_based_predictions_grounding(self):
        pass

    def test_analysis_is_none_path(self):
        # When analysis is None (LLM failure path), building the response should stay clean.
        from response_builder import build_response

        intent = {
            "category": "Performance",
            "subcategory": "TopPerformers",
            "confidence": "high",
        }
        resp = build_response(
            message="Intro",
            formatted_data={"type": "TABLE", "data": {}},
            metadata={},
            session_id="session-1",
        )
        # When analysis is not supplied, it defaults to empty string (not None)
        self.assertEqual(resp.get("analysis"), "")

    def test_comparison_prediction_direction(self):
        from prediction_engine import generate_predictions

        answer = {
            "left": {"label": "Company A", "data": [{"score": 90}]},
            "right": {"label": "Company B", "data": [{"score": 70}]},
            "comparison": {"difference": 20.0, "higher": "Company A"},
        }
        res = generate_predictions(answer, "compare", {"category": "Performance"})
        # Since Company A (left_label) is higher, shortTerm should be "increasing" / trend "Increasing"
        self.assertEqual(res["trend"], "Increasing")

        answer_decreasing = {
            "left": {"label": "Company A", "data": [{"score": 70}]},
            "right": {"label": "Company B", "data": [{"score": 90}]},
            "comparison": {"difference": 20.0, "higher": "Company B"},
        }
        res_dec = generate_predictions(
            answer_decreasing, "compare", {"category": "Performance"}
        )
        # Since Company B (right_label) is higher, trend should be "Decreasing"
        self.assertEqual(res_dec["trend"], "Decreasing")


if __name__ == "__main__":
    unittest.main()
