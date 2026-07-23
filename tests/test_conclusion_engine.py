"""
Characterization tests for conclusion_engine.py, written as a regression net
BEFORE consolidating its duplicated helpers (_record_label, _percentile)
into intelligence_common.py. These pin today's actual computed values so any
accidental behavior change during the move is caught immediately.

Also locks in a deliberate non-merge decision: conclusion_engine.py's
_extract_scores is the SIMPLE (non-nested-aware) variant and must continue
to ignore nested attempt scores — merging it with analysis_engine.py's/
prediction_engine.py's nested-aware variant would be a real behavior change.
"""

from __future__ import annotations

import unittest

from conclusion_engine import (
    _detect_trend,
    _extract_scores,
    _percentile,
    _record_label,
    generate_conclusion,
)


class TestStatisticalHelpers(unittest.TestCase):
    def test_percentile_known_values(self):
        data = sorted([10, 20, 30, 40])
        self.assertEqual(_percentile(data, 25), 17.5)
        self.assertEqual(_percentile(data, 75), 32.5)

    def test_record_label_prefers_name_over_id(self):
        record = {"fullName": "Amit Kumar", "agniveerNo": "A01"}
        self.assertEqual(_record_label(record), "Amit Kumar")

    def test_extract_scores_ignores_nested_attempt_scores(self):
        """conclusion_engine's _extract_scores is deliberately the SIMPLE
        variant — it must NOT pick up nested attempt scores when there's no
        top-level score field. This pins that behavior so a future
        consolidation doesn't accidentally merge it with the nested-aware
        variant used by analysis_engine.py/prediction_engine.py."""
        records = [
            {"attempts": [{"sections": [{"score": 90}]}]},  # nested only, no top-level
            {"bestTotal": 70},  # top-level present
        ]
        scores = _extract_scores(records)
        self.assertEqual(scores, [70.0])

    def test_detect_trend_rising(self):
        scores = [40.0, 50.0, 60.0, 70.0, 80.0]
        label, slope = _detect_trend(scores)
        self.assertEqual(label, "rising")
        self.assertGreater(slope, 0)

    def test_detect_trend_falling(self):
        scores = [80.0, 70.0, 60.0, 50.0, 40.0]
        label, slope = _detect_trend(scores)
        self.assertEqual(label, "falling")
        self.assertLess(slope, 0)

    def test_detect_trend_stable(self):
        scores = [60.0, 61.0, 60.0, 61.0, 60.0]
        label, _ = _detect_trend(scores)
        self.assertEqual(label, "stable")

    def test_detect_trend_volatile(self):
        scores = [10.0, 90.0, 15.0, 85.0, 20.0]
        label, _ = _detect_trend(scores)
        self.assertEqual(label, "volatile")

    def test_detect_trend_insufficient_data(self):
        label, slope = _detect_trend([50.0, 55.0])
        self.assertEqual(label, "insufficient_data")


class TestGenerateConclusion(unittest.TestCase):
    def test_simple_query_with_scores(self):
        combined_result = {
            "sections": [
                {
                    "label": "Result",
                    "data": [
                        {"fullName": "A", "bestTotal": 90},
                        {"fullName": "B", "bestTotal": 40},
                        {"fullName": "C", "bestTotal": 60},
                        {"fullName": "D", "bestTotal": 80},
                    ],
                }
            ]
        }
        result = generate_conclusion(
            combined_result, "simple", {"category": "Performance"}
        )
        self.assertIn("summary", result)
        self.assertIsInstance(result["bullets"], list)
        self.assertTrue(
            any("90" in b or "40" in b or "avg" in b for b in result["bullets"])
        )

    def test_empty_combined_result_returns_no_match(self):
        result = generate_conclusion({}, "simple", {"category": "Performance"})
        self.assertEqual(
            result["summary"], "I couldn't find any matching records for that request."
        )
        self.assertEqual(result["bullets"], [])

    def test_none_combined_result_handled_gracefully(self):
        result = generate_conclusion(None, "simple", {"category": "Performance"})
        self.assertIn("summary", result)

    def test_trend_prediction_bullet_present_for_simple_query(self):
        """Note: generate_conclusion's "simple" branch calls
        _detect_trend(sorted(scores)) — trend is computed on the
        sorted-ascending distribution, not encounter/temporal order, so it
        is always non-decreasing here regardless of input order (a
        pre-existing characteristic of this code path, out of scope to
        change in this pass). This test pins that a trend/prediction bullet
        is produced at all, not a specific direction."""
        combined_result = {
            "sections": [
                {
                    "label": "Result",
                    "data": [
                        {"fullName": f"P{i}", "bestTotal": v}
                        for i, v in enumerate([80, 70, 60, 50, 40])
                    ],
                }
            ]
        }
        result = generate_conclusion(
            combined_result, "simple", {"category": "Performance"}
        )
        bullets_text = " ".join(result["bullets"])
        self.assertIn("projected avg", bullets_text.lower())
        self.assertIn("trend is upward", bullets_text.lower())

    def test_compare_query_type(self):
        combined_result = {
            "left": {"label": "Alpha", "data": [{"bestTotal": 80}, {"bestTotal": 90}]},
            "right": {"label": "Bravo", "data": [{"bestTotal": 60}, {"bestTotal": 70}]},
        }
        result = generate_conclusion(
            combined_result, "compare", {"category": "Performance"}
        )
        self.assertIn("Alpha", " ".join(result["bullets"]))
        self.assertIn("Bravo", " ".join(result["bullets"]))


if __name__ == "__main__":
    unittest.main()
