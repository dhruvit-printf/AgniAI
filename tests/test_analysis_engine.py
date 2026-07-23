"""
Characterization tests for analysis_engine.py, written as a regression net
BEFORE consolidating its duplicated helpers (_record_label, _percentile,
_extract_nested_scores, _extract_scores, threshold constants) into
intelligence_common.py. These pin today's actual computed values so any
accidental behavior change during the move is caught immediately.
"""

from __future__ import annotations

import unittest

from analysis_engine import (
    _extract_nested_scores,
    _extract_scores,
    _percentile,
    _record_label,
    generate_analysis,
)


class TestStatisticalHelpers(unittest.TestCase):
    def test_percentile_known_values(self):
        data = sorted([10, 20, 30, 40])
        self.assertEqual(_percentile(data, 25), 17.5)
        self.assertEqual(_percentile(data, 75), 32.5)
        self.assertEqual(_percentile(data, 50), 25.0)

    def test_percentile_empty(self):
        self.assertEqual(_percentile([], 50), 0.0)

    def test_record_label_prefers_name_over_id(self):
        record = {"fullName": "Amit Kumar", "agniveerNo": "A01"}
        self.assertEqual(_record_label(record), "Amit Kumar")

    def test_record_label_falls_back_to_id(self):
        record = {"agniveerNo": "A01"}
        self.assertEqual(_record_label(record), "A01")

    def test_record_label_none_when_no_fields(self):
        self.assertIsNone(_record_label({}))

    def test_extract_nested_scores_walks_attempts_structure(self):
        record = {
            "attempts": [
                {
                    "sections": [
                        {
                            "score": 60,
                            "subItems": [{"score": 55}, {"marksObtained": 58}],
                        }
                    ]
                }
            ]
        }
        scores = _extract_nested_scores(record)
        self.assertEqual(sorted(scores), [55.0, 58.0, 60.0])

    def test_extract_nested_scores_includes_top_level_score(self):
        record = {"bestTotal": 80, "attempts": [{"sections": [{"score": 70}]}]}
        scores = _extract_nested_scores(record)
        self.assertEqual(sorted(scores), [70.0, 80.0])

    def test_extract_scores_preserve_order_true(self):
        records = [{"bestTotal": 30}, {"bestTotal": 10}, {"bestTotal": 20}]
        self.assertEqual(_extract_scores(records, preserve_order=True), [30.0, 10.0, 20.0])

    def test_extract_scores_preserve_order_false_sorts(self):
        records = [{"bestTotal": 30}, {"bestTotal": 10}, {"bestTotal": 20}]
        self.assertEqual(_extract_scores(records, preserve_order=False), [10.0, 20.0, 30.0])

    def test_extract_scores_uses_max_nested_when_no_top_level(self):
        records = [
            {"attempts": [{"sections": [{"score": 40}, {"score": 65}]}]},
        ]
        self.assertEqual(_extract_scores(records), [65.0])


class TestGenerateAnalysis(unittest.TestCase):
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
        result = generate_analysis(combined_result, "simple", {"category": "Performance"})
        stats = result["statistics"]
        self.assertEqual(stats["record_count"], 4)
        self.assertEqual(stats["average_score"], 67.5)
        self.assertEqual(stats["min_score"], 40.0)
        self.assertEqual(stats["max_score"], 90.0)
        self.assertEqual(stats["low_count"], 1)  # 40 < 50
        self.assertEqual(stats["high_count"], 2)  # 90, 80 > 75
        self.assertIn("summary", result)
        self.assertIsInstance(result["insights"], list)

    def test_empty_combined_result_returns_no_match(self):
        result = generate_analysis({}, "simple", {"category": "Performance"})
        self.assertEqual(result["statistics"], {"record_count": 0})
        self.assertEqual(
            result["summary"], "I couldn't find any matching records for that request."
        )
        self.assertEqual(result["insights"], [])

    def test_none_combined_result_handled_gracefully(self):
        result = generate_analysis(None, "simple", {"category": "Performance"})
        self.assertEqual(result["statistics"]["record_count"], 0)

    def test_compare_query_type(self):
        combined_result = {
            "left": {"label": "Alpha", "data": [{"bestTotal": 80}, {"bestTotal": 90}]},
            "right": {"label": "Bravo", "data": [{"bestTotal": 60}, {"bestTotal": 70}]},
        }
        result = generate_analysis(combined_result, "compare", {"category": "Performance"})
        stats = result["statistics"]
        self.assertEqual(stats["left_average"], 85.0)
        self.assertEqual(stats["right_average"], 65.0)
        self.assertEqual(stats["left_count"], 2)
        self.assertEqual(stats["right_count"], 2)

    def test_nested_attempt_scores_feed_into_stats(self):
        combined_result = {
            "sections": [
                {
                    "label": "Result",
                    "data": [
                        {
                            "fullName": "A",
                            "attempts": [{"sections": [{"score": 88}]}],
                        }
                    ],
                }
            ]
        }
        result = generate_analysis(combined_result, "simple", {"category": "Performance"})
        self.assertEqual(result["statistics"]["average_score"], 88.0)


if __name__ == "__main__":
    unittest.main()
