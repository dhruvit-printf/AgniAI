import unittest

from result_combiner import compare_results, intersect_results, merge_results


class TestResultCombiner(unittest.TestCase):

    def test_intersect_results(self):
        result_a = [
            {"agniveerId": 101, "fullName": "AMIT KUMAR", "score": 95},
            {"agniveerId": 102, "fullName": "KAPIL DEV", "score": 88},
        ]
        result_b = [
            {"agniveerId": 102, "sports": "Cricket"},
            {"agniveerId": 103, "sports": "Football"},
        ]

        combined = intersect_results([result_a, result_b], primary_index=0)
        self.assertEqual(combined["matchCount"], 1)
        self.assertEqual(combined["records"][0]["agniveerId"], 102)

    def test_merge_results(self):
        labeled = [("Attendance", {"total": 100}), ("Equipment", {"overdue": 5})]
        merged = merge_results(labeled)
        self.assertEqual(merged["sectionCount"], 2)
        self.assertEqual(merged["sections"][0]["label"], "Attendance")

    def test_compare_results(self):
        labeled = [
            ("PPT", {"average": 85.5, "total": 50}),
            ("BPET", {"average": 79.2, "total": 50}),
        ]
        compared = compare_results(labeled)
        self.assertIn("average", compared["comparedMetrics"])
        self.assertEqual(len(compared["sides"]), 2)


if __name__ == "__main__":
    unittest.main()
