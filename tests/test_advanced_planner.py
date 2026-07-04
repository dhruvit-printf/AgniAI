"""
tests/test_advanced_planner.py
================================
Test suite for:
  1. Nested / N-way cross filters
  2. Multi-filter cross category (leave, medical, attendance)
  3. Filtered comparison queries
  4. ANALYTICS query type
  5. Follow-up query support (admin_context)
  6. Aggregation helper (result_combiner)
  7. Comparison engine improvements
  8. Backward-compatibility guard for SIMPLE / existing types

Run with:
    pytest tests/test_advanced_planner.py -v
"""

from __future__ import annotations

import os
import sys
import types
import unittest

# ── Minimal stubs so modules import without a full Flask app ────────────────
_STUB_MODS = [
    "flask",
    "flask_cors",
    "flask_limiter",
    "flask_limiter.util",
    "dotenv",
    "requests",
]
for mod in _STUB_MODS:
    try:
        __import__(mod)
    except ImportError:
        if mod not in sys.modules:
            stub = types.ModuleType(mod)
            if mod == "dotenv":
                stub.load_dotenv = lambda *a, **kw: None  # type: ignore[attr-defined]
            sys.modules[mod] = stub

# ── Point import resolution at the repo root ─────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import admin_context as ac
import intent_engine.query_planner as qp
import result_combiner as rc

QueryType = qp.QueryType
plan_query = qp.plan_query


# =============================================================================
# 1. NESTED / N-WAY CROSS FILTERS
# =============================================================================


class TestNestedCrossFilters(unittest.TestCase):

    def test_two_way_cross_filter_performance_sport(self):
        """Classic 2-way: top performer × sport — must still work."""
        plan = plan_query("Show top performer in PPT who plays cricket")
        self.assertEqual(plan.query_type, QueryType.CROSS_FILTER)
        self.assertEqual(len(plan.operations), 2)
        self.assertEqual(plan.operations[0].intent_result["category"], "Performance")
        self.assertEqual(plan.operations[1].intent_result["category"], "Skills")
        self.assertEqual(plan.operations[1].intent_result["sport"], "Cricket")

    def test_three_way_cross_filter(self):
        """
        3-way: top performer in PPT × cricket × currently on leave.
        Expects 3 operations.
        """
        plan = plan_query(
            "Show top performer in PPT who plays cricket and is currently on leave"
        )
        self.assertEqual(plan.query_type, QueryType.CROSS_FILTER)
        self.assertEqual(len(plan.operations), 3)
        self.assertEqual(plan.operations[0].intent_result["category"], "Performance")
        self.assertEqual(plan.operations[1].intent_result["category"], "Skills")
        self.assertEqual(plan.operations[1].intent_result["sport"], "Cricket")
        self.assertEqual(plan.operations[2].intent_result["category"], "Leave")
        self.assertEqual(plan.operations[2].intent_result["operation"], "Current")

    def test_n_way_intersect_three_result_sets(self):
        """result_combiner: 3-way ID intersection."""
        set_a = [{"agniveerId": 101}, {"agniveerId": 102}, {"agniveerId": 103}]
        set_b = [{"agniveerId": 102}, {"agniveerId": 103}]
        set_c = [{"agniveerId": 103}, {"agniveerId": 104}]
        result = rc.intersect_results([set_a, set_b, set_c], primary_index=0)
        self.assertEqual(result["matchCount"], 1)
        self.assertEqual(result["records"][0]["agniveerId"], 103)
        self.assertEqual(result["filterDepth"], 3)

    def test_intersect_empty_set_gives_zero_matches(self):
        set_a = [{"agniveerId": 101}, {"agniveerId": 102}]
        set_b: list = []
        result = rc.intersect_results([set_a, set_b])
        self.assertEqual(result["matchCount"], 0)

    def test_intersect_no_common_ids(self):
        set_a = [{"agniveerId": 1}]
        set_b = [{"agniveerId": 2}]
        result = rc.intersect_results([set_a, set_b])
        self.assertEqual(result["matchCount"], 0)


# =============================================================================
# 2. MULTI-FILTER CROSS CATEGORY (leave, medical, attendance)
# =============================================================================


class TestMultiFilterCrossCategory(unittest.TestCase):

    def test_sport_cross_with_leave(self):
        """Football players currently on leave."""
        plan = plan_query("Show football players currently on leave")
        self.assertEqual(plan.query_type, QueryType.CROSS_FILTER)
        cats = {op.intent_result.get("category") for op in plan.operations}
        self.assertTrue(len(cats) >= 1)

    def test_sport_cross_with_medical(self):
        """Football players with BMI analysis."""
        plan = plan_query("Show football players with BMI analysis")
        self.assertEqual(plan.query_type, QueryType.CROSS_FILTER)

    def test_cricket_cross_with_leave(self):
        """Cricket players currently on leave."""
        plan = plan_query("Show cricket players currently on leave")
        self.assertEqual(plan.query_type, QueryType.CROSS_FILTER)

    def test_performer_with_medical_leave(self):
        """Top performers with medical leave."""
        plan = plan_query("Show top performers with medical leave")
        self.assertEqual(plan.query_type, QueryType.CROSS_FILTER)


# =============================================================================
# 3. FILTERED COMPARISON QUERIES
# =============================================================================


class TestFilteredComparison(unittest.TestCase):

    def test_compare_sections_among_cricket_players(self):
        """Compare PPT and BPET performance among cricket players."""
        plan = plan_query("Compare PPT and BPET performance among cricket players")
        self.assertEqual(plan.query_type, QueryType.COMPARISON)
        self.assertEqual(len(plan.operations), 2)

    def test_compare_two_sections_for_sikh_class(self):
        """Compare BPET and PPT for Sikh class."""
        plan = plan_query("Compare BPET and PPT for Sikh class")
        self.assertEqual(plan.query_type, QueryType.COMPARISON)
        self.assertEqual(len(plan.operations), 2)
        # Each operation fragment should mention the filter
        frags = [op.raw_fragment for op in plan.operations]
        self.assertTrue(any("sikh" in f.lower() for f in frags))

    def test_compare_results_with_record_lists(self):
        """compare_results() must extract metrics from record lists, not just dicts."""
        set_a = [
            {"agniveerId": 1, "bestTotal": 90},
            {"agniveerId": 2, "bestTotal": 80},
        ]
        set_b = [
            {"agniveerId": 3, "bestTotal": 70},
            {"agniveerId": 4, "bestTotal": 60},
        ]
        result = rc.compare_results([("PPT", set_a), ("BEPT", set_b)])
        self.assertIn("averageScore", result["comparedMetrics"])
        self.assertIn("recordCount", result["comparedMetrics"])
        # PPT should have higher averageScore
        ppt_side = next(s for s in result["sides"] if s["label"] == "PPT")
        bept_side = next(s for s in result["sides"] if s["label"] == "BEPT")
        self.assertGreater(
            ppt_side["metrics"]["averageScore"], bept_side["metrics"]["averageScore"]
        )

    def test_compare_results_top_score(self):
        """compare_results() should surface topScore for each side."""
        set_a = [{"agniveerId": 1, "bestTotal": 95}, {"agniveerId": 2, "bestTotal": 85}]
        set_b = [{"agniveerId": 3, "bestTotal": 70}]
        result = rc.compare_results([("Side A", set_a), ("Side B", set_b)])
        self.assertIn("topScore", result["comparedMetrics"])


# =============================================================================
# 4. ANALYTICS QUERY TYPE
# =============================================================================


class TestAnalyticsQueryType(unittest.TestCase):

    def test_highest_average_by_section(self):
        plan = plan_query("Which section has the highest average score?")
        self.assertEqual(plan.query_type, QueryType.ANALYTICS)
        self.assertIsNotNone(plan.analytics_hint)

    def test_rank_sections_by_pass_percentage(self):
        plan = plan_query("Rank sections by pass percentage")
        self.assertEqual(plan.query_type, QueryType.ANALYTICS)
        self.assertIn(plan.analytics_hint, ("rank", "highest", "lowest", "aggregate"))

    def test_which_unit_has_most_absconded(self):
        plan = plan_query("Which unit has the most absconded persons?")
        self.assertEqual(plan.query_type, QueryType.DISTRIBUTION)
        self.assertEqual(plan.analytics_hint, None)

    def test_which_sport_has_best_performers(self):
        plan = plan_query("Which sport has the best performers?")
        self.assertEqual(plan.query_type, QueryType.ANALYTICS)

    def test_average_by_section_aggregate(self):
        plan = plan_query("Average score by section")
        self.assertIn(plan.query_type, (QueryType.ANALYTICS, QueryType.DISTRIBUTION))

    def test_attendance_by_unit(self):
        plan = plan_query("Attendance by unit")
        self.assertEqual(plan.query_type, QueryType.DISTRIBUTION)


# =============================================================================
# 5. FOLLOW-UP QUERY SUPPORT
# =============================================================================


class TestFollowUpQuerySupport(unittest.TestCase):

    def setUp(self):
        self.ctx = ac.AdminSessionContext()

    def test_no_history_not_followup(self):
        self.assertFalse(self.ctx.is_followup_query("sess-1", "Who plays cricket?"))

    def test_followup_detected_after_update(self):
        self.ctx.update(
            "sess-1",
            "Show top PPT performers",
            {"category": "Performance"},
            [{"agniveerId": 1}, {"agniveerId": 2}],
        )
        self.assertTrue(
            self.ctx.is_followup_query("sess-1", "Which of them play cricket?")
        )

    def test_non_followup_phrasing_not_detected(self):
        self.ctx.update(
            "sess-1",
            "Show top PPT performers",
            {"category": "Performance"},
            [{"agniveerId": 1}],
        )
        self.assertFalse(
            self.ctx.is_followup_query("sess-1", "Show latest distribution")
        )

    def test_previous_ids_returned(self):
        self.ctx.update(
            "sess-2",
            "Show football players",
            {"category": "Skills"},
            [{"agniveerId": 10}, {"agniveerId": 20}],
        )
        ids = self.ctx.get_previous_ids("sess-2")
        self.assertIsNotNone(ids)
        self.assertIn(10, ids)
        self.assertIn(20, ids)

    def test_different_sessions_isolated(self):
        self.ctx.update(
            "sess-A",
            "Show top PPT performers",
            {"category": "Performance"},
            [{"agniveerId": 99}],
        )
        # sess-B has no history
        self.assertFalse(
            self.ctx.is_followup_query("sess-B", "Which of them play cricket?")
        )

    def test_ids_extracted_from_nested_result(self):
        """IDs from .NET wrapper shape {data: {records: [...]}}"""
        wrapped = {"data": {"records": [{"agniveerId": 42}]}}
        ids = ac._extract_ids_from_result(wrapped)
        self.assertIn(42, ids)

    def test_history_cleared_on_clear(self):
        self.ctx.update(
            "sess-3",
            "q",
            {"category": "Leave"},
            [{"agniveerId": 1}],
        )
        self.ctx.clear("sess-3")
        self.assertIsNone(self.ctx.get_previous_ids("sess-3"))


# =============================================================================
# 6. GROUPING AND AGGREGATION
# =============================================================================


class TestAggregationHelper(unittest.TestCase):

    def _make_records(self):
        return [
            {"agniveerId": 1, "sectionName": "PPT", "bestTotal": 90},
            {"agniveerId": 2, "sectionName": "PPT", "bestTotal": 80},
            {"agniveerId": 3, "sectionName": "BPET", "bestTotal": 70},
            {"agniveerId": 4, "sectionName": "BPET", "bestTotal": 60},
            {"agniveerId": 5, "sectionName": "Firing", "bestTotal": 95},
        ]

    def test_aggregate_by_section_returns_rows(self):
        rows = rc.aggregate_records(self._make_records(), group_by="section")
        self.assertGreater(len(rows), 0)
        groups = {r["group"] for r in rows}
        self.assertIn("PPT", groups)
        self.assertIn("BPET", groups)

    def test_aggregate_average_score(self):
        rows = rc.aggregate_records(self._make_records(), group_by="section")
        ppt_row = next(r for r in rows if r["group"] == "PPT")
        self.assertEqual(ppt_row["averageScore"], 85.0)

    def test_aggregate_sorted_descending(self):
        rows = rc.aggregate_records(self._make_records(), group_by="section")
        scores = [r.get("averageScore", 0) for r in rows]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_aggregate_by_sport_with_comma_separated(self):
        records = [
            {"agniveerId": 1, "sports": "Cricket, Kabaddi", "bestTotal": 80},
            {"agniveerId": 2, "sports": "Cricket", "bestTotal": 90},
            {"agniveerId": 3, "sports": "Football", "bestTotal": 70},
        ]
        rows = rc.aggregate_records(records, group_by="sport")
        groups = {r["group"] for r in rows}
        self.assertIn("Cricket", groups)
        self.assertIn("Football", groups)

    def test_aggregate_empty_records(self):
        rows = rc.aggregate_records([], group_by="section")
        self.assertEqual(rows, [])

    def test_aggregate_unknown_dimension(self):
        rows = rc.aggregate_records(self._make_records(), group_by="xyz")
        self.assertEqual(rows, [])


# =============================================================================
# 7. COMPARISON ENGINE IMPROVEMENTS
# =============================================================================


class TestComparisonEngineImprovements(unittest.TestCase):

    def test_compare_scalar_dicts_still_works(self):
        """Backward compat: scalar dict comparison from v1."""
        result = rc.compare_results(
            [
                ("PPT", {"average": 85.5, "total": 50}),
                ("BPET", {"average": 79.2, "total": 50}),
            ]
        )
        self.assertIn("average", result["comparedMetrics"])

    def test_compare_record_count(self):
        set_a = [{"agniveerId": i} for i in range(30)]
        set_b = [{"agniveerId": i} for i in range(10)]
        result = rc.compare_results([("Big", set_a), ("Small", set_b)])
        self.assertIn("recordCount", result["comparedMetrics"])
        big_side = next(s for s in result["sides"] if s["label"] == "Big")
        small_side = next(s for s in result["sides"] if s["label"] == "Small")
        self.assertEqual(big_side["metrics"]["recordCount"], 30)
        self.assertEqual(small_side["metrics"]["recordCount"], 10)

    def test_compare_distribution_comparisons(self):
        """Distributions (dict with list inside) are counted by record."""
        set_a = {"data": [{"agniveerId": 1}, {"agniveerId": 2}]}
        set_b = {"data": [{"agniveerId": 3}]}
        result = rc.compare_results([("A", set_a), ("B", set_b)])
        a_side = next(s for s in result["sides"] if s["label"] == "A")
        b_side = next(s for s in result["sides"] if s["label"] == "B")
        self.assertEqual(a_side["metrics"]["recordCount"], 2)
        self.assertEqual(b_side["metrics"]["recordCount"], 1)


# =============================================================================
# 8. BACKWARD COMPATIBILITY
# =============================================================================


class TestBackwardCompatibility(unittest.TestCase):

    def test_simple_top_performers(self):
        plan = plan_query("Show top 10 performers in PPT")
        self.assertEqual(plan.query_type, QueryType.FILTER_QUERY)

    def test_simple_monthly_attendance(self):
        plan = plan_query("Show monthly attendance stats")
        self.assertEqual(plan.query_type, QueryType.TREND)

    def test_existing_cross_filter_unchanged(self):
        plan = plan_query("Show top performer in PPT who plays cricket")
        self.assertEqual(plan.query_type, QueryType.CROSS_FILTER)

    def test_no_split_phrase_guard(self):
        plan = plan_query("show approved and pending leave records")
        self.assertEqual(plan.query_type, QueryType.FILTER_QUERY)

    def test_multi_independent_unchanged(self):
        plan = plan_query("Show attendance stats as well as equipment overdue records")
        self.assertEqual(plan.query_type, QueryType.MULTI_OPERATION)

    def test_intersect_two_sets_backward_compat(self):
        """v1 call signature must still work."""
        set_a = [{"agniveerId": 1}, {"agniveerId": 2}]
        set_b = [{"agniveerId": 2}]
        result = rc.intersect_results([set_a, set_b])
        self.assertEqual(result["matchCount"], 1)
        self.assertEqual(result["records"][0]["agniveerId"], 2)

    def test_merge_results_unchanged(self):
        labeled = [("Attendance", {"total": 100}), ("Equipment", {"overdue": 5})]
        merged = rc.merge_results(labeled)
        self.assertEqual(merged["sectionCount"], 2)

    def test_compare_results_backward_compat(self):
        labeled = [("PPT", {"average": 85.5}), ("BPET", {"average": 79.2})]
        result = rc.compare_results(labeled)
        self.assertIn("average", result["comparedMetrics"])
        self.assertEqual(len(result["sides"]), 2)

    def test_query_plan_to_dict_schema(self):
        """plan.to_dict() must include standard keys."""
        plan = plan_query("Show top 5 performers")
        d = plan.to_dict()
        for key in (
            "queryType",
            "confidence",
            "operationCount",
            "reasoning",
            "operations",
        ):
            self.assertIn(key, d)

    def test_sub_operation_to_dict_schema(self):
        plan = plan_query("Show top 5 performers in BPET")
        op = plan.operations[0]
        d = op.to_dict()
        for key in ("rawFragment", "intentResult", "dotnetPayload"):
            self.assertIn(key, d)


if __name__ == "__main__":
    unittest.main(verbosity=2)
