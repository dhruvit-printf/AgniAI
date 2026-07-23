"""
Query type detection coverage for query_understanding_engine.understand_query().

This used to test a standalone, unwired `intent_engine/query_type_detector.py`
that duplicated (and diverged from) the real classification logic. That module
was folded into query_understanding_engine.py — its distinct marker vocabulary
(medical/leave/verification/equipment status phrases, comparison phrases,
multi-independent separation words) was merged into the canonical
_COMPARISON_MARKERS / _CROSS_FILTER_MARKERS / _CROSS_FILTER_STATUS_MARKERS /
_MULTI_INDEPENDENT_MARKERS tuples there, and this file now exercises that one
real engine instead of a second, unused implementation.

Unlike the old standalone detector, understand_query() requires >= 2 distinct
categories (not just a marker hit) before classifying a query as cross_filter
or multi_independent — a single-category filter like "Agniveers whose
verification is pending" is correctly "simple" (a filtered lookup), not
cross_filter. Test cases below reflect that real gating behavior.
"""

from __future__ import annotations

from query_understanding_engine import understand_query


def test_comparison_detection():
    """Comparison markers, including ones merged in from the old detector
    (vs., v/s, as compared to, relative to, contrasted with, superior to)."""
    comparison_queries = [
        "Compare BPET and PPT",
        "Comparison between BPET and PPT",
        "BPET vs PPT",
        "BPET vs. PPT",
        "BPET versus PPT",
        "v/s comparison of BPET and PPT",
        "Difference between BPET and PPT",
        "Contrast BPET with PPT",
        "BPET compared to PPT",
        "BPET performance as compared to PPT performance",
        "BPET relative to PPT scores",
        "Side by side comparison of BPET and PPT",
        "Compare Lakhwinder company vs Jaswant company",
    ]

    for query in comparison_queries:
        result = understand_query(query)
        assert (
            result["query_type"] == "comparison"
        ), f"Failed: {query} -> {result['query_type']}"


def test_cross_filter_detection():
    """Cross-filter requires a relative-clause/status marker AND >= 2 distinct
    categories (or performance sections) — that's what distinguishes it from
    a single-category filtered lookup, which stays 'simple'."""
    cross_filter_queries = [
        "People currently on medical leave",
        "Agniveers belonged to Dogra class who are on medical leave",
        "Agniveers treated for fever and having pending verification",
        "Attendance along with verification status",
        "Show BPET scores together with medical records",
        # Repeated relative pronoun ("whose ... whose ...") — regression
        # test for the splitter retrying a later occurrence of the same
        # separator instead of abandoning it after a generic-lead first hit
        # (previously mis-split into one fragment classified as
        # Verification only, losing the BMI/Medical condition entirely).
        "Agniveers whose BMI is Normal whose Police verification is Verified",
    ]

    for query in cross_filter_queries:
        result = understand_query(query)
        assert result["query_type"] == "cross_filter", (
            f"Failed: {query} -> {result['query_type']} "
            f"({result.get('complexity')})"
        )


def test_multi_independent_detection():
    """Multi-independent markers, including ones merged in from the old
    detector (respectively, individually, one by one, in the meantime)."""
    multi_independent_queries = [
        "Show attendance and leave records",
        "Give me performance stats as well as equipment details",
        "Display leave status in addition to BMI",
        "Show attendance, leave, and medical records",
        "Show both attendance and leave",
        "Show attendance and leave records respectively",
        "Show attendance and leave records individually",
    ]

    for query in multi_independent_queries:
        result = understand_query(query)
        assert (
            result["query_type"] == "multi_independent"
        ), f"Failed: {query} -> {result['query_type']}"


def test_simple_queries():
    """Single-category queries — including ones with a filter/status word —
    stay 'simple' because they don't reference a second distinct category."""
    simple_queries = [
        "Show attendance",
        "BPET scores",
        "Medical records",
        "Leave history",
        "Equipment stats",
        "Top 10 performers",
        "Agniveers whose verification is pending",
        "Candidates hospitalized",
        # Single category (Leave) — the relative clause narrows *within*
        # one category rather than intersecting a second one.
        "Show Agniveers who have medical leave",
    ]

    for query in simple_queries:
        result = understand_query(query)
        assert (
            result["query_type"] == "simple"
        ), f"Failed: {query} -> {result['query_type']}"
