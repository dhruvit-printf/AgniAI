from __future__ import annotations

import pytest
from intent_engine.query_type_detector import (
    detect_query_type,
    _has_cross_filter_marker,
    _has_comparison_marker,
    _has_multi_independent_marker,
)


def test_cross_filter_detection():
    """Test all cross-filter variations."""
    cross_filter_queries = [
        "Show Agniveers who have medical leave",
        "Agniveers whose verification is pending",
        "Candidates which are overweight",
        "People that are on leave",
        "Agniveers with BMI over 25",
        "Candidates having excellent grading",
        "People has issued equipment",
        "Agniveers holding overdue equipment",
        "Agniveers belonging to Dogra class",
        "People belongs to Platoon 3",
        "Candidates part of Alpha company",
        "Agniveers suffering from fever",
        "People diagnosed with malaria",
        "Candidates hospitalized",
        "Agniveers on leave",
        "People currently on medical leave",
        "Candidates absent",
        "Agniveers with pending verification",
        "People rejected in police verification",
        "Agniveers holding damaged equipment",
        "Candidates with overdue equipment",
        "Agniveers who scored Excellent",
        "People who improved in BPET",
    ]
    
    for query in cross_filter_queries:
        result = detect_query_type(query)
        assert result["type"] == "cross_filter", f"Failed: {query} -> {result['type']} ({result['reasoning']})"


def test_comparison_detection():
    """Test all comparison variations."""
    comparison_queries = [
        "Compare BPET and PPT",
        "Comparison between BPET and PPT",
        "BPET vs PPT",
        "BPET versus PPT",
        "Which is better BPET or PPT?",
        "Who scored higher in BPET?",
        "Which section has more pass percentage?",
        "Who is the best performer?",
        "Which is the worst section?",
        "Who scored highest marks?",
        "Difference between BPET and PPT",
        "Contrast BPET with PPT",
        "BPET compared to PPT",
        "Side by side comparison of BPET and PPT",
        "Compare Lakhwinder company vs Jaswant company",
        "Platoon 1 and Platoon 2 performance",
        "A0701882L and A0701883M comparison",
    ]
    
    for query in comparison_queries:
        result = detect_query_type(query)
        assert result["type"] == "comparison", f"Failed: {query} -> {result['type']} ({result['reasoning']})"


def test_multi_independent_detection():
    """Test all multi-independent variations."""
    multi_independent_queries = [
        "Show attendance and leave records",
        "Give me performance stats as well as equipment details",
        "Attendance along with verification status",
        "Show BPET scores together with medical records",
        "Display leave status in addition to BMI",
        "Attendance and also verification",
        "Show attendance, leave, and medical records",
        "First show attendance, then show leave",
        "Show BPET scores, then PPT scores",
        "First performance, then medical, finally equipment",
        "Show both attendance and leave",
        "Show attendance, leave, medical, equipment stats",
        "BPET, PPT, Firing, Drill scores",
        "Show attendance and give me leave records",
        "List BPET scores and display medical records",
    ]
    
    for query in multi_independent_queries:
        result = detect_query_type(query)
        assert result["type"] == "multi_independent", f"Failed: {query} -> {result['type']} ({result['reasoning']})"


def test_simple_queries():
    """Test simple queries that should stay simple."""
    simple_queries = [
        "Show attendance",
        "BPET scores",
        "Medical records",
        "Leave history",
        "Equipment stats",
        "Top 10 performers",
    ]
    
    for query in simple_queries:
        result = detect_query_type(query)
        assert result["type"] == "simple", f"Failed: {query} -> {result['type']}"
