"""
tests/test_context_engine.py
============================
Unit tests for context_engine.py (ConversationContextEngine).
"""

from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock

from context_engine import (
    ContextResolution,
    ConversationContextEngine,
    InteractionRecord,
    _compute_follow_up_score,
    _compute_relevance,
    _detect_follow_up_kind,
    _jaccard,
    _normalize,
    _reconstruct_query,
    _tokenize,
)


class TestContextEngineHelpers(unittest.TestCase):

    def test_normalize_and_tokenize(self):
        self.assertEqual(_normalize("  Hello   World! "), "hello world!")
        self.assertEqual(_tokenize("Hello, World 123!"), ["hello", "world", "123"])
        self.assertEqual(_tokenize(""), [])

    def test_jaccard_similarity(self):
        self.assertAlmostEqual(_jaccard(["a", "b"], ["b", "c"]), 1 / 3)
        self.assertEqual(_jaccard([], []), 0.0)
        self.assertEqual(_jaccard(["a"], []), 0.0)
        self.assertEqual(_jaccard(["a", "b"], ["a", "b"]), 1.0)

    def test_compute_follow_up_score(self):
        # Short queries without domain keywords should be classified as follow-ups
        self.assertGreater(_compute_follow_up_score("show them"), 0.55)
        self.assertGreater(_compute_follow_up_score("only platoon 2"), 0.55)

        # Explicit follow-up phrases
        self.assertGreater(_compute_follow_up_score("compare with bravo company"), 0.55)
        self.assertGreater(_compute_follow_up_score("top 5"), 0.55)

        # Continuation tokens
        self.assertGreater(_compute_follow_up_score("now show"), 0.55)

        # Pure number
        self.assertGreater(_compute_follow_up_score("10"), 0.55)

        # Independent domain query should be low score
        self.assertLess(
            _compute_follow_up_score("Show BPET scores for company alpha"), 0.35
        )

    def test_compute_relevance(self):
        record = InteractionRecord(
            user_message="Show BPET results for platoon 1",
            resolved_query="Show BPET results for platoon 1",
            intent={"category": "Performance", "section": "BPET"},
            entities={"platoon_id": "1"},
            filters={"section": "BPET"},
            category="Performance",
            section="BPET",
            operation="Summary",
            payload_summary="BPET performance list",
        )

        # Category match
        score_cat = _compute_relevance("Performance details", record)
        self.assertGreater(score_cat, 0.4)

        # Section match
        score_sec = _compute_relevance("Show bpet list", record)
        self.assertGreater(score_sec, 0.4)

        # Token overlap
        score_overlap = _compute_relevance("results for platoon 1", record)
        self.assertGreater(score_overlap, 0.2)

        # Irrelevant query
        score_irrelevant = _compute_relevance("Show equipment issues", record)
        self.assertLess(score_irrelevant, 0.1)

    def test_detect_follow_up_kind(self):
        record = InteractionRecord(
            user_message="Show BPET results",
            resolved_query="Show BPET results",
            intent={"category": "Performance", "section": "BPET"},
            entities={},
            filters={"section": "BPET"},
            category="Performance",
            section="BPET",
            operation="Summary",
            payload_summary="",
        )

        self.assertEqual(
            _detect_follow_up_kind("show as bar chart", record), "visualization"
        )
        self.assertEqual(
            _detect_follow_up_kind("compare with PPT", record), "comparison"
        )
        self.assertEqual(_detect_follow_up_kind("top 5", record), "ranking")
        self.assertEqual(_detect_follow_up_kind("show them", record), "pronoun")
        self.assertEqual(
            _detect_follow_up_kind("show PPT results", record), "section_switch"
        )
        self.assertEqual(_detect_follow_up_kind("only platoon 2", record), "filter")
        self.assertEqual(_detect_follow_up_kind("again", record), "continuation")
        self.assertEqual(_detect_follow_up_kind("obese trainees", record), "unknown")

    def test_reconstruct_query(self):
        record = InteractionRecord(
            user_message="Show BPET results",
            resolved_query="Show BPET results",
            intent={"category": "Performance", "section": "BPET"},
            entities={},
            filters={"section": "BPET"},
            category="Performance",
            section="BPET",
            operation="Summary",
            payload_summary="",
        )

        # Visualization
        self.assertEqual(
            _reconstruct_query("show as bar chart", record, "visualization"),
            "Show BPET results as bar chart",
        )

        # Comparison
        self.assertEqual(
            _reconstruct_query("compare with PPT", record, "comparison"),
            "Compare BPET with PPT",
        )

        # Ranking
        self.assertEqual(
            _reconstruct_query("top 5", record, "ranking"), "Show top 5 BPET results"
        )

        # Filter
        self.assertEqual(
            _reconstruct_query("only platoon 2", record, "filter"),
            "Show BPET results for platoon 2",
        )

        # Pronoun
        self.assertEqual(
            _reconstruct_query("show them", record, "pronoun"), "Show BPET results"
        )

        # Section Switch
        self.assertEqual(
            _reconstruct_query("show PPT", record, "section_switch"), "Show PPT results"
        )

        # Continuation
        self.assertEqual(
            _reconstruct_query("again", record, "continuation"), "Show BPET results"
        )
        self.assertEqual(
            _reconstruct_query("same but for platoon 3", record, "continuation"),
            "Show BPET results for platoon 3",
        )


class TestConversationContextEngine(unittest.TestCase):

    def setUp(self):
        self.engine = ConversationContextEngine()

    def test_add_and_retrieve_history(self):
        session_id = "session-123"
        self.engine.add_interaction(
            session_id,
            user_message="Show BPET",
            resolved_query="Show BPET",
            intent={"category": "Performance"},
            entities={},
            filters={"section": "BPET"},
            category="Performance",
            section="BPET",
            operation="Summary",
            payload_summary="BPET records",
        )

        history = self.engine.history(session_id)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].user_message, "Show BPET")

    def test_session_isolation(self):
        self.engine.add_interaction(
            "session-1",
            user_message="BPET",
            resolved_query="BPET",
            intent={},
            entities={},
            filters={},
        )
        self.engine.add_interaction(
            "session-2",
            user_message="PPT",
            resolved_query="PPT",
            intent={},
            entities={},
            filters={},
        )

        self.assertEqual(len(self.engine.history("session-1")), 1)
        self.assertEqual(self.engine.history("session-1")[0].user_message, "BPET")
        self.assertEqual(len(self.engine.history("session-2")), 1)
        self.assertEqual(self.engine.history("session-2")[0].user_message, "PPT")

    def test_max_interactions_rolling_memory(self):
        session_id = "session-rolling"
        for i in range(12):
            self.engine.add_interaction(
                session_id,
                user_message=f"Query {i}",
                resolved_query=f"Query {i}",
                intent={},
                entities={},
                filters={},
            )

        history = self.engine.history(session_id)
        self.assertEqual(len(history), 10)
        self.assertEqual(history[0].user_message, "Query 2")
        self.assertEqual(history[-1].user_message, "Query 11")

    def test_max_sessions_lru_eviction(self):
        # Add MAX_SESSIONS + 5 sessions
        for s in range(505):
            self.engine.add_interaction(
                f"session-{s}",
                user_message="Query",
                resolved_query="Query",
                intent={},
                entities={},
                filters={},
            )

        self.assertEqual(len(self.engine), 500)
        # Oldest sessions (0 to 4) should be evicted
        self.assertEqual(len(self.engine.history("session-0")), 0)
        self.assertEqual(len(self.engine.history("session-4")), 0)
        self.assertEqual(len(self.engine.history("session-5")), 1)

    def test_clear_session(self):
        session_id = "session-clear"
        self.engine.add_interaction(
            session_id,
            user_message="Query",
            resolved_query="Query",
            intent={},
            entities={},
            filters={},
        )
        self.assertEqual(len(self.engine.history(session_id)), 1)
        self.engine.clear(session_id)
        self.assertEqual(len(self.engine.history(session_id)), 0)

    def test_resolve_fresh_query(self):
        res = self.engine.resolve("session-fresh", "Show BPET results")
        self.assertEqual(res.context_source, "fresh")
        self.assertEqual(res.resolved_query, "Show BPET results")
        self.assertFalse(res.needs_clarification)

    def test_resolve_follow_up(self):
        session_id = "session-followup"
        self.engine.add_interaction(
            session_id,
            user_message="Show BPET results",
            resolved_query="Show BPET results",
            intent={"category": "Performance"},
            entities={"platoon_id": "1"},
            filters={"section": "BPET"},
            category="Performance",
            section="BPET",
            operation="Summary",
            payload_summary="",
        )

        res = self.engine.resolve(session_id, "only platoon 2")
        self.assertEqual(res.context_source, "interaction_1")
        self.assertEqual(res.resolved_query, "Show BPET results for platoon 2")
        self.assertEqual(res.carry_forward_filters.get("section"), "BPET")
        self.assertEqual(res.resolved_entities.get("platoon_id"), "1")

    def test_resolve_ambiguity_clarification(self):
        session_id = "session-ambiguity"
        # Add two past interactions with different sections
        self.engine.add_interaction(
            session_id,
            user_message="Show BPET skills",
            resolved_query="Show BPET skills",
            intent={"category": "Skills", "section": "BPET"},
            entities={},
            filters={"section": "BPET"},
            category="Skills",
            section="BPET",
        )
        # Sleep briefly to ensure timestamps differ, or artificially boost recency
        self.engine.add_interaction(
            session_id,
            user_message="Show PPT skills",
            resolved_query="Show PPT skills",
            intent={"category": "Skills", "section": "PPT"},
            entities={},
            filters={"section": "PPT"},
            category="Skills",
            section="PPT",
        )

        # Resolve a moderate follow-up with close relevance to both
        res = self.engine.resolve(session_id, "skills of company alpha")
        # Since follow_up_score is 0.45 (len tokens <= 4, no domain keyword -> +0.45)
        # and both PPT and BPET are relevant, it triggers ambiguity detection.
        self.assertTrue(res.needs_clarification)
        self.assertIn(
            "Do you mean the PPT results or the BPET records?",
            res.clarification_question,
        )
