"""
query_intelligence_engine.py
============================
The Query Intelligence Engine (QIE) acts as a preprocessing and normalization pipeline
before the intent classifier. It standardizes language, corrects spelling/grammar,
resolves military abbreviations, and constructs a canonical query.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import sys
import os

# Ensure we can import query_normalizer which is at the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import query_normalizer


@dataclass
class NormalizedQuery:
    original_query: str
    normalized_text: str
    canonical_text: str
    corrections: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 1.0


class QueryIntelligenceEngine:
    def __init__(self):
        pass

    def _canonicalize_terms(self, text: str) -> str:
        """Phase 7: Canonical Query Builder. Replaces known synonyms/aliases with canonical forms."""
        # Simple whole-word replacement using vocabulary manager
        from intent_engine.vocabulary_manager import vocab_manager

        aliases = vocab_manager.get_unit_aliases()
        # sort by length descending to match longest first
        for alias, canonical in sorted(
            aliases.items(), key=lambda x: len(x[0]), reverse=True
        ):
            # regex word boundary replace
            import re

            text = re.sub(rf"(?i)\b{re.escape(alias)}\b", canonical, text)
        return text

    def process(self, raw_query: str) -> NormalizedQuery:
        """
        Execute the full QIE pipeline on the raw query.
        """
        if not raw_query or not raw_query.strip():
            return NormalizedQuery(
                original_query=raw_query,
                normalized_text="",
                canonical_text="",
            )

        # Phase 2 & 4: Normalization (Noise, Unicode, Whitespace, Spell Correction)
        normalization_result = query_normalizer.normalize_query_detailed(raw_query)

        # Phase 7: Canonical Query Builder
        canonical_text = self._canonicalize_terms(normalization_result.normalized)

        # Phase 8: Confidence Engine
        confidence = normalization_result.confidence
        if canonical_text != normalization_result.normalized:
            # We made a strong canonical substitution, boost confidence of exact match
            confidence = min(1.0, confidence + 0.05)

        # Build initial response
        result = NormalizedQuery(
            original_query=normalization_result.original,
            normalized_text=normalization_result.normalized,
            canonical_text=canonical_text,
            corrections=[{"type": c} for c in normalization_result.corrections],
            confidence_score=confidence,
        )

        return result


# Global instance for easy import
qie = QueryIntelligenceEngine()


def process_query(raw_query: str) -> NormalizedQuery:
    return qie.process(raw_query)
