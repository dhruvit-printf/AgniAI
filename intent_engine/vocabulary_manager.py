"""
vocabulary_manager.py
=====================
Centralized manager for domain vocabulary.
It extracts keywords, operations, synonyms, and aliases from the intent schema
and serves them for spell correction, canonical mapping, and fuzzy searching.
"""
from __future__ import annotations
import re
from typing import FrozenSet, Dict, Optional, Set

from intent_engine.intent_schema import (
    BMI_CATEGORIES,
    CATEGORY_KEYWORDS,
    CLASSES,
    FUZZY_VOCAB,
    GRADING_CATEGORIES,
    LEAVE_TYPES,
    OPERATION_SYNONYMS,
    OPERATIONS_BY_CATEGORY,
    SECTION,
    SPORTS,
    SUBSECTION_ALIASES,
    UNIT_ALIASES,
    VERIFICATION_STATUSES,
)

_COMMON_WORDS: FrozenSet[str] = frozenset(
    {
        "show", "give", "list", "who", "what", "when", "where", "which", "how",
        "is", "are", "was", "were", "the", "a", "an", "of", "in", "on", "for",
        "me", "my", "please", "need", "do", "did", "does", "come", "came",
        "report", "today", "yesterday", "tomorrow", "week", "month", "year",
        "top", "best", "worst", "lowest", "highest", "most", "least", "all",
        "current", "status", "summary", "detail", "details", "and", "with",
        "from", "to", "count", "number", "total", "overall", "each", "per",
        "result", "results", "record", "records", "score", "scores", "data",
    }
)

class VocabularyManager:
    def __init__(self):
        self._domain_vocab: Optional[FrozenSet[str]] = None

    def _add_phrase(self, words: Set[str], phrase: str) -> None:
        for tok in re.findall(r"[a-zA-Z]+", str(phrase).lower()):
            if len(tok) >= 3:
                words.add(tok)

    def get_domain_vocab(self) -> FrozenSet[str]:
        if self._domain_vocab is not None:
            return self._domain_vocab

        words: Set[str] = set(_COMMON_WORDS)

        for phrases in CATEGORY_KEYWORDS.values():
            for phrase in phrases:
                self._add_phrase(words, phrase)

        for op_map in OPERATION_SYNONYMS.values():
            for phrases in op_map.values():
                for phrase in phrases:
                    self._add_phrase(words, phrase)

        for source in (
            SPORTS, CLASSES, GRADING_CATEGORIES, LEAVE_TYPES,
            BMI_CATEGORIES, VERIFICATION_STATUSES, UNIT_ALIASES,
        ):
            for key, value in source.items():
                self._add_phrase(words, key)
                self._add_phrase(words, str(value))

        for aliases in SUBSECTION_ALIASES.values():
            for phrase in aliases:
                self._add_phrase(words, phrase)

        for category, ops in OPERATIONS_BY_CATEGORY.items():
            self._add_phrase(words, category)
            for op in ops:
                self._add_phrase(words, op)

        for section_name in SECTION.keys():
            self._add_phrase(words, section_name)

        for canonical in FUZZY_VOCAB.values():
            self._add_phrase(words, canonical)

        self._domain_vocab = frozenset(words)
        return self._domain_vocab

    def get_fuzzy_vocab(self) -> Dict[str, str]:
        return dict(FUZZY_VOCAB)
        
    def get_unit_aliases(self) -> Dict[str, str]:
        return dict(UNIT_ALIASES)
        
# Global Instance
vocab_manager = VocabularyManager()
