"""
query_normalizer.py
===================

Shared query normalization helpers for admin intent classification.

Pipeline (per clean_query() call):
  1. Strip invisible/zero-width unicode, normalize to NFC.
  2. Strip banners, HTML/code fences, transport-message prefixes.
  3. Normalize curly quotes/dashes to their plain ASCII equivalents.
  4. Collapse runs of 3+ repeated letters down to 2 ("ffffiring" -> "ffiring") —
     a safe, general reduction since real English essentially never triple-
     repeats a letter; the remaining single/double discrepancy (if any) is
     resolved by the fuzzy correction stage below.
  5. Split concatenated words with no space between them ("showattendance"
     -> "show attendance") via dictionary segmentation against the domain
     vocabulary already defined in intent_engine.intent_schema.
  6. Apply the curated FUZZY_VOCAB substring map (fast, precise, handles the
     misspellings someone already anticipated and listed).
  7. Generic adaptive fuzzy correction: any remaining token not already a
     recognized vocabulary word gets matched against the domain vocabulary
     by edit distance, catching typos nobody explicitly enumerated. Only
     applied when there's a single unambiguous best match within a tight
     distance threshold — ambiguous or distant candidates are left alone
     rather than guessed at.
  8. Collapse whitespace/punctuation runs.

clean_query()/admin_normalize_query() keep returning a plain string (every
existing caller chains .lower()/.strip() directly onto the result), so none
of this requires touching the Intent Classifier, Entity Extractor, Query
Planner, or any other downstream component. normalize_query_detailed()
exposes the same pipeline with confidence scoring and a correction trace for
callers that want it (e.g. logging), without changing the plain-string API.

Deliberately out of scope (no current use case in this app): OCR-specific
correction (no image/scan input path), multi-language support beyond the
Hinglish tokens already in FUZZY_VOCAB, and the embedding/LLM-based semantic
matching that already lives in intent_engine/semantic_classifier.py.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

from intent_engine.intent_schema import FUZZY_VOCAB
from intent_engine.vocabulary_manager import vocab_manager

_FUZZY_PATTERNS = [
    (re.compile(rf"(?<!\w){re.escape(misspelling)}(?!\w)", re.IGNORECASE), canonical)
    for misspelling, canonical in sorted(
        FUZZY_VOCAB.items(), key=lambda item: len(item[0]), reverse=True
    )
]

_INVISIBLE_CHARS = ("​", "﻿", "‎", "‏")
_BANNER_PATTERNS = (
    r"(?i)\b(?:agni\s*ai|agniai)\b.*?\bmay make mistakes\b(?:\.\s*|\s*)?",
    r"(?i)\bverify important information\b\.?",
    r"(?i)\bplease verify before use\b\.?",
    r"(?i)\bverify before use\b\.?",
    r"(?i)agniai can make mistakes\.?\s*please verify before use\.?",
)
_BANNER_PATTERNS_RE = tuple(re.compile(p) for p in _BANNER_PATTERNS)
_MESSAGE_PREFIXES = r"(?im)^(?:ws:|socket:|event:|message:)\s*"
_MESSAGE_PREFIXES_RE = re.compile(_MESSAGE_PREFIXES)

# Precompiled — these run on every normalize_query_detailed() call (some,
# like _NON_ALPHA_RE, once per token), so avoiding re-compilation matters:
# clean_query() is called from hot loops (e.g. scoring ~300 equipment-item
# candidates per query in entity_extractor.py), not just once per user query.
_WHITESPACE_RUN_RE = re.compile(r"[^\S\r\n]+")
_REPEATED_PUNCT_RE = re.compile(r"([!?.,;:])\1{1,}")
_PUNCT_SPACING_RE = re.compile(r"\s*([!?.,;:])\s*")
_MULTI_SPACE_RE = re.compile(r"\s+")
_CODE_FENCE_RE = re.compile(r"(?is)```.*?```")
_HTML_TAG_RE = re.compile(r"(?is)<[^>]+>")
_MD_HEADER_RE = re.compile(r"(?m)^#{1,6}\s*")
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z]")


def _collapse_spaces(text: str) -> str:
    text = _WHITESPACE_RUN_RE.sub(" ", text)
    # Collapse repeated punctuation ("???", "!!!!", ".....") BEFORE adding
    # trailing spacing around it — doing it after would leave the repeats
    # separated by spaces ("? ? ?") and unable to match as consecutive.
    text = _REPEATED_PUNCT_RE.sub(r"\1", text)
    text = _PUNCT_SPACING_RE.sub(r"\1 ", text)
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def _apply_fuzzy_vocab(text: str) -> str:
    if not text:
        return text
    corrected = text
    for pattern, canonical in _FUZZY_PATTERNS:
        corrected = pattern.sub(canonical, corrected)
    return corrected


def _strip_banner_text(text: str) -> str:
    stripped = text
    for pattern in _BANNER_PATTERNS_RE:
        stripped = pattern.sub(" ", stripped)
    return stripped


# ---------------------------------------------------------------------------
# Duplicate-character collapsing
# ---------------------------------------------------------------------------

_REPEATED_CHAR_RE = re.compile(r"([a-zA-Z])\1{2,}")


def _collapse_repeated_chars(text: str) -> str:
    """"ffffiring" -> "ffiring", "attttendance" -> "attendance". Only ever
    collapses a run of 3+ to 2 — never to 1 — so a legitimate double letter
    ("attendance", "class") is never destroyed; any remaining single/double
    mismatch is left for the fuzzy-correction stage to resolve."""
    return _REPEATED_CHAR_RE.sub(lambda m: m.group(1) * 2, text)


# ---------------------------------------------------------------------------
# Domain vocabulary — harvested from the schema that already exists, not
# duplicated. Used by both the word-segmentation and fuzzy-correction stages.
# ---------------------------------------------------------------------------


def _vocab() -> FrozenSet[str]:
    return vocab_manager.get_domain_vocab()


_VOCAB_INDEX: Optional[Dict[Tuple[int, str], Tuple[str, ...]]] = None


def _vocab_index() -> Dict[Tuple[int, str], Tuple[str, ...]]:
    """Vocabulary indexed by (length, first letter), built once from the
    static domain vocabulary. Fuzzy correction only ever needs candidates
    within +/-max_dist of the input word's length that also share its first
    letter — real-world typos essentially never change the first letter —
    so this index turns an O(vocab_size) scan per token into a handful of
    lookups, which matters because clean_query() is called from hot loops
    (e.g. scoring ~300 equipment-item candidates per query in
    entity_extractor.py), not just once per user query."""
    global _VOCAB_INDEX
    if _VOCAB_INDEX is None:
        buckets: Dict[Tuple[int, str], List[str]] = {}
        for word in _vocab():
            buckets.setdefault((len(word), word[0]), []).append(word)
        _VOCAB_INDEX = {key: tuple(words) for key, words in buckets.items()}
    return _VOCAB_INDEX


def _candidates_near_length(length: int, max_dist: int, first_letter: str) -> List[str]:
    index = _vocab_index()
    out: List[str] = []
    for candidate_len in range(length - max_dist, length + max_dist + 1):
        out.extend(index.get((candidate_len, first_letter), ()))
    return out


# ---------------------------------------------------------------------------
# Missing-space splitting ("showattendance" -> "show attendance")
# ---------------------------------------------------------------------------


def _segment_word(token: str, vocab: FrozenSet[str]) -> Optional[List[str]]:
    """DP word-break: return a full segmentation of *token* into >=2 known
    vocabulary words, or None if no full segmentation exists. Among valid
    segmentations, prefers fewer (i.e. longer) segments."""
    n = len(token)
    lower = token.lower()
    # best[i] = fewest-segment split of lower[:i], or None if unreachable
    best: List[Optional[List[str]]] = [None] * (n + 1)
    best[0] = []
    for i in range(1, n + 1):
        for j in range(max(0, i - 20), i):
            if best[j] is None:
                continue
            piece = lower[j:i]
            if len(piece) < 3 or piece not in vocab:
                continue
            candidate = best[j] + [piece]
            if best[i] is None or len(candidate) < len(best[i]):
                best[i] = candidate
    result = best[n]
    if result is None or len(result) < 2:
        return None
    return result


def _split_concatenated_words(text: str) -> str:
    vocab = _vocab()
    out_tokens: List[str] = []
    for tok in text.split(" "):
        core = _NON_ALPHA_RE.sub("", tok)
        if len(core) < 8 or core.lower() in vocab:
            out_tokens.append(tok)
            continue
        segments = _segment_word(core, vocab)
        if segments:
            out_tokens.append(" ".join(segments))
        else:
            out_tokens.append(tok)
    return " ".join(out_tokens)


# ---------------------------------------------------------------------------
# Generic fuzzy correction against the domain vocabulary (beyond the curated
# FUZZY_VOCAB map) — reuses the Damerau-Levenshtein implementation already
# in intent_engine.intent_classifier rather than duplicating it.
# ---------------------------------------------------------------------------


def _max_distance_for(word_len: int) -> int:
    return 1 if word_len <= 5 else 2


# Ordinary English words that must NEVER be "corrected" against the domain
# vocabulary, no matter how close the edit distance — the domain vocabulary
# is narrow (a few hundred military/admin terms), so an everyday word can
# easily land 1 edit away from an unrelated domain term by pure coincidence
# (e.g. "food" is one substitution from "foot", which is itself a term from
# a BPET subsection). Without a full English dictionary to confirm a token
# is already a real word, this denylist is the guard against that class of
# false positive — expand it as new collisions turn up.
_PROTECTED_COMMON_WORDS: FrozenSet[str] = frozenset(
    {
        "food", "foods", "water", "house", "world", "point", "place", "thing",
        "things", "people", "person", "family", "friend", "friends", "money",
        "phone", "email", "address", "school", "office", "field", "party",
        "night", "morning", "evening", "hour", "hours", "minute", "minutes",
        "second", "seconds", "book", "books", "paper", "papers", "letter",
        "letters", "road", "roads", "city", "cities", "state", "states",
        "country", "countries", "story", "stories", "movie", "music", "game",
        "games", "team", "teams", "line", "lines", "area", "areas", "case",
        "cases", "fact", "facts", "level", "levels", "hand", "hands", "part",
        "parts", "eye", "eyes", "life", "lives", "child", "children", "care",
        "word", "words", "power", "moment", "moments", "issue", "issues",
        "side", "sides", "kind", "kinds", "head", "heads", "help", "want",
        "wants", "wanted", "look", "looks", "looking", "looked", "seem",
        "seems", "seemed", "feel", "feels", "feeling", "felt", "leave",
        "leaves", "call", "calls", "called", "calling", "try", "tries",
        "tried", "trying", "ask", "asks", "asked", "asking", "work", "works",
        "worked", "working", "town", "towns", "matter", "matters", "problem",
        "problems",
    }
)


def _fuzzy_correct_tokens(text: str) -> Tuple[str, List[str]]:
    try:
        from intent_engine.intent_classifier import _damerau_levenshtein
    except Exception:
        return text, []

    vocab = _vocab()
    corrections: List[str] = []
    out_words: List[str] = []
    for raw_word in text.split(" "):
        core = _NON_ALPHA_RE.sub("", raw_word)
        if len(core) < 4 or not core.isalpha():
            out_words.append(raw_word)
            continue
        lower = core.lower()
        if lower in _PROTECTED_COMMON_WORDS:
            out_words.append(raw_word)
            continue
        if lower in vocab:
            out_words.append(raw_word)
            continue

        max_dist = _max_distance_for(len(lower))
        best_word: Optional[str] = None
        best_dist = max_dist + 1
        tie = False
        for candidate in _candidates_near_length(len(lower), max_dist, lower[0]):
            dist = _damerau_levenshtein(lower, candidate)
            if dist < best_dist:
                best_dist = dist
                best_word = candidate
                tie = False
            elif dist == best_dist and candidate != best_word:
                tie = True

        if best_word and best_dist <= max_dist and not tie:
            replaced = raw_word.replace(core, best_word) if core else best_word
            out_words.append(replaced)
            corrections.append(f"'{raw_word}' -> '{replaced}' (distance={best_dist})")
        else:
            out_words.append(raw_word)

    return " ".join(out_words), corrections


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class NormalizationResult:
    original: str
    normalized: str
    confidence: float
    corrections: List[str] = field(default_factory=list)


def clean_query(query: str) -> str:
    """Normalize query text before intent classification."""
    return normalize_query_detailed(query).normalized


def normalize_query_detailed(query: str) -> NormalizationResult:
    """Same normalization as clean_query(), plus a confidence score and a
    human-readable trace of every correction applied — for logging/
    observability, not required by any downstream component."""
    if not query:
        return NormalizationResult(original="", normalized="", confidence=1.0)

    original = str(query)
    text = original
    for char in _INVISIBLE_CHARS:
        text = text.replace(char, "")
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CODE_FENCE_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _MESSAGE_PREFIXES_RE.sub("", text)
    text = _strip_banner_text(text)
    text = _MD_HEADER_RE.sub("", text)
    text = text.replace("â€™", "'").replace("â€œ", '"').replace("â€", '"')
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    corrections: List[str] = []

    before = text
    text = _collapse_repeated_chars(text)
    if text != before:
        corrections.append("collapsed repeated characters")

    before = text
    text = _split_concatenated_words(text)
    if text != before:
        corrections.append("split concatenated words")

    before = text
    text = _apply_fuzzy_vocab(text)
    if text != before:
        corrections.append("applied curated typo map")

    text, fuzzy_corrections = _fuzzy_correct_tokens(text)
    corrections.extend(fuzzy_corrections)

    text = _collapse_spaces(text)

    confidence = max(0.3, 1.0 - 0.08 * len(corrections))
    return NormalizationResult(
        original=original,
        normalized=text,
        confidence=round(confidence, 2),
        corrections=corrections,
    )


def admin_normalize_query(query: str) -> str:
    """Backward-compatible alias for legacy pipeline normalization."""
    return clean_query(query)
