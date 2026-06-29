"""
context_engine.py
=================
Conversation Context Engine for AgniAI.

Maintains rolling memory of the last 10 complete interactions per session.
Resolves follow-up queries by reconstructing the full query from conversation
context before passing to the Intent Engine.

Design goals:
  - No external ML dependencies — lightweight, runs in-path
  - Thread-safe per-session storage (mirrors memory.py conventions)
  - Never blindly appends previous queries — always understands intent first
  - Matches the most semantically relevant past interaction, not just the latest
"""

from __future__ import annotations

import re
import time
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, field
from threading import RLock
from typing import Any, Deque, Dict, List, Optional, Tuple

MAX_SESSIONS = 500
MAX_INTERACTIONS = 10

# ─── Follow-up Signal Patterns ───────────────────────────────────────────────

_PRONOUN_TOKENS = frozenset(
    {
        "them",
        "those",
        "these",
        "their",
        "they",
        "him",
        "her",
        "it",
        "its",
        "which of",
        "who among",
        "any of",
        "some of",
        "each of",
    }
)

_CONTINUATION_TOKENS = frozenset(
    {
        "now",
        "then",
        "again",
        "instead",
        "likewise",
        "also",
        "then show",
        "now show",
        "next",
        "similarly",
    }
)

_COMPARISON_MARKERS = (
    "compare with",
    "comparison with",
    "vs ",
    "versus ",
    "compare against",
    "now compare",
    "also compare",
)

_VISUALIZATION_MARKERS = (
    "show as",
    "make it a",
    "as a ",
    "display as",
    "bar chart",
    "pie chart",
    "line chart",
    "as table",
    "as chart",
    "as bar",
    "as pie",
    "as line",
    "in a bar",
    "in a table",
    "convert to chart",
    "convert to table",
)

_RANKING_PATTERN = re.compile(
    r"\b(top|bottom|highest|lowest|best|worst)\s+(\d+)\b", re.IGNORECASE
)

_NUMBER_ONLY = re.compile(r"^\d+$")

_SECTION_ALIASES: Dict[str, str] = {
    "bpet": "BPET",
    "bept": "BPET",
    "ppt": "PPT",
    "firing": "Firing",
    "drill": "Drill",
}

# Domain keywords — their presence signals an independent topic
_DOMAIN_KEYWORDS = frozenset(
    {
        "bpet",
        "ppt",
        "firing",
        "drill",
        "leave",
        "medical",
        "attendance",
        "performance",
        "score",
        "marks",
        "grade",
        "grading",
        "football",
        "cricket",
        "blood",
        "bmi",
        "distribution",
        "trend",
        "verification",
        "equipment",
        "issued",
        "procured",
        "absent",
        "absconded",
        "strength",
        "overdue",
        "pending",
        "approved",
        "rejected",
        "overall",
        "average",
        "pass",
        "fail",
        "improvement",
        "decline",
        "drop",
        "attempt",
        "summary",
        "verify",
        "cleared",
        "responded",
        "roster",
        "sport",
        "sports",
        "medical leave",
        "sick",
        "disease",
        "fever",
        "malaria",
        "injury",
        "current leave",
        "on leave",
    }
)

# Phrases that are unambiguously continuations regardless of domain keywords
_EXPLICIT_FOLLOWUP_PHRASES = (
    "top 5",
    "top 10",
    "top 3",
    "top 20",
    "top 50",
    "bottom 5",
    "bottom 10",
    "bottom 3",
    "only these",
    "list them",
    "show them",
    "their details",
    "compare with",
    "vs ",
    "versus ",
    "same company",
    "same platoon",
    "same batch",
    "this platoon",
    "that company",
    "those candidates",
    "again",
    "likewise",
    "show only",
    "only the",
)


# ─── Data Structures ─────────────────────────────────────────────────────────


@dataclass
class InteractionRecord:
    """One complete resolved interaction stored in conversation memory."""

    user_message: str
    resolved_query: str
    intent: Dict[str, Any]
    entities: Dict[str, Any]
    filters: Dict[str, Any]
    category: Optional[str]
    section: Optional[str]
    operation: str
    payload_summary: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ContextResolution:
    """Result returned by ConversationContextEngine.resolve()."""

    resolved_query: str
    resolved_entities: Dict[str, Any]
    context_source: str  # "fresh" | "interaction_N" | "clarification"
    needs_clarification: bool
    clarification_question: Optional[str]
    carry_forward_filters: Dict[str, Any]
    matched_interaction: Optional[InteractionRecord]


# ─── Follow-up Detection ─────────────────────────────────────────────────────


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", _normalize(text))


def _jaccard(a: List[str], b: List[str]) -> float:
    """Token-level Jaccard similarity."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _has_domain_keyword(tokens: List[str]) -> bool:
    return any(t in _DOMAIN_KEYWORDS for t in tokens)


def _has_pronoun(tokens: List[str]) -> bool:
    return any(t in _PRONOUN_TOKENS for t in tokens)


def _extract_new_section(text: str) -> Optional[str]:
    """Return the first section alias found in text, or None."""
    norm = _normalize(text)
    for alias, label in _SECTION_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", norm):
            return label
    return None


def _compute_follow_up_score(msg: str) -> float:
    """
    Return a score in [0, 1] indicating how likely the message is a follow-up.
    > 0.55 → treat as follow-up.
    0.35–0.55 → uncertain.
    < 0.35 → treat as fresh query.
    """
    norm = _normalize(msg)
    tokens = _tokenize(msg)
    score = 0.0

    # Very short messages with no domain keyword → very likely follow-up
    if len(tokens) <= 2 and not _has_domain_keyword(tokens):
        score += 0.7
    elif len(tokens) <= 4 and not _has_domain_keyword(tokens):
        score += 0.45

    # Explicit follow-up phrases
    for phrase in _EXPLICIT_FOLLOWUP_PHRASES:
        if phrase in norm:
            score += 0.5
            break

    # Pronoun references
    if _has_pronoun(tokens):
        score += 0.4

    # Continuation tokens at the start
    for tok in _CONTINUATION_TOKENS:
        if norm.startswith(tok + " ") or norm == tok:
            score += 0.35
            break

    # Comparison markers (could be follow-up "compare with X" or standalone)
    for marker in _COMPARISON_MARKERS:
        if marker in norm:
            score += 0.3
            break

    # Visualization change signals
    for marker in _VISUALIZATION_MARKERS:
        if marker in norm:
            score += 0.5
            break

    # Pure number (e.g., user typed "10")
    if _NUMBER_ONLY.match(norm):
        score += 0.6

    # Filter starters without domain ("only platoon 2", "for company alpha")
    if re.match(r"^(only|just|for|from|show only|show for)\s+", norm):
        if not _has_domain_keyword(tokens):
            score += 0.45

    return min(score, 1.0)


# ─── Relevance Scoring ────────────────────────────────────────────────────────


def _compute_relevance(msg: str, record: InteractionRecord) -> float:
    """
    Score how relevant a past interaction is to the current message.
    Higher = more relevant.
    """
    norm = _normalize(msg)
    query_terms = _tokenize(msg)
    if not query_terms:
        return 0.0
    score = 0.0

    # Category match
    if record.category:
        cat_norm = record.category.lower()
        # Check if any token relates to the category
        cat_tokens = _tokenize(record.category)
        if any(t in query_terms for t in cat_tokens):
            score += 0.4
        # More specific: does message mention the category directly
        if cat_norm in norm:
            score += 0.1

    # Section match (BPET, PPT, etc.)
    if record.section:
        sec_norm = record.section.lower()
        if sec_norm in norm:
            score += 0.45
        # Also check aliases
        for alias, label in _SECTION_ALIASES.items():
            if label == record.section and alias in norm:
                score += 0.35
                break

    # Operation match
    if record.operation and record.operation != "lookup":
        op_tokens = _tokenize(record.operation)
        if any(t in query_terms for t in op_tokens):
            score += 0.1

    # Token overlap with previous resolved query
    prev_tokens = _tokenize(record.resolved_query)
    overlap = _jaccard(query_terms, prev_tokens)
    score += overlap * 0.3

    # Recency boost (more recent = slightly preferred when scores are tied)
    age_secs = time.time() - record.timestamp
    score += max(0.0, 0.05 * (1 - age_secs / 3600))  # decays over 1 hour

    return score


# ─── Query Reconstruction ─────────────────────────────────────────────────────


def _detect_follow_up_kind(msg: str, matched: InteractionRecord) -> str:
    """
    Classify the type of follow-up to guide reconstruction.
    Returns one of: ranking, comparison, visualization, filter, section_switch,
    pronoun, continuation, unknown.
    """
    norm = _normalize(msg)
    tokens = _tokenize(msg)

    # Visualization change
    for marker in _VISUALIZATION_MARKERS:
        if marker in norm:
            return "visualization"

    # Comparison addition
    for marker in _COMPARISON_MARKERS:
        if marker in norm:
            return "comparison"

    # Ranking refinement
    if _RANKING_PATTERN.search(msg):
        return "ranking"

    # Pronoun reference
    if _has_pronoun(tokens):
        return "pronoun"

    # Section switch (new section, same operation)
    new_sec = _extract_new_section(msg)
    if new_sec and new_sec != matched.section:
        return "section_switch"

    # Filter addition/override
    if re.match(r"^(only|just|for|from|in)\s+", norm):
        return "filter"

    # Pure continuation ("again", "likewise")
    for tok in ("again", "likewise", "same"):
        if norm == tok or norm.startswith(tok + " "):
            return "continuation"

    return "unknown"


def _reconstruct_query(
    raw_msg: str,
    matched: InteractionRecord,
    follow_up_kind: str,
) -> str:
    """
    Build a complete, intent-engine-parseable query by merging the current
    message with context from the matched past interaction.
    """
    base = matched.resolved_query.strip()
    msg = raw_msg.strip()

    if follow_up_kind == "visualization":
        # Strip redundant leading "show" / "make it" before appending
        hint = re.sub(
            r"^(show\s+as|make\s+it\s+a?|display\s+as|convert\s+to)\s*",
            "",
            msg,
            flags=re.IGNORECASE,
        ).strip()
        return f"{base} as {hint}" if hint else f"{base} {msg}"

    if follow_up_kind == "comparison":
        # "compare with X" → "Compare {prev_section} with X"
        norm = _normalize(msg)
        for marker in _COMPARISON_MARKERS:
            if marker in norm:
                tail = norm.split(marker, 1)[1].strip()
                source = matched.section or matched.category or "previous results"
                # Capitalise the comparison target from the original casing
                tail_original = re.split(marker, msg, maxsplit=1, flags=re.IGNORECASE)[
                    -1
                ].strip()
                return f"Compare {source} with {tail_original}"
        return f"{base} {msg}"

    if follow_up_kind == "ranking":
        rank_m = _RANKING_PATTERN.search(msg)
        if rank_m:
            rank_phrase = rank_m.group(0)
            # Strip existing rank from base
            base_no_rank = _RANKING_PATTERN.sub("", base).strip()
            # Remove leading verb to get clean topic
            topic = re.sub(r"^show\s+", "", base_no_rank, flags=re.IGNORECASE).strip()
            return f"Show {rank_phrase} {topic}"
        return f"{base} {msg}"

    if follow_up_kind == "filter":
        # "only platoon 2" → "{base} for platoon 2"
        stripped = re.sub(
            r"^(only|just|show\s+only|show\s+for)\s+", "", msg, flags=re.IGNORECASE
        ).strip()
        connector = "for" if not stripped.lower().startswith("for ") else ""
        return f"{base} {connector} {stripped}".strip()

    if follow_up_kind == "pronoun":
        # Pronoun refers to the previous query's results — re-execute same base
        # but include the current verb/action if meaningful
        action_verbs = {"list", "show", "display", "get", "fetch", "give", "find"}
        tokens = _tokenize(msg)
        if tokens and tokens[0] in action_verbs and len(tokens) <= 3:
            return base
        return f"{base} — {msg}"

    if follow_up_kind == "section_switch":
        new_sec = _extract_new_section(msg)
        if new_sec and matched.section:
            # Replace old section with new one in base query
            replaced = re.sub(
                re.escape(matched.section), new_sec, base, flags=re.IGNORECASE
            )
            if replaced != base:
                return replaced
        # Fall through: append new message
        return f"{msg} {matched.operation or ''}".strip()

    if follow_up_kind == "continuation":
        # "again" → repeat the previous query
        norm = _normalize(msg)
        if norm in ("again", "same", "likewise", "repeat"):
            return base
        # "same but for X" → base + " for X"
        for marker in ("but for", "but only", "but with"):
            if marker in norm:
                tail = norm.split(marker, 1)[1].strip()
                return f"{base} for {tail}"
        return f"{base} {msg}"

    # unknown — safest fallback: prepend base context
    return f"{base} {msg}"


# ─── Main Engine ─────────────────────────────────────────────────────────────


class ConversationContextEngine:
    """
    Thread-safe, session-aware conversation context engine.

    Stores up to MAX_INTERACTIONS (10) interaction records per session.
    Resolves incoming messages as either fresh queries or follow-ups, and
    returns a fully reconstructed query ready for the Intent Engine.
    """

    def __init__(self) -> None:
        self._sessions: "OrderedDict[str, Deque[InteractionRecord]]" = OrderedDict()
        self._lock = RLock()

    # ── Write ──────────────────────────────────────────────────────────────

    def add_interaction(
        self,
        session_id: str,
        *,
        user_message: str,
        resolved_query: str,
        intent: Dict[str, Any],
        entities: Dict[str, Any],
        filters: Dict[str, Any],
        category: Optional[str] = None,
        section: Optional[str] = None,
        operation: str = "lookup",
        payload_summary: str = "",
    ) -> None:
        """Record a completed interaction in session memory."""
        record = InteractionRecord(
            user_message=user_message,
            resolved_query=resolved_query,
            intent=intent,
            entities=entities,
            filters=filters,
            category=category,
            section=section,
            operation=operation,
            payload_summary=payload_summary,
        )
        with self._lock:
            key = session_id or "default"
            bucket = self._sessions.get(key)
            if bucket is None:
                bucket = deque(maxlen=MAX_INTERACTIONS)
                self._sessions[key] = bucket
            else:
                self._sessions.move_to_end(key)
            bucket.append(record)
            # Evict oldest sessions when limit exceeded
            while len(self._sessions) > MAX_SESSIONS:
                self._sessions.popitem(last=False)

    # ── Read ───────────────────────────────────────────────────────────────

    def history(self, session_id: str) -> List[InteractionRecord]:
        """Return the interaction history for a session (oldest first)."""
        with self._lock:
            return list(self._sessions.get(session_id or "default", ()))

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id or "default", None)

    # ── Resolve ────────────────────────────────────────────────────────────

    def resolve(self, session_id: str, raw_message: str) -> ContextResolution:
        """
        Resolve raw_message against conversation history.

        Returns a ContextResolution with:
          - resolved_query: the fully reconstructed query for the Intent Engine
          - resolved_entities: entities carried forward from context
          - context_source: "fresh" | "interaction_N" | "clarification"
          - needs_clarification: True if ambiguity requires a user question
          - clarification_question: the question to ask (when needs_clarification)
          - carry_forward_filters: filters from the matched past interaction
          - matched_interaction: the InteractionRecord used, or None for fresh
        """
        history = self.history(session_id)
        _fresh = ContextResolution(
            resolved_query=raw_message,
            resolved_entities={},
            context_source="fresh",
            needs_clarification=False,
            clarification_question=None,
            carry_forward_filters={},
            matched_interaction=None,
        )

        if not history:
            return _fresh

        # Step 1 — decide if this is a follow-up
        follow_up_score = _compute_follow_up_score(raw_message)
        if follow_up_score < 0.35:
            return _fresh

        # Step 2 — score each past interaction for relevance
        # Note: for strong follow-ups the message intentionally omits domain keywords,
        # so we treat recency (most recent = index 0) as the default match.
        scored: List[Tuple[float, int, InteractionRecord]] = []
        for idx, record in enumerate(reversed(history)):  # idx=0 is most recent
            rel = _compute_relevance(raw_message, record)
            scored.append((rel, idx, record))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_idx, best_record = scored[0]

        # For STRONG follow-ups (unambiguous continuation), skip the relevance
        # threshold — the message is short precisely because it inherits context.
        # For MODERATE follow-ups, require at least a weak relevance signal.
        if follow_up_score < 0.6 and best_score < 0.1:
            return _fresh

        # Step 3 — ambiguity check: two equally relevant past interactions
        # Only meaningful for moderate follow-ups with real relevance signals.
        if (
            len(scored) >= 2
            and follow_up_score < 0.7  # not an obvious follow-up
            and best_score >= 0.2  # both candidates are actually relevant
            and (best_score - scored[1][0]) < 0.08  # nearly tied
        ):
            a = scored[0][2]
            b = scored[1][2]
            topic_a = a.section or a.category or a.resolved_query
            topic_b = b.section or b.category or b.resolved_query
            if topic_a != topic_b:
                question = (
                    f"Do you mean the {topic_a} results or the {topic_b} records?"
                )
                return ContextResolution(
                    resolved_query=raw_message,
                    resolved_entities={},
                    context_source="clarification",
                    needs_clarification=True,
                    clarification_question=question,
                    carry_forward_filters={},
                    matched_interaction=None,
                )

        # For strong follow-ups prefer the most recent interaction (idx=0) over a
        # higher-scoring older one, unless the older one is clearly more relevant.
        if follow_up_score >= 0.6 and best_idx != 0:
            recent_entry = next(((s, i, r) for s, i, r in scored if i == 0), None)
            if recent_entry is not None and (best_score - recent_entry[0]) < 0.2:
                best_record = recent_entry[2]
                best_idx = 0

        # Step 4 — reconstruct the full query
        kind = _detect_follow_up_kind(raw_message, best_record)
        reconstructed = _reconstruct_query(raw_message, best_record, kind)

        # Step 5 — collect carry-forward filters (do not override explicit values)
        carry_filters: Dict[str, Any] = {}
        for key in (
            "section",
            "category",
            "company_id",
            "companyId",
            "platoon_id",
            "platoonId",
            "batch_id",
            "batchId",
            "group_by",
            "sort_by",
            "top_n",
        ):
            val = best_record.filters.get(key) or best_record.entities.get(key)
            if val is not None:
                carry_filters[key] = val

        source_label = f"interaction_{best_idx + 1}"
        return ContextResolution(
            resolved_query=reconstructed,
            resolved_entities=dict(best_record.entities),
            context_source=source_label,
            needs_clarification=False,
            clarification_question=None,
            carry_forward_filters=carry_filters,
            matched_interaction=best_record,
        )

    def __len__(self) -> int:
        with self._lock:
            return sum(len(b) for b in self._sessions.values())


# Singleton — imported by admin_pipeline.py
context_engine = ConversationContextEngine()
