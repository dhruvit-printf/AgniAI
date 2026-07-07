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

import logging
import re
import time
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass, field
from threading import RLock
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

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
        "who among",
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

_NUMERIC_ID_PATTERN = re.compile(
    r"\b(?:company|coy|platoon|plt|batch|agniveer)\s+\d+\b", re.IGNORECASE
)
_SPORT_TERMS = frozenset(
    {
        "cricket", "football", "soccer", "volleyball", "hockey", "basketball",
        "kabaddi", "tennis", "badminton", "swimming", "wrestling", "boxing",
        "archery", "shooting", "weightlifting", "gymnastics", "cycling", "chess",
        "carrom", "squash", "billiards", "snooker", "judo", "karate", "taekwondo",
        "handball", "rugby", "golf", "rowing", "sailing", "fencing", "polo",
        "marathon", "running", "athletics",
    }
)
_DATE_ENTITY_PATTERN = re.compile(
    r"\b(?:january|february|march|april|may|june|july|august|september|"
    r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
    r"(?:\s+\d{4})?\b",
    re.IGNORECASE,
)
_AGNIVEER_NO_PATTERN = re.compile(r"\b[A-Z]?\d{5,8}[A-Z]?\b")


def _is_explicit_continuation_phrase(msg: str) -> bool:
    """
    True when the message is structurally a filter/continuation/pronoun
    phrase ("only platoon 2", "same but for company alpha", "show them").
    In that case any entity inside it is refining the previous query, not
    stating a brand-new standalone one, so the entity guard below must not
    override it.
    """
    norm = _normalize(msg)
    tokens = _tokenize(msg)
    if re.match(r"^(only|just|for|from|show only|show for)\s+", norm):
        return True
    for phrase in _EXPLICIT_FOLLOWUP_PHRASES:
        if phrase in norm:
            return True
    if _has_pronoun(tokens):
        return True
    for tok in _CONTINUATION_TOKENS:
        if norm.startswith(tok + " ") or norm == tok:
            return True
    return False


def _has_new_entity(msg: str) -> bool:
    """Return True if the message contains a freshly typed entity that makes it an
    independent query, not a follow-up.  Uses simple patterns to avoid importing the
    full entity extractor (which would be a circular dependency)."""
    norm = _normalize(msg)
    if _NUMERIC_ID_PATTERN.search(msg):
        return True
    if _DATE_ENTITY_PATTERN.search(norm):
        return True
    if _AGNIVEER_NO_PATTERN.search(msg):
        return True
    tokens = _tokenize(msg)
    if any(t in _SPORT_TERMS for t in tokens):
        return True
    return False


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
    >= 0.55 → treat as follow-up (see resolve()'s `< 0.55` reject check).
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
        pronoun_idx = next((i for i, t in enumerate(tokens) if t in _PRONOUN_TOKENS), -1)
        if 0 <= pronoun_idx <= 3:
            score += 0.4
        else:
            score += 0.1

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

    # unknown — do not fabricate continuity by fusing unrelated text; classify
    # the user's literal message on its own.
    return msg


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
            key = session_id or ""
            if not key:
                return  # ephemeral / no-session — nothing to persist
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
        if not session_id:
            return []
        with self._lock:
            return list(self._sessions.get(session_id, ()))

    def clear(self, session_id: str) -> None:
        if not session_id:
            return
        with self._lock:
            self._sessions.pop(session_id, None)

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

        # Step 1 — decide if this is a follow-up. Scores >= 0.35 are at least
        # "uncertain" (see _compute_follow_up_score docstring) and are worth
        # checking against history — the weak-relevance guard below and the
        # ambiguity check handle the low end of that range, so a message
        # doesn't need to clear the full 0.55 "confirmed follow-up" bar just
        # to be eligible for ambiguity detection.
        follow_up_score = _compute_follow_up_score(raw_message)
        if follow_up_score < 0.35:
            return _fresh

        # Entity guard: a message with freshly stated entities is a new
        # question — UNLESS the message is structurally a filter/continuation
        # phrase ("only platoon 2"), where the entity is refining the
        # previous query rather than starting an unrelated one.
        if _has_new_entity(raw_message) and not _is_explicit_continuation_phrase(
            raw_message
        ):
            logger.debug(
                "context_engine.resolve: entity guard triggered — treating as fresh "
                "| session=%s | follow_up_score=%.2f | msg=%r",
                session_id, follow_up_score, raw_message,
            )
            return _fresh

        # Step 2 — score each past interaction for relevance
        # Note: for strong follow-ups the message intentionally omits domain keywords,
        # so we treat recency (most recent = index 0) as the default match.
        scored: List[Tuple[float, int, InteractionRecord]] = []
        current_time = time.time()
        for idx, record in enumerate(reversed(history)):  # idx=0 is most recent
            if current_time - record.timestamp > 300:
                continue  # TTL: ignore interactions older than 5 minutes
            rel = _compute_relevance(raw_message, record)
            scored.append((rel, idx, record))
            
        if not scored:
            return _fresh

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

        # Step 4 — determine whether the matched interaction is actually the same
        # topic as the current message. This is computed BEFORE reconstruction so
        # that a genuine topic change never gets its query text fused with stale
        # text, and never carries forward stale entities/filters (prevents stale
        # companyId/platoonId/etc. leaking across an unrelated follow-up).
        _matched_cat = best_record.category
        _classification_failed = False
        _cur_cat: Optional[str] = None

        # Only reclassify when the message itself contains an explicit domain
        # keyword. A bare filter fragment ("only platoon 2", "for company
        # alpha") has no category signal of its own — re-classifying it in
        # isolation lets the fuzzy matcher latch onto an unrelated category
        # (e.g. "platoon" alone gets fuzzy-matched to "Strength" at medium
        # confidence) purely because a unit/number was mentioned, which then
        # incorrectly looks like a topic change and discards a legitimate
        # follow-up. A real topic change ("now show leave records") always
        # carries its own domain keyword, so this still catches genuine
        # switches without misfiring on plain filters.
        if _has_domain_keyword(_tokenize(raw_message)):
            # Import inline to avoid circular dependency at module level
            try:
                from intent_engine.intent_classifier import classify_intent as _clf
                _cur_intent = _clf(raw_message)
                _cur_cat = _cur_intent.get("category")
            except Exception:
                logger.warning(
                    "context_engine.resolve: inline classify_intent failed during "
                    "carry-forward category check | session=%s | msg=%r",
                    session_id, raw_message, exc_info=True,
                )
                _classification_failed = True

        # A classification failure must fail CLOSED (treat as a topic change, do
        # not carry forward) rather than silently defaulting to "same category".
        _same_category = not _classification_failed and (
            _matched_cat is None
            or _cur_cat is None
            or _cur_cat == "Unknown"
            or _matched_cat == _cur_cat
        )

        # Step 5 — reconstruct the query text. Only fuse the current message with
        # the matched interaction's resolved query when the topic actually
        # matches — otherwise classify the user's literal message on its own.
        if _same_category:
            kind = _detect_follow_up_kind(raw_message, best_record)
            reconstructed = _reconstruct_query(raw_message, best_record, kind)
        else:
            reconstructed = raw_message

        # Step 6 — collect carry-forward filters (do not override explicit values).
        carry_filters: Dict[str, Any] = {}

        if _same_category:
            for key, val in {**best_record.entities, **best_record.filters}.items():
                if key not in (
                    "raw_query",
                    "query_type",
                    "user_message",
                    "timestamp",
                    "payload_summary",
                ):
                    # Normalize keys to camelCase for backend payload mapping
                    camel_key = "".join(
                        part[0].upper() + part[1:] if idx > 0 else part
                        for idx, part in enumerate(key.split("_"))
                    )
                    if camel_key == "class":
                        camel_key = "class_"
                    if val is not None:
                        carry_filters[camel_key] = val

        source_label = f"interaction_{best_idx + 1}"
        logger.info(
            "context_engine.resolve: %s | session=%s | followup_score=%.2f | "
            "resolved_query=%r | carried_filters=%r",
            source_label, session_id, follow_up_score, reconstructed, carry_filters,
        )
        return ContextResolution(
            resolved_query=reconstructed,
            resolved_entities=dict(best_record.entities) if _same_category else {},
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
