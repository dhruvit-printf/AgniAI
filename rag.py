"""Retrieval helpers for AgniAI."""

from __future__ import annotations

import json
import logging
import os
import pickle
import hashlib
import re
import time
import warnings
import threading
from difflib import SequenceMatcher
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from runtime_cache import TTLCache

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
_DOCSTORE_CACHE = None

from config import (
    BM25_INDEX_PATH,
    BM25_WEIGHT,
    DEFAULT_MODEL,
    DENSE_WEIGHT,
    DOCSTORE_PATH,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    FALLBACK_MODELS,
    MIN_SCORE,
    OLLAMA_TAGS_URL,
    OLLAMA_URL,
    REFERENCE_FALLBACK,
    RERANKER_MODEL,
    RERANK_TOP_K,
    REQUEST_TIMEOUT,
    STRICT_MIN_SCORE,
    STRICT_RAG_PROMPT,
    STRICT_RAG_PROMPT_COMPUTE,
    STRICT_TOP_K,
    MIN_RETRIEVAL_CONFIDENCE,
    SYSTEM_PROMPT,
    TOP_K,
    USE_HYBRID,
    USE_RERANKER,
    RETRIEVAL_CACHE_TTL,
    RESPONSE_CACHE_TTL,
    EMBED_CACHE_TTL,
    MAX_CACHE_ENTRIES,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_CHARS_DEFAULT,
    MAX_TOKENS_STYLE,
    HIGH_RETRIEVAL_CONFIDENCE,
    LOW_RETRIEVAL_CONFIDENCE,
    STYLE_POINT_TOKEN_BUDGET,
    style_structure_instruction,
    trim_to_complete_sentence,
)

logger = logging.getLogger(__name__)

_MODEL: Optional[SentenceTransformer] = None
_RERANKER = None
_RERANKER_FAILED = False
_INDEX: Optional[faiss.Index] = None
_DOCS: List[Dict[str, str]] = []
_BM25 = None
_INDEX_LOCK = threading.RLock()
_QUERY_EMBED_CACHE = TTLCache(maxsize=MAX_CACHE_ENTRIES, ttl=EMBED_CACHE_TTL)
_RETRIEVAL_CACHE = TTLCache(maxsize=MAX_CACHE_ENTRIES, ttl=RETRIEVAL_CACHE_TTL)
_RESPONSE_CACHE = TTLCache(maxsize=MAX_CACHE_ENTRIES, ttl=RESPONSE_CACHE_TTL)
_MAX_STRUCTURED_POINTS = int(os.getenv("MAX_STRUCTURED_POINTS", "12"))

_STEP_TEMPLATE = [
    "Preparation",
    "Online CEE",
    "Rally Process",
    "Medical Examination",
    "Review Opportunity",
    "Final Selection",
    "Training",
]

_STEP_PATTERNS = [
    (
        "Preparation",
        [
            r"\bpreparation\b",
            r"\bbefore rally\b",
            r"\bpre[- ]?rally\b",
            r"\bregistration\b",
            r"\bapply\b",
            r"\bapplication\b",
            r"\beligibility\b",
            r"\bdocuments?\b",
            r"\badmit card\b",
        ],
    ),
    (
        "Online CEE",
        [
            r"\bcee\b",
            r"\bcommon entrance exam\b",
            r"\bonline exam\b",
            r"\bonline entrance exam\b",
            r"\bwritten exam\b",
            r"\bcomputer[- ]based\b",
            r"\bcbt\b",
        ],
    ),
    (
        "Rally Process",
        [
            r"\brally\b",
            r"\bphysical test\b",
            r"\bphysical fitness test\b",
            r"\bpft\b",
            r"\bfitness test\b",
            r"\brun\b",
            r"\bmeasurement\b",
            r"\bheight\b",
            r"\bchest\b",
            r"\bweight\b",
        ],
    ),
    (
        "Medical Examination",
        [
            r"\bmedical examination\b",
            r"\bmedical test\b",
            r"\bmedical\b",
            r"\bdoctor\b",
            r"\bhospital\b",
            r"\bfitness certificate\b",
        ],
    ),
    (
        "Review Opportunity",
        [
            r"\bre[- ]?medical\b",
            r"\breview\b",
            r"\brecheck\b",
            r"\bappeal\b",
            r"\bverification\b",
            r"\bclarification\b",
            r"\breconsider\b",
        ],
    ),
    (
        "Final Selection",
        [
            r"\bfinal selection\b",
            r"\bselection list\b",
            r"\bmerit\b",
            r"\bresult\b",
            r"\bdispatch\b",
            r"\bdespatch\b",
            r"\bjoining\b",
            r"\bappointment\b",
            r"\boffer\b",
        ],
    ),
    (
        "Training",
        [
            r"\btraining\b",
            r"\binduction\b",
            r"\bbasic training\b",
            r"\bregimental\b",
            r"\borientation\b",
            r"\bcentre\b",
        ],
    ),
]

_STEP_ORDER = {label: idx for idx, label in enumerate(_STEP_TEMPLATE)}
_STEP_GENERIC_NOISE = {
    "army",
    "result",
    "results",
    "test",
    "tests",
    "part",
    "chapter",
    "section",
    "annexure",
    "appendix",
    "schedule",
    "table",
    "figure",
    "contents",
    "introduction",
    "overview",
    "detail",
    "details",
    "important",
    "general",
}
_STEP_SECTION_NOISE = (
    r"^part[-\s]*[ivxlcdm0-9]+$",
    r"^chapter[-\s]*[ivxlcdm0-9]+$",
    r"^section[-\s]*[ivxlcdm0-9]+$",
    r"^annexure[-\s]*[ivxlcdm0-9]+$",
    r"^appendix[-\s]*[ivxlcdm0-9]+$",
    r"^result[s]?$",
    r"^army$",
    r"^tests?$",
    r"^[ivxlcdm]+$",
)


def _ensure_dirs() -> None:
    FAISS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCSTORE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _new_index() -> faiss.Index:
    return faiss.IndexFlatIP(EMBEDDING_DIM)


def _escape_control_chars_in_json_strings(raw: str) -> str:
    repaired: List[str] = []
    in_string = False
    escaped = False
    for ch in raw:
        if in_string:
            if escaped:
                repaired.append(ch)
                escaped = False
                continue
            if ch == "\\":
                repaired.append(ch)
                escaped = True
                continue
            if ch == '"':
                repaired.append(ch)
                in_string = False
                continue
            if ch == "\n":
                repaired.append("\\n")
                continue
            if ch == "\r":
                repaired.append("\\r")
                continue
            if ch == "\t":
                repaired.append("\\t")
                continue
            if ord(ch) < 32:
                repaired.append(f"\\u{ord(ch):04x}")
                continue
            repaired.append(ch)
            continue
        repaired.append(ch)
        if ch == '"':
            in_string = True
    return "".join(repaired)


def _extract_json_scalar(line: str) -> str:
    value = line.split(":", 1)[1].strip()
    if value.endswith(","):
        value = value[:-1].rstrip()
    if value.startswith('"') and value.endswith('"'):
        value = value[1:-1]
    return value


def _repair_docstore_from_lines(raw: str) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    obj: Dict[str, str] = {}
    text_lines: List[str] = []
    in_object = False
    in_text = False
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped == "[" or not stripped:
            continue
        if stripped == "]":
            break
        if stripped.startswith("{"):
            obj = {}
            text_lines = []
            in_object = True
            in_text = False
            continue
        if not in_object:
            continue
        if in_text:
            if stripped in {"}", "},"}:
                obj["text"] = "\n".join(text_lines)
                docs.append(obj)
                obj = {}
                text_lines = []
                in_object = False
                in_text = False
                continue
            text_lines.append(line)
            continue
        if stripped.startswith('"source":'):
            obj["source"] = _extract_json_scalar(line)
        elif stripped.startswith('"doc_type":'):
            obj["doc_type"] = _extract_json_scalar(line)
        elif stripped.startswith('"chunk_id":'):
            obj["chunk_id"] = _extract_json_scalar(line)
        elif stripped.startswith('"text":'):
            fragment = line.split(":", 1)[1].lstrip()
            if fragment.startswith('"'):
                fragment = fragment[1:]
            if fragment.endswith('",'):
                fragment = fragment[:-2]
                obj["text"] = fragment
                text_lines = []
            elif fragment.endswith('"'):
                fragment = fragment[:-1]
                obj["text"] = fragment
                text_lines = []
            else:
                text_lines = [fragment]
                in_text = True
            continue
        elif stripped in {"}", "},"}:
            if "text" not in obj and text_lines:
                obj["text"] = "\n".join(text_lines)
            if obj:
                docs.append(obj)
            obj = {}
            text_lines = []
            in_object = False
            in_text = False
    if in_object and obj:
        if "text" not in obj and text_lines:
            obj["text"] = "\n".join(text_lines)
        docs.append(obj)
    return docs


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9\u0900-\u097F]+", text.lower())


def _meaningful_tokens(text: str) -> List[str]:
    stopwords = {
        "a", "an", "and", "are", "be", "by", "for", "from", "how", "i", "in",
        "is", "it", "me", "my", "of", "on", "or", "please", "show", "tell",
        "the", "to", "what", "when", "where", "which", "who", "why", "with",
        "you", "your", "can", "could", "would", "should", "do", "does", "did",
        "this", "that", "these", "those", "as", "at", "was", "were", "will",
        "just", "about", "into", "over", "under", "up", "down",
    }
    return [t for t in _tokenize(text) if t not in stopwords and len(t) > 1]


def _chunk_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    a_norm = _normalise_text(a)
    b_norm = _normalise_text(b)
    ratio = SequenceMatcher(None, a_norm, b_norm).ratio()
    if ratio >= 0.95:
        return ratio
    a_tokens = set(_meaningful_tokens(a_norm))
    b_tokens = set(_meaningful_tokens(b_norm))
    if not a_tokens or not b_tokens:
        return ratio
    jaccard = len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))
    return max(ratio, jaccard)


def _dedupe_docs(docs: List[Dict[str, str]], similarity_threshold: float = 0.88):
    deduped = []
    seen_hashes = set()

    for doc in docs:
        text = _normalise_text(doc.get("text", ""))
        if not text:
            continue

        h = hashlib.md5(text.encode()).hexdigest()
        if h in seen_hashes:
            continue

        if any(_chunk_similarity(text, d.get("text", "")) >= similarity_threshold for d in deduped):
            continue

        seen_hashes.add(h)
        deduped.append(doc)

    return deduped


def _sentence_split(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    chunks = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [chunk.strip() for chunk in chunks if chunk and chunk.strip()]


def _split_section_candidates(text: str) -> List[str]:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    sections: List[str] = []
    current: List[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            if current:
                sections.append(" ".join(current).strip())
                current = []
            continue
        bullet_like = bool(re.match(r"^(\d+[\).\:-]|\-|\*|•)\s+", line))
        heading_like = _looks_like_heading(line)
        if heading_like and current:
            sections.append(" ".join(current).strip())
            current = [line]
            continue
        if bullet_like and current:
            sections.append(" ".join(current).strip())
            current = [line]
            continue
        current.append(line)
    if current:
        sections.append(" ".join(current).strip())
    return [section for section in sections if section]


def _looks_like_heading(line: str) -> bool:
    line = re.sub(r"^\s*(?:\d+[\).\:-]?\s*|\-|\*|•\s*)", "", (line or "").strip())
    if not line:
        return False
    if len(line) > 80:
        return False
    words = line.split()
    if len(words) > 12:
        return False
    alpha_words = [w for w in words if re.search(r"[A-Za-z\u0900-\u097F]", w)]
    if not alpha_words:
        return False
    if line.endswith(":"):
        return True
    if line.isupper() and len(words) <= 8:
        return True
    titleish = sum(1 for w in alpha_words if w[:1].isupper())
    if titleish >= max(1, len(alpha_words) - 1):
        return True
    return False


def _is_noise_step_text(text: str) -> bool:
    candidate = _clean_point_title(text)
    if not candidate:
        return True
    lowered = _normalise_text(candidate)
    if any(re.fullmatch(pattern, lowered) for pattern in _STEP_SECTION_NOISE):
        return True
    if lowered in _STEP_GENERIC_NOISE:
        return True
    words = lowered.split()
    if len(words) == 1:
        return _canonical_step_label(candidate) is None
    if len(words) <= 2 and _canonical_step_label(candidate) is None:
        return True
    alpha_words = [w for w in words if re.search(r"[a-z\u0900-\u097f]", w)]
    if not alpha_words:
        return True
    if candidate.isupper() and len(words) <= 4 and _canonical_step_label(candidate) is None:
        return True
    return False

# ── TOC / garbage-line detector ───────────────────────────────────────────────
_TOC_LINE_RE = re.compile(
    r"""
    \b\d{1,3}(?:-\d{1,3})?\s*$          # trailing page numbers: "5-11", "78-82"
    | ^[A-Z][A-Z\s/()&]{4,}\s+\d{1,3}   # ALL-CAPS heading + page number
    | \(CEE\)\s*\d                        # "(CEE) 36-69"
    | ^PART[-\s]*[IVX\d]+\b              # "PART-I", "PART - III"
    | Next,\s+it\s+Will\s+proceed        # UI nav text
    | ^(?:Enter|Click|Choose|Select|Fill|Read)\s+\w  # form-fill instructions
    """,
    re.VERBOSE | re.IGNORECASE,
)

def _is_toc_or_garbage(text: str) -> bool:
    """Return True if the line looks like a TOC entry, page ref, or UI fragment."""
    text = (text or "").strip()
    if not text or len(text) < 8:
        return True
    if _TOC_LINE_RE.search(text):
        return True
    if text.isupper() and len(text.split()) <= 5:   # short ALL-CAPS headers
        return True
    return False

def _canonical_step_label(text: str, context: str = "") -> Optional[str]:
    haystack = f"{text or ''} {context or ''}".lower()
    if not haystack.strip():
        return None
    for label, patterns in _STEP_PATTERNS:
        if any(re.search(pattern, haystack, flags=re.IGNORECASE) for pattern in patterns):
            return label
    return None


def _step_support_snippet(text: str, label: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    label_patterns = dict(_STEP_PATTERNS).get(label, [])
    sentences = _sentence_split(text)
    hits = []
    for sentence in sentences:
        lowered = sentence.lower()
        if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in label_patterns):
            cleaned = sentence.strip()
            if cleaned:
                hits.append(cleaned)
    if hits:
        return " ".join(hits[:2]).strip()
    sections = _split_section_candidates(text)
    if sections:
        first = sections[0].strip()
        if len(first) > 220:
            return first[:220].rstrip()
        return first
    if len(text) > 220:
        return text[:220].rstrip()
    return text


def _merge_step_point(
    existing: Dict[str, str],
    *,
    support: str,
    raw: str,
    score: float,
    source: str,
) -> Dict[str, str]:
    if support:
        existing_support = (existing.get("support") or "").strip()
        if not existing_support:
            existing["support"] = support
        elif support not in existing_support:
            existing["support"] = f"{existing_support} {support}".strip()
    if raw and not existing.get("raw"):
        existing["raw"] = raw
    if source and not existing.get("source"):
        existing["source"] = source
    existing["score"] = str(max(float(existing.get("score", 0.0)), score))
    return existing


def _structured_step_template(query: str = "", docs: Sequence[Dict[str, str]] | None = None) -> List[Dict[str, str]]:
    docs = list(docs or [])
    points: List[Dict[str, str]] = []
    for label in _STEP_TEMPLATE:
        support = ""
        source = ""
        if docs:
            for doc in sorted(docs, key=lambda d: float(d.get("score", 0.0)), reverse=True):
                text = (doc.get("text") or "").strip()
                if not text:
                    continue
                snippet = _step_support_snippet(text, label)
                if snippet:
                    support = snippet
                    source = doc.get("source", "")
                    break
        points.append({
            "title": label,
            "support": support,
            "raw": support or label,
            "source": source,
            "score": "0.0",
        })
    return points


def _order_structured_points(points: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    buckets: Dict[str, Dict[str, str]] = {}
    for point in points:
        title = _canonical_step_label(point.get("title", ""), point.get("support", "")) or _canonical_step_label(point.get("raw", ""))
        if not title:
            continue
        cleaned_title = title
        candidate_support = (point.get("support") or "").strip()
        candidate_raw = (point.get("raw") or "").strip()
        candidate_source = (point.get("source") or "").strip()
        candidate_score = float(point.get("score", 0.0))
        if cleaned_title in buckets:
            buckets[cleaned_title] = _merge_step_point(
                buckets[cleaned_title],
                support=candidate_support,
                raw=candidate_raw,
                score=candidate_score,
                source=candidate_source,
            )
        else:
            buckets[cleaned_title] = {
                "title": cleaned_title,
                "support": candidate_support,
                "raw": candidate_raw,
                "source": candidate_source,
                "score": str(candidate_score),
            }

    ordered = [buckets[label] for label in _STEP_TEMPLATE if label in buckets]
    return ordered[: len(_STEP_TEMPLATE)]


def _strip_leading_marker(text: str) -> str:
    return re.sub(r"^\s*(?:\d+[\).\:-]?\s*|\-|\*|•\s*)", "", (text or "").strip())


def _clean_point_title(title: str) -> str:
    title = _strip_leading_marker(title)
    title = title.strip(" \t:-—")
    title = re.sub(r"\s+", " ", title)
    if not title:
        return ""
    title = re.split(r"[.!?]", title)[0].strip()
    for separator in (" : ", " - ", " — ", " – "):
        if separator in title:
            left, right = title.split(separator, 1)
            if len(left.split()) <= 8:
                title = left.strip()
                break
            if len(right.split()) <= 8:
                title = right.strip()
                break
    if ":" in title:
        left, right = title.split(":", 1)
        if len(left.split()) <= 8:
            title = left.strip()
        elif right.strip():
            title = right.strip()
    words = title.split()
    if len(words) > 10:
        title = " ".join(words[:10]).strip()
    return title.strip(" \t:-—")


def _infer_support_text(section: str, title: str) -> str:
    section = (section or "").strip()
    if not section:
        return ""
    if title:
        title_norm = _normalise_text(title)
        section_norm = _normalise_text(section)
        if section_norm.startswith(title_norm):
            section = section[len(title):].lstrip(" :-—\n\t")
        else:
            section = re.sub(re.escape(title), "", section, count=1, flags=re.IGNORECASE).strip(" :-—\n\t")
    sentences = _sentence_split(section)
    if not sentences:
        return section.strip()
    if len(sentences) == 1:
        return sentences[0]
    return " ".join(sentences[:3]).strip()


def _section_to_point(section: str) -> Optional[Dict[str, str]]:
    section = (section or "").strip()
    if not section:
        return None
    lines = [line.strip() for line in section.splitlines() if line.strip()]
    first_line = lines[0] if lines else section
    if _is_noise_step_text(first_line):
        return None
    title = _canonical_step_label(first_line, section)
    if not title:
        title = _canonical_step_label(section)
    if not title:
        return None
    support = _step_support_snippet(section, title)
    if not support:
        support = _infer_support_text(section, title)
    return {"title": title, "support": support, "raw": section}


def _dedupe_points(points: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    deduped: List[Dict[str, str]] = []
    seen: set[str] = set()
    for point in points:
        title = _clean_point_title(point.get("title", ""))
        if not title:
            continue
        key = _normalise_text(title)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append({
            "title": title,
            "support": (point.get("support") or "").strip(),
            "raw": (point.get("raw") or "").strip(),
        })
    return deduped


def _fallback_points_from_docs(docs: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    return _structured_step_template(docs=docs)


def extract_key_points(
    docs: Sequence[Dict[str, str]],
    *,
    query: str = "",
    max_points: int = _MAX_STRUCTURED_POINTS,
) -> List[Dict[str, str]]:
    _PROCESS_KEYWORDS = (
        "process", "step", "procedure", "how to join", "recruitment process",
        "selection process", "how do i apply", "how to apply",
    )
    _query_lower = (query or "").lower()
    _is_process_query = any(kw in _query_lower for kw in _PROCESS_KEYWORDS)

    if not docs:
        return _structured_step_template(query=query, docs=docs) if _is_process_query else []

    ordered = sorted(docs, key=lambda doc: float(doc.get("score", 0.0)), reverse=True)
    candidates: List[Dict[str, str]] = []
    for doc in ordered:
        text = (doc.get("text") or "").strip()
        if not text:
            continue
        sections = _split_section_candidates(text)
        if not sections:
            sections = [text]
        for section in sections:
            point = _section_to_point(section)
            if not point:
                continue
            point["score"] = str(float(doc.get("score", 0.0)))
            point["source"] = doc.get("source", "")
            candidates.append(point)

    deduped = _order_structured_points(_dedupe_points(candidates))

    # First fallback (after initial dedup):
    if len(deduped) < 3:
        fallback = _fallback_points_from_docs(ordered)
        fallback_ordered = _order_structured_points(fallback)
        if len(fallback_ordered) > len(deduped):
            deduped = fallback_ordered

    if not deduped and _is_process_query:
        deduped = _structured_step_template(query=query, docs=ordered)

    if len(deduped) < 3 and _is_process_query:
        deduped = _structured_step_template(query=query, docs=ordered)

    if max_points <= 0:
        return []

    limit = min(max_points, len(_STEP_TEMPLATE))
    return deduped[:limit]


def _style_point_token_budget(style: str) -> int:
    style_key = (style or "").strip().lower()
    return int(STYLE_POINT_TOKEN_BUDGET.get(style_key, STYLE_POINT_TOKEN_BUDGET["elaborate"]))


def _limit_sentence_count(text: str, max_sentences: int) -> str:
    text = (text or "").strip()
    if not text or max_sentences <= 0:
        return ""
    sentences = _sentence_split(text)
    if not sentences:
        return text
    return " ".join(sentences[:max_sentences]).strip()


def _limit_words(text: str, max_words: int) -> str:
    text = (text or "").strip()
    if not text or max_words <= 0:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).strip()


def _shape_explanation(text: str, style: str) -> str:
    style_key = (style or "").strip().lower()
    text = (text or "").strip()
    if not text:
        return ""
    if style_key == "short":
        return ""
    text = trim_to_complete_sentence(text)
    if not text:
        return ""
    if style_key == "elaborate":
        return _limit_sentence_count(text, 5)
    return text


def _clean_generated_explanation(text: str, title: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    text = text.strip("*_` ")
    text = _strip_leading_marker(text)
    title_norm = _normalise_text(title)
    text_norm = _normalise_text(text)
    if title_norm and text_norm.startswith(title_norm):
        text = text[len(title):].lstrip(" :-—\n\t")
    return text.strip()


def _build_support_explanation(point: Dict[str, str], style: str) -> str:
    text = (point.get("support") or point.get("raw") or "").strip()
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)

    if style == "short":
        return _limit_words(text, 20)

    if style == "elaborate":
        return _limit_sentence_count(text, 3)

    return text  # detailed = full


def _build_point_messages(
    *,
    query: str,
    point: Dict[str, str],
    style: str,
    reasoning: bool = False,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    system_content = STRICT_RAG_PROMPT_COMPUTE if reasoning else STRICT_RAG_PROMPT
    system_content = (
        f"{system_content}\n\n"
        f"{style_structure_instruction(style)}\n"
        "You are writing the explanation for a single numbered point in a fixed structured answer. "
        "Do not add, remove, or rename points. Output only the explanation body for this one point. "
        "Use only the supplied point title and supporting context. "
        "If the supporting context is thin, stay conservative and summarize only what is explicit."
    )
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_content}]
    if history:
        for msg in history:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
    support = (point.get("support") or point.get("raw") or "").strip()
    user_content = (
        f"Question: {query}\n"
        f"Point title: {point.get('title', '').strip()}\n"
        f"Supporting context:\n{support}\n\n"
        f"Style: {style}\n"
        "Return only the explanation text for this point."
    )
    messages.append({"role": "user", "content": user_content})
    return messages


def _generate_point_explanation(
    *,
    session,
    model: str,
    query: str,
    point: Dict[str, str],
    style: str,
    reasoning: bool = False,
    history: Optional[List[Dict[str, str]]] = None,
) -> str:
    style_key = (style or "").strip().lower()
    if style_key == "short":
        return ""

    support = (point.get("support") or point.get("raw") or "").strip()
    if not support:
        return ""

    try:
        messages = _build_point_messages(
            query=query,
            point=point,
            style=style,
            reasoning=reasoning,
            history=history,
        )

        from ollama_cpu_chat import chat_with_fallback

        response = chat_with_fallback(
            session,
            model,
            messages,
            stream_tokens=False,
            max_tokens_override=_style_point_token_budget(style),
        )
        return _clean_generated_explanation((response.text or "").strip(), point.get("title", ""))
    except Exception as exc:
        logger.warning("Point explanation generation failed: %s", exc)
        return _build_support_explanation(point, style)


def format_structured_answer(points, explanations, style: str) -> str:
    style_key = (style or "").strip().lower()
    lines = []
    for idx, point in enumerate(points, start=1):
        title = _clean_point_title(point.get("title", "")) or f"Point {idx}"
        lines.append(f"{idx}. {title}")
        if style_key == "short":
            continue
        explanation = ""
        if idx - 1 < len(explanations):
            explanation = _shape_explanation(
                _clean_generated_explanation(explanations[idx - 1], title),
                style,
            )
        # NEW: fallback to raw/support text if LLM explanation is empty
        if not explanation:
            explanation = _shape_explanation(
                point.get("support") or point.get("raw") or "",
                style,
            )
        if explanation:
            lines.append(f"   {explanation}")
    return "\n".join(lines).strip()


def generate_structured_answer(
    query: str,
    *,
    docs: Sequence[Dict[str, str]],
    style: str,
    model: str,
    session,
    reasoning: bool = False,
    history: Optional[List[Dict[str, str]]] = None,
    max_points: int = _MAX_STRUCTURED_POINTS,
) -> Dict[str, object]:
    """Generate a complete structured answer using a single LLM call."""
    if not docs:
        return {
            "answer": REFERENCE_FALLBACK,
            "points": [],
            "explanations": [],
            "structured": False,
            "token_budget_per_point": 0,
        }

    context = build_context(
        docs,
        max_chunks=max(STRICT_TOP_K, min(5, len(docs))),
        min_score=LOW_RETRIEVAL_CONFIDENCE,
        max_chars=MAX_CONTEXT_CHARS.get(style, MAX_CONTEXT_CHARS_DEFAULT)
        if isinstance(MAX_CONTEXT_CHARS, dict)
        else MAX_CONTEXT_CHARS_DEFAULT,
    )

    if not context.strip():
        return {
            "answer": REFERENCE_FALLBACK,
            "points": [],
            "explanations": [],
            "structured": False,
            "token_budget_per_point": 0,
        }

    system_prompt = STRICT_RAG_PROMPT_COMPUTE if reasoning else STRICT_RAG_PROMPT
    system_prompt = f"{system_prompt}\n\n{style_structure_instruction(style)}"

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    if history:
        for msg in history[-6:]:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})

    user_content = (
        f"Reference information:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Using ONLY the reference information above, write a complete structured answer. "
        "Do not use any knowledge outside the reference information. "
        "Do not repeat a numbered list inside a numbered list."
    )
    messages.append({"role": "user", "content": user_content})

    token_budget = MAX_TOKENS_STYLE.get(style, MAX_TOKENS_DEFAULT)

    try:
        from ollama_cpu_chat import chat_with_fallback
        result = chat_with_fallback(
            session,
            model,
            messages,
            stream_tokens=False,
            max_tokens_override=token_budget,
        )
        answer = trim_to_complete_sentence((result.text or "").strip())
    except Exception as exc:
        logger.warning("Single-shot LLM call failed: %s", exc)
        answer = ""

    if not answer:
        answer = REFERENCE_FALLBACK

    return {
        "answer": answer,
        "points": [],
        "explanations": [],
        "structured": bool(answer and answer != REFERENCE_FALLBACK),
        "token_budget_per_point": token_budget,
    }


def _normalize_query_for_retrieval(query: str) -> str:
    cleaned = query.lower()
    filler_phrases = (
        "in short", "briefly", "brief", "quick answer", "short answer",
        "summarise", "summarize", "tldr", "tl;dr", "in brief",
        "give me short", "one line", "one-line",
        "give a short", "keep it short", "in detail", "detailed",
        "explain in detail", "full detail", "comprehensive", "exhaustive",
        "step by step", "step-by-step", "explain fully", "tell me everything",
        "give me detail", "elaborate in detail", "full explanation",
        "full breakdown", "break it down", "please", "can you", "could you",
    )
    for phrase in sorted(filler_phrases, key=len, reverse=True):
        cleaned = re.sub(rf"\b{re.escape(phrase)}\b", " ", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .,!?:;")
    if len(cleaned.split()) < 3:
        cleaned = query.strip().lower()

    expansions = (
        (r"\bage limit\b", "required age eligibility"),
        (r"\bage\b", "required age eligibility"),
        (r"\beligibilit", "eligibility criteria required age qualification"),
        (r"\bselection process\b", "recruitment process merit medical physical fitness"),
        (r"\bhow.*select", "recruitment process merit medical physical fitness"),
        (r"\brecruitment process\b", "registration rally medical"),
        (r"\bhow.*appl", "registration application"),
        (r"\bsalary\b", "customised package in hand seva nidhi monthly"),
        (r"\bpay\b", "customised package in hand monthly"),
        (r"\bphysical test\b", "physical fitness test pft 1.6 km run"),
        (r"\bpft\b", "physical fitness test 1.6 km run"),
        (r"\bbonus mark", "bonus marks ncc sports"),
        (r"\binsurance\b", "life insurance cover 48 lakhs"),
        (r"\bseva nidhi\b", "seva nidhi corpus fund exit after 4 year lakh"),
        (r"\btraining\b", "military training regimental centre"),
        (r"\bdocument", "documents required matric aadhaar domicile"),
        (r"\bmedical\b", "medical examination army medical standards"),
    )
    for pattern, extra in expansions:
        if re.search(pattern, cleaned):
            cleaned = f"{cleaned} {extra}"
            break
    return re.sub(r"\s+", " ", cleaned).strip() or query.strip()


def _rewrite_query_candidates(query: str) -> List[str]:
    q = query.strip().lower()
    candidates = [q]
    if any(word in q for word in ("calculate", "total", "sum", "overall", "aggregate", "combined")):
        candidates.append(
            re.sub(r"\b(calculate|total|sum|overall|aggregate|combined)\b", " ", q)
        )
    return [re.sub(r"\s+", " ", cand).strip() for cand in candidates if cand.strip()]


def _query_similarity(a: str, b: str) -> float:
    try:
        av = _cache_query_embedding(a)
        bv = _cache_query_embedding(b)
        return float(np.dot(av[0], bv[0]))
    except Exception:
        return 0.0


def safe_rewrite_query(query: str) -> str:
    candidates = _rewrite_query_candidates(query)
    if len(candidates) == 1:
        return candidates[0]
    original = candidates[0]
    best = original
    best_score = -1.0
    for candidate in candidates:
        score = _query_similarity(original, candidate)
        if score > best_score:
            best_score = score
            best = candidate
    if best != original and best_score < 0.88:
        logger.debug("Rewrite rejected for query=%r: best_score=%.3f", query, best_score)
        return original
    if best != original:
        logger.debug("Rewrite accepted for query=%r -> %r (score=%.3f)", query, best, best_score)
    return best


def _query_cache_key(query: str) -> str:
    return re.sub(r"\s+", " ", query).strip().lower()


def _cache_query_embedding(query: str) -> np.ndarray:
    key = _query_cache_key(query)
    cached = _QUERY_EMBED_CACHE.get(key)
    if cached is not None:
        return cached
    vec = embed_query(query)
    _QUERY_EMBED_CACHE.set(key, vec)
    return vec


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def make_retrieval_cache_key(query: str, top_k: int) -> str:
    return f"{_query_cache_key(query)}|k={top_k}"


def make_response_cache_key(
    query: str,
    *,
    style: str,
    model: str,
    context: str,
    session_id: str = "",
) -> str:
    del session_id
    payload = "|".join([style, model, _query_cache_key(query), _hash_text(context)])
    return _hash_text(payload)


def get_cached_retrieval(query: str, top_k: int):
    normalized = _normalize_query_for_retrieval(query)
    return _RETRIEVAL_CACHE.get(make_retrieval_cache_key(normalized, top_k))


def set_cached_retrieval(query: str, top_k: int, docs):
    normalized = _normalize_query_for_retrieval(query)
    _RETRIEVAL_CACHE.set(make_retrieval_cache_key(normalized, top_k), docs)


def get_cached_response(key: str) -> Optional[str]:
    return _RESPONSE_CACHE.get(key)


def set_cached_response(key: str, value: str) -> None:
    _RESPONSE_CACHE.set(key, value)


def load_embedding_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _MODEL = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    return _MODEL


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    model = load_embedding_model()
    vecs = model.encode(
        list(texts),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 20,
        batch_size=32,
    )
    return np.asarray(vecs, dtype="float32")


def embed_query(query: str) -> np.ndarray:
    model = load_embedding_model()
    vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=1,
    )
    return np.asarray(vec, dtype="float32")


def _reranker_local_files_available(model_name: str) -> bool:
    model_path = Path(model_name)
    if model_path.exists():
        return True
    hf_hub_cache = os.getenv("HF_HUB_CACHE")
    hf_home = os.getenv("HF_HOME")
    if hf_hub_cache:
        cache_root = Path(hf_hub_cache)
    elif hf_home:
        cache_root = Path(hf_home) / "hub"
    else:
        cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    repo_dir = cache_root / f"models--{model_name.replace('/', '--')}"
    if not repo_dir.exists():
        return False
    snapshots = repo_dir / "snapshots"
    if not snapshots.exists():
        return False
    try:
        return any(
            snapshot.is_dir() and any(snapshot.iterdir())
            for snapshot in snapshots.iterdir()
        )
    except (PermissionError, OSError, StopIteration):
        return False


def load_reranker():
    global _RERANKER, _RERANKER_FAILED
    if _RERANKER is not None:
        return _RERANKER
    if _RERANKER_FAILED or not USE_RERANKER:
        return None
    if not _reranker_local_files_available(RERANKER_MODEL):
        _RERANKER_FAILED = True
        logger.info("Reranker not available locally, skipping rerank step.")
        return None
    try:
        from sentence_transformers import CrossEncoder  # type: ignore

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _RERANKER = CrossEncoder(RERANKER_MODEL, local_files_only=True)
        return _RERANKER
    except Exception as exc:
        _RERANKER_FAILED = True
        logger.warning("Could not load reranker: %s", exc)
        return None


def rerank(query: str, docs: List[Dict], top_n: int = RERANK_TOP_K) -> List[Dict]:
    if not docs:
        return docs
    reranker = load_reranker()
    if reranker is None:
        return docs[:top_n]
    try:
        pairs = [(query, d.get("text", "")) for d in docs]
        scores = reranker.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        reranked = []
        for score, doc in ranked[:top_n]:
            new_doc = dict(doc)
            new_doc["rerank_score"] = round(float(score), 4)
            reranked.append(new_doc)
        return reranked
    except Exception as exc:
        logger.warning("Re-ranking failed, using original order: %s", exc)
        return docs[:top_n]


def load_bm25():
    global _BM25
    if _BM25 is not None:
        return _BM25
    if not USE_HYBRID or not BM25_INDEX_PATH.exists():
        return None
    try:
        with open(BM25_INDEX_PATH, "rb") as f:
            _BM25 = pickle.load(f)
        return _BM25
    except Exception as exc:
        logger.warning("Could not load BM25 index: %s", exc)
        return None


def save_bm25(docs: List[Dict[str, str]]) -> None:
    if not USE_HYBRID:
        return
    docs_snapshot = list(docs or [])

    def _build() -> None:
        global _BM25
        try:
            from rank_bm25 import BM25Okapi  # type: ignore

            corpus = [_tokenize(d.get("text", "")) for d in docs_snapshot]
            bm25 = BM25Okapi(corpus)
            _ensure_dirs()
            with open(BM25_INDEX_PATH, "wb") as f:
                pickle.dump(bm25, f)
            with _INDEX_LOCK:
                _BM25 = bm25
        except ModuleNotFoundError:
            return
        except Exception as exc:
            logger.warning("BM25 index build failed: %s", exc)

    threading.Thread(target=_build, daemon=True).start()


def load_docstore():
    global _DOCSTORE_CACHE
    if _DOCSTORE_CACHE is not None:
        return _DOCSTORE_CACHE

    if not DOCSTORE_PATH.exists():
        return []

    raw = DOCSTORE_PATH.read_text(encoding="utf-8", errors="replace")

    try:
        docs = json.loads(raw)
    except:
        docs = _repair_docstore_from_lines(raw)

    _DOCSTORE_CACHE = docs
    return docs


def _save_docstore(docs: List[Dict[str, str]]) -> None:
    _ensure_dirs()
    with DOCSTORE_PATH.open("w", encoding="utf-8") as fh:
        json.dump(docs, fh, ensure_ascii=False, indent=2)


def _rebuild_index_from_docs(docs: List[Dict[str, str]]) -> faiss.Index:
    index = _new_index()
    if not docs:
        return index
    texts = [d.get("text", "") for d in docs]
    vectors = embed_texts(texts)
    if vectors.size == 0:
        return index
    if vectors.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dimension mismatch while rebuilding: expected {EMBEDDING_DIM}, got {vectors.shape[1]}"
        )
    index.add(vectors)
    save_index(index, docs)
    return index


def load_index() -> faiss.Index:
    global _INDEX, _DOCS
    if _INDEX is not None and _DOCS:
        return _INDEX
    with _INDEX_LOCK:
        if _INDEX is not None and _DOCS:
            return _INDEX
        _ensure_dirs()
        _DOCS = load_docstore()
        if FAISS_INDEX_PATH.exists():
            _INDEX = faiss.read_index(str(FAISS_INDEX_PATH))
        else:
            _INDEX = _new_index()
        if _INDEX.d != EMBEDDING_DIM:
            _INDEX = _rebuild_index_from_docs(_DOCS)
        if _INDEX.ntotal > 0 and len(_DOCS) == 0:
            logger.warning("FAISS index has vectors but docstore is empty.")
        return _INDEX


def save_index(index: faiss.Index, docs: List[Dict[str, str]]) -> None:
    global _DOCS, _INDEX, _DOCSTORE_CACHE
    docs_snapshot = list(docs or [])
    with _INDEX_LOCK:
        _ensure_dirs()
        faiss.write_index(index, str(FAISS_INDEX_PATH))
        _save_docstore(docs_snapshot)
        _DOCS = docs_snapshot
        _INDEX = index
        _DOCSTORE_CACHE = docs_snapshot
    save_bm25(docs_snapshot)


def _index_snapshot() -> tuple[Optional[faiss.Index], List[Dict[str, str]]]:
    with _INDEX_LOCK:
        return _INDEX, list(_DOCS)


def _bm25_scores(query: str) -> np.ndarray:
    bm25 = load_bm25()
    _, docs = _index_snapshot()
    if bm25 is None or not docs:
        return np.zeros(len(docs), dtype="float32")
    try:
        scores = np.array(bm25.get_scores(_tokenize(query)), dtype="float32")
        if scores.shape[0] != len(docs):
            return np.zeros(len(docs), dtype="float32")
        max_s = scores.max()
        if max_s > 0:
            scores /= max_s
        return scores
    except Exception:
        return np.zeros(len(docs), dtype="float32")


def _min_max_normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo < 1e-8:
        return np.ones_like(values, dtype="float32")
    return ((values - lo) / (hi - lo)).astype("float32")


_DOMAIN_BOOSTS = [
    (r"\bage\b", r"required age|eligibility", 0.70),
    (r"eligibilit", r"eligibility criteria|required age", 0.70),
    (r"selection|how.*select", r"recruitment process|flow chart", 0.90),
    (r"how.*appl|apply", r"registration|application", 0.70),
    (r"salary|pay|package", r"seva nidhi|in hand|monthly", 0.80),
    (r"physical|pft", r"physical fitness test|1\.6 km run", 0.80),
    (r"bonus mark", r"bonus marks|ncc|sports", 0.80),
    (r"insurance", r"48 lakh|life insurance", 0.80),
    (r"training", r"military training|regimental", 0.70),
    (r"document", r"documents required|matric|aadhaar|domicile", 0.70),
    (r"medical", r"medical examination|army medical", 0.70),
    (r"ncc", r"ncc.*certificate|bonus.*ncc", 0.70),
]


def _apply_domain_boosts(query_lower: str, doc_text_lower: str) -> float:
    best = 0.0
    for q_pat, d_pat, boost in _DOMAIN_BOOSTS:
        if re.search(q_pat, query_lower) and re.search(d_pat, doc_text_lower):
            best = max(best, boost)
    return best


def search(query: str, top_k: int = TOP_K) -> List[Dict[str, str]]:
    cached = get_cached_retrieval(query, top_k)
    if cached is not None:
        logger.debug("Retrieval cache hit for query=%r", query)
        return [dict(doc) for doc in cached]

    index = load_index()
    if index.ntotal == 0:
        return []
    _, docs_snapshot = _index_snapshot()

    rewritten_query = _normalize_query_for_retrieval(query)
    retrieval_query = safe_rewrite_query(rewritten_query)
    if retrieval_query != rewritten_query:
        logger.debug(
            "Query rewrite downgraded to preserve intent. original=%r rewritten=%r final=%r",
            query,
            rewritten_query,
            retrieval_query,
        )
    logger.debug("Retrieval query: original=%r rewritten=%r final=%r", query, rewritten_query, retrieval_query)
    qvec = _cache_query_embedding(retrieval_query)

    candidate_k = min(max(top_k * 4, 10), 40, index.ntotal)
    scores_dense, ids = index.search(qvec, candidate_k)
    dense_scores = scores_dense[0]
    doc_ids = ids[0]
    dense_map = {
        int(doc_id): float(score)
        for doc_id, score in zip(doc_ids, dense_scores)
        if doc_id >= 0
    }

    if USE_HYBRID and docs_snapshot:
        bm25_all = _bm25_scores(retrieval_query)
        bm25_top_ids = np.argsort(bm25_all)[::-1][:candidate_k]

        token_count = len(retrieval_query.split())
        if token_count <= 3:
            dense_weight, bm25_weight = 0.25, 0.75
        elif token_count <= 6:
            dense_weight, bm25_weight = 0.40, 0.60
        else:
            dense_weight, bm25_weight = DENSE_WEIGHT, BM25_WEIGHT

        candidate_ids: List[int] = []
        seen_ids: set[int] = set()
        for doc_id in list(doc_ids) + [int(x) for x in bm25_top_ids]:
            if doc_id < 0 or doc_id >= len(docs_snapshot) or doc_id in seen_ids:
                continue
            candidate_ids.append(int(doc_id))
            seen_ids.add(int(doc_id))

        dense_values = np.array(
            [dense_map.get(doc_id, 0.0) for doc_id in candidate_ids], dtype="float32"
        )
        dense_values = _min_max_normalize(dense_values)
        bm25_values = np.array(
            [float(bm25_all[doc_id]) for doc_id in candidate_ids], dtype="float32"
        )
        query_terms = set(_meaningful_tokens(retrieval_query))
        query_lower = retrieval_query.lower()

        fused: List[tuple] = []
        for doc_id, ds, bs in zip(candidate_ids, dense_values, bm25_values):
            combined = dense_weight * float(ds) + bm25_weight * float(bs)
            doc_text = docs_snapshot[doc_id].get("text", "")
            if query_terms:
                doc_terms = set(_meaningful_tokens(doc_text))
                overlap = len(query_terms & doc_terms) / max(1, len(query_terms))
                combined += 0.15 * overlap
            combined += _apply_domain_boosts(query_lower, doc_text.lower())
            if combined >= MIN_SCORE:
                fused.append((combined, int(doc_id)))

        fused.sort(key=lambda item: item[0], reverse=True)
        candidates = []
        for combined, doc_id in fused:
            doc = dict(docs_snapshot[doc_id])
            doc["score"] = round(float(combined), 4)
            candidates.append(doc)
    else:
        candidates = []
        for doc_id, score in zip(doc_ids, dense_scores):
            if doc_id < 0 or doc_id >= len(docs_snapshot):
                continue
            if float(score) < MIN_SCORE:
                continue
            doc = dict(docs_snapshot[doc_id])
            doc["score"] = round(float(score), 4)
            candidates.append(doc)

    if not candidates:
        return []

    candidates = _dedupe_docs(candidates)

    if USE_RERANKER:
        candidates = rerank(query, candidates, top_n=min(max(top_k, RERANK_TOP_K), len(candidates)))

    candidates = sorted(candidates, key=lambda doc: float(doc.get("score", 0.0)), reverse=True)

    logger.debug(
        "Retrieved chunks for query=%r: %s",
        query,
        [
            {"score": doc.get("score"), "source": doc.get("source"), "chunk_id": doc.get("chunk_id")}
            for doc in candidates[: max(top_k, STRICT_TOP_K)]
        ],
    )
    final = candidates[: max(top_k, STRICT_TOP_K)]
    set_cached_retrieval(query, top_k, final)
    return [dict(doc) for doc in final]


def build_context(
    docs: Sequence[Dict[str, str]],
    *,
    max_chunks: int = STRICT_TOP_K,
    min_score: float = STRICT_MIN_SCORE,
    max_chars: int = 3000,
) -> str:
    if not docs:
        return ""

    ordered = sorted(docs, key=lambda doc: float(doc.get("score", 0.0)), reverse=True)
    ordered = [doc for doc in ordered if float(doc.get("score", 0.0)) >= min_score]
    ordered = _dedupe_docs(ordered)[:max_chunks]
    if not ordered:
        ordered = _dedupe_docs(sorted(docs, key=lambda doc: float(doc.get("score", 0.0)), reverse=True))[:max_chunks]

    if not ordered:
        return ""

    def _truncate_to_limit(text: str, limit: int) -> str:
        text = (text or "").strip()
        if limit <= 0 or len(text) <= limit:
            return text
        sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
        if len(sentences) <= 1:
            return text[:limit].rstrip()
        pieces: List[str] = []
        total = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            extra = len(sentence) + (1 if pieces else 0)
            if total + extra > limit:
                break
            pieces.append(sentence)
            total += extra
        if pieces:
            return " ".join(pieces).strip()
        return text[:limit].rstrip()

    blocks: List[str] = []
    total_chars = 0
    for i, doc in enumerate(ordered, start=1):
        text = (doc.get("text") or "").strip()
        if not text:
            continue
        source = doc.get("source", "unknown")
        if len(source) > 60:
            source = "..." + source[-57:]
        score = float(doc.get("score", 0.0))
        header = f"[{i}] score={score:.3f} source={source}\n"
        remaining = max_chars - total_chars - len(header)
        if remaining <= 0:
            break
        truncated_text = _truncate_to_limit(text, remaining)
        block = f"{header}{truncated_text}".strip()
        if not truncated_text:
            continue
        blocks.append(block)
        total_chars += len(block)
        if total_chars >= max_chars:
            break

    logger.debug(
        "Final context blocks: %s",
        [{"score": doc.get("score"), "source": doc.get("source")} for doc in ordered[: len(blocks)]],
    )
    return "\n\n---\n\n".join(blocks)


def retrieval_confidence(docs: Sequence[Dict[str, str]], query: str) -> float:
    if not docs:
        return 0.0
    ordered = sorted(docs, key=lambda doc: float(doc.get("score", 0.0)), reverse=True)
    top_score = float(ordered[0].get("score", 0.0))
    query_terms = set(_meaningful_tokens(query))
    if not query_terms:
        return min(1.0, top_score)
    top_text = ordered[0].get("text", "")
    overlap = len(query_terms & set(_meaningful_tokens(top_text))) / max(1, len(query_terms))
    confidence = (0.65 * top_score) + (0.35 * overlap)
    if len(ordered) > 1:
        confidence = min(1.0, confidence + 0.05 * min(2, len(ordered) - 1))
    return round(float(confidence), 4)


def is_reasoning_query(query: str) -> bool:
    q = query.lower()
    reasoning_terms = (
        "calculate", "total", "sum", "overall", "aggregate", "combined",
        "how much", "how many", "after 4 years", "over 4 years", "for 4 years",
        "what happens after", "in total",
    )
    return any(term in q for term in reasoning_terms)


def decide_answer_mode(
    *,
    query: str,
    docs: Sequence[Dict[str, str]],
    confidence: float,
) -> str:
    if not docs:
        return "reject"
    if confidence < LOW_RETRIEVAL_CONFIDENCE:
        return "strict_answer"
    if confidence < HIGH_RETRIEVAL_CONFIDENCE:
        return "strict_answer"
    return "normal_answer"


def prepare_rag_bundle(
    query: str,
    *,
    top_k: int = TOP_K,
    style: str = "elaborate",
    max_context_chars: Optional[int] = None,
) -> Dict[str, object]:
    retrieval_query = _normalize_query_for_retrieval(query)
    docs = search(retrieval_query, top_k=top_k)
    context_limit = (
        MAX_CONTEXT_CHARS.get(style, MAX_CONTEXT_CHARS_DEFAULT)
        if isinstance(MAX_CONTEXT_CHARS, dict)
        else MAX_CONTEXT_CHARS_DEFAULT
    )
    if max_context_chars is not None:
        context_limit = max(0, min(int(context_limit), int(max_context_chars)))
    confidence = retrieval_confidence(docs, query)
    mode = decide_answer_mode(query=query, docs=docs, confidence=confidence)
    context_min_score = STRICT_MIN_SCORE if mode == "normal_answer" else LOW_RETRIEVAL_CONFIDENCE
    context = build_context(
        docs,
        max_chunks=max(STRICT_TOP_K, min(5, top_k)),
        min_score=context_min_score,
        max_chars=context_limit,
    )
    points = extract_key_points(docs, query=query)
    logger.debug(
        "RAG bundle: confidence=%.3f low=%.2f high=%.2f mode=%s context_min=%.2f docs=%s",
        confidence,
        LOW_RETRIEVAL_CONFIDENCE,
        HIGH_RETRIEVAL_CONFIDENCE,
        mode,
        context_min_score,
        [
            {"score": d.get("score"), "source": d.get("source")}
            for d in docs[: min(5, len(docs))]
        ],
    )
    return {
        "query": query,
        "retrieval_query": retrieval_query,
        "docs": docs,
        "context": context,
        "points": points,
        "confidence": confidence,
        "mode": mode,
        "reasoning": is_reasoning_query(query),
        "style": style,
    }


def answer_is_grounded(answer: str, context: str) -> bool:
    answer = (answer or "").strip()
    if not answer:
        return False
    if not context.strip():
        return answer.lower() == REFERENCE_FALLBACK.lower()

    answer_norm = _normalise_text(answer)
    context_norm = _normalise_text(context)
    numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", answer_norm)
    for num in numbers:
        if num not in context_norm:
            return False
    tokens = [tok for tok in _meaningful_tokens(answer_norm) if len(tok) >= 5]
    if not tokens:
        return True
    supported = sum(1 for tok in tokens if tok in context_norm)
    return supported / max(1, len(tokens)) >= 0.75


def build_strict_messages(
    query: str,
    *,
    context: str,
    style: str = "elaborate",
    reasoning: bool = False,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    system_content = STRICT_RAG_PROMPT_COMPUTE if reasoning else STRICT_RAG_PROMPT
    system_content = f"{system_content}\n\n{style_structure_instruction(style)}"
    messages = [{"role": "system", "content": system_content}]
    if history:
        for msg in history:
            role = msg.get("role")
            content = msg.get("content", "").strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
    user_content = f"Reference information:\n{context}\n\nQuestion: {query}"
    messages.append({"role": "user", "content": user_content})
    return messages


def index_stats() -> Dict[str, int]:
    index, docs = _index_snapshot()
    if index is None:
        index = load_index()
        _, docs = _index_snapshot()
    return {"vectors": int(index.ntotal), "chunks": len(docs)}


def _installed_models(session: requests.Session) -> List[str]:
    try:
        resp = session.get(OLLAMA_TAGS_URL, timeout=(8, 10))
        resp.raise_for_status()
        models = resp.json().get("models", [])
        models.sort(key=lambda m: m.get("size", 99_000_000_000))
        return [m["name"] for m in models if m.get("name")]
    except Exception:
        return []


def _candidate_models(requested: str, installed: List[str]) -> List[str]:
    installed_set = set(installed)
    ordered: List[str] = []

    def _add(name: str) -> None:
        if name and name not in ordered:
            ordered.append(name)

    _add(requested)
    for fb in FALLBACK_MODELS:
        if fb in installed_set:
            _add(fb)
    for model in installed:
        _add(model)
    return ordered


def _build_messages(prompt: str, history: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, str]]:
    return build_strict_messages(prompt, context="", history=history)


def call_llm(
    prompt: str,
    history: Optional[List[Dict[str, str]]] = None,
    model: Optional[str] = None,
) -> str:
    session = requests.Session()
    requested_models: List[str] = []
    if model:
        requested_models.append(model)
    requested_models.append(DEFAULT_MODEL)
    requested_models.extend(FALLBACK_MODELS)

    installed = _installed_models(session)
    candidate_models: List[str] = []
    for requested in requested_models:
        for candidate in _candidate_models(requested, installed):
            if candidate not in candidate_models:
                candidate_models.append(candidate)
    if not candidate_models:
        raise RuntimeError("No Ollama models found.")

    messages = _build_messages(prompt, history)
    last_error: Optional[str] = None
    for candidate in candidate_models:
        try:
            resp = session.post(
                OLLAMA_URL,
                json={
                    "model": candidate,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 400, "num_ctx": 2048},
                },
                timeout=(8, REQUEST_TIMEOUT),
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content", "")
        except Exception as exc:
            last_error = str(exc)
    raise RuntimeError(last_error or "Ollama request failed")
