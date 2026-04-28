"""Central configuration for AgniAI."""

from __future__ import annotations

import os
import re
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"

DOCSTORE_PATH    = INDEX_DIR / "docstore.json"
FAISS_INDEX_PATH = INDEX_DIR / "agni.index"
BM25_INDEX_PATH  = INDEX_DIR / "bm25.pkl"

# ── Embeddings ─────────────────────────────────────────────────────────────
# CHANGED: switched from all-mpnet-base-v2 (420MB, 768-dim, ~300ms/query)
#          to multi-qa-MiniLM-L6-cos-v1   (80MB,  384-dim, ~40ms/query)
#
# multi-qa-MiniLM-L6-cos-v1 was trained specifically on question-answer
# retrieval pairs — exactly what AgniAI does. It outperforms the general
# all-MiniLM-L6-v2 on closed-domain factual QA retrieval benchmarks.
#
# IMPORTANT: After changing this you MUST re-ingest all documents because
# stored FAISS vectors are dimension-specific:
#   python -c "from ingest import clear_index; clear_index()"
#   python -c "from ingest import ingest_pdf; ingest_pdf('Agniveer.pdf')"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
EMBEDDING_DIM   = int(os.getenv("EMBEDDING_DIM", "384"))  # was 768

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
# Reranker disabled by default — adds latency without meaningful accuracy gain
# for a small single-domain knowledge base. Set USE_RERANKER=1 to enable.
USE_RERANKER = os.getenv("USE_RERANKER", "0") not in {"0", "false", "False"}

# ── Ollama ─────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_URL      = os.getenv("OLLAMA_CHAT_URL", f"{OLLAMA_BASE_URL}/api/chat")
OLLAMA_TAGS_URL = os.getenv("OLLAMA_TAGS_URL", f"{OLLAMA_BASE_URL}/api/tags")

# CHANGED: q4_K_M quantization — 2-3x faster than full model, retains ~99%
# accuracy on closed-domain factual QA. RAM: ~4.1GB vs ~5.5GB for full model.
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct-q4_K_M")

# Context window — 4096 is the safe default for mistral/llama3 7B models
MODEL_MAX_CONTEXT_TOKENS = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv(
        "OLLAMA_FALLBACK_MODELS",
        "mistral:7b-instruct-q4_K_M,mistral:7b-instruct,mistral:7b-instruct-q4_0,"
        "llama3:8b,llama3:8b-instruct,llama3.1:8b,llama3.2:3b,gemma2:2b",
    ).split(",")
    if m.strip()
]

# ── Chunking ───────────────────────────────────────────────────────────────
CHUNK_WORDS    = int(os.getenv("CHUNK_WORDS",    "420"))
CHUNK_OVERLAP  = int(os.getenv("CHUNK_OVERLAP",  "80"))
CHUNK_MIN_WORDS = int(os.getenv("CHUNK_MIN_WORDS", "12"))

# ── Retrieval ──────────────────────────────────────────────────────────────
TOP_K         = int(os.getenv("TOP_K",         "5"))
RERANK_TOP_K  = int(os.getenv("RERANK_TOP_K",  "4"))
MIN_SCORE     = float(os.getenv("MIN_SCORE",    "0.20"))
STRICT_MIN_SCORE       = float(os.getenv("STRICT_MIN_SCORE",        "0.55"))
STRICT_TOP_K           = int(os.getenv("STRICT_TOP_K",              "4"))
LOW_RETRIEVAL_CONFIDENCE  = float(os.getenv("LOW_RETRIEVAL_CONFIDENCE",  "0.35"))
HIGH_RETRIEVAL_CONFIDENCE = float(os.getenv("HIGH_RETRIEVAL_CONFIDENCE", "0.60"))
MIN_RETRIEVAL_CONFIDENCE  = LOW_RETRIEVAL_CONFIDENCE

# Hybrid retrieval
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.60"))
BM25_WEIGHT  = float(os.getenv("BM25_WEIGHT",  "0.40"))
USE_HYBRID   = os.getenv("USE_HYBRID", "1") not in {"0", "false", "False"}

# ── Memory ─────────────────────────────────────────────────────────────────
MEMORY_MAX_MESSAGES = int(os.getenv("MEMORY_MAX_MESSAGES", "10"))

# ── Network and cache ──────────────────────────────────────────────────────
REQUEST_TIMEOUT     = int(os.getenv("REQUEST_TIMEOUT",     "120"))
FIRST_TOKEN_TIMEOUT = int(os.getenv("FIRST_TOKEN_TIMEOUT", "30"))
STREAM_TIMEOUT      = int(os.getenv("STREAM_TIMEOUT",      "300"))

# CHANGED: cache TTLs extended — Agniveer recruitment info changes annually,
# so long TTLs are safe and dramatically reduce repeated-query latency.
RETRIEVAL_CACHE_TTL = int(os.getenv("RETRIEVAL_CACHE_TTL", "3600"))   # was 300  → 1 hour
RESPONSE_CACHE_TTL  = int(os.getenv("RESPONSE_CACHE_TTL",  "86400"))  # was 300  → 24 hours
EMBED_CACHE_TTL     = int(os.getenv("EMBED_CACHE_TTL",     "86400"))  # was 3600 → 24 hours
MAX_CACHE_ENTRIES   = int(os.getenv("MAX_CACHE_ENTRIES",   "2048"))

SESSION_HEADER = os.getenv("SESSION_HEADER", "X-Session-Id")

# ── Context char budgets ───────────────────────────────────────────────────
# CHANGED: tightened slightly — FAISS+BM25 already retrieves the most
# relevant chunks; extra context adds noise more than signal at these ranges.
# Tighter context = faster LLM processing + more focused answers.
MAX_CONTEXT_CHARS = {
    "short":     int(os.getenv("MAX_CONTEXT_CHARS_SHORT",     "1000")),  # was 1500
    "elaborate": int(os.getenv("MAX_CONTEXT_CHARS_ELABORATE", "1800")),  # was 2500
    "detail":    int(os.getenv("MAX_CONTEXT_CHARS_DETAIL",    "2800")),  # was 3500
}
MAX_CONTEXT_CHARS_DEFAULT = int(os.getenv("MAX_CONTEXT_CHARS_DEFAULT", "1800"))

# ── Token budgets for completion ───────────────────────────────────────────
MAX_TOKENS_STYLE = {
    "short":     int(os.getenv("MAX_TOKENS_SHORT",     "250")),
    "elaborate": int(os.getenv("MAX_TOKENS_ELABORATE", "500")),
    "detail":    int(os.getenv("MAX_TOKENS_DETAIL",    "800")),
}
MAX_TOKENS_DEFAULT = int(os.getenv("MAX_TOKENS_DEFAULT", "500"))

# Safety buffer (tokens reserved for model overhead / special tokens)
TOKEN_SAFETY_BUFFER = int(os.getenv("TOKEN_SAFETY_BUFFER", "100"))

# ── CORS ───────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

# ── Answer-style keywords ──────────────────────────────────────────────────
STYLE_SHORT_KEYWORDS = [
    "in short", "briefly", "brief", "quick answer", "short answer",
    "summarise", "summarize", "tldr", "tl;dr", "in brief",
    "give me short", "one line", "one-line",
    "give a short", "keep it short", "summary", "summarise it",
    "quick summary",
]

STYLE_DETAIL_KEYWORDS = [
    "in detail", "detailed", "explain in detail", "full detail",
    "comprehensive", "thoroughly", "exhaustive", "step by step",
    "step-by-step", "explain fully", "tell me everything",
    "give me detail", "elaborate in detail", "full explanation",
    "complete explanation", "everything about", "all about",
    "full breakdown", "break it down",
]

STYLE_ELABORATE_KEYWORDS = [
    "elaborate", "explain", "elaborate on", "tell me more",
    "expand on", "describe", "give more", "more info", "more detail",
    "walk me through", "how does", "how do",
]

# ── Style output guidance ──────────────────────────────────────────────────
STYLE_OUTPUT_GUIDANCE = {
    "short": (
        "OUTPUT FORMAT — SHORT (strict):\n"
        "Write exactly ONE concise paragraph of 60–90 words.\n"
        "State only the key facts. No elaboration, no repetition, no summary sentence at the end.\n"
        "Use ONLY the provided reference information.\n"
        "If the answer is not in the reference, respond exactly: 'Answer not found in the document.'"
    ),
    "elaborate": (
        "OUTPUT FORMAT — ELABORATE (strict):\n"
        "Write 2 to 3 paragraphs totalling 150–250 words.\n"
        "First paragraph: core facts. Second paragraph: supporting details or context. "
        "Third paragraph (optional): practical implications or exceptions.\n"
        "Use ONLY the provided reference information. No bullet points unless listing more than 4 items.\n"
        "If the answer is not in the reference, respond exactly: 'Answer not found in the document.'"
    ),
    "detail": (
        "OUTPUT FORMAT — DETAIL (strict):\n"
        "Write 3 to 5 paragraphs totalling 300–450 words.\n"
        "Cover: (1) core eligibility/facts, (2) specific requirements or conditions, "
        "(3) exceptions or special cases, (4) process or procedure if applicable, "
        "(5) important notes or warnings.\n"
        "Use ONLY the provided reference information. Be thorough but do not invent information.\n"
        "If the answer is not in the reference, respond exactly: 'Answer not found in the document.'"
    ),
}

# Per-point token budgets (kept for backward compat)
STYLE_POINT_TOKEN_BUDGET = {
    "short":     int(os.getenv("STYLE_SHORT_POINT_TOKENS",     "0")),
    "elaborate": int(os.getenv("STYLE_ELABORATE_POINT_TOKENS", "80")),
    "detail":    int(os.getenv("STYLE_DETAIL_POINT_TOKENS",    "150")),
}

# ── System prompts ─────────────────────────────────────────────────────────
_PARAGRAPH_RULES = (
    "RULES (follow strictly for every response):\n"
    "1. Use ONLY the provided context.\n"
    "2. Do NOT use prior knowledge.\n"
    "3. Do NOT add explanations, tips, assumptions, or examples that are not present in the context.\n"
    "4. Do NOT infer missing information.\n"
    "5. If the answer is incomplete or missing, respond exactly: "
    "'Answer not found in the document.'\n"
    "6. Do NOT repeat the question.\n"
    "7. Do NOT include generic filler sentences.\n"
    "8. Write in clear structured paragraphs only. Avoid bullet overload unless it is truly needed.\n"
    "9. Deliver all key facts first, compress the remainder if needed, and never cut mid-sentence.\n"
    "10. Every sentence must be grammatically complete and the final sentence must end cleanly."
)


def style_structure_instruction(style: str) -> str:
    style_key = (style or "").strip().lower()
    guidance = STYLE_OUTPUT_GUIDANCE.get(style_key, STYLE_OUTPUT_GUIDANCE["elaborate"])
    return f"{_PARAGRAPH_RULES}\n\n{guidance}"


STRICT_RAG_PROMPT = (
    "You are a strict document-based QA system for Agniveer / Agnipath recruitment content. "
    "Use ONLY the provided context. Do NOT use prior knowledge. Do NOT infer missing information. "
    "Do NOT add explanations, tips, assumptions, or generic filler. "
    "Deliver all key facts first, keep the answer compact, and avoid unnecessary repetition. "
    "If the answer is incomplete or missing, respond exactly with: "
    "'Answer not found in the document.' "
    "Write in clear structured paragraphs only. Do not repeat the question."
)

STRICT_RAG_PROMPT_COMPUTE = (
    "You are a strict document-based QA system for Agniveer / Agnipath recruitment content. "
    "Use ONLY the provided context. You may compute or aggregate values only from figures explicitly present in the context. "
    "Do NOT use prior knowledge. Do NOT infer missing information. "
    "Do NOT add explanations, tips, assumptions, or generic filler. "
    "Deliver all key facts first, keep the answer compact, and avoid unnecessary repetition. "
    "If the answer is incomplete or missing, respond exactly with: "
    "'Answer not found in the document.' "
    "Write in clear structured paragraphs only. Do not repeat the question."
)

# Aliases for backward compatibility
SYSTEM_PROMPT_SHORT    = STRICT_RAG_PROMPT
SYSTEM_PROMPT_ELABORATE = STRICT_RAG_PROMPT
SYSTEM_PROMPT_DETAIL   = STRICT_RAG_PROMPT
SYSTEM_PROMPT          = STRICT_RAG_PROMPT

# Fallback answer when no relevant context is found
REFERENCE_FALLBACK = "Answer not found in the document."


# ── Token estimation utilities ─────────────────────────────────────────────

def estimate_text_tokens(text: str) -> int:
    text = (text or "").strip()
    if not text:
        return 0
    raw = max(1, (len(text) + 2) // 3)
    return int(raw * 1.10)


def estimate_message_tokens(messages: list[dict]) -> int:
    total = 0
    for message in messages or []:
        if not isinstance(message, dict):
            continue
        total += 6
        total += estimate_text_tokens(str(message.get("role", "")))
        total += estimate_text_tokens(str(message.get("content", "")))
    return total


def trim_to_complete_sentence(text: str) -> str:
    """
    Return text trimmed to the last COMPLETE sentence.

    Rules:
    1. If text already ends with . ! ? — return as-is.
    2. Find the last sentence boundary and trim there.
    3. Only trim if the result keeps >= 70% of original length.
    4. If no boundary found, return original unchanged.
    """
    text = (text or "").strip()
    if not text:
        return text

    if text[-1] in ".!?":
        return text

    matches = list(re.finditer(r"[.!?](?:\s|$)", text))
    if not matches:
        return text

    last_end = matches[-1].end()
    trimmed  = text[:last_end].strip()

    if trimmed and len(trimmed) >= len(text) * 0.70:
        return trimmed

    return text