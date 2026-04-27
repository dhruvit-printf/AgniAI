"""Central configuration for AgniAI."""

from __future__ import annotations

import os
import re
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"

DOCSTORE_PATH = INDEX_DIR / "docstore.json"
FAISS_INDEX_PATH = INDEX_DIR / "agni.index"
BM25_INDEX_PATH = INDEX_DIR / "bm25.pkl"

# Embeddings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
# Reranker disabled by default — adds latency without meaningful accuracy gain
# for a small single-domain knowledge base. Set USE_RERANKER=1 to enable.
USE_RERANKER = os.getenv("USE_RERANKER", "0") not in {"0", "false", "False"}

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_URL = os.getenv("OLLAMA_CHAT_URL", f"{OLLAMA_BASE_URL}/api/chat")
OLLAMA_TAGS_URL = os.getenv("OLLAMA_TAGS_URL", f"{OLLAMA_BASE_URL}/api/tags")

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct")

# Context window — 4096 is the safe default for mistral/llama3 7B models
MODEL_MAX_CONTEXT_TOKENS = int(os.getenv("OLLAMA_NUM_CTX", "4096"))

FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv(
        "OLLAMA_FALLBACK_MODELS",
        "mistral:7b-instruct,mistral:7b-instruct-q4_0,llama3:8b,llama3:8b-instruct,"
        "llama3.1:8b,llama3.2:3b,gemma2:2b",
    ).split(",")
    if m.strip()
]

# Chunking
CHUNK_WORDS = int(os.getenv("CHUNK_WORDS", "420"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
CHUNK_MIN_WORDS = int(os.getenv("CHUNK_MIN_WORDS", "12"))

# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL
# Reduced TOP_K from 8 → 5: fetches fewer chunks, speeds up embedding + context
# building without losing relevant content for a focused domain knowledge base.
# ─────────────────────────────────────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", "5"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "4"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.20"))
STRICT_MIN_SCORE = float(os.getenv("STRICT_MIN_SCORE", "0.55"))
STRICT_TOP_K = int(os.getenv("STRICT_TOP_K", "4"))
LOW_RETRIEVAL_CONFIDENCE = float(os.getenv("LOW_RETRIEVAL_CONFIDENCE", "0.35"))
HIGH_RETRIEVAL_CONFIDENCE = float(os.getenv("HIGH_RETRIEVAL_CONFIDENCE", "0.60"))
MIN_RETRIEVAL_CONFIDENCE = LOW_RETRIEVAL_CONFIDENCE

# Hybrid retrieval
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.60"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.40"))
USE_HYBRID = os.getenv("USE_HYBRID", "1") not in {"0", "false", "False"}

# Memory
MEMORY_MAX_MESSAGES = int(os.getenv("MEMORY_MAX_MESSAGES", "10"))

# Network and cache
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "120"))
FIRST_TOKEN_TIMEOUT = int(os.getenv("FIRST_TOKEN_TIMEOUT", "30"))
STREAM_TIMEOUT = int(os.getenv("STREAM_TIMEOUT", "300"))
RETRIEVAL_CACHE_TTL = int(os.getenv("RETRIEVAL_CACHE_TTL", "300"))
RESPONSE_CACHE_TTL = int(os.getenv("RESPONSE_CACHE_TTL", "300"))
EMBED_CACHE_TTL = int(os.getenv("EMBED_CACHE_TTL", "3600"))
MAX_CACHE_ENTRIES = int(os.getenv("MAX_CACHE_ENTRIES", "2048"))
SESSION_HEADER = os.getenv("SESSION_HEADER", "X-Session-Id")

# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT CHAR BUDGETS — how many characters of retrieved context to send to LLM
# Keeping these moderate prevents prompt token bloat which slows inference.
# ─────────────────────────────────────────────────────────────────────────────
MAX_CONTEXT_CHARS = {
    "short":     int(os.getenv("MAX_CONTEXT_CHARS_SHORT",     "1500")),
    "elaborate": int(os.getenv("MAX_CONTEXT_CHARS_ELABORATE", "2500")),
    "detail":    int(os.getenv("MAX_CONTEXT_CHARS_DETAIL",    "3500")),
}
MAX_CONTEXT_CHARS_DEFAULT = int(os.getenv("MAX_CONTEXT_CHARS_DEFAULT", "2500"))

# ─────────────────────────────────────────────────────────────────────────────
# TOKEN BUDGETS FOR COMPLETION
# Reduced significantly vs previous values:
#   short    ≈ 80-120  words  → 200  tokens (was 420)
#   elaborate≈ 180-250 words  → 350  tokens (was 820)
#   detail   ≈ 300-400 words  → 500  tokens (was 1250)
#
# Smaller budgets = faster inference. Paragraph answers are more dense
# than numbered lists, so fewer tokens convey more information.
# ─────────────────────────────────────────────────────────────────────────────
MAX_TOKENS_STYLE = {
    "short":     int(os.getenv("MAX_TOKENS_SHORT",     "200")),
    "elaborate": int(os.getenv("MAX_TOKENS_ELABORATE", "350")),
    "detail":    int(os.getenv("MAX_TOKENS_DETAIL",    "500")),
}
MAX_TOKENS_DEFAULT = int(os.getenv("MAX_TOKENS_DEFAULT", "350"))

# Safety buffer (tokens reserved for model overhead / special tokens)
TOKEN_SAFETY_BUFFER = int(os.getenv("TOKEN_SAFETY_BUFFER", "200"))

# CORS
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

# Answer-style keywords
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

# ─────────────────────────────────────────────────────────────────────────────
# STYLE OUTPUT GUIDANCE — paragraph format, no numbered lists
# These instructions are injected into the system prompt.
# ─────────────────────────────────────────────────────────────────────────────
STYLE_OUTPUT_GUIDANCE = {
    "short": (
        "OUTPUT FORMAT — SHORT:\n"
        "Write one concise, complete paragraph.\n"
        "Use only the provided reference information.\n"
        "Do not add explanations, tips, assumptions, or examples that are not in the reference.\n"
        "If the answer is incomplete or missing, respond exactly: "
        "'Answer not found in the document.'"
    ),
    "elaborate": (
        "OUTPUT FORMAT — ELABORATE:\n"
        "Write 2 to 3 complete paragraphs.\n"
        "Use only the provided reference information.\n"
        "Do not add explanations, tips, assumptions, or examples that are not in the reference.\n"
        "If the answer is incomplete or missing, respond exactly: "
        "'Answer not found in the document.'"
    ),
    "detail": (
        "OUTPUT FORMAT — DETAIL:\n"
        "Write 3 to 4 complete paragraphs.\n"
        "Use only the provided reference information.\n"
        "Do not add explanations, tips, assumptions, or examples that are not in the reference.\n"
        "If the answer is incomplete or missing, respond exactly: "
        "'Answer not found in the document.'"
    ),
}

# Per-point token budgets (kept for backward compat but not used in paragraph mode)
STYLE_POINT_TOKEN_BUDGET = {
    "short":     int(os.getenv("STYLE_SHORT_POINT_TOKENS",     "0")),
    "elaborate": int(os.getenv("STYLE_ELABORATE_POINT_TOKENS", "80")),
    "detail":    int(os.getenv("STYLE_DETAIL_POINT_TOKENS",    "150")),
}

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

# Core paragraph-format instruction prepended to every style guidance block
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
    "9. Every sentence must be grammatically complete and the final sentence must end cleanly."
)


def style_structure_instruction(style: str) -> str:
    style_key = (style or "").strip().lower()
    guidance = STYLE_OUTPUT_GUIDANCE.get(style_key, STYLE_OUTPUT_GUIDANCE["elaborate"])
    return f"{_PARAGRAPH_RULES}\n\n{guidance}"


# ─────────────────────────────────────────────────────────────────────────────
# STRICT RAG SYSTEM PROMPTS
# Two variants: standard and compute (allows aggregation of numbers from context)
# ─────────────────────────────────────────────────────────────────────────────
STRICT_RAG_PROMPT = (
    "You are a strict document-based QA system for Agniveer / Agnipath recruitment content. "
    "Use ONLY the provided context. Do NOT use prior knowledge. Do NOT infer missing information. "
    "Do NOT add explanations, tips, assumptions, or generic filler. "
    "If the answer is incomplete or missing, respond exactly with: "
    "'Answer not found in the document.' "
    "Write in clear structured paragraphs only. Do not repeat the question."
)

STRICT_RAG_PROMPT_COMPUTE = (
    "You are a strict document-based QA system for Agniveer / Agnipath recruitment content. "
    "Use ONLY the provided context. You may compute or aggregate values only from figures explicitly present in the context. "
    "Do NOT use prior knowledge. Do NOT infer missing information. "
    "Do NOT add explanations, tips, assumptions, or generic filler. "
    "If the answer is incomplete or missing, respond exactly with: "
    "'Answer not found in the document.' "
    "Write in clear structured paragraphs only. Do not repeat the question."
)

# Aliases kept for backward compatibility with other modules
SYSTEM_PROMPT_SHORT = STRICT_RAG_PROMPT
SYSTEM_PROMPT_ELABORATE = STRICT_RAG_PROMPT
SYSTEM_PROMPT_DETAIL = STRICT_RAG_PROMPT
SYSTEM_PROMPT = STRICT_RAG_PROMPT

# Fallback answer when no relevant context is found
REFERENCE_FALLBACK = "Answer not found in the document."


# ─────────────────────────────────────────────────────────────────────────────
# TOKEN ESTIMATION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

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
    trimmed = text[:last_end].strip()

    if trimmed and len(trimmed) >= len(text) * 0.70:
        return trimmed

    return text
