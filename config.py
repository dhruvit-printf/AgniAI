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
USE_RERANKER = os.getenv("USE_RERANKER", "1") not in {"0", "false", "False"}

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_URL = os.getenv("OLLAMA_CHAT_URL", f"{OLLAMA_BASE_URL}/api/chat")
OLLAMA_TAGS_URL = os.getenv("OLLAMA_TAGS_URL", f"{OLLAMA_BASE_URL}/api/tags")

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b-instruct")
# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT WINDOW
# Raised default for longer structured answers; override via env var if needed.
# ─────────────────────────────────────────────────────────────────────────────
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

# Retrieval
TOP_K = int(os.getenv("TOP_K", "8"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.20"))
STRICT_MIN_SCORE = float(os.getenv("STRICT_MIN_SCORE", "0.70"))
STRICT_TOP_K = int(os.getenv("STRICT_TOP_K", "5"))
LOW_RETRIEVAL_CONFIDENCE = float(os.getenv("LOW_RETRIEVAL_CONFIDENCE", "0.45"))
HIGH_RETRIEVAL_CONFIDENCE = float(os.getenv("HIGH_RETRIEVAL_CONFIDENCE", "0.70"))
MIN_RETRIEVAL_CONFIDENCE = LOW_RETRIEVAL_CONFIDENCE

# Hybrid retrieval
DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "0.55"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.45"))
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
# CONTEXT CHAR BUDGETS
# How many characters of retrieved context to pass to the LLM per style.
# These budgets leave more room for the longer completion targets.
# ─────────────────────────────────────────────────────────────────────────────
MAX_CONTEXT_CHARS = {
    "short":     int(os.getenv("MAX_CONTEXT_CHARS_SHORT",     "2000")),
    "elaborate": int(os.getenv("MAX_CONTEXT_CHARS_ELABORATE", "3500")),
    "detail":    int(os.getenv("MAX_CONTEXT_CHARS_DETAIL",    "5000")),
}
MAX_CONTEXT_CHARS_DEFAULT = int(os.getenv("MAX_CONTEXT_CHARS_DEFAULT", "5000"))

# ─────────────────────────────────────────────────────────────────────────────
# TOKEN BUDGETS FOR COMPLETION
# Target word counts from spec:
#   short    ≈ 250–300 words  → ~320–400 tokens
#   elaborate≈ 500–600 words  → ~640–780 tokens
#   detail   ≈ 850–900 words  → ~1100–1170 tokens
#
# With a 4096-token context window there is more room for longer completions
# while still leaving headroom for prompt and retrieved context.
# ─────────────────────────────────────────────────────────────────────────────
MAX_TOKENS_STYLE = {
    "short":     int(os.getenv("MAX_TOKENS_SHORT",     "420")),
    "elaborate": int(os.getenv("MAX_TOKENS_ELABORATE", "820")),
    "detail":    int(os.getenv("MAX_TOKENS_DETAIL",   "1250")),
}
MAX_TOKENS_DEFAULT = int(os.getenv("MAX_TOKENS_DEFAULT", "820"))

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
# STYLE OUTPUT GUIDANCE
# These instructions are injected into the system prompt.  They tell the LLM
# how much to write per numbered point so that the total answer lands in the
# correct word-count band.
# ─────────────────────────────────────────────────────────────────────────────
STYLE_OUTPUT_GUIDANCE = {
    "short": (
        "OUTPUT FORMAT — SHORT:\n"
        "Target: 250 to 300 words total.\n"
        "Format: Numbered list only. Each item = one point title + one complete sentence.\n"
        "Rule: Cover only the specific topic asked. Do NOT cover unrelated topics.\n"
        "Rule: Do NOT write sub-bullets or paragraphs under any point.\n"
        "End: One complete concluding sentence after the last numbered point."
    ),
    "elaborate": (
        "OUTPUT FORMAT — ELABORATE:\n"
        "Target: 500 to 600 words total.\n"
        "Format: Numbered list. Each item = point title + 2 to 3 complete sentences.\n"
        "Content per point: (a) what it means, (b) how it applies to Agniveer candidates, "
        "(c) any specific figure or rule from the reference material.\n"
        "Rule: Cover only the specific topic asked. Do NOT cover unrelated topics.\n"
        "Rule: Do NOT nest a numbered list inside a point explanation.\n"
        "End: One complete concluding sentence after the last numbered point."
    ),
    "detail": (
        "OUTPUT FORMAT — DETAIL:\n"
        "Target: 850 to 900 words total.\n"
        "Format: Numbered list. Each item = point title + a full paragraph of 4 to 6 complete sentences.\n"
        "Content per point: (a) clear definition, (b) detailed explanation with reference data, "
        "(c) specific figures/dates/rules, (d) practical meaning for a candidate, "
        "(e) any exceptions or special cases.\n"
        "Rule: Cover only the specific topic asked. Do NOT cover unrelated topics.\n"
        "Rule: Do NOT nest a numbered list inside a point explanation.\n"
        "End: A concluding paragraph of 2 to 3 complete sentences."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# PER-POINT TOKEN BUDGETS (used when generating each point's explanation)
# ─────────────────────────────────────────────────────────────────────────────
STYLE_POINT_TOKEN_BUDGET = {
    "short":     int(os.getenv("STYLE_SHORT_POINT_TOKENS",     "0")),
    "elaborate": int(os.getenv("STYLE_ELABORATE_POINT_TOKENS", "120")),
    "detail":    int(os.getenv("STYLE_DETAIL_POINT_TOKENS",    "220")),
}

# Strict prompt
REFERENCE_FALLBACK = "Not available in the document"

_COMPLETION_GUARD = (
    "\n\nFINAL INSTRUCTION: You MUST write a complete answer. "
    "If you are approaching your limit, shorten earlier points slightly "
    "but ALWAYS finish the current sentence and write a concluding sentence. "
    "An answer that ends mid-sentence will be rejected. "
    "An answer shorter than the required word count will be rejected."
)

STRICT_RAG_PROMPT = (
    "You are a strict question-answering system. "
    "Use only the provided context. Do not add external knowledge. "
    "If context exists, always produce a structured answer. "
    "If the answer is partially available, use what is present and expand intelligently. "
    "Be complete and thorough. Prioritize all relevant key points."
) + _COMPLETION_GUARD

STRICT_RAG_PROMPT_COMPUTE = (
    "You are a strict question-answering system. "
    "Use only the provided context. Do not add external knowledge. "
    "You may compute or aggregate values only from the provided context. "
    "If context exists, always produce a structured answer. "
    "If the answer is partially available, use what is present and expand intelligently. "
    "Be complete and thorough. Prioritize all relevant key points."
) + _COMPLETION_GUARD

STRUCTURE_FIRST_PROMPT = (
    "CRITICAL RULES (apply to every response):\n"
    "1. Answer ONLY the specific question asked. Do not answer a different question.\n"
    "2. Extract key facts from the reference information provided — do not invent facts.\n"
    "3. Use numbered points only. Each point must be a real topic from the context.\n"
    "4. SHORT style: numbered point title + one complete sentence. No explanation.\n"
    "5. ELABORATE style: numbered point title + 2 to 3 sentences of explanation using "
    "context data (figures, rules, dates where available).\n"
    "6. DETAIL style: numbered point title + a full paragraph (4 to 6 sentences) with "
    "specific data, eligibility rules, figures, and practical implications.\n"
    "7. NEVER invent information not present in the reference material.\n"
    "8. NEVER copy the exact wording of earlier points into later points.\n"
    "9. NEVER repeat a numbered list inside an explanation paragraph.\n"
    "10. Every sentence must be grammatically complete. Never stop mid-sentence.\n"
    "11. After the last numbered point, write one complete concluding sentence.\n"
    "12. Respect the word-count target — stay within 20 words of the target range."
)


def style_structure_instruction(style: str) -> str:
    style_key = (style or "").strip().lower()
    guidance = STYLE_OUTPUT_GUIDANCE.get(style_key, STYLE_OUTPUT_GUIDANCE["elaborate"])
    return f"{STRUCTURE_FIRST_PROMPT}\n\n{guidance}"


SYSTEM_PROMPT_SHORT = STRICT_RAG_PROMPT
SYSTEM_PROMPT_ELABORATE = STRICT_RAG_PROMPT
SYSTEM_PROMPT_DETAIL = STRICT_RAG_PROMPT
SYSTEM_PROMPT = STRICT_RAG_PROMPT


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

    # Already ends cleanly
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
