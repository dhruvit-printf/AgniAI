"""
config.py
=========
Central configuration for AgniAI.
All values can be overridden by environment variables where noted.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"

DOCSTORE_PATH    = INDEX_DIR / "docstore.json"
FAISS_INDEX_PATH = INDEX_DIR / "agni.index"

# ── Embeddings ─────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384

# ── Ollama ─────────────────────────────────────────────────────────────────
OLLAMA_URL      = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

# Prefer smallest capable models for CPU-only hardware.
# llama3:latest (8B) is deliberately last — too slow on most CPUs.
DEFAULT_MODEL   = "phi3:mini"
FALLBACK_MODELS = [
    "phi3:mini",
    "phi3:3.8b",
    "gemma2:2b",
    "llama3.2:3b",
    "llama3.2:1b",
    "mistral:7b-instruct-q4_0",
    "llama3:latest",
]

# ── Chunking ───────────────────────────────────────────────────────────────
CHUNK_WORDS   = 400
CHUNK_OVERLAP = 40

# ── Retrieval ──────────────────────────────────────────────────────────────
TOP_K     = 3
MIN_SCORE = 0.01   # very low — never silently drop chunks; let the LLM decide relevance

# ── Memory ─────────────────────────────────────────────────────────────────
MEMORY_MAX_MESSAGES = 6   # 3 user + 3 assistant turns

# ── Network ────────────────────────────────────────────────────────────────
REQUEST_TIMEOUT = 90

# ── Context budget sent to the LLM ─────────────────────────────────────────
MAX_CONTEXT_CHARS = 2000   # ~500 tokens — fits in num_ctx=1024

# ── Prompts ────────────────────────────────────────────────────────────────
# IMPORTANT: Keep this prompt SHORT and POSITIVE for small models (phi3, gemma2).
# Long rule lists cause small models to trigger the refusal clause even when
# context IS present. Tell the model what TO do, not what NOT to do.
SYSTEM_PROMPT = """\
You are AgniAI, an expert assistant for India's Agniveer / Agnipath military recruitment scheme.

Instructions:
- The user's question and relevant reference text are provided below.
- Read the reference text carefully and answer the question from it.
- Be concise and structured. Use bullet points (•) or numbered steps where helpful.
- End your answer with "📌 Source:" followed by the filename or URL from the reference text.
- If the reference text does not contain the answer, say: "The provided documents do not cover this topic. Please ingest the relevant document."
"""