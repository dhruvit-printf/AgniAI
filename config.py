"""
config.py
=========
Central configuration for AgniAI.
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
CHUNK_WORDS   = 200  # BUG-8 FIX
CHUNK_OVERLAP = 40

# ── Retrieval ──────────────────────────────────────────────────────────────
TOP_K     = 4      # retrieve up to 4 chunks for richer answers
MIN_SCORE = 0.01   # near-zero — let the LLM judge relevance, not the score filter

# ── Memory ─────────────────────────────────────────────────────────────────
MEMORY_MAX_MESSAGES = 6

# ── Network ────────────────────────────────────────────────────────────────
REQUEST_TIMEOUT = 90

# ── Context budget ─────────────────────────────────────────────────────────
# 2400 chars ~ 600 tokens — enough for 2-3 full chunks, fits in num_ctx=1024
MAX_CONTEXT_CHARS = 2400

# ── System prompt ──────────────────────────────────────────────────────────
# Short and positive — small models (phi3, gemma2) respond better to
# "here is what you should do" than to long lists of "never do X" rules.
SYSTEM_PROMPT = """\
You are AgniAI, an assistant for India's Agniveer recruitment.
Below is reference information. Read it carefully, then answer the question using ONLY that information.
Format your answer as bullet points. On the last line write: Source: <the source name from the reference>.
If the answer is genuinely not in the reference, say only: "Not found in provided documents."
"""
