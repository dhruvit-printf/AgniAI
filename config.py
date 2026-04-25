"""Central configuration for AgniAI."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "index"

DOCSTORE_PATH = INDEX_DIR / "docstore.json"
FAISS_INDEX_PATH = INDEX_DIR / "agni.index"
BM25_INDEX_PATH = INDEX_DIR / "bm25.pkl"

# Embeddings
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
EMBEDDING_DIM = 768

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
USE_RERANKER = False

# Ollama
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

DEFAULT_MODEL = "mistral:7b-instruct"
FALLBACK_MODELS = [
    "mistral:7b-instruct",
    "mistral:7b-instruct-q4_0",
    "llama3:8b",
    "llama3:8b-instruct",
    "llama3.1:8b",
    "llama3.2:3b",
    "gemma2:2b",
]

# Chunking
CHUNK_WORDS = 420
CHUNK_OVERLAP = 80
CHUNK_MIN_WORDS = 12

# Retrieval
TOP_K = 4
RERANK_TOP_K = 3
MIN_SCORE = 0.20
STRICT_MIN_SCORE = 0.70
STRICT_TOP_K = 3

# Hybrid retrieval weights
DENSE_WEIGHT = 0.55
BM25_WEIGHT = 0.45
USE_HYBRID = True

# Memory
MEMORY_MAX_MESSAGES = 10

# Network
REQUEST_TIMEOUT = 120

# Context budget
MAX_CONTEXT_CHARS = {
    "short": 1800,
    "elaborate": 2400,
    "detail": 3000,
}
MAX_CONTEXT_CHARS_DEFAULT = 3000

# Token budget
MAX_TOKENS_STYLE = {
    "short": 300,
    "elaborate": 400,
    "detail": 500,
}
MAX_TOKENS_DEFAULT = 400

# CORS
ALLOWED_ORIGINS = "*"

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

# Strict prompt
REFERENCE_FALLBACK = "Not available in the document"

STRICT_RAG_PROMPT = """You are a strict question-answering system.

Rules:
- Answer ONLY using the provided context
- Do NOT add any external knowledge
- If answer is not present, say: 'Not available in the document'
- Do NOT hallucinate
- Keep answers precise and structured
- Ignore irrelevant context

Format:
- Use bullet points
- Be concise and factual"""

SYSTEM_PROMPT_SHORT = STRICT_RAG_PROMPT
SYSTEM_PROMPT_ELABORATE = STRICT_RAG_PROMPT
SYSTEM_PROMPT_DETAIL = STRICT_RAG_PROMPT
SYSTEM_PROMPT = STRICT_RAG_PROMPT
