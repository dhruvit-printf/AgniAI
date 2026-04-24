"""
rag.py
======
Core retrieval-augmented generation layer — accuracy-optimised edition.

Accuracy improvements over baseline:
  1. Upgraded embedding model: all-mpnet-base-v2 (768-dim vs 384-dim)
     → Better semantic understanding, ~5-8pp NDCG improvement
  2. Hybrid retrieval: dense cosine + BM25 sparse, score-fused
     → Catches keyword matches that dense retrieval misses (e.g. exact names)
  3. Cross-encoder re-ranking: ms-marco-MiniLM-L-6-v2
     → Re-scores top candidates with full query×chunk interaction
     → Typically +5-10pp precision@k vs bi-encoder alone
  4. MMR diversity: removes near-duplicate chunks before sending to LLM
     → Reduces repetition and frees token budget for distinct evidence
  5. Tighter MIN_SCORE threshold (0.20 vs 0.01) — drops noisy matches
  6. Larger context budget (3500 vs 1500 chars) — more evidence per query
  7. Context formatted with explicit chunk numbering [1], [2] so LLM
     can cite sources reliably
"""

import json
import logging
import os
import pickle
import warnings
from typing import Dict, List, Optional, Sequence

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

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
    RERANKER_MODEL,
    RERANK_TOP_K,
    REQUEST_TIMEOUT,
    SYSTEM_PROMPT,
    TOP_K,
    USE_HYBRID,
    USE_RERANKER,
)

# ── Module-level singletons ────────────────────────────────────────────────
_MODEL: Optional[SentenceTransformer] = None
_RERANKER = None          # CrossEncoder, loaded lazily
_INDEX: Optional[faiss.Index] = None
_DOCS: List[Dict[str, str]] = []
_BM25 = None              # rank_bm25.BM25Okapi, loaded lazily


# ── Helpers ────────────────────────────────────────────────────────────────

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
                repaired.append(ch); escaped = False; continue
            if ch == "\\":
                repaired.append(ch); escaped = True; continue
            if ch == '"':
                repaired.append(ch); in_string = False; continue
            if ch == "\n": repaired.append("\\n"); continue
            if ch == "\r": repaired.append("\\r"); continue
            if ch == "\t": repaired.append("\\t"); continue
            if ord(ch) < 32: repaired.append(f"\\u{ord(ch):04x}"); continue
            repaired.append(ch); continue
        repaired.append(ch)
        if ch == '"':
            in_string = True
    return "".join(repaired)


def _extract_json_scalar(line: str) -> str:
    value = line.split(":", 1)[1].strip()
    if value.endswith(","): value = value[:-1].rstrip()
    if value.startswith('"') and value.endswith('"'): value = value[1:-1]
    return value


def _repair_docstore_from_lines(raw: str) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    obj: Dict[str, str] = {}
    text_lines: List[str] = []
    in_object = False
    in_text = False
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped == "[" or not stripped: continue
        if stripped == "]": break
        if stripped.startswith("{"):
            obj = {}; text_lines = []; in_object = True; in_text = False; continue
        if not in_object: continue
        if in_text:
            if stripped in {"}", "},"}:
                obj["text"] = "\n".join(text_lines)
                docs.append(obj); obj = {}; text_lines = []
                in_object = False; in_text = False; continue
            text_lines.append(line); continue
        if stripped.startswith('"source":'): obj["source"] = _extract_json_scalar(line)
        elif stripped.startswith('"doc_type":'): obj["doc_type"] = _extract_json_scalar(line)
        elif stripped.startswith('"chunk_id":'): obj["chunk_id"] = _extract_json_scalar(line)
        elif stripped.startswith('"text":'):
            fragment = line.split(":", 1)[1].lstrip()
            if fragment.startswith('"'): fragment = fragment[1:]
            if fragment.endswith('",'):
                fragment = fragment[:-2]; obj["text"] = fragment; text_lines = []
            elif fragment.endswith('"'):
                fragment = fragment[:-1]; obj["text"] = fragment; text_lines = []
            else:
                text_lines = [fragment]; in_text = True
            continue
        elif stripped in {"}", "},"}:
            if "text" not in obj and text_lines: obj["text"] = "\n".join(text_lines)
            if obj: docs.append(obj)
            obj = {}; text_lines = []; in_object = False; in_text = False
    if in_object and obj:
        if "text" not in obj and text_lines: obj["text"] = "\n".join(text_lines)
        docs.append(obj)
    return docs


# ── Embedding model ────────────────────────────────────────────────────────

def load_embedding_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _MODEL = SentenceTransformer(EMBEDDING_MODEL)
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
    vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True, batch_size=1)
    return np.asarray(vec, dtype="float32")


# ── Cross-encoder re-ranker ────────────────────────────────────────────────

def load_reranker():
    """Lazy-load the cross-encoder re-ranker."""
    global _RERANKER
    if _RERANKER is not None:
        return _RERANKER
    if not USE_RERANKER:
        return None
    try:
        from sentence_transformers import CrossEncoder  # type: ignore
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _RERANKER = CrossEncoder(RERANKER_MODEL)
        return _RERANKER
    except Exception as exc:
        print(f"[WARNING] Could not load re-ranker ({exc}). Using bi-encoder scores only.")
        return None


def rerank(query: str, docs: List[Dict], top_n: int = RERANK_TOP_K) -> List[Dict]:
    """
    Re-rank *docs* using a cross-encoder. Returns top_n highest-scoring docs.
    Falls back to original order if the re-ranker isn't available.
    """
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
            doc = dict(doc)
            doc["rerank_score"] = round(float(score), 4)
            reranked.append(doc)
        return reranked
    except Exception as exc:
        print(f"[WARNING] Re-ranking failed ({exc}). Using original order.")
        return docs[:top_n]


# ── BM25 sparse index ──────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    import re
    return re.findall(r"[a-zA-Z0-9\u0900-\u097F]+", text.lower())


def load_bm25():
    """Lazy-load BM25 index from disk."""
    global _BM25
    if _BM25 is not None:
        return _BM25
    if not USE_HYBRID:
        return None
    if not BM25_INDEX_PATH.exists():
        return None
    try:
        with open(BM25_INDEX_PATH, "rb") as f:
            _BM25 = pickle.load(f)
        return _BM25
    except Exception as exc:
        print(f"[WARNING] Could not load BM25 index ({exc}).")
        return None


def save_bm25(docs: List[Dict[str, str]]) -> None:
    """Build and persist BM25 index from docstore."""
    if not USE_HYBRID:
        return
    try:
        from rank_bm25 import BM25Okapi  # type: ignore
        corpus = [_tokenize(d.get("text", "")) for d in docs]
        bm25 = BM25Okapi(corpus)
        _ensure_dirs()
        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump(bm25, f)
        global _BM25
        _BM25 = bm25
    except ModuleNotFoundError:
        pass  # rank_bm25 not installed — hybrid disabled silently
    except Exception as exc:
        print(f"[WARNING] BM25 index build failed: {exc}")


# ── MMR diversity filter ───────────────────────────────────────────────────

def _mmr_filter(
    query_vec: np.ndarray,
    doc_vecs: np.ndarray,
    docs: List[Dict],
    k: int,
    lambda_: float = 0.6,
) -> List[Dict]:
    """
    Maximal Marginal Relevance: balances relevance and diversity.
    lambda_=1.0 → pure relevance; lambda_=0.0 → pure diversity.
    lambda_=0.6 is the standard balanced setting.
    """
    if len(docs) <= k:
        return docs

    selected_idx: List[int] = []
    remaining = list(range(len(docs)))

    for _ in range(min(k, len(docs))):
        if not remaining:
            break
        if not selected_idx:
            # First pick: highest relevance
            scores = doc_vecs[remaining] @ query_vec.T
            best = remaining[int(np.argmax(scores))]
        else:
            rel_scores = doc_vecs[remaining] @ query_vec.T
            sim_to_selected = np.max(doc_vecs[remaining] @ doc_vecs[selected_idx].T, axis=1)
            mmr_scores = lambda_ * rel_scores.flatten() - (1 - lambda_) * sim_to_selected.flatten()
            best = remaining[int(np.argmax(mmr_scores))]
        selected_idx.append(best)
        remaining.remove(best)

    return [docs[i] for i in selected_idx]


# ── Index persistence ──────────────────────────────────────────────────────

def load_docstore() -> List[Dict[str, str]]:
    if not DOCSTORE_PATH.exists():
        return []
    raw = DOCSTORE_PATH.read_text(encoding="utf-8", errors="replace")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        repaired = _escape_control_chars_in_json_strings(raw)
        try:
            docs = json.loads(repaired)
        except json.JSONDecodeError:
            docs = _repair_docstore_from_lines(raw)
            if not docs:
                raise
        _save_docstore(docs)
        print("[WARNING] Repaired malformed docstore.json and saved cleaned copy.")
        return docs


def _save_docstore(docs: List[Dict[str, str]]) -> None:
    _ensure_dirs()
    with DOCSTORE_PATH.open("w", encoding="utf-8") as fh:
        json.dump(docs, fh, ensure_ascii=False, indent=2)


def load_index() -> faiss.Index:
    global _INDEX, _DOCS
    if _INDEX is not None and _DOCS:
        return _INDEX
    _ensure_dirs()
    if FAISS_INDEX_PATH.exists():
        _INDEX = faiss.read_index(str(FAISS_INDEX_PATH))
    else:
        _INDEX = _new_index()
    _DOCS = load_docstore()
    if _INDEX.ntotal > 0 and len(_DOCS) == 0:
        print("[WARNING] FAISS index has vectors but docstore is empty. "
              "Run /reset and re-ingest your documents.")
    return _INDEX


def save_index(index: faiss.Index, docs: List[Dict[str, str]]) -> None:
    global _DOCS
    _ensure_dirs()
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    _save_docstore(docs)
    _DOCS = docs
    # Rebuild BM25 index whenever docstore changes
    save_bm25(docs)


# ── Search ─────────────────────────────────────────────────────────────────

def _bm25_scores(query: str, n: int) -> np.ndarray:
    """
    Return BM25 scores for all docs, normalised to [0, 1].
    Returns zeros array if BM25 is unavailable.
    """
    bm25 = load_bm25()
    if bm25 is None or not _DOCS:
        return np.zeros(len(_DOCS), dtype="float32")
    try:
        tokens = _tokenize(query)
        scores = np.array(bm25.get_scores(tokens), dtype="float32")
        max_s = scores.max()
        if max_s > 0:
            scores /= max_s
        return scores
    except Exception:
        return np.zeros(len(_DOCS), dtype="float32")


def search(query: str, top_k: int = TOP_K) -> List[Dict[str, str]]:
    """
    Retrieve top-k most relevant chunks using hybrid dense+sparse search,
    followed by cross-encoder re-ranking and MMR diversity filtering.

    Pipeline:
      1. Dense cosine search → candidate_k results (2× top_k)
      2. BM25 sparse score fusion (if USE_HYBRID)
      3. Cross-encoder re-ranking (if USE_RERANKER) → RERANK_TOP_K
      4. MMR diversity filter → final top_k
    """
    index = load_index()
    if index.ntotal == 0:
        return []

    qvec = embed_query(query)
    candidate_k = min(top_k * 2, index.ntotal)   # retrieve 2× for re-ranking pool

    # ── Dense retrieval ────────────────────────────────────────
    scores_dense, ids = index.search(qvec, candidate_k)
    dense_scores = scores_dense[0]
    doc_ids = ids[0]

    # ── BM25 fusion ────────────────────────────────────────────
    if USE_HYBRID and len(_DOCS) > 0:
        bm25_all = _bm25_scores(query, len(_DOCS))
        fused: List[tuple] = []
        for i, (doc_id, ds) in enumerate(zip(doc_ids, dense_scores)):
            if doc_id < 0 or doc_id >= len(_DOCS):
                continue
            bs = float(bm25_all[doc_id])
            combined = DENSE_WEIGHT * float(ds) + BM25_WEIGHT * bs
            if combined < MIN_SCORE:
                continue
            fused.append((combined, int(doc_id)))
        fused.sort(key=lambda x: x[0], reverse=True)
        candidates = []
        for combined, doc_id in fused:
            doc = dict(_DOCS[doc_id])
            doc["score"] = round(combined, 4)
            candidates.append(doc)
    else:
        candidates = []
        for doc_id, score in zip(doc_ids, dense_scores):
            if doc_id < 0 or doc_id >= len(_DOCS):
                continue
            if float(score) < MIN_SCORE:
                continue
            doc = dict(_DOCS[doc_id])
            doc["score"] = round(float(score), 4)
            candidates.append(doc)

    if not candidates:
        return []

    # ── Cross-encoder re-ranking ───────────────────────────────
    if USE_RERANKER:
        candidates = rerank(query, candidates, top_n=min(RERANK_TOP_K, len(candidates)))

    # ── MMR diversity ──────────────────────────────────────────
    if len(candidates) > 1:
        texts = [c.get("text", "") for c in candidates]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cand_vecs = embed_texts(texts)
        candidates = _mmr_filter(qvec, cand_vecs, candidates, k=min(top_k, len(candidates)))

    return candidates[:top_k]


# ── Context builder ────────────────────────────────────────────────────────

def build_context(docs: Sequence[Dict[str, str]]) -> str:
    """
    Format retrieved docs into a numbered context block for the LLM.

    Accuracy improvements:
    - Explicit [N] numbering so the LLM can cite sources
    - Deduplication of near-identical chunks
    - Source label kept compact
    """
    if not docs:
        return ""

    blocks: List[str] = []
    seen_texts: set = set()

    for i, doc in enumerate(docs, start=1):
        text = (doc.get("text") or "").strip()
        # Deduplicate by first 100 chars fingerprint (handles near-dupes)
        fingerprint = text[:100].lower()
        if not text or fingerprint in seen_texts:
            continue
        seen_texts.add(fingerprint)
        source = doc.get("source", "unknown")
        # Compact source display
        if len(source) > 60:
            source = "…" + source[-58:]
        blocks.append(f"[{i}] Source: {source}\n{text}")

    return "\n\n---\n\n".join(blocks)


def index_stats() -> Dict[str, int]:
    index = load_index()
    return {"vectors": int(index.ntotal), "chunks": len(_DOCS)}


# ── Ollama client (non-streaming, used by rag.call_llm) ───────────────────

def _fetch_ollama_models() -> List[str]:
    try:
        resp = requests.get(OLLAMA_TAGS_URL, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", []) if m.get("name")]
    except requests.RequestException:
        return []


def _available_models() -> List[str]:
    installed = _fetch_ollama_models()
    return installed if installed else FALLBACK_MODELS


def _build_messages(
    prompt: str,
    history: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for msg in history:
            role = msg.get("role")
            content = msg.get("content", "").strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": prompt})
    return messages


def call_llm(
    prompt: str,
    history: Optional[List[Dict[str, str]]] = None,
    model: Optional[str] = None,
) -> str:
    candidates: List[str] = []
    if model: candidates.append(model)
    if DEFAULT_MODEL not in candidates: candidates.append(DEFAULT_MODEL)
    for m in _available_models():
        if m not in candidates: candidates.append(m)

    messages = _build_messages(prompt, history=history)
    last_error: Optional[str] = None

    for candidate in candidates:
        body = {"model": candidate, "messages": messages, "stream": False}
        try:
            resp = requests.post(OLLAMA_URL, json=body, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "").strip()
            if content:
                return content
            last_error = f"Empty response from '{candidate}'."
        except requests.RequestException as exc:
            last_error = str(exc)

    raise RuntimeError(
        "Ollama is unreachable or no installed model responded.\n"
        f"Last error: {last_error}\nMake sure Ollama is running: ollama serve"
    )