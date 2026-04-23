"""
debug_rag.py
============
Run this to diagnose why AgniAI says "I don't have that information"
even when the docstore has the answer.

Usage:
    python debug_rag.py
"""

import sys
from pathlib import Path

# Make sure we can import from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from rag import build_context, embed_query, index_stats, load_docstore, load_index, search
from config import MIN_SCORE, TOP_K, MAX_CONTEXT_CHARS

QUERY = "Selection Criteria for agniveer"

print("=" * 60)
print(f"Query: {QUERY!r}")
print("=" * 60)

# 1. Index stats
stats = index_stats()
print(f"\n[1] Index stats: {stats['vectors']} vectors, {stats['chunks']} chunks")

# 2. Raw FAISS search (bypass MIN_SCORE filter to see raw scores)
index = load_index()
docs  = load_docstore()
qvec  = embed_query(QUERY)
k     = min(5, index.ntotal)
scores, ids = index.search(qvec, k)
print(f"\n[2] Raw FAISS top-{k} results (before score filter, MIN_SCORE={MIN_SCORE}):")
for rank, (doc_id, score) in enumerate(zip(ids[0], scores[0]), 1):
    if doc_id < 0 or doc_id >= len(docs):
        print(f"  #{rank}  id={doc_id}  score={score:.4f}  ← INVALID id")
        continue
    src   = docs[doc_id].get("source", "?")[:60]
    snippet = docs[doc_id].get("text", "")[:80].replace("\n", " ")
    kept  = "✔ KEPT" if score >= MIN_SCORE else f"✘ DROPPED (< {MIN_SCORE})"
    print(f"  #{rank}  id={doc_id}  score={score:.4f}  {kept}")
    print(f"       src    : {src}")
    print(f"       snippet: {snippet}…")

# 3. search() result (with filter applied)
results = search(QUERY, top_k=TOP_K)
print(f"\n[3] search() returned {len(results)} chunk(s) after MIN_SCORE filter:")
for r in results:
    print(f"  chunk_id={r['chunk_id']}  score={r['score']}  src={r.get('source','?')[:60]}")
    print(f"  text[:120]: {r.get('text','')[:120].replace(chr(10),' ')}…")

# 4. build_context output
context = build_context(results)
print(f"\n[4] build_context() length: {len(context)} chars  (MAX_CONTEXT_CHARS={MAX_CONTEXT_CHARS})")
if context:
    print(f"\n--- Context sent to LLM (first 600 chars) ---\n{context[:600]}\n---")
else:
    print("  ⚠  EMPTY CONTEXT — this is why the LLM says 'I don't know'")

# 5. Diagnosis
print("\n[5] DIAGNOSIS:")
if stats["vectors"] == 0:
    print("  ✘  FAISS index is empty. Run /ingest to add documents.")
elif not results:
    raw_scores = [float(s) for s in scores[0] if s > 0]
    if raw_scores and max(raw_scores) < MIN_SCORE:
        print(f"  ✘  All scores below MIN_SCORE ({MIN_SCORE}).")
        print(f"     Best score was {max(raw_scores):.4f}.")
        print(f"     FIX: lower MIN_SCORE in config.py to ~0.01")
    else:
        print("  ✘  search() returned no results. Check docstore alignment.")
elif not context:
    print("  ✘  Chunks found but build_context returned empty string.")
    print("     Possible cause: all chunk 'text' fields are empty.")
else:
    print("  ✔  RAG pipeline is working. Context IS being built.")
    print("  ⚠  The LLM is ignoring the context — likely a prompt issue.")
    print("     Check that the user_content in main.py includes the context.")
