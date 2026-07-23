"""
Regression test: SentenceTransformer constructor must be called exactly once
under concurrent access.
"""
from __future__ import annotations

import threading
import time

import pytest


def test_concurrent_semantic_catalog_build_loads_model_once():
    import rag
    from intent_engine import semantic_classifier as sc

    if sc._build_thread.is_alive():
        sc._build_thread.join(timeout=10)

    rag._MODEL = None
    sc._catalog_embeddings = None
    sc._catalog_labels = []

    from sentence_transformers import SentenceTransformer

    init_counter = {"count": 0}
    original_init = SentenceTransformer.__init__

    def counting_init(self, *args, **kwargs):
        init_counter["count"] += 1
        return original_init(self, *args, **kwargs)

    SentenceTransformer.__init__ = counting_init
    init_counter["count"] = 0

    _orig_load = rag.load_embedding_model
    start_event = threading.Event()

    def delayed_load():
        start_event.wait(timeout=10)
        time.sleep(0.3)
        return _orig_load()

    rag.load_embedding_model = delayed_load

    try:
        num_threads = 4
        queries = [
            "Who has cancer?",
            "Show top performers",
            "Who is on leave?",
            "Show BMI analysis",
        ]

        errors = []
        results = []
        lock = threading.Lock()

        def worker(query):
            try:
                result = sc.classify_semantic(query)
                with lock:
                    results.append(result)
            except Exception as exc:
                with lock:
                    errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(q,))
            for q in queries[:num_threads]
        ]

        t0 = time.perf_counter()
        for t in threads:
            t.start()
        time.sleep(0.1)
        start_event.set()
        for t in threads:
            t.join(timeout=60)
        elapsed = time.perf_counter() - t0

        assert len(errors) == 0, f"Worker errors: {errors}"
        assert len(results) == num_threads
        assert init_counter["count"] == 1, (
            f"SentenceTransformer constructed {init_counter['count']} times "
            f"under {num_threads} concurrent threads"
        )
    finally:
        SentenceTransformer.__init__ = original_init
        rag.load_embedding_model = _orig_load
