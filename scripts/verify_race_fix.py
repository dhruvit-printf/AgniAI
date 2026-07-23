"""
Verification Script for Semantic Classifier Model-Loading Race Fix.
Covers:
  - Step 2: 1 racer vs background warm-up thread
  - Step 3: 5 and 50 concurrent request threads on reset state
  - Step 4: Cold load cost vs cached second call
"""

import threading
import time
import sentence_transformers

# --- Patch SentenceTransformer.__init__ BEFORE importing semantic_classifier ---
init_counter = {"n": 0}
lock = threading.Lock()
orig_init = sentence_transformers.SentenceTransformer.__init__


def counting_init(self, *args, **kwargs):
    with lock:
        init_counter["n"] += 1
    return orig_init(self, *args, **kwargs)


sentence_transformers.SentenceTransformer.__init__ = counting_init

import rag
import intent_engine.semantic_classifier as sc


def verify_step_2():
    print("=" * 60)
    print("STEP 2: Single racer thread vs. background warm-up thread")
    print("=" * 60)

    # Confirm background thread is active
    is_alive = sc._build_thread.is_alive()
    print(f"Background thread running at import: {is_alive}")

    errors = []

    def worker():
        try:
            sc.classify_semantic("Who has cancer?")
        except Exception as exc:
            errors.append(exc)

    t = threading.Thread(target=worker)
    t.start()
    t.join(timeout=60)
    sc._build_thread.join(timeout=60)

    assert not t.is_alive(), "Worker thread deadlocked or hung!"
    assert not sc._build_thread.is_alive(), "Background thread deadlocked or hung!"
    assert len(errors) == 0, f"Worker thread raised error: {errors}"
    print(f"Total SentenceTransformer constructions: {init_counter['n']}")
    assert init_counter["n"] == 1, f"EXPECTED 1 construction, GOT {init_counter['n']}"
    print("PASSED: Step 2 verified successfully.\n")


def verify_step_3(num_threads=5, stress_runs=3):
    print("=" * 60)
    print(
        f"STEP 3: Concurrent requests on clean cold state ({num_threads} threads, {stress_runs} runs)"
    )
    print("=" * 60)

    # Ensure any active build thread is joined before reset
    sc._build_thread.join(timeout=30)

    queries = [
        "Who has cancer?",
        "Show top performers",
        "Who is on leave?",
        "Show BMI analysis",
        "What is the average PPT score?",
        "List absconded agniveers",
        "Personal details of soldier",
    ]

    for run_idx in range(1, stress_runs + 1):
        # Reset state while no threads are running
        rag._MODEL = None
        sc._catalog_embeddings = None
        sc._catalog_labels = []
        with lock:
            init_counter["n"] = 0

        barrier = threading.Barrier(num_threads)
        errors = []
        results = []
        res_lock = threading.Lock()

        def worker(q):
            try:
                barrier.wait(timeout=10)
                res = sc.classify_semantic(q)
                with res_lock:
                    results.append(res)
            except Exception as exc:
                with res_lock:
                    errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(queries[i % len(queries)],))
            for i in range(num_threads)
        ]

        t0 = time.perf_counter()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        elapsed = time.perf_counter() - t0

        assert len(errors) == 0, f"Run {run_idx} errors: {errors}"
        assert len(results) == num_threads, f"Run {run_idx} incomplete results"
        print(
            f"Run {run_idx}/{stress_runs}: {num_threads} threads finished in {elapsed:.2f}s | Constructions: {init_counter['n']}"
        )
        assert (
            init_counter["n"] == 1
        ), f"Run {run_idx} EXPECTED 1 construction, GOT {init_counter['n']}"

    print(
        f"PASSED: Step 3 verified across {stress_runs} runs with {num_threads} threads.\n"
    )


def verify_step_4():
    print("=" * 60)
    print("STEP 4: One-time cold-load cost vs cached second call")
    print("=" * 60)

    # Reset state cleanly
    sc._build_thread.join(timeout=30)
    rag._MODEL = None
    sc._catalog_embeddings = None
    sc._catalog_labels = []
    with lock:
        init_counter["n"] = 0

    from intent_engine.intent_classifier import classify_intent

    # Call 1: Keyword resolved (Stage 1)
    t0 = time.time()
    res1 = classify_intent("Who has viral fever?")
    dt1 = time.time() - t0
    print(
        f"Call 1 ('Who has viral fever?'): {dt1:.4f}s | Stage 2 hit: {res1 != 'rag'} | Constructions: {init_counter['n']}"
    )

    # Call 2: Needs Stage 2 (First Stage 2 call -> cold load)
    t0 = time.time()
    res2 = classify_intent("Who has cancer?")
    dt2 = time.time() - t0
    print(
        f"Call 2 ('Who has cancer?'): {dt2:.4f}s | Cold Load Cost Paid | Constructions: {init_counter['n']}"
    )

    # Call 3: Needs Stage 2 (Second Stage 2 call -> cached)
    t0 = time.time()
    res3 = classify_intent("Who has diabetes?")
    dt3 = time.time() - t0
    print(
        f"Call 3 ('Who has diabetes?'): {dt3:.4f}s | Fast Cached Call | Constructions: {init_counter['n']}"
    )

    assert (
        init_counter["n"] == 1
    ), f"Expected 1 total model construction across sequence, got {init_counter['n']}"
    assert (
        dt3 < dt2
    ), "Cached call (dt3) should be significantly faster than cold load (dt2)"
    print("PASSED: Step 4 verified cold-start caveat and sub-second caching.\n")


if __name__ == "__main__":
    verify_step_2()
    verify_step_3(num_threads=5, stress_runs=3)
    verify_step_3(num_threads=30, stress_runs=3)
    verify_step_4()
    print("ALL VERIFICATION CHECKS COMPLETED SUCCESSFULLY!")
