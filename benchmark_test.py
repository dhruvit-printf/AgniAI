from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Callable, Optional

import numpy as np
import psutil
import requests

import rag
import ollama_cpu_chat as oc
from config import DEFAULT_MODEL, estimate_message_tokens, estimate_text_tokens


# =============================================================================
# LOGGING
# =============================================================================

LOG = logging.getLogger("benchmark")


def timed_step(label: str) -> Callable[[Callable[..., Any]], Callable[..., tuple[Any, float]]]:
    """
    Optional timing decorator.

    Wraps a function and returns ``(result, duration_seconds)``.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., tuple[Any, float]]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> tuple[Any, float]:
            start = perf_counter()
            result = fn(*args, **kwargs)
            elapsed = perf_counter() - start
            LOG.debug("%s took %.6fs", label, elapsed)
            return result, elapsed

        return wrapper

    return decorator


def fmt_s(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def format_mb(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def format_pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def throughput_label(tokens_per_second: Optional[float]) -> str:
    if tokens_per_second is None:
        return "n/a"
    if tokens_per_second < 5:
        return "slow"
    if tokens_per_second < 20:
        return "moderate"
    return "fast"


class CpuSampler:
    """Sample system CPU usage while a query is generating."""

    def __init__(self, interval_s: float = 0.1) -> None:
        self.interval_s = interval_s
        self.samples: list[float] = []
        self._samples_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = psutil.cpu_percent(interval=self.interval_s)
            with self._samples_lock:
                self.samples.append(sample)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval_s * 3)

    def average(self) -> Optional[float]:
        with self._samples_lock:
            if not self.samples:
                return None
            return float(sum(self.samples) / len(self.samples))


# =============================================================================
# BENCHMARK DATA
# =============================================================================

TEST_CASES = [
    {
        "name": "short_answer",
        "query": "What is the age limit for Agniveer?",
        "style": "short",
    },
    {
        "name": "elaborate_answer",
        "query": "Explain the selection process of Agniveer in elaborate form.",
        "style": "elaborate",
    },
    {
        "name": "detailed_answer",
        "query": "Explain the Agnipath scheme in detailed form including all benefits and conditions.",
        "style": "detail",
    },
    {
        "name": "calculation_based",
        "query": "Calculate total salary of Agniveer over 4 years.",
        "style": "detail",
    },
    {
        "name": "edge_case_missing_answer",
        "query": "What is the retirement pension after 20 years?",
        "style": "short",
    },
]


@dataclass
class StartupMetrics:
    embedding_load_s: float
    embedding_warmup_encode_s: float
    faiss_index_load_s: float
    bm25_load_s: float
    ollama_preload_s: float
    total_startup_s: float


@dataclass
class QueryMetrics:
    name: str
    query: str
    style: str
    pass_type: str
    cold_or_warm: str
    total_pipeline_time_s: float
    query_normalization_s: float
    query_embedding_s: float
    retrieval_cache_lookup_s: float
    retrieval_compute_s: float
    retrieval_total_s: float
    retrieval_s: float
    context_building_s: float
    llm_first_token_latency_s: Optional[float]
    llm_true_first_token_latency_s: Optional[float]
    total_response_generation_s: Optional[float]
    tokens_generated: Optional[int]
    tokens_per_second: Optional[float]
    effective_throughput_label: str
    tokens_source: str
    retrieval_cache_hit: bool
    cpu_percent_before_query: float
    cpu_percent_during_generation: Optional[float]
    memory_usage_mb_before: float
    memory_usage_mb_after: float
    answer_char_count: Optional[int]
    answer_preview: str
    model_used: str
    error: Optional[str] = None


@dataclass
class BenchmarkReport:
    started_at: str
    startup: StartupMetrics
    system_info: dict[str, Any] = field(default_factory=dict)
    cold_run: list[QueryMetrics] = field(default_factory=list)
    warm_run: list[QueryMetrics] = field(default_factory=list)
    cache_effect: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# BENCHMARK CORE
# =============================================================================


class BenchmarkRunner:
    def __init__(self, *, verbose: bool = False, model: Optional[str] = None) -> None:
        self.verbose = verbose
        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_MODEL)
        self.session = requests.Session()
        self.process = psutil.Process()
        psutil.cpu_percent(interval=None)
        self.process.cpu_percent(interval=None)

    def _log(self, message: str, *args: Any) -> None:
        if self.verbose:
            LOG.info(message, *args)

    def _normalize_query(self, query: str) -> tuple[str, float]:
        start = perf_counter()
        normalized = rag._normalize_query_for_retrieval(query)  # local benchmark hook
        normalized = rag.safe_rewrite_query(normalized)
        return normalized, perf_counter() - start

    def _embed_query(self, query: str) -> tuple[np.ndarray, float]:
        start = perf_counter()
        vec = rag.embed_query(query)
        return vec, perf_counter() - start

    def _retrieve_docs(self, normalized_query: str, qvec: np.ndarray, top_k: int) -> tuple[list[dict], float, bool]:
        """
        Measure FAISS + BM25 retrieval only.

        The query normalization and embedding are measured separately, so this
        method focuses on retrieval, fusion, and caching behavior.
        """
        cached = rag.get_cached_retrieval(normalized_query, top_k, normalized=True)
        if cached is not None:
            return [dict(doc) for doc in cached], 0.0, True

        start = perf_counter()
        index, docs_snapshot = rag._index_snapshot()
        if index is None:
            index = rag.load_index()
            index, docs_snapshot = rag._index_snapshot()

        if index is None or index.ntotal == 0:
            docs: list[dict] = []
        else:
            candidate_k = min(max(top_k * 4, 10), 40, index.ntotal)
            scores_dense, ids = index.search(qvec, candidate_k)
            dense_scores = scores_dense[0]
            doc_ids = ids[0]
            dense_map = {
                int(doc_id): float(score)
                for doc_id, score in zip(doc_ids, dense_scores)
                if doc_id >= 0
            }

            if rag.USE_HYBRID and docs_snapshot:
                bm25_all = rag._bm25_scores(normalized_query)
                bm25_top_ids = np.argsort(bm25_all)[::-1][:candidate_k]

                token_count = len(normalized_query.split())
                if token_count <= 3:
                    dense_weight, bm25_weight = 0.25, 0.75
                elif token_count <= 6:
                    dense_weight, bm25_weight = 0.40, 0.60
                else:
                    dense_weight, bm25_weight = rag.DENSE_WEIGHT, rag.BM25_WEIGHT

                candidate_ids: list[int] = []
                seen_ids: set[int] = set()
                for doc_id in list(doc_ids) + [int(x) for x in bm25_top_ids]:
                    if doc_id < 0 or doc_id >= len(docs_snapshot) or doc_id in seen_ids:
                        continue
                    candidate_ids.append(int(doc_id))
                    seen_ids.add(int(doc_id))

                dense_values = np.array(
                    [dense_map.get(doc_id, 0.0) for doc_id in candidate_ids], dtype="float32"
                )
                dense_values = rag._min_max_normalize(dense_values)
                bm25_values = np.array(
                    [float(bm25_all[doc_id]) for doc_id in candidate_ids], dtype="float32"
                )
                query_terms = set(rag._meaningful_tokens(normalized_query))
                query_lower = normalized_query.lower()

                fused: list[tuple[float, int]] = []
                for doc_id, ds, bs in zip(candidate_ids, dense_values, bm25_values):
                    combined = dense_weight * float(ds) + bm25_weight * float(bs)
                    doc_text = docs_snapshot[doc_id].get("text", "")
                    if query_terms:
                        doc_terms = set(rag._meaningful_tokens(doc_text))
                        overlap = len(query_terms & doc_terms) / max(1, len(query_terms))
                        combined += 0.15 * overlap
                    combined += rag._apply_domain_boosts(query_lower, doc_text.lower())
                    if combined >= rag.MIN_SCORE:
                        fused.append((combined, int(doc_id)))

                fused.sort(key=lambda item: item[0], reverse=True)
                docs = []
                for combined, doc_id in fused:
                    doc = dict(docs_snapshot[doc_id])
                    doc["score"] = round(float(combined), 4)
                    docs.append(doc)
            else:
                docs = []
                for doc_id, score in zip(doc_ids, dense_scores):
                    if doc_id < 0 or doc_id >= len(docs_snapshot):
                        continue
                    if float(score) < rag.MIN_SCORE:
                        continue
                    doc = dict(docs_snapshot[doc_id])
                    doc["score"] = round(float(score), 4)
                    docs.append(doc)

        rag.set_cached_retrieval(normalized_query, top_k, docs, normalized=True)
        return docs, perf_counter() - start, False

    def _build_context(self, docs: list[dict], query: str, style: str, *, max_context_chars: Optional[int] = None) -> tuple[str, float]:
        start = perf_counter()
        context_limit = (
            rag.MAX_CONTEXT_CHARS.get(style, rag.MAX_CONTEXT_CHARS_DEFAULT)
            if isinstance(rag.MAX_CONTEXT_CHARS, dict)
            else rag.MAX_CONTEXT_CHARS_DEFAULT
        )
        if max_context_chars is not None:
            context_limit = max(0, min(int(context_limit), int(max_context_chars)))
        confidence = rag.retrieval_confidence(docs, query)
        mode = rag.decide_answer_mode(query=query, docs=docs, confidence=confidence)
        context_min_score = rag.STRICT_MIN_SCORE if mode == "normal_answer" else rag.LOW_RETRIEVAL_CONFIDENCE
        context = rag.build_context(
            docs,
            max_chunks=max(rag.STRICT_TOP_K, min(5, len(docs))),
            min_score=context_min_score,
            max_chars=context_limit,
        )
        return context, perf_counter() - start

    def _build_messages(self, query: str, context: str, style: str) -> list[dict]:
        reasoning = rag.is_reasoning_query(query)
        return rag.build_strict_messages(
            query,
            context=context,
            style=style,
            reasoning=reasoning,
            history=None,
        )

    def _process_memory_mb(self) -> float:
        return float(self.process.memory_info().rss / (1024 * 1024))

    def _stream_ollama_candidate(
        self,
        model_name: str,
        messages: list[dict],
        *,
        max_tokens_override: Optional[int] = None,
    ) -> dict[str, Any]:
        prompt_tokens_estimate = estimate_message_tokens(messages)
        effective_max_tokens = max_tokens_override if max_tokens_override is not None else oc.MAX_TOKENS
        available_tokens = oc.MODEL_MAX_CONTEXT_TOKENS - prompt_tokens_estimate - oc.TOKEN_SAFETY_BUFFER
        if available_tokens > 0:
            effective_max_tokens = min(effective_max_tokens, available_tokens)
        else:
            effective_max_tokens = max(16, min(effective_max_tokens, 64))

        payload = {
            "model": model_name,
            "messages": messages,
            "stream": True,
            "keep_alive": oc.KEEP_ALIVE,
            "options": {
                "temperature": oc.TEMPERATURE,
                "num_ctx": oc.NUM_CTX,
                "num_predict": effective_max_tokens,
                "top_k": oc._SAMPLING_TOP_K,
                "top_p": oc.TOP_P,
                "repeat_penalty": oc.REPEAT_PENALTY,
                "num_thread": oc._default_num_thread(),
            },
        }

        start = perf_counter()
        first_byte_s: Optional[float] = None
        first_token_s: Optional[float] = None
        pieces: list[str] = []
        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None
        done = False
        buffer = b""

        with self.session.post(
            oc.CHAT_ENDPOINT,
            json=payload,
            stream=True,
            timeout=(oc.TIMEOUT_CONNECT, oc.STREAM_TIMEOUT),
        ) as resp:
            if resp.status_code == 404:
                raise RuntimeError(f"Model '{model_name}' not found.")
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"Ollama HTTP {resp.status_code} for '{model_name}': {resp.text[:300]}"
                )

            for chunk in resp.iter_content(chunk_size=512):
                if not chunk:
                    continue
                if first_byte_s is None:
                    first_byte_s = perf_counter() - start
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    event = json.loads(line.decode("utf-8", errors="replace"))
                    if "error" in event:
                        raise RuntimeError(str(event["error"]))
                    if event.get("done"):
                        prompt_tokens = event.get("prompt_eval_count", prompt_tokens)
                        completion_tokens = event.get("eval_count", completion_tokens)
                        done = True
                        break
                    token = event.get("message", {}).get("content", "")
                    if token:
                        if first_token_s is None:
                            first_token_s = perf_counter() - start
                        pieces.append(token)
                if done:
                    break

            tail = buffer.strip()
            if tail:
                event = json.loads(tail.decode("utf-8", errors="replace"))
                if "error" in event:
                    raise RuntimeError(str(event["error"]))
                if event.get("done"):
                    prompt_tokens = event.get("prompt_eval_count", prompt_tokens)
                    completion_tokens = event.get("eval_count", completion_tokens)
                    done = True

            if not done:
                raise RuntimeError(f"Ollama stream for '{model_name}' ended before completion.")

        return {
            "model": model_name,
            "text": "".join(pieces),
            "duration_s": perf_counter() - start,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "true_first_token_s": first_byte_s,
            "callback_first_token_s": first_token_s,
        }

    def _generate_answer(
        self,
        messages: list[dict],
        *,
        max_tokens_override: Optional[int] = None,
    ) -> dict[str, Any]:
        installed = oc._installed_models(self.session)
        candidates: list[str] = []
        for requested in (self.model, self.model, *oc.FALLBACK_MODELS):
            for candidate in oc._candidate_models(requested, installed):
                if candidate not in candidates:
                    candidates.append(candidate)

        if not candidates:
            raise RuntimeError("No Ollama models found.")

        last_error: Optional[str] = None
        for candidate in candidates:
            for attempt in range(1, oc.MAX_RETRIES + 1):
                try:
                    return self._stream_ollama_candidate(
                        candidate,
                        messages,
                        max_tokens_override=max_tokens_override,
                    )
                except Exception as exc:
                    last_error = str(exc)
                    if attempt < oc.MAX_RETRIES:
                        sleep(0.5 * attempt)
                        continue
                    break

        raise RuntimeError(last_error or "Ollama request failed")

    def _warm_embedding_model(self) -> float:
        start = perf_counter()
        model = rag.load_embedding_model()
        model.encode(
            ["Agniveer age eligibility salary benefits recruitment"],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return perf_counter() - start

    def _warm_ollama_model(self) -> float:
        """
        Preload the Ollama model using the same session as the benchmarked
        queries so connection setup is not measured later as query latency.
        """
        start = perf_counter()
        self.session.post(
            rag.OLLAMA_URL,
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "keep_alive": "10m",
                "options": {
                    "num_predict": 1,
                    "temperature": 0.0,
                    "num_ctx": 512,
                },
            },
            timeout=(10, 180),
        )
        return perf_counter() - start

    def run_startup(self) -> StartupMetrics:
        
        startup_total = perf_counter()

        embedding_load_start = perf_counter()
        rag.load_embedding_model()
        embedding_load_s = perf_counter() - embedding_load_start

        embedding_warmup_encode_s = self._warm_embedding_model()

        index_load_start = perf_counter()
        rag.load_index()
        faiss_index_load_s = perf_counter() - index_load_start

        bm25_load_start = perf_counter()
        rag.load_bm25()
        bm25_load_s = perf_counter() - bm25_load_start

        ollama_preload_s = self._warm_ollama_model()

        total_startup_s = perf_counter() - startup_total

        return StartupMetrics(
            embedding_load_s=embedding_load_s,
            embedding_warmup_encode_s=embedding_warmup_encode_s,
            faiss_index_load_s=faiss_index_load_s,
            bm25_load_s=bm25_load_s,
            ollama_preload_s=ollama_preload_s,
            total_startup_s=total_startup_s,
        )

    def run_query(
        self,
        *,
        name: str,
        query: str,
        style: str,
        pass_type: str,
        cold_or_warm: str,
        top_k: int = rag.TOP_K,
    ) -> QueryMetrics:
        pipeline_start = perf_counter()
        cpu_before = float(psutil.cpu_percent(interval=None))
        memory_before = self._process_memory_mb()

        error: Optional[str] = None
        answer_text = ""
        tokens_generated: Optional[int] = None
        tokens_source = "ollama"
        callback_first_token_s: Optional[float] = None
        true_first_token_s: Optional[float] = None
        total_generation_s: Optional[float] = None
        retrieval_cache_hit = False
        retrieval_lookup_s = 0.0
        retrieval_compute_s = 0.0
        retrieval_total_s = 0.0
        context_s = 0.0
        cpu_during_generation: Optional[float] = None
        memory_after = memory_before
        sampler: Optional[CpuSampler] = None

        try:
            normalized_query, normalization_s = self._normalize_query(query)
            qvec, embedding_s = self._embed_query(normalized_query)

            retrieval_lookup_start = perf_counter()
            cached = rag.get_cached_retrieval(normalized_query, top_k, normalized=True)
            retrieval_lookup_s = perf_counter() - retrieval_lookup_start

            if cached is not None:
                docs = [dict(doc) for doc in cached]
                retrieval_cache_hit = True
            else:
                retrieval_compute_start = perf_counter()
                docs, _, retrieval_cache_hit = self._retrieve_docs(normalized_query, qvec, top_k)
                retrieval_compute_s = perf_counter() - retrieval_compute_start
            retrieval_total_s = retrieval_lookup_s + retrieval_compute_s

            context, context_s = self._build_context(docs, query, style)
            messages = self._build_messages(query, context, style)

            sampler = CpuSampler(interval_s=0.1)
            sampler.start()
            try:
                result = self._generate_answer(messages)
            finally:
                sampler.stop()

            total_generation_s = float(result["duration_s"])
            answer_text = result["text"] or ""
            true_first_token_s = result.get("true_first_token_s")
            callback_first_token_s = result.get("callback_first_token_s")

            if result.get("completion_tokens") is not None:
                tokens_generated = int(result["completion_tokens"])
                tokens_source = "ollama"
            elif answer_text:
                tokens_generated = estimate_text_tokens(answer_text)
                tokens_source = "estimated"

            if tokens_generated is not None and total_generation_s and total_generation_s > 0:
                tokens_per_second = float(tokens_generated / total_generation_s)
            else:
                tokens_per_second = None
            throughput = throughput_label(tokens_per_second)

            cpu_during_generation = sampler.average() if sampler else None
            memory_after = self._process_memory_mb()
            total_pipeline_time_s = perf_counter() - pipeline_start

            return QueryMetrics(
                name=name,
                query=query,
                style=style,
                pass_type=pass_type,
                cold_or_warm=cold_or_warm,
                total_pipeline_time_s=total_pipeline_time_s,
                query_normalization_s=normalization_s,
                query_embedding_s=embedding_s,
                retrieval_cache_lookup_s=retrieval_lookup_s,
                retrieval_compute_s=retrieval_compute_s,
                retrieval_total_s=retrieval_total_s,
                retrieval_s=retrieval_total_s,
                context_building_s=context_s,
                llm_first_token_latency_s=callback_first_token_s,
                llm_true_first_token_latency_s=true_first_token_s,
                total_response_generation_s=total_generation_s,
                tokens_generated=tokens_generated,
                tokens_per_second=tokens_per_second,
                effective_throughput_label=throughput,
                tokens_source=tokens_source,
                retrieval_cache_hit=retrieval_cache_hit,
                cpu_percent_before_query=cpu_before,
                cpu_percent_during_generation=cpu_during_generation,
                memory_usage_mb_before=memory_before,
                memory_usage_mb_after=memory_after,
                answer_char_count=len(answer_text) if answer_text else None,
                answer_preview=answer_text[:220].replace("\n", " ").strip(),
                model_used=result.get("model", self.model) or self.model,
            )
        except Exception as exc:
            error = str(exc)
            LOG.exception("Query %s failed", name)
            cpu_during_generation = sampler.average() if sampler else None
            memory_after = self._process_memory_mb()
            total_pipeline_time_s = perf_counter() - pipeline_start
            return QueryMetrics(
                name=name,
                query=query,
                style=style,
                pass_type=pass_type,
                cold_or_warm=cold_or_warm,
                total_pipeline_time_s=total_pipeline_time_s,
                query_normalization_s=0.0,
                query_embedding_s=0.0,
                retrieval_cache_lookup_s=retrieval_lookup_s,
                retrieval_compute_s=retrieval_compute_s,
                retrieval_total_s=retrieval_total_s,
                retrieval_s=retrieval_total_s,
                context_building_s=context_s,
                llm_first_token_latency_s=None,
                llm_true_first_token_latency_s=None,
                total_response_generation_s=None,
                tokens_generated=None,
                tokens_per_second=None,
                effective_throughput_label="n/a",
                tokens_source="n/a",
                retrieval_cache_hit=retrieval_cache_hit,
                cpu_percent_before_query=cpu_before,
                cpu_percent_during_generation=cpu_during_generation,
                memory_usage_mb_before=memory_before,
                memory_usage_mb_after=memory_after,
                answer_char_count=None,
                answer_preview="",
                model_used=self.model,
                error=error,
            )


# =============================================================================
# OUTPUT
# =============================================================================


def print_startup_metrics(startup: StartupMetrics) -> None:
    print("STARTUP METRICS:")
    print(f"- Embedding Load: {fmt_s(startup.embedding_load_s)} sec")
    print(f"- Embedding Warmup Encode: {fmt_s(startup.embedding_warmup_encode_s)} sec")
    print(f"- Index Load: {fmt_s(startup.faiss_index_load_s)} sec")
    print(f"- BM25 Load: {fmt_s(startup.bm25_load_s)} sec")
    print(f"- Ollama Load: {fmt_s(startup.ollama_preload_s)} sec")
    print(f"- Total Startup: {fmt_s(startup.total_startup_s)} sec")
    print()


def collect_system_info(model_name: str) -> dict[str, Any]:
    vm = psutil.virtual_memory()
    return {
        "cpu_cores_logical": os.cpu_count(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "available_ram_mb": round(vm.available / (1024 * 1024), 2),
        "python_version": platform.python_version(),
        "python_build": platform.python_build()[0],
        "model_name": model_name,
    }


def print_system_info(system_info: dict[str, Any]) -> None:
    print("SYSTEM INFO:")
    print(f"- CPU Cores (logical): {system_info.get('cpu_cores_logical')}")
    print(f"- CPU Cores (physical): {system_info.get('cpu_cores_physical')}")
    print(f"- Available RAM: {system_info.get('available_ram_mb')} MB")
    print(f"- Python Version: {system_info.get('python_version')}")
    print(f"- Model Name: {system_info.get('model_name')}")
    print()


def print_query_metrics(metrics: QueryMetrics) -> None:
    print(f"Query: {metrics.query}")
    print(f"- Style: {metrics.style} | Pass: {metrics.cold_or_warm} | Cache Hit: {metrics.retrieval_cache_hit}")
    print(f"- Total Pipeline Time: {fmt_s(metrics.total_pipeline_time_s)} sec")
    print(f"- Query Normalization: {fmt_s(metrics.query_normalization_s)} sec")
    print(f"- Query Embedding: {fmt_s(metrics.query_embedding_s)} sec")
    print(f"- Retrieval Cache Lookup: {fmt_s(metrics.retrieval_cache_lookup_s)} sec")
    print(f"- Retrieval Compute: {fmt_s(metrics.retrieval_compute_s)} sec")
    print(f"- Retrieval Total: {fmt_s(metrics.retrieval_total_s)} sec")
    print(f"- Context Building: {fmt_s(metrics.context_building_s)} sec")
    print(f"- LLM First Token (callback): {fmt_s(metrics.llm_first_token_latency_s)} sec")
    print(f"- LLM True First Token: {fmt_s(metrics.llm_true_first_token_latency_s)} sec")
    print(f"- Total Response Generation: {fmt_s(metrics.total_response_generation_s)} sec")
    if metrics.tokens_generated is not None:
        print(f"- Tokens Generated: {metrics.tokens_generated} ({metrics.tokens_source})")
        print(f"- Tokens / Second: {fmt_s(metrics.tokens_per_second)} [{metrics.effective_throughput_label}]")
    else:
        print("- Tokens Generated: n/a")
        print("- Tokens / Second: n/a")
    print(f"- CPU % Before Query: {fmt_s(metrics.cpu_percent_before_query)}")
    print(f"- CPU % During Generation: {fmt_s(metrics.cpu_percent_during_generation)}")
    print(f"- Memory Before: {format_mb(metrics.memory_usage_mb_before)} MB")
    print(f"- Memory After: {format_mb(metrics.memory_usage_mb_after)} MB")
    if metrics.error:
        print(f"- Error: {metrics.error}")
    print(f"- Answer Preview: {metrics.answer_preview or 'n/a'}")
    print()


def save_results(report: BenchmarkReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"benchmark_results_{timestamp}.json"
    csv_path = output_dir / f"benchmark_results_{timestamp}.csv"

    payload = asdict(report)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for phase_name, phase_rows in (("cold_run", report.cold_run), ("warm_run", report.warm_run)):
        for row in phase_rows:
            rows.append({
                "phase": phase_name,
                **asdict(row),
            })

    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    return json_path, csv_path


def build_cache_effect(report: BenchmarkReport) -> dict[str, Any]:
    if not report.cold_run or not report.warm_run:
        return {}

    paired = list(zip(report.cold_run, report.warm_run))
    cold_generation = [row.total_response_generation_s for row in report.cold_run if row.total_response_generation_s is not None]
    warm_generation = [row.total_response_generation_s for row in report.warm_run if row.total_response_generation_s is not None]
    cold_pipeline = [row.total_pipeline_time_s for row in report.cold_run if row.total_pipeline_time_s is not None]
    warm_pipeline = [row.total_pipeline_time_s for row in report.warm_run if row.total_pipeline_time_s is not None]

    def _avg(values: list[float]) -> Optional[float]:
        if not values:
            return None
        return float(sum(values) / len(values))

    avg_cold_generation_s = _avg(cold_generation)
    avg_warm_generation_s = _avg(warm_generation)
    avg_cold_pipeline_s = _avg(cold_pipeline)
    avg_warm_pipeline_s = _avg(warm_pipeline)

    improvement_percentage = None
    if (
        avg_cold_generation_s is not None
        and avg_warm_generation_s is not None
        and avg_cold_generation_s > 0
    ):
        improvement_percentage = float(
            ((avg_cold_generation_s - avg_warm_generation_s) / avg_cold_generation_s) * 100.0
        )

    per_query_comparison: list[dict[str, Any]] = []
    for cold, warm in paired:
        per_query_comparison.append({
            "name": cold.name,
            "cold_generation_s": cold.total_response_generation_s,
            "warm_generation_s": warm.total_response_generation_s,
            "generation_delta_s": (
                None
                if cold.total_response_generation_s is None or warm.total_response_generation_s is None
                else cold.total_response_generation_s - warm.total_response_generation_s
            ),
            "cold_pipeline_s": cold.total_pipeline_time_s,
            "warm_pipeline_s": warm.total_pipeline_time_s,
            "pipeline_delta_s": (
                None
                if cold.total_pipeline_time_s is None or warm.total_pipeline_time_s is None
                else cold.total_pipeline_time_s - warm.total_pipeline_time_s
            ),
        })

    effect = {
        "avg_cold_generation_s": avg_cold_generation_s,
        "avg_warm_generation_s": avg_warm_generation_s,
        "avg_cold_pipeline_s": avg_cold_pipeline_s,
        "avg_warm_pipeline_s": avg_warm_pipeline_s,
        "improvement_percentage": improvement_percentage,
        "per_query_comparison": per_query_comparison,
    }
    return effect


# =============================================================================
# MAIN
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark AgniAI RAG startup, retrieval, and Ollama generation latency."
    )
    parser.add_argument("--verbose", action="store_true", help="Print detailed benchmark logs.")
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Directory for JSON and CSV benchmark outputs.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional Ollama model override. Defaults to OLLAMA_MODEL or the repo default.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    runner = BenchmarkRunner(verbose=args.verbose, model=args.model)
    system_info = collect_system_info(runner.model)
    print_system_info(system_info)

    startup = runner.run_startup()
    print_startup_metrics(startup)

    report = BenchmarkReport(
        started_at=datetime.now().isoformat(timespec="seconds"),
        startup=startup,
        system_info=system_info,
    )

    print("QUERY METRICS:")
    for case in TEST_CASES:
        result = runner.run_query(
            name=case["name"],
            query=case["query"],
            style=case["style"],
            pass_type="cold",
            cold_or_warm="cold",
        )
        report.cold_run.append(result)
        print_query_metrics(result)

    print("WARM RUN METRICS:")
    for case in TEST_CASES:
        result = runner.run_query(
            name=case["name"],
            query=case["query"],
            style=case["style"],
            pass_type="warm",
            cold_or_warm="warm",
        )
        report.warm_run.append(result)
        print_query_metrics(result)

    report.cache_effect = build_cache_effect(report)

    if report.cache_effect:
        print("CACHE EFFECT:")
        print(f"- Avg Cold Generation: {fmt_s(report.cache_effect['avg_cold_generation_s'])} sec")
        print(f"- Avg Warm Generation: {fmt_s(report.cache_effect['avg_warm_generation_s'])} sec")
        print(f"- Avg Cold Pipeline: {fmt_s(report.cache_effect['avg_cold_pipeline_s'])} sec")
        print(f"- Avg Warm Pipeline: {fmt_s(report.cache_effect['avg_warm_pipeline_s'])} sec")
        print(f"- Improvement: {format_pct(report.cache_effect['improvement_percentage'])}%")
        print("- Per Query:")
        for item in report.cache_effect["per_query_comparison"]:
            print(
                f"  - {item['name']}: "
                f"cold_gen={fmt_s(item['cold_generation_s'])}s, "
                f"warm_gen={fmt_s(item['warm_generation_s'])}s, "
                f"cold_pipeline={fmt_s(item['cold_pipeline_s'])}s, "
                f"warm_pipeline={fmt_s(item['warm_pipeline_s'])}s"
            )
        print()

    json_path, csv_path = save_results(report, Path(args.output_dir))
    print("SAVED RESULTS:")
    print(f"- JSON: {json_path}")
    print(f"- CSV: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
