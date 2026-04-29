"""Flask REST API for AgniAI."""

from __future__ import annotations

import json
import logging
import threading
import time
from hashlib import sha1
from queue import Empty, Queue

import requests as _requests
from flask import Flask, Response, g, jsonify, request, stream_with_context
from flask_cors import CORS

from api_models import (
    err,
    ok_chat,
    ok_health,
    ok_ingest,
    ok_message,
    ok_sources,
    ok_stats,
)
from config import (
    ALLOWED_ORIGINS,
    API_SECRET_KEY,
    FIRST_TOKEN_TIMEOUT,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_CHARS_DEFAULT,
    MAX_TOKENS_STYLE,
    MAX_TOKENS_DEFAULT,
    MODEL_MAX_CONTEXT_TOKENS,
    MIN_RETRIEVAL_CONFIDENCE,
    OLLAMA_TAGS_URL,
    REFERENCE_FALLBACK,
    SESSION_HEADER,
    TOKEN_SAFETY_BUFFER,
    STYLE_DETAIL_KEYWORDS,
    STYLE_ELABORATE_KEYWORDS,
    STYLE_SHORT_KEYWORDS,
    TOP_K,
    classify_intent,
    estimate_message_tokens,
    style_structure_instruction,
    trim_to_complete_sentence,
)
from ingest import (
    clear_index,
    ingest_docx,
    ingest_pdf,
    ingest_text,
    ingest_txt,
    ingest_url,
    list_sources,
)
from memory import ConversationMemory
from ollama_cpu_chat import MODEL_NAME as DEFAULT_MODEL
from ollama_cpu_chat import PartialResponseError, chat_with_fallback
from rag import (
    build_strict_messages,
    get_cached_response,
    index_stats,
    make_response_cache_key,
    decide_answer_mode,
    generate_structured_answer,
    is_reasoning_query,
    prepare_rag_bundle,
    warmup_runtime,
    set_cached_response,
)

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
CORS(app, origins=ALLOWED_ORIGINS)

_memory = ConversationMemory()
_session = _requests.Session()
_active_model = DEFAULT_MODEL
_lock = threading.Lock()

_STYLE_MIN_WORDS = {
    "short": 200,
    "elaborate": 400,
    "detail": 680,
}


def _validate_answer_length(answer: str, style: str) -> bool:
    min_words = _STYLE_MIN_WORDS.get((style or "").strip().lower(), 0)
    if min_words == 0:
        return True
    return len((answer or "").split()) >= min_words


def _log_answer_quality(answer: str, style: str) -> None:
    word_count = len((answer or "").split())
    min_words = _STYLE_MIN_WORDS.get((style or "").strip().lower(), 0)
    if not _validate_answer_length(answer, style):
        logger.warning(
            "Answer below minimum length: style=%s words=%d min=%d",
            style,
            word_count,
            min_words,
        )
    else:
        logger.debug("Answer length OK: style=%s words=%d", style, word_count)


@app.before_request
def _start_timer() -> None:
    g.request_start = time.time()


@app.after_request
def _log_request(response):
    elapsed_ms = (time.time() - getattr(g, "request_start", time.time())) * 1000.0
    response.headers["X-Request-Duration-Ms"] = f"{elapsed_ms:.1f}"
    logger.info(
        "%s %s -> %s in %.1fms",
        request.method,
        request.path,
        response.status_code,
        elapsed_ms,
    )
    return response


def _require_secret(fn):
    """Decorator: reject request if API_SECRET_KEY is set and header doesn't match."""
    from functools import wraps

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if API_SECRET_KEY:
            provided = request.headers.get("X-Api-Key", "")
            if provided != API_SECRET_KEY:
                return jsonify(*err("Unauthorized. Provide X-Api-Key header.", 401))
        return fn(*args, **kwargs)

    return wrapper


def _kw_match(query_lower: str, keywords: list) -> bool:
    for kw in keywords:
        if " " in kw:
            if kw in query_lower:
                return True
        elif f" {kw} " in f" {query_lower} ":
            return True
    return False


def detect_answer_style(query: str) -> tuple[str, str]:
    q = query.lower()
    if _kw_match(q, STYLE_SHORT_KEYWORDS):
        return "short", "short"
    if _kw_match(q, STYLE_DETAIL_KEYWORDS):
        return "detail", "detail"
    if _kw_match(q, STYLE_ELABORATE_KEYWORDS):
        return "elaborate", "elaborate"
    return "elaborate", "elaborate"


def _get_session_id(data: dict) -> str:
    session_id = (
        data.get("session_id") or request.headers.get(SESSION_HEADER) or ""
    ).strip()
    return session_id or "default"


def _get_context_limit(style: str) -> int:
    if isinstance(MAX_CONTEXT_CHARS, dict):
        return MAX_CONTEXT_CHARS.get(style, MAX_CONTEXT_CHARS_DEFAULT)
    return int(MAX_CONTEXT_CHARS)


def _get_token_limit(style: str) -> int:
    return MAX_TOKENS_STYLE.get(style, MAX_TOKENS_DEFAULT)


def _build_budget_probe_messages(
    *,
    query: str,
    style: str,
    history: list[dict] | None,
    reasoning: bool,
    use_rag: bool,
) -> list[dict]:
    if use_rag:
        return build_strict_messages(
            query,
            context="",
            style=style,
            reasoning=reasoning,
            history=history,
        )
    messages = [
        {
            "role": "system",
            "content": (
                "You are AgniAI, a helpful assistant for India's Agniveer Training scheme. "
                "Respond naturally and concisely."
            ),
        }
    ]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": query})
    return messages


def _compute_context_char_budget(
    *,
    query: str,
    style: str,
    history: list[dict] | None,
    reasoning: bool,
    use_rag: bool,
) -> tuple[int, int]:
    style_budget = _get_token_limit(style)
    probe_messages = _build_budget_probe_messages(
        query=query,
        style=style,
        history=history,
        reasoning=reasoning,
        use_rag=use_rag,
    )
    prompt_tokens = estimate_message_tokens(probe_messages)
    available_after_prompt = (
        MODEL_MAX_CONTEXT_TOKENS - prompt_tokens - TOKEN_SAFETY_BUFFER
    )
    completion_budget = (
        max(1, min(style_budget, available_after_prompt))
        if available_after_prompt > 0
        else 1
    )
    context_tokens = max(
        0,
        MODEL_MAX_CONTEXT_TOKENS
        - prompt_tokens
        - completion_budget
        - TOKEN_SAFETY_BUFFER,
    )
    return completion_budget, context_tokens * 4


def _finalize_answer(answer: str) -> str:
    final = trim_to_complete_sentence(answer)
    return final or REFERENCE_FALLBACK


def _build_messages(
    *,
    query: str,
    style: str,
    context: str,
    reasoning: bool,
    history: list[dict] | None,
    use_rag: bool,
) -> list[dict]:
    if use_rag:
        return build_strict_messages(
            query,
            context=context,
            style=style,
            reasoning=reasoning,
            history=history,
        )
    messages = [
        {
            "role": "system",
            "content": (
                "You are AgniAI, a helpful assistant for India's Agniveer recruitment scheme. "
                "Respond naturally and concisely."
                f"\n\n{style_structure_instruction(style)}"
            ),
        }
    ]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": query})
    return messages


def _generate_structured_rag_answer(
    *,
    query: str,
    style: str,
    docs: list[dict],
    context: str | None,
    model: str,
    session,
    reasoning: bool,
    history: list[dict] | None,
) -> dict:
    return generate_structured_answer(
        query,
        docs=docs,
        context=context,
        style=style,
        model=model,
        session=session,
        reasoning=reasoning,
        history=history,
    )


def _answer_via_llm(
    *,
    messages: list[dict],
    model: str,
    token_limit: int,
    stream: bool,
    on_token=None,
) -> str:
    result = chat_with_fallback(
        _session,
        model,
        messages,
        stream_tokens=stream,
        on_token=on_token,
        max_tokens_override=token_limit,
    )
    return result.text


def _stream_answer_response(answer_generator, status_payload: dict) -> Response:
    def generate():
        yield f"event: meta\ndata: {json.dumps(status_payload, ensure_ascii=False)}\n\n"
        try:
            for token in answer_generator():
                if isinstance(token, str) and token.startswith("event:"):
                    yield token
                else:
                    yield (
                        f"event: token\ndata: "
                        f"{json.dumps({'token': token}, ensure_ascii=False)}\n\n"
                    )
        except Exception as exc:
            yield (
                f"event: error\ndata: "
                f"{json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"
            )
        yield "event: done\ndata: {}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


# =============================================================================
# ROUTES
# =============================================================================


@app.route("/api/health")
def health():
    stats_data = index_stats()
    ollama_ok = True
    try:
        _session.get(OLLAMA_TAGS_URL, timeout=3)
    except Exception:
        ollama_ok = False

    if not ollama_ok:
        return jsonify(
            ok_health(
                vectors=stats_data["vectors"],
                chunks=stats_data["chunks"],
                model=_active_model,
                status="ollama_unreachable",
            )
        ), 503

    return jsonify(
        ok_health(
            vectors=stats_data["vectors"],
            chunks=stats_data["chunks"],
            model=_active_model,
            status="ok",
        )
    )


@app.route("/api/chat", methods=["POST"])
def chat():
    global _active_model

    data = request.get_json(force=True, silent=True) or {}
    message = (data.get("message") or "").strip()
    model = (data.get("model") or "").strip()
    stream_value = data.get("stream")
    stream = (
        str(stream_value).lower() in {"1", "true", "yes", "on"}
        if stream_value is not None
        else False
    )
    session_id = _get_session_id(data)

    if not message:
        return jsonify(*err("message field is required and cannot be empty.", 400))

    with _lock:
        if model:
            _active_model = model
        current_model = _active_model

    style_name, _ = detect_answer_style(message)
    # Use shared classifier from config.py — single source of truth
    intent = classify_intent(message)
    use_rag = intent == "rag"
    history = _memory.history(session_id)
    reasoning = is_reasoning_query(message) if use_rag else False

    # ── Reject / out-of-domain ─────────────────────────────────────────────
    if intent == "reject":
        response_key = make_response_cache_key(
            message,
            style=style_name,
            model=_active_model,
            context="",
            session_id=session_id,
        )
        answer = REFERENCE_FALLBACK
        _memory.add("user", message, session_id=session_id)
        _memory.add("assistant", answer, session_id=session_id)
        _log_answer_quality(answer, style_name)
        set_cached_response(response_key, answer)
        if stream:
            return _stream_answer_response(
                answer_generator=lambda: iter([answer]),
                status_payload={
                    "success": True,
                    "style": style_name,
                    "session_id": session_id,
                    "cached": False,
                    "grounded": False,
                    "confidence": 0.0,
                    "mode": "reject",
                },
            )
        return jsonify(ok_chat(answer=answer, style=style_name, session_id=session_id))

    # ── Compute budgets ────────────────────────────────────────────────────
    token_limit, context_char_budget = _compute_context_char_budget(
        query=message,
        style=style_name,
        history=history,
        reasoning=reasoning,
        use_rag=use_rag,
    )

    # ── RAG retrieval ──────────────────────────────────────────────────────
    bundle: dict = {"docs": [], "context": "", "confidence": 0.0}
    if use_rag:
        bundle = prepare_rag_bundle(
            message,
            top_k=TOP_K,
            style=style_name,
            max_context_chars=context_char_budget,
            include_points=False,
        )
        context = bundle.get("context", "") if isinstance(bundle, dict) else ""
        confidence = (
            float(bundle.get("confidence", 0.0)) if isinstance(bundle, dict) else 0.0
        )
        mode = bundle.get("mode", "reject") if isinstance(bundle, dict) else "reject"
        reasoning = (
            bool(bundle.get("reasoning", False)) if isinstance(bundle, dict) else False
        )
    else:
        context = ""
        confidence = 0.0
        mode = "chat"

    response_key = make_response_cache_key(
        message,
        style=style_name,
        model=current_model,
        context=context,
        session_id=session_id,
    )

    # ── Cache check ────────────────────────────────────────────────────────
    cached_answer = get_cached_response(response_key)
    if cached_answer is not None:
        _memory.add("user", message, session_id=session_id)
        _memory.add("assistant", cached_answer, session_id=session_id)
        if stream:
            return _stream_answer_response(
                answer_generator=lambda: iter([cached_answer]),
                status_payload={
                    "success": True,
                    "style": style_name,
                    "session_id": session_id,
                    "cached": True,
                    "grounded": bool(use_rag),
                    "confidence": confidence,
                    "mode": mode,
                },
            )
        return jsonify(
            ok_chat(
                answer=cached_answer,
                style=style_name,
                session_id=session_id,
            )
        )

    # ── RAG answer generation ──────────────────────────────────────────────
    if use_rag:
        structured_history = history[-6:] if history else None
        docs = bundle.get("docs", []) if isinstance(bundle, dict) else []

        if stream:
            token_queue: Queue[str | None] = Queue()
            outcome: dict[str, object] = {}
            rag_messages = _build_messages(
                query=message,
                style=style_name,
                context=context,
                reasoning=reasoning,
                history=structured_history,
                use_rag=True,
            )

            def _rag_worker() -> None:
                try:
                    outcome["answer"] = _answer_via_llm(
                        messages=rag_messages,
                        model=current_model,
                        token_limit=token_limit,
                        stream=True,
                        on_token=token_queue.put,
                    )
                except Exception as exc:
                    outcome["error"] = str(exc)
                finally:
                    token_queue.put(None)

            threading.Thread(target=_rag_worker, daemon=True).start()

            def _rag_generator():
                pieces: list[str] = []
                while True:
                    try:
                        token = token_queue.get(timeout=FIRST_TOKEN_TIMEOUT)
                    except Empty:
                        yield (
                            f"event: error\ndata: "
                            f"{json.dumps({'error': 'First token timeout'}, ensure_ascii=False)}\n\n"
                        )
                        return
                    if token is None:
                        break
                    pieces.append(token)
                    yield token

                if "error" in outcome:
                    yield (
                        f"event: error\ndata: "
                        f"{json.dumps({'error': str(outcome['error'])}, ensure_ascii=False)}\n\n"
                    )
                    return

                answer = (
                    "".join(pieces).strip() or str(outcome.get("answer", "")).strip()
                )
                if mode == "strict_answer" and not context.strip() and not docs:
                    answer = REFERENCE_FALLBACK
                answer = _finalize_answer(answer)
                if not pieces and answer:
                    yield answer
                _memory.add("user", message, session_id=session_id)
                _memory.add("assistant", answer, session_id=session_id)
                _log_answer_quality(answer, style_name)
                set_cached_response(response_key, answer)

            return _stream_answer_response(
                answer_generator=_rag_generator,
                status_payload={
                    "success": True,
                    "style": style_name,
                    "session_id": session_id,
                    "cached": False,
                    "grounded": True,
                    "confidence": confidence,
                    "mode": mode,
                },
            )

        # Non-streaming RAG
        structured = _generate_structured_rag_answer(
            query=message,
            style=style_name,
            docs=docs,
            context=context,
            model=current_model,
            session=_session,
            reasoning=reasoning,
            history=structured_history,
        )
        answer = str(structured.get("answer", "")).strip()
        if not answer or (not context.strip() and not docs):
            answer = REFERENCE_FALLBACK
        _memory.add("user", message, session_id=session_id)
        _memory.add("assistant", answer, session_id=session_id)
        _log_answer_quality(answer, style_name)
        set_cached_response(response_key, answer)
        return jsonify(ok_chat(answer=answer, style=style_name, session_id=session_id))

    # ── Non-RAG (chat / greeting) path ────────────────────────────────────
    messages = _build_messages(
        query=message,
        style=style_name,
        context=context,
        reasoning=reasoning,
        history=history[-6:] if history else None,
        use_rag=False,
    )

    if stream:
        _token_queue: Queue[str | None] = Queue()
        _outcome: dict[str, object] = {}

        def _chat_worker() -> None:
            try:
                _outcome["answer"] = _answer_via_llm(
                    messages=messages,
                    model=current_model,
                    token_limit=token_limit,
                    stream=True,
                    on_token=_token_queue.put,
                )
            except PartialResponseError as exc:
                _outcome["answer"] = exc.partial_text or REFERENCE_FALLBACK
            except Exception as exc:
                _outcome["error"] = str(exc)
            finally:
                _token_queue.put(None)

        threading.Thread(target=_chat_worker, daemon=True).start()

        def _chat_generator():
            pieces: list[str] = []
            while True:
                try:
                    token = _token_queue.get(timeout=FIRST_TOKEN_TIMEOUT)
                except Empty:
                    yield (
                        f"event: error\ndata: "
                        f"{json.dumps({'error': 'First token timeout'}, ensure_ascii=False)}\n\n"
                    )
                    return
                if token is None:
                    break
                pieces.append(token)
                yield token

            if "error" in _outcome:
                yield (
                    f"event: error\ndata: "
                    f"{json.dumps({'error': str(_outcome['error'])}, ensure_ascii=False)}\n\n"
                )
                return

            answer = "".join(pieces).strip() or str(_outcome.get("answer", "")).strip()
            answer = _finalize_answer(answer)
            if not pieces and answer:
                yield answer
            _memory.add("user", message, session_id=session_id)
            _memory.add("assistant", answer, session_id=session_id)
            _log_answer_quality(answer, style_name)
            set_cached_response(response_key, answer)

        return _stream_answer_response(
            answer_generator=_chat_generator,
            status_payload={
                "success": True,
                "style": style_name,
                "session_id": session_id,
                "cached": False,
                "grounded": False,
                "confidence": confidence,
                "mode": mode,
            },
        )

    try:
        answer = _answer_via_llm(
            messages=messages,
            model=current_model,
            token_limit=token_limit,
            stream=False,
        )
    except PartialResponseError as exc:
        answer = exc.partial_text or "Partial response received. Please try again."
    except RuntimeError as exc:
        return jsonify(*err(f"LLM service unavailable: {exc}", 503))

    answer = _finalize_answer(answer)
    _memory.add("user", message, session_id=session_id)
    _memory.add("assistant", answer, session_id=session_id)
    _log_answer_quality(answer, style_name)
    set_cached_response(response_key, answer)
    return jsonify(ok_chat(answer=answer, style=style_name, session_id=session_id))


@app.route("/api/ingest", methods=["POST"])
def ingest():
    data = request.get_json(force=True, silent=True) or {}
    kind = (data.get("kind") or "").strip().lower()
    target = (data.get("target") or "").strip()

    if not kind:
        return jsonify(*err("kind field is required (pdf|url|txt|text|docx).", 400))
    if not target:
        return jsonify(*err("target field is required (file path or URL).", 400))

    fn_map = {
        "pdf": ingest_pdf,
        "url": ingest_url,
        "txt": ingest_txt,
        "text": ingest_text,
        "docx": ingest_docx,
    }
    if kind not in fn_map:
        return jsonify(
            *err(
                f"Unknown kind '{kind}'. Valid values: pdf, url, txt, text, docx.", 400
            )
        )

    try:
        count = fn_map[kind](target)
    except FileNotFoundError as exc:
        return jsonify(*err(f"File not found: {exc}", 404))
    except Exception as exc:
        return jsonify(*err(f"Ingestion failed: {exc}", 500))

    if count == 0:
        return jsonify(
            ok_ingest(
                message="Source was already ingested. No new chunks added.",
                chunks=0,
                source=target,
            )
        )
    return jsonify(
        ok_ingest(
            message=f"Successfully ingested {count} chunks.",
            chunks=count,
            source=target,
        )
    )


@app.route("/api/sources")
def sources():
    return jsonify(ok_sources(list_sources()))


@app.route("/api/stats")
def stats():
    s = index_stats()
    return jsonify(ok_stats(vectors=s["vectors"], chunks=s["chunks"]))


@app.route("/api/clear_memory", methods=["POST"])
def clear_memory():
    session_id = _get_session_id(request.get_json(force=True, silent=True) or {})
    _memory.clear(session_id if session_id != "default" else None)
    return jsonify(ok_message("Conversation memory cleared."))


@app.route("/api/reset_index", methods=["POST"])
@_require_secret
def reset_index():
    """Destructive: deletes entire knowledge base. Protected by API_SECRET_KEY if set."""
    clear_index()
    return jsonify(ok_message("Knowledge base reset. Re-ingest documents to continue."))


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    warmup_runtime(async_load=False)

    from ollama_cpu_chat import _start_keepalive_heartbeat

    _start_keepalive_heartbeat(_session, interval_seconds=300)

    stats_data = index_stats()

    print("\n  AgniAI REST API")
    print("  Listening on  http://0.0.0.0:5000")
    print("  Health check  http://localhost:5000/api/health")
    print("  Chat endpoint http://localhost:5000/api/chat  [POST]")
    if API_SECRET_KEY:
        print("  Auth          X-Api-Key header required for /api/reset_index")
    if stats_data["vectors"] == 0:
        print("  Warning: Knowledge base is empty.")
        print("  POST /api/ingest to add documents before chatting.\n")
    else:
        print(f"  Knowledge base ready: {stats_data['vectors']} vectors.\n")

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
