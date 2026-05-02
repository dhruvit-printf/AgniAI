"""Flask REST API for AgniAI.

Key changes vs original:
  - Non-RAG (chat) path now uses CHAT_SYSTEM_PROMPT from config.py so
    greetings, patriotic slogans and aspirant motivation replies match
    what main.py CLI produces.
  - RAG user message now includes the explicit "Using ONLY the reference
    information above …" instruction (matches main.py's _build_rag_messages).
  - Non-streaming RAG now calls chat_with_fallback directly (same path as
    main.py) instead of going through generate_structured_answer.
  - New FALLBACK_GENERAL path: when RAG retrieval finds nothing relevant
    (mode == "strict_answer" with empty context) the bot answers from its
    own general knowledge via CHAT_SYSTEM_PROMPT so the user always gets a
    helpful reply instead of "Answer not found in the document."
  - build_strict_messages in rag.py is monkey-patched locally via a wrapper
    so we don't have to touch rag.py.
"""

from __future__ import annotations

import json
import logging
import threading
import time
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
    CHAT_SYSTEM_PROMPT,
    FIRST_TOKEN_TIMEOUT,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_CHARS_DEFAULT,
    MAX_TOKENS_STYLE,
    MAX_TOKENS_DEFAULT,
    MODEL_MAX_CONTEXT_TOKENS,
    OLLAMA_TAGS_URL,
    REFERENCE_FALLBACK,
    SESSION_HEADER,
    STRICT_RAG_PROMPT,
    STRICT_RAG_PROMPT_COMPUTE,
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
    LOW_RETRIEVAL_CONFIDENCE,
    STRICT_TOP_K,
    build_context,
    build_strict_messages,
    get_cached_response,
    index_stats,
    is_reasoning_query,
    make_response_cache_key,
    prepare_rag_bundle,
    set_cached_response,
    warmup_runtime,
)

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
_cors_origins = [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]

CORS(
    app,
    origins=_cors_origins if _cors_origins != ["*"] else "*",
    allow_headers=["Content-Type", "X-Api-Key", "X-Session-Id", "ngrok-skip-browser-warning"],
    methods=["GET", "POST", "OPTIONS"],
    supports_credentials=True,
)

_memory = ConversationMemory()
_session = _requests.Session()
_active_model = DEFAULT_MODEL
_lock = threading.Lock()

_STYLE_MIN_WORDS = {
    "short": 50,
    "elaborate": 120,
    "detail": 250,
}


# =============================================================================
# HELPERS
# =============================================================================


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
            "content": CHAT_SYSTEM_PROMPT,
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


def _build_rag_messages(
    *,
    query: str,
    style: str,
    context: str,
    reasoning: bool,
    history: list[dict] | None,
) -> list[dict]:
    """
    Build RAG messages — matches main.py's _build_rag_messages exactly,
    including the explicit instruction sentence in the user turn.
    """
    system_prompt = STRICT_RAG_PROMPT_COMPUTE if reasoning else STRICT_RAG_PROMPT
    system_prompt = f"{system_prompt}\n\n{style_structure_instruction(style)}"

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    if history:
        for msg in history[-6:]:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})

    if context.strip():
        user_content = (
            f"Reference information:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Using ONLY the reference information above, write a complete answer. "
            "Do not use any knowledge outside the reference information."
        )
    else:
        user_content = query

    messages.append({"role": "user", "content": user_content})
    return messages


def _build_general_messages(
    *,
    query: str,
    style: str,
    history: list[dict] | None,
) -> list[dict]:
    """
    Builds messages for out-of-KB queries.
    Detects whether the query is factual/subject-based or casual/personal
    and sets the tone accordingly.
    """
    query_lower = query.lower().strip()

    # Detect factual / subject question
    _FACTUAL_SIGNALS = (
        "what is", "what are", "who is", "who was", "when did", "when was",
        "where is", "where was", "how does", "how do", "why does", "why do",
        "explain", "define", "difference between", "full form", "meaning of",
        "capital of", "president of", "prime minister", "how many", "how much",
        "which country", "which state", "formula", "equation", "calculate",
        "theorem", "law of", "principle of", "history of", "founder of",
        "invented by", "discovered by", "what happens", "why is", "how is",
    )
    is_factual = any(signal in query_lower for signal in _FACTUAL_SIGNALS)

    if is_factual:
        system_content = (
            f"{CHAT_SYSTEM_PROMPT}\n\n"
            "The user is asking a factual or subject-based question outside your "
            "Agniveer knowledge base. Answer it like a knowledgeable teacher or mentor — "
            "accurate, clear, and complete. Do not restrict yourself to Agniveer topics. "
            "If the topic connects to Agniveer or Indian Army, mention that naturally at the end. "
            "Do not say 'Answer not found in the document.' Just answer the question properly.\n\n"
            f"{style_structure_instruction(style)}"
        )
    else:
        # Casual / personal / emotional — reply like a human
        system_content = (
            f"{CHAT_SYSTEM_PROMPT}\n\n"
            "The user is talking to you naturally — sharing something personal, "
            "asking for advice, or just having a conversation. "
            "Respond like a warm, caring human — not like a document reader. "
            "Be genuine, encouraging, and natural. Keep it conversational. "
            "Do not use bullet points or structured formatting. "
            "Do not say 'Answer not found in the document.' Just talk to them.\n\n"
            f"{style_structure_instruction(style)}"
        )

    messages: list[dict] = [{"role": "system", "content": system_content}]
    if history:
        for msg in (history or [])[-6:]:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": query})
    return messages


def _build_chat_messages(
    *,
    query: str,
    style: str,
    history: list[dict] | None,
) -> list[dict]:
    """
    Build messages for non-RAG (chat / greeting) path.
    Uses CHAT_SYSTEM_PROMPT — matches main.py exactly now.
    """
    messages: list[dict] = [
        {
            "role": "system",
            "content": CHAT_SYSTEM_PROMPT,
        }
    ]
    if history:
        for msg in (history or [])[-6:]:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": query})
    return messages


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
    intent = classify_intent(message)
    use_rag = intent == "rag"
    history = _memory.history(session_id)
    reasoning = is_reasoning_query(message) if use_rag else False

    # ── Reject / out-of-domain ─────────────────────────────────────────────
    if intent == "reject":
        # Instead of a hard fallback, attempt a general-knowledge answer
        gen_messages = _build_general_messages(
            query=message,
            style=style_name,
            history=history[-6:] if history else None,
        )
        response_key = make_response_cache_key(
            message,
            style=style_name,
            model=current_model,
            context="general",
            session_id=session_id,
        )
        cached_answer = get_cached_response(response_key)
        if cached_answer:
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
                        "grounded": False,
                        "confidence": 0.0,
                        "mode": "general",
                    },
                )
            return jsonify(ok_chat(answer=cached_answer, style=style_name, session_id=session_id))

        token_limit = _get_token_limit(style_name)

        if stream:
            token_queue: Queue[str | None] = Queue()
            outcome: dict[str, object] = {}

            def _gen_worker() -> None:
                try:
                    outcome["answer"] = _answer_via_llm(
                        messages=gen_messages,
                        model=current_model,
                        token_limit=token_limit,
                        stream=True,
                        on_token=token_queue.put,
                    )
                except PartialResponseError as exc:
                    outcome["answer"] = exc.partial_text or REFERENCE_FALLBACK
                except Exception as exc:
                    outcome["error"] = str(exc)
                finally:
                    token_queue.put(None)

            threading.Thread(target=_gen_worker, daemon=True).start()

            def _gen_generator():
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

                answer = (
                    "".join(pieces).strip() or str(outcome.get("answer", "")).strip()
                )
                answer = _finalize_answer(answer) or REFERENCE_FALLBACK
                _memory.add("user", message, session_id=session_id)
                _memory.add("assistant", answer, session_id=session_id)
                set_cached_response(response_key, answer)

            return _stream_answer_response(
                answer_generator=_gen_generator,
                status_payload={
                    "success": True,
                    "style": style_name,
                    "session_id": session_id,
                    "cached": False,
                    "grounded": False,
                    "confidence": 0.0,
                    "mode": "general",
                },
            )

        try:
            answer = _answer_via_llm(
                messages=gen_messages,
                model=current_model,
                token_limit=token_limit,
                stream=False,
            )
        except PartialResponseError as exc:
            answer = exc.partial_text or REFERENCE_FALLBACK
        except RuntimeError:
            answer = REFERENCE_FALLBACK

        answer = _finalize_answer(answer) or REFERENCE_FALLBACK
        _memory.add("user", message, session_id=session_id)
        _memory.add("assistant", answer, session_id=session_id)
        set_cached_response(response_key, answer)
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

        # ── Determine if we should fall through to general knowledge ──────
        # If context is empty (nothing retrieved) go to general knowledge
        # so the user gets a helpful answer instead of the fallback string.
        use_general_fallback = not context.strip()

        if use_general_fallback:
            rag_messages = _build_general_messages(
                query=message,
                style=style_name,
                history=structured_history,
            )
            effective_mode = "general"
        else:
            rag_messages = _build_rag_messages(
                query=message,
                style=style_name,
                context=context,
                reasoning=reasoning,
                history=structured_history,
            )
            effective_mode = mode

        if stream:
            token_queue2: Queue[str | None] = Queue()
            outcome2: dict[str, object] = {}

            def _rag_worker() -> None:
                try:
                    outcome2["answer"] = _answer_via_llm(
                        messages=rag_messages,
                        model=current_model,
                        token_limit=token_limit,
                        stream=True,
                        on_token=token_queue2.put,
                    )
                except PartialResponseError as exc:
                    outcome2["answer"] = exc.partial_text or REFERENCE_FALLBACK
                except Exception as exc:
                    outcome2["error"] = str(exc)
                finally:
                    token_queue2.put(None)

            threading.Thread(target=_rag_worker, daemon=True).start()

            def _rag_generator():
                pieces: list[str] = []
                while True:
                    try:
                        token = token_queue2.get(timeout=FIRST_TOKEN_TIMEOUT)
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

                if "error" in outcome2:
                    yield (
                        f"event: error\ndata: "
                        f"{json.dumps({'error': str(outcome2['error'])}, ensure_ascii=False)}\n\n"
                    )
                    return

                answer = (
                    "".join(pieces).strip() or str(outcome2.get("answer", "")).strip()
                )
                answer = _finalize_answer(answer) or REFERENCE_FALLBACK
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
                    "grounded": not use_general_fallback,
                    "confidence": confidence,
                    "mode": effective_mode,
                },
            )

        # Non-streaming RAG — use chat_with_fallback directly (matches main.py)
        try:
            answer = _answer_via_llm(
                messages=rag_messages,
                model=current_model,
                token_limit=token_limit,
                stream=False,
            )
        except PartialResponseError as exc:
            answer = exc.partial_text or REFERENCE_FALLBACK
        except RuntimeError as exc:
            return jsonify(*err(f"LLM service unavailable: {exc}", 503))

        answer = _finalize_answer(answer) or REFERENCE_FALLBACK
        _memory.add("user", message, session_id=session_id)
        _memory.add("assistant", answer, session_id=session_id)
        _log_answer_quality(answer, style_name)
        set_cached_response(response_key, answer)
        return jsonify(ok_chat(answer=answer, style=style_name, session_id=session_id))

    # ── Non-RAG (chat / greeting) path ────────────────────────────────────
    # Now uses CHAT_SYSTEM_PROMPT — matches main.py CLI output exactly.
    messages = _build_chat_messages(
        query=message,    


        
        style=style_name,
        history=history[-6:] if history else None,
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
            answer = _finalize_answer(answer) or REFERENCE_FALLBACK
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
        answer = exc.partial_text or REFERENCE_FALLBACK
    except RuntimeError as exc:
        return jsonify(*err(f"LLM service unavailable: {exc}", 503))

    answer = _finalize_answer(answer) or REFERENCE_FALLBACK
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
    from logging.handlers import RotatingFileHandler

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            RotatingFileHandler(
                "agniai.log",
                maxBytes=5 * 1024 * 1024,
                backupCount=10,
                encoding="utf-8",
            )
        ],
    )

    warmup_runtime(async_load=False)

    from ollama_cpu_chat import _start_keepalive_heartbeat

    _start_keepalive_heartbeat(_session, interval_seconds=300)

    stats_data = index_stats()

    print("\n  AgniAI REST API")
    print("  Listening on  http://0.0.0.0:7257")
    print("  Health check  http://localhost:7257/api/health")
    print("  Chat endpoint http://localhost:7257/api/chat  [POST]")
    if API_SECRET_KEY:
        print("  Auth  X-Api-Key header required for /api/reset_index")
    if stats_data["vectors"] == 0:
        print("  Warning: Knowledge base is empty.\n")
    else:
        print(f"  Knowledge base ready: {stats_data['vectors']} vectors.\n")

    app.run(host="0.0.0.0", port=7257, debug=False, threaded=True)