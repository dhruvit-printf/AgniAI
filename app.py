"""Flask REST API for AgniAI.

PORT CONFIGURATION:
  Python / Flask listens on port 5000  (this file — app.run port=5000)
  .NET AiCommand API runs on port 7257 (set via DOTNET_API_BASE_URL in .env)
  These MUST be different ports. Never point DOTNET_API_BASE_URL at 5000.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from queue import Empty, Queue

import requests as _requests
from flask import Flask, Response, g, jsonify, request, stream_with_context
from flask_cors import CORS
from flask_socketio import SocketIO
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename

try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
except Exception:  # pragma: no cover
    Limiter = None
    get_remote_address = None

# ── Blueprints ─────────────────────────────────────────────────────────────
from admin_routes import _register_rate_limits, admin_bp
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
    API_FIRST_TOKEN_TIMEOUT,
    API_SECRET_KEY,
    CHAT_SYSTEM_PROMPT,
    GENERAL_KNOWLEDGE_FALLBACK_PROMPT,
    INGEST_ALLOWED_ROOT,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_CHARS_DEFAULT,
    MAX_TOKENS_DEFAULT,
    MAX_TOKENS_STYLE,
    MODEL_MAX_CONTEXT_TOKENS,
    OLLAMA_TAGS_URL,
    REFERENCE_FALLBACK,
    SESSION_HEADER,
    STRICT_RAG_PROMPT,
    STRICT_RAG_PROMPT_COMPUTE,
    STYLE_MIN_WORDS,
    TOKEN_SAFETY_BUFFER,
    TOP_K,
    _fuzzy_normalize_query,
    _get_current_date_response,
    _is_date_query,
    classify_intent,
    detect_answer_style,
    estimate_message_tokens,
    style_structure_instruction,
    trim_to_complete_sentence,
)
from ingest import (
    clear_index,
    ingest_doc,
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
    deterministic_policy_answer,
    get_cached_response,
    index_stats,
    is_ready,
    is_reasoning_query,
    make_response_cache_key,
    prepare_rag_bundle,
    set_cached_response,
    warmup_runtime,
)

# ── Sentry (CRITICAL FIX: wire in at import time) ─────────────────────────
from sentry_integration import init_sentry
from swagger_ui import swagger_bp

# ── WebSocket ──────────────────────────────────────────────────────────────
from websocket_manager import ws_manager
from websocket_routes import register_socketio_events

init_sentry()

# ── Startup guard (CRITICAL FIX: fail fast if critical env vars missing) ──
from settings import get_dotnet_config, validate_critical_env

validate_critical_env(dict(os.environ))

logger = logging.getLogger(__name__)

# ── App creation (must come before any blueprint registration) ─────────────
app = Flask(__name__)
app.config["JSON_AS_ASCII"] = False
if hasattr(app, "json"):
    app.json.sort_keys = False
else:
    app.config["JSON_SORT_KEYS"] = False
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

# ── CORS ───────────────────────────────────────────────────────────────────
CORS(
    app,
    origins="*",
    allow_headers=[
        "Content-Type",
        "X-Api-Key",
        "X-Session-Id",
        "ngrok-skip-browser-warning",
    ],
    methods=["GET", "POST", "OPTIONS"],
    supports_credentials=False,
)

# ── Register blueprints ────────────────────────────────────────────────────
app.register_blueprint(admin_bp)
app.register_blueprint(swagger_bp)

# ── Socket.IO (WebSocket) ──────────────────────────────────────────────────
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    logger=False,
    engineio_logger=False,
)
ws_manager.init(socketio)
register_socketio_events(socketio)

# ── Shared state ───────────────────────────────────────────────────────────
_memory = ConversationMemory()
_session = _requests.Session()
_active_model = DEFAULT_MODEL
_lock = threading.Lock()
_STREAM_SEMAPHORE = threading.Semaphore(10)
_STREAM_WORKER_ACQUIRE_TIMEOUT = float(os.getenv("STREAM_WORKER_ACQUIRE_TIMEOUT", "5"))

# Rate-limit configuration
RATE_LIMIT_CHAT = os.getenv("RATE_LIMIT_CHAT", "30 per minute")
RATE_LIMIT_INGEST = os.getenv("RATE_LIMIT_INGEST", "10 per minute")
RATE_LIMIT_DEFAULT = os.getenv("RATE_LIMIT_DEFAULT", "60 per minute")

# ── File Upload Helpers ────────────────────────────────────────────────────
ALLOWED_EXTENSIONS = {"pdf", "txt", "docx", "doc"}


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _proxy_aware_remote_address() -> str:
    if request.access_route:
        return request.access_route[0]
    if callable(get_remote_address):
        return get_remote_address()
    return request.remote_addr or "127.0.0.1"


_limiter = None
if Limiter is not None:
    try:
        _limiter = Limiter(
            key_func=_proxy_aware_remote_address,
            default_limits=[RATE_LIMIT_DEFAULT],
            storage_uri="memory://",
            headers_enabled=True,
        )
        _limiter.init_app(app)
    except Exception as exc:  # pragma: no cover
        logger.warning("Rate limiter disabled due to initialization failure: %s", exc)
        _limiter = None
else:  # pragma: no cover
    logger.warning("Flask-Limiter is not installed; rate limiting is disabled.")


def _limit_route(limit_value: str):
    def _decorator(fn):
        return fn

    if _limiter is None:
        return _decorator
    return _limiter.limit(limit_value, override_defaults=True)


# ── Wire admin rate limits AFTER limiter is initialised ───────────────────
_register_rate_limits(app, _limiter)


# =============================================================================
# HELPERS
# =============================================================================


def _validate_answer_length(answer: str, style: str) -> bool:
    min_words = STYLE_MIN_WORDS.get((style or "").strip().lower(), 0)
    if min_words == 0:
        return True
    return len((answer or "").split()) >= min_words


def _log_answer_quality(answer: str, style: str) -> None:
    word_count = len((answer or "").split())
    min_words = STYLE_MIN_WORDS.get((style or "").strip().lower(), 0)
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
                return _json_error("Unauthorized. Provide X-Api-Key header.", 401)
        return fn(*args, **kwargs)

    return wrapper


def _json_error(message: str, status_code: int):
    payload, code = err(message, status_code)
    response = jsonify(payload)
    response.status_code = code
    return response


@app.errorhandler(429)
def _handle_rate_limit(_exc):
    return _json_error("Rate limit exceeded. Please retry later.", 429)


@app.errorhandler(404)
def _handle_not_found(_exc):
    return _json_error("Route not found.", 404)


@app.errorhandler(405)
def _handle_method_not_allowed(_exc):
    return _json_error("Method not allowed for this route.", 405)


def _client_accepts_sse() -> bool:
    accept = request.headers.get("Accept", "")
    return "text/event-stream" in accept.lower()


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
    query_lower = query.lower().strip()
    style_key = (style or "").strip().lower()
    if style_key == "short":
        fallback_style = "Keep the answer concise, usually 1 short paragraph."
    elif style_key == "detail":
        fallback_style = (
            "Give a clear, well-organized answer without inventing specifics."
        )
    else:
        fallback_style = "Give a clear answer in 1 to 3 short paragraphs."

    _FACTUAL_SIGNALS = (
        "what is",
        "what are",
        "who is",
        "who was",
        "when did",
        "when was",
        "where is",
        "where was",
        "how does",
        "how do",
        "why does",
        "why do",
        "explain",
        "define",
        "difference between",
        "full form",
        "meaning of",
        "capital of",
        "president of",
        "prime minister",
        "how many",
        "how much",
        "which country",
        "which state",
        "formula",
        "equation",
        "calculate",
        "theorem",
        "law of",
        "principle of",
        "history of",
        "founder of",
        "invented by",
        "discovered by",
        "what happens",
        "why is",
        "how is",
    )
    is_factual = any(signal in query_lower for signal in _FACTUAL_SIGNALS)

    if is_factual:
        system_content = (
            f"{CHAT_SYSTEM_PROMPT}\n\n"
            f"{GENERAL_KNOWLEDGE_FALLBACK_PROMPT}\n\n"
            "The user is asking a factual or subject-based question. If the knowledge base "
            "does not contain the answer, use only safe general knowledge. Keep the answer "
            "clear, realistic, and conservative. Do not guess. Do not mix reference facts "
            "with your own knowledge. If exact specifics are uncertain, say clearly that you "
            "do not have that information.\n\n" + fallback_style
        )
    else:
        system_content = (
            f"{CHAT_SYSTEM_PROMPT}\n\n"
            "The user is talking to you naturally — sharing something personal, "
            "asking for advice, or just having a conversation. "
            "Respond like a warm, caring human — not like a document reader. "
            "Be genuine, encouraging, and natural. Keep it conversational. "
            "Do not use bullet points or structured formatting. "
            "Do not say 'Answer not found in the document.' Just talk to them.\n\n"
            + fallback_style
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


def _start_stream_worker(target) -> None:
    acquired = _STREAM_SEMAPHORE.acquire(timeout=_STREAM_WORKER_ACQUIRE_TIMEOUT)
    if not acquired:
        raise RuntimeError(
            "Too many concurrent streaming requests. Please retry shortly."
        )

    def _wrapped_target() -> None:
        try:
            target()
        finally:
            _STREAM_SEMAPHORE.release()

    try:
        worker = threading.Thread(target=_wrapped_target, daemon=True)
        worker.start()
    except Exception:
        _STREAM_SEMAPHORE.release()
        raise


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
        return (
            jsonify(
                ok_health(
                    vectors=stats_data["vectors"],
                    chunks=stats_data["chunks"],
                    model=_active_model,
                    status="ollama_unreachable",
                )
            ),
            503,
        )

    return jsonify(
        ok_health(
            vectors=stats_data["vectors"],
            chunks=stats_data["chunks"],
            model=_active_model,
            status="ok",
        )
    )


@app.route("/api/ready")
def ready():
    if not is_ready():
        return jsonify({"success": False, "status": "warming_up"}), 503
    return jsonify({"success": True, "status": "ready"})


@app.route("/metrics")
@app.route("/api/metrics")
def prometheus_metrics():
    from metrics import metrics_collector

    return (
        metrics_collector.generate_prometheus_text(),
        200,
        {"Content-Type": "text/plain; version=0.0.4"},
    )


@app.route("/api/chat", methods=["POST"])
@app.route("/api/chat/", methods=["POST"])
@app.route("/chat", methods=["POST"])
@app.route("/chat/", methods=["POST"])
@_limit_route(RATE_LIMIT_CHAT)
def chat():
    global _active_model

    data = request.get_json(force=True, silent=True) or {}
    message = (data.get("message") or "").strip()
    message = _fuzzy_normalize_query(message)
    model = (data.get("model") or "").strip()
    stream_value = data.get("stream")
    requested_stream = (
        str(stream_value).lower() in {"1", "true", "yes", "on"}
        if stream_value is not None
        else False
    )
    stream = requested_stream and _client_accepts_sse()
    if requested_stream and not stream:
        logger.warning(
            "Ignoring stream=true for JSON client. Send Accept: text/event-stream to use SSE."
        )
    session_id = _get_session_id(data)

    if not message:
        return _json_error("message field is required and cannot be empty.", 400)

    with _lock:
        if model:
            _active_model = model
            from ollama_cpu_chat import _ACTIVE_MODEL_REF

            _ACTIVE_MODEL_REF[0] = model
        current_model = _active_model

    # ── Hardcoded date/time response ───────────────────────────────────────
    if _is_date_query(message):
        answer = _get_current_date_response()
        _memory.add("user", message, session_id=session_id)
        _memory.add("assistant", answer, session_id=session_id)
        if stream:

            def _gen():
                yield answer

            return _stream_answer_response(
                answer_generator=_gen,
                status_payload={
                    "success": True,
                    "style": "elaborate",
                    "session_id": session_id,
                    "cached": False,
                    "grounded": False,
                    "confidence": 0.0,
                    "mode": "hardcoded_date",
                },
            )
        return jsonify(ok_chat(answer=answer, style="elaborate", session_id=session_id))

    style_name, _ = detect_answer_style(message)
    intent = classify_intent(message)
    use_rag = intent == "rag"
    history = _memory.history(session_id)
    reasoning = is_reasoning_query(message) if use_rag else False

    # ── Reject / out-of-domain ─────────────────────────────────────────────
    if intent == "reject":
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
            return jsonify(
                ok_chat(answer=cached_answer, style=style_name, session_id=session_id)
            )

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

            try:
                _start_stream_worker(_gen_worker)
            except RuntimeError as exc:
                return _json_error(str(exc), 429)

            def _gen_generator(
                _message=message,
                _session_id=session_id,
                _response_key=response_key,
            ):
                pieces: list[str] = []
                while True:
                    try:
                        token = token_queue.get(timeout=API_FIRST_TOKEN_TIMEOUT)
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
                _memory.add("user", _message, session_id=_session_id)
                _memory.add("assistant", answer, session_id=_session_id)
                set_cached_response(_response_key, answer)

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
        context = "chat"
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
    deterministic_answer = (
        deterministic_policy_answer(message, context) if use_rag else None
    )
    if deterministic_answer:
        _memory.add("user", message, session_id=session_id)
        _memory.add("assistant", deterministic_answer, session_id=session_id)
        set_cached_response(response_key, deterministic_answer)
        if stream:
            return _stream_answer_response(
                answer_generator=lambda: iter([deterministic_answer]),
                status_payload={
                    "success": True,
                    "style": style_name,
                    "session_id": session_id,
                    "cached": False,
                    "grounded": True,
                    "confidence": confidence,
                    "mode": "deterministic_salary",
                },
            )
        return jsonify(
            ok_chat(
                answer=deterministic_answer,
                style=style_name,
                session_id=session_id,
            )
        )

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

            try:
                _start_stream_worker(_rag_worker)
            except RuntimeError as exc:
                return _json_error(str(exc), 429)

            def _rag_generator(
                _message=message,
                _session_id=session_id,
                _response_key=response_key,
                _style_name=style_name,
            ):
                pieces: list[str] = []
                while True:
                    try:
                        token = token_queue2.get(timeout=API_FIRST_TOKEN_TIMEOUT)
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
                _memory.add("user", _message, session_id=_session_id)
                _memory.add("assistant", answer, session_id=_session_id)
                _log_answer_quality(answer, _style_name)
                set_cached_response(_response_key, answer)

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
            return _json_error(f"LLM service unavailable: {exc}", 503)

        answer = _finalize_answer(answer) or REFERENCE_FALLBACK
        _memory.add("user", message, session_id=session_id)
        _memory.add("assistant", answer, session_id=session_id)
        _log_answer_quality(answer, style_name)
        set_cached_response(response_key, answer)
        return jsonify(ok_chat(answer=answer, style=style_name, session_id=session_id))

    # ── Non-RAG (chat / greeting) path ────────────────────────────────────
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

        try:
            _start_stream_worker(_chat_worker)
        except RuntimeError as exc:
            return _json_error(str(exc), 429)

        def _chat_generator(
            _message=message,
            _session_id=session_id,
            _response_key=response_key,
            _style_name=style_name,
        ):
            pieces: list[str] = []
            while True:
                try:
                    token = _token_queue.get(timeout=API_FIRST_TOKEN_TIMEOUT)
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
            _memory.add("user", _message, session_id=_session_id)
            _memory.add("assistant", answer, session_id=_session_id)
            _log_answer_quality(answer, _style_name)
            set_cached_response(_response_key, answer)

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
        return _json_error(f"LLM service unavailable: {exc}", 503)

    answer = _finalize_answer(answer) or REFERENCE_FALLBACK
    _memory.add("user", message, session_id=session_id)
    _memory.add("assistant", answer, session_id=session_id)
    _log_answer_quality(answer, style_name)
    set_cached_response(response_key, answer)
    return jsonify(ok_chat(answer=answer, style=style_name, session_id=session_id))


@app.route("/api/ingest", methods=["POST"])
# Security Fix: The old code lacked authentication, allowing any client to trigger ingestion.
@_require_secret
@_limit_route(RATE_LIMIT_INGEST)
def ingest():
    data = request.get_json(force=True, silent=True) or {}
    kind = (data.get("kind") or "").strip().lower()
    target = (data.get("target") or "").strip()

    if not kind:
        return _json_error("kind field is required (pdf|url|txt|text|docx).", 400)
    if not target:
        return _json_error("target field is required (file path or URL).", 400)

    fn_map = {
        "pdf": ingest_pdf,
        "url": ingest_url,
        "txt": ingest_txt,
        "text": ingest_text,
        "docx": ingest_docx,
        "doc": ingest_doc,
    }
    if kind not in fn_map:
        return _json_error(
            f"Unknown kind '{kind}'. Valid values: pdf, url, txt, text, doc, docx.",
            400,
        )

    try:
        count = fn_map[kind](target)
    except FileNotFoundError as exc:
        return _json_error(f"File not found: {exc}", 404)
    except Exception as exc:
        return _json_error(f"Ingestion failed: {exc}", 500)

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


@app.route("/api/upload", methods=["POST"])
# Security Fix: The old code lacked authentication, allowing any client to upload files.
@_require_secret
@_limit_route(RATE_LIMIT_INGEST)
def upload_file():
    if "file" not in request.files:
        return _json_error(
            "No file part in request. Send multipart/form-data with field name 'file'.",
            400,
        )

    uploaded = request.files["file"]

    if not uploaded.filename:
        return _json_error(
            "No filename provided. The 'file' field appears to be empty.", 400
        )

    filename = secure_filename(uploaded.filename)
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if not _allowed_file(filename):
        return _json_error(
            f"Unsupported file type '.{ext}'. Allowed types: pdf, txt, docx.",
            415,
        )

    kind_override = (request.form.get("kind") or "").strip().lower()
    kind = kind_override if kind_override in {"pdf", "txt", "docx"} else ext

    fn_map = {
        "pdf": ingest_pdf,
        "txt": ingest_txt,
        "docx": ingest_docx,
        "doc": ingest_doc,
    }

    if kind not in fn_map:
        return _json_error(
            f"Cannot determine how to process file type '.{ext}'. "
            f"Pass 'kind' as one of: pdf, txt, docx.",
            415,
        )

    tmp_path = None
    try:
        # Security Fix: The old code stored temp files in the OS temp dir, failing the confinement check.
        with tempfile.NamedTemporaryFile(
            suffix=f".{ext}",
            delete=False,
            dir=str(INGEST_ALLOWED_ROOT),
        ) as tmp:
            uploaded.save(tmp.name)
            tmp_path = tmp.name

        logger.info(
            "Ingesting uploaded file: filename=%s kind=%s tmp=%s",
            filename,
            kind,
            tmp_path,
        )

        count = fn_map[kind](tmp_path, original_filename=filename)

    except FileNotFoundError as exc:
        logger.error("Temp file missing during upload ingestion: %s", exc)
        return _json_error(f"File handling error: {exc}", 500)
    except ValueError as exc:
        logger.warning("Content error for uploaded file %s: %s", filename, exc)
        return _json_error(f"Content error: {exc}", 422)
    except RuntimeError as exc:
        logger.error("Runtime error ingesting %s: %s", filename, exc)
        return _json_error(f"Ingestion runtime error: {exc}", 500)
    except Exception as exc:
        logger.exception("Unexpected error ingesting uploaded file %s", filename)
        return _json_error(f"Ingestion failed: {exc}", 500)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError as cleanup_err:
                logger.warning(
                    "Could not delete temp file %s: %s", tmp_path, cleanup_err
                )

    if count == 0:
        return jsonify(
            ok_ingest(
                message=(
                    f"'{filename}' was already ingested or contained no new content. "
                    "No new chunks were added."
                ),
                chunks=0,
                source=filename,
            )
        )

    return jsonify(
        ok_ingest(
            message=f"Successfully ingested '{filename}' — {count} chunk(s) added.",
            chunks=count,
            source=filename,
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
                os.getenv("AGNI_LOG_FILE", "agniai.log"),
                maxBytes=5 * 1024 * 1024,
                backupCount=10,
                encoding="utf-8",
            ),
        ],
    )

    warmup_runtime(async_load=False)

    from ollama_cpu_chat import _start_keepalive_heartbeat

    _start_keepalive_heartbeat(_session, interval_seconds=300)

    stats_data = index_stats()

    logger.info("AgniAI REST API")
    logger.info("=" * 50)
    logger.info("Python / Flask listens on  http://0.0.0.0:5000")
    logger.info(
        ".NET AiCommand runs on     %s  (set DOTNET_API_BASE_URL in .env)",
        get_dotnet_config().BASE_URL,
    )
    logger.info("=" * 50)
    logger.info("Health check  http://localhost:5000/api/health")
    logger.info("Chat endpoint http://localhost:5000/api/chat  [POST]")
    logger.info("Upload        http://localhost:5000/api/upload  [POST multipart]")
    logger.info("Admin chat    http://localhost:5000/api/admin/chat  [POST]")
    logger.info("Admin WS      ws://localhost:5000/socket.io  [WebSocket]")
    logger.info("Swagger UI    http://localhost:5000/docs")
    logger.info("CORS          origins=* (all frontends allowed)")
    if API_SECRET_KEY:
        logger.info("Auth  X-Api-Key header required for /api/reset_index")
    if stats_data["vectors"] == 0:
        logger.warning("Knowledge base is empty.")
    else:
        logger.info("Knowledge base ready: %s vectors.", stats_data["vectors"])

    socketio.run(app, host="0.0.0.0", port=5000, debug=False)
