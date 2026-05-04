"""CLI chatbot for AgniAI."""

from __future__ import annotations

import logging
import re
import sys
from typing import Optional, Tuple

import requests

from config import (
    CHAT_SYSTEM_PROMPT,
    DATA_DIR,
    detect_answer_style,
    GENERAL_KNOWLEDGE_FALLBACK_PROMPT,
    INDEX_DIR,
    MAX_CONTEXT_CHARS,
    MAX_CONTEXT_CHARS_DEFAULT,
    MAX_TOKENS_DEFAULT,
    MAX_TOKENS_STYLE,
    MODEL_MAX_CONTEXT_TOKENS,
    REFERENCE_FALLBACK,
    STRICT_RAG_PROMPT,
    STRICT_RAG_PROMPT_COMPUTE,
    TOKEN_SAFETY_BUFFER,
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
from ollama_cpu_chat import MODEL_NAME as DEFAULT_MODEL_NAME
from ollama_cpu_chat import PartialResponseError, chat_with_fallback
from rag import (
    build_context,
    build_strict_messages,
    deterministic_policy_answer,
    get_cached_response,
    index_stats,
    make_response_cache_key,
    is_reasoning_query,
    prepare_rag_bundle,
    warmup_runtime,
    set_cached_response,
    STRICT_TOP_K,
    LOW_RETRIEVAL_CONFIDENCE,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# =============================================================================
# COLOUR HELPERS
# =============================================================================


def _c(code: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def dim(t):
    return _c("2", t)


def bold(t):
    return _c("1", t)


def cyan(t):
    return _c("96", t)


def green(t):
    return _c("92", t)


def yellow(t):
    return _c("93", t)


def red(t):
    return _c("91", t)


def blue(t):
    return _c("94", t)


BANNER = cyan(r"""
   ___                  _ ___    ___
  / _ |___  ___ _  ___ (_) _ \  / _ \
 / __ / _ \/ _ \ |/ _ \| | | | | (_) |
/_/ |_\___/_//_/___|___/|_|___/  \___/
""") + bold("  Agniveer AI Assistant  - Offline · Local · Private\n")

HELP_TEXT = """
Available commands:

  /ingest pdf  <path>     Add a PDF to the knowledge base
  /ingest url  <url>      Add a web page to the knowledge base
  /ingest txt  <path>     Add a plain .txt file
  /ingest text <content>  Add raw text
  /ingest docx <path>     Add a Word (.docx) file
  /sources                List all ingested sources
  /stats                  Show index vector count
  /clear                  Clear conversation memory
  /reset                  Delete the entire knowledge base
  /model <name>           Switch the Ollama model
  /help                   Show this help
  /exit  or  /quit        Exit AgniAI

Answer style is detected automatically from your question.
  - "briefly" / "in short"   → SHORT
  - "explain" / "elaborate"  → ELABORATE
  - "in detail" / "step by step" → DETAIL
"""

_STYLE_LABEL = {"short": "SHORT", "elaborate": "ELABORATE", "detail": "DETAIL"}
_STYLE_COLOR = {"short": yellow, "elaborate": cyan, "detail": blue}


# =============================================================================
# HELPERS
# =============================================================================


def get_context_limit(style: str) -> int:
    if isinstance(MAX_CONTEXT_CHARS, dict):
        return MAX_CONTEXT_CHARS.get(style, MAX_CONTEXT_CHARS_DEFAULT)
    return int(MAX_CONTEXT_CHARS)


def get_token_limit(style: str) -> int:
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
    style_budget = get_token_limit(style)
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
    docs: list[dict],
    style: str,
    reasoning: bool,
    history: list[dict] | None,
    context_char_budget: int,
    context: str | None = None,
) -> tuple[list[dict], str]:
    if context is None:
        context = build_context(
            docs,
            max_chunks=max(STRICT_TOP_K, min(5, len(docs))),
            min_score=LOW_RETRIEVAL_CONFIDENCE,
            max_chars=context_char_budget,
        )

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
    return messages, context


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
        fallback_style = "Give a clear, well-organized answer without inventing specifics."
    else:
        fallback_style = "Give a clear answer in 1 to 3 short paragraphs."

    factual_signals = (
        "what is", "what are", "who is", "who was", "when did", "when was",
        "where is", "where was", "how does", "how do", "why does", "why do",
        "explain", "define", "difference between", "full form", "meaning of",
        "capital of", "president of", "prime minister", "how many", "how much",
        "which country", "which state", "formula", "equation", "calculate",
        "theorem", "law of", "principle of", "history of", "founder of",
        "invented by", "discovered by", "what happens", "why is", "how is",
    )
    is_factual = any(signal in query_lower for signal in factual_signals)

    if is_factual:
        system_content = (
            f"{CHAT_SYSTEM_PROMPT}\n\n"
            f"{GENERAL_KNOWLEDGE_FALLBACK_PROMPT}\n\n"
            "The user is asking a factual or subject-based question outside your "
            "Agniveer knowledge base. Answer it like a knowledgeable teacher or mentor - "
            "clear, conservative, and complete only where generic knowledge is enough. "
            "Do not restrict yourself to Agniveer topics. "
            "If the topic connects to Agniveer or Indian Army, mention that naturally at the end. "
            "Do not say 'Answer not found in the document' in this fallback path, but clearly "
            "state when exact values are unavailable in the knowledge base.\n\n"
            f"{fallback_style}"
        )
    else:
        system_content = (
            f"{CHAT_SYSTEM_PROMPT}\n\n"
            "The user is talking to you naturally - sharing something personal, "
            "asking for advice, or just having a conversation. "
            "Respond like a warm, caring human - not like a document reader. "
            "Be genuine, encouraging, and natural. Keep it conversational. "
            "Do not use bullet points or structured formatting. "
            "Do not say 'Answer not found in the document.' Just talk to them.\n\n"
            f"{fallback_style}"
        )

    messages: list[dict] = [{"role": "system", "content": system_content}]
    if history:
        for msg in history[-6:]:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": query})
    return messages


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# COMMAND HANDLERS
# =============================================================================


def _handle_ingest(command: str) -> None:
    parts = command.split(maxsplit=2)
    if len(parts) < 3:
        print(
            yellow(
                "Usage: /ingest pdf <path> | /ingest url <url> | "
                "/ingest txt <path> | /ingest text <content> | /ingest docx <path>"
            )
        )
        return

    kind = parts[1].lower()
    target = parts[2].strip()
    fn_map = {
        "pdf": ingest_pdf,
        "url": ingest_url,
        "txt": ingest_txt,
        "text": ingest_text,
        "docx": ingest_docx,
    }
    if kind not in fn_map:
        print(yellow(f"Unknown type '{kind}'. Use: pdf, url, txt, text, docx."))
        return
    try:
        print(dim(f"  Ingesting {kind}..."))
        count = fn_map[kind](target)
        if count == 0:
            print(yellow("  Source already ingested (use /reset to re-ingest)."))
        else:
            print(green(f"  Ingested {count} chunk(s) successfully."))
    except FileNotFoundError as exc:
        print(red(f"  File not found: {exc}"))
    except Exception as exc:
        print(red(f"  Ingestion failed: {exc}"))


def _handle_sources() -> None:
    sources = list_sources()
    if not sources:
        print(yellow("  No sources ingested yet."))
        return
    print(bold(f"\n  Ingested Sources ({len(sources)} total):"))
    for s in sources:
        chunk_info = dim(f"({s['chunk_count']} chunks)")
        print(f"    - [{s['doc_type'].upper()}] {s['source']}  {chunk_info}")


def _handle_stats() -> None:
    stats = index_stats()
    print(f"\n  Index stats: {stats['vectors']} vectors / {stats['chunks']} chunks")


def _handle_reset(memory: ConversationMemory) -> None:
    confirm = input(
        yellow("  This will DELETE the entire knowledge base. Type YES to confirm: ")
    ).strip()
    if confirm == "YES":
        clear_index()
        memory.clear()
        print(green("  Knowledge base cleared."))
    else:
        print(dim("  Reset cancelled."))


# =============================================================================
# MAIN CHAT LOOP
# =============================================================================


def run_chat() -> None:
    _ensure_dirs()
    memory = ConversationMemory()
    active_model: Optional[str] = DEFAULT_MODEL_NAME
    session = requests.Session()

    from ollama_cpu_chat import _start_keepalive_heartbeat

    _start_keepalive_heartbeat(session, interval_seconds=300)

    print(BANNER)
    stats = index_stats()
    if stats["vectors"] == 0:
        print(
            yellow(
                "  Knowledge base is empty. Use /ingest to add PDFs, URLs, or text.\n"
            )
        )
    else:
        print(dim(f"  Knowledge base ready: {stats['vectors']} vectors loaded.\n"))
    print(dim(f"  Active model: {active_model}  (type /model <name> to switch)\n"))

    while True:
        try:
            raw = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not raw:
            continue

        low = raw.lower()

        # ── Built-in commands ──────────────────────────────────────────────
        if low in {"/exit", "/quit"}:
            print("Goodbye.")
            break
        if low == "/help":
            print(HELP_TEXT)
            continue
        if low == "/sources":
            _handle_sources()
            continue
        if low == "/stats":
            _handle_stats()
            continue
        if low == "/clear":
            memory.clear()
            print(green("  Conversation memory cleared."))
            continue
        if low == "/reset":
            _handle_reset(memory)
            continue
        if low.startswith("/model"):
            parts = raw.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                active_model = parts[1].strip()
                from ollama_cpu_chat import _ACTIVE_MODEL_REF

                _ACTIVE_MODEL_REF[0] = active_model
                print(green(f"  Model switched to '{active_model}'."))
            else:
                print(yellow("  Usage: /model <model-name>"))
            continue
        if low.startswith("/ingest "):
            _handle_ingest(raw)
            continue
        if low.startswith("/"):
            print(yellow(f"  Unknown command: {raw} (type /help for a list)"))
            continue

        # ── Determine style and intent ─────────────────────────────────────
        style_name, _ = detect_answer_style(raw)
        style_label = _STYLE_LABEL[style_name]
        style_color = _STYLE_COLOR[style_name]
        context_limit = get_context_limit(style_name)
        history = memory.history()

        # Use shared classifier from config.py — single source of truth
        intent = classify_intent(raw)
        use_rag = intent == "rag"
        reasoning = is_reasoning_query(raw) if use_rag else False

        print(
            dim("  Answer style: ")
            + style_color(style_label)
            + dim(f"  [ctx={context_limit} chars]")
        )

        # ── Reject out-of-domain queries ───────────────────────────────────
        if intent == "reject":
            messages = _build_general_messages(
                query=raw,
                style=style_name,
                history=history[-6:] if history else None,
            )
            try:
                print("\nAgniAI: ", end="", flush=True)
                result = chat_with_fallback(
                    session,
                    active_model or DEFAULT_MODEL_NAME,
                    messages,
                    stream_tokens=True,
                    max_tokens_override=get_token_limit(style_name),
                )
                print()
                answer = _finalize_answer(result.text)
            except PartialResponseError as exc:
                print(f"\n  Partial response: {exc}\n")
                answer = _finalize_answer(exc.partial_text or REFERENCE_FALLBACK)
            except RuntimeError as exc:
                print(f"\n  LLM Error: {exc}\n")
                continue
            memory.add("user", raw)
            memory.add("assistant", answer)
            continue

        # ── Compute token / context budgets ────────────────────────────────
        token_limit, context_char_budget = _compute_context_char_budget(
            query=raw,
            style=style_name,
            history=history,
            reasoning=reasoning,
            use_rag=use_rag,
        )

        if not use_rag:
            response_key = make_response_cache_key(
                raw,
                style=style_name,
                model=active_model or DEFAULT_MODEL_NAME,
                context="chat",
                session_id="cli",
            )
            cached_answer = get_cached_response(response_key)
            if cached_answer is not None:
                print(dim("  [cache hit]"))
                print(f"\nAgniAI: {cached_answer}\n")
                memory.add("user", raw)
                memory.add("assistant", cached_answer)
                continue

        # ── RAG retrieval ──────────────────────────────────────────────────
        bundle: dict = {
            "docs": [],
            "context": "",
            "confidence": 0.0,
            "mode": "reject",
            "reasoning": False,
        }
        if use_rag:
            print(dim("  Preparing retrieval..."))
            bundle = prepare_rag_bundle(
                raw,
                top_k=TOP_K,
                style=style_name,
                max_context_chars=context_char_budget,
                include_points=False,
            )
            docs = bundle.get("docs", []) if isinstance(bundle, dict) else []
            confidence = (
                float(bundle.get("confidence", 0.0))
                if isinstance(bundle, dict)
                else 0.0
            )
            mode = (
                bundle.get("mode", "reject") if isinstance(bundle, dict) else "reject"
            )
            reasoning = (
                bool(bundle.get("reasoning", False))
                if isinstance(bundle, dict)
                else False
            )
            context_for_cache = (
                bundle.get("context", "") if isinstance(bundle, dict) else ""
            )
            response_key = make_response_cache_key(
                raw,
                style=style_name,
                model=active_model or DEFAULT_MODEL_NAME,
                context=context_for_cache,
                session_id="cli",
            )
            print(
                dim(
                    f"  Retrieval confidence: {confidence:.3f} | "
                    f"mode={mode} | reasoning={reasoning}"
                )
            )

            deterministic_answer = deterministic_policy_answer(raw, context_for_cache)
            if deterministic_answer:
                print(f"\nAgniAI: {deterministic_answer}\n")
                memory.add("user", raw)
                memory.add("assistant", deterministic_answer)
                set_cached_response(response_key, deterministic_answer)
                continue

            cached_answer = get_cached_response(response_key)
            if cached_answer is not None:
                print(dim("  [cache hit]"))
                print(f"\nAgniAI: {cached_answer}\n")
                memory.add("user", raw)
                memory.add("assistant", cached_answer)
                continue
        else:
            response_key = make_response_cache_key(
                raw,
                style=style_name,
                model=active_model or DEFAULT_MODEL_NAME,
                context="chat",
                session_id="cli",
            )

        docs = bundle.get("docs", []) if isinstance(bundle, dict) else []
        confidence = (
            float(bundle.get("confidence", 0.0)) if isinstance(bundle, dict) else 0.0
        )
        mode = bundle.get("mode", "reject") if isinstance(bundle, dict) else "reject"
        reasoning = (
            bool(bundle.get("reasoning", False)) if isinstance(bundle, dict) else False
        )

        # ── Generate answer (streaming to stdout) ──────────────────────────
        try:
            print("\nAgniAI: ", end="", flush=True)

            if use_rag:
                messages, _ = _build_rag_messages(
                    query=raw,
                    docs=docs,
                    style=style_name,
                    reasoning=reasoning,
                    history=history[-6:] if history else None,
                    context_char_budget=context_char_budget,
                    context=bundle.get("context", "")
                    if isinstance(bundle, dict)
                    else "",
                )
            else:
                messages = [
                    {
                        "role": "system",
                        "content": CHAT_SYSTEM_PROMPT,
                    }
                ]
                if history:
                    messages.extend(history[-6:])
                messages.append({"role": "user", "content": raw})

            result = chat_with_fallback(
                session,
                active_model or DEFAULT_MODEL_NAME,
                messages,
                stream_tokens=True,
                max_tokens_override=token_limit,
            )
            print()
            answer = _finalize_answer(result.text)

        except PartialResponseError as exc:
            print(f"\n  Partial response: {exc}\n")
            answer = _finalize_answer(exc.partial_text or REFERENCE_FALLBACK)
        except RuntimeError as exc:
            print(f"\n  LLM Error: {exc}\n")
            continue
        except KeyboardInterrupt:
            print("\n\n  [Generation stopped]\n")
            continue

        print()
        memory.add("user", raw)
        memory.add("assistant", answer)
        set_cached_response(response_key, answer)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    warmup_runtime(async_load=False)
    run_chat()
