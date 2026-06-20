---
language: en
tags:
  - rag
  - ollama
  - offline
  - agniveer
  - agnipath
  - chatbot
  - retrieval-augmented-generation
  - faiss
  - sentence-transformers
  - admin-intelligence
  - websocket
  - prometheus
license: mit
---

# AgniAI — Offline Agniveer Chatbot + Admin Intelligence Layer

AgniAI is a fully local, offline-first platform with **two chatbots in one service**:

1. **User RAG Chatbot** — answers **Agniveer / Agnipath recruitment & training** questions, grounded entirely in documents you ingest. No cloud, no API keys.
2. **Admin Command Console** — a natural-language interface for commanding officers that classifies a question, routes it through a **.NET AiCommand backend**, and returns formatted, grounded reports (Performance, Leave, Medical, Attendance, Verification, Equipment, Distribution, Skills).

Both run inside the same Flask server and share the same Ollama LLM.

### Core technology

- 🦙 **Ollama** — runs LLMs (Mistral, Llama 3, Phi-3, …) 100% locally
- 🔍 **FAISS + BM25** — hybrid dense + keyword retrieval with optional cross-encoder reranking
- 🧠 **Sentence Transformers** — local embeddings, downloaded once then fully offline
- 📄 **Dynamic RAG** — ingest PDFs, URLs, Word docs, or raw text at any time
- 🌐 **REST + WebSocket APIs** — Flask + Socket.IO for .NET / React / mobile frontends
- 🛡️ **Production hardening** — circuit breaker, retries, rate limiting, audit logs, Prometheus metrics, optional Sentry + OpenTelemetry

---

## Architecture at a glance

```
                         ┌──────────────────────────────┐
   User question ───────▶│  /api/chat   (RAG pipeline)  │──▶ FAISS+BM25 ─▶ Ollama ─▶ grounded answer
                         └──────────────────────────────┘

                         ┌──────────────────────────────┐
   Admin question ──────▶│ /api/admin/chat  +  WebSocket│
                         │     (admin_pipeline.py)      │
                         └──────────────┬───────────────┘
                                        │
        plan_query ─▶ classify_intent ─▶ .NET AiCommand ─▶ combine ─▶ report ─▶ response
        (query_planner) (admin_intent)  (dotnet_executor) (result_  (report_   (response_
                                                            combiner) generator)  builder)
```

`admin_pipeline.execute_admin_query()` is the **single source of truth** — both the HTTP route and the WebSocket route call it.

---

## Requirements

| Tool | Minimum Version |
|------|----------------|
| Python | 3.9+ (3.11 used in CI) |
| Ollama | 0.1.x+ |
| RAM | 8 GB (16 GB recommended) |
| Disk | ~5 GB for model weights |
| .NET backend | Required **only** for the Admin Console |

---

## Step-by-step Setup

### 1 — Install Ollama

**Linux / macOS:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
**Windows:** Download the installer from https://ollama.com/download

### 2 — Pull a local LLM

```bash
ollama pull mistral:7b-instruct-q4_K_M   # ~4.1 GB — default, fast & efficient
ollama pull llama3                        # ~4.7 GB — best quality
ollama pull phi3                          # ~2.3 GB — lightest option
```

### 3 — Start Ollama

```bash
ollama serve
```
Keep this terminal open. AgniAI calls it on `http://127.0.0.1:11434`.

### 4 — Clone AgniAI

```bash
git clone https://github.com/florencygajera/AgniAI.git
cd AgniAI
```

### 5 — Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 6 — Install Python dependencies

```bash
pip install -r requirements.txt
```
The first run downloads the embedding model (~90 MB), after which everything runs offline.

### 7 — Configure environment

```bash
cp .env.example .env
```
At minimum set `DOTNET_API_BASE_URL` — the app **fails fast at startup** if it is missing (see `settings.py`). For the user RAG chatbot alone you can point it at any reachable URL; it is only used by the Admin Console.

### 8 — Ingest your documents

AgniAI starts with an empty knowledge base. Add documents before asking RAG questions:

```bash
# In the CLI (see below), or via the REST API
/ingest pdf /path/to/agniveer_notification.pdf
/ingest url https://joinindianarmy.nic.in/
```

### 9 — Run AgniAI

**CLI mode (interactive terminal, user RAG only):**
```bash
python main.py
```

**REST + WebSocket API mode (frontends / .NET integration):**
```bash
python app.py
# Server starts at http://0.0.0.0:5000
```

---

## The two chatbots

### User RAG Chatbot

Every incoming message is classified by `config.classify_intent()` into **chat**, **rag**, or **reject**, then handled accordingly:

- **chat** — greetings, small talk, patriotic phrases, aspirant casual talk → warm conversational reply.
- **rag** — domain questions (eligibility, salary, training, documents, timelines, …) → FAISS+BM25 retrieval, grounded answer with strict number-accuracy rules.
- **reject / general fallback** — out-of-domain factual questions answered conservatively from general knowledge, or redirected.

It also includes:
- **Deterministic policy answers** for high-stakes facts (salary tables, age eligibility, marital-status rules) computed arithmetically rather than left to the LLM (`rag.deterministic_policy_answer`).
- **Conflict handling** — when the knowledge base contains disagreeing figures (e.g. Year 3 in-hand salary), both values are reported with a verification note.
- **Answer-style detection** — `short` / `elaborate` / `detail` inferred from the wording, controlling length and token budgets.
- **SSE streaming** — set `"stream": true` and send `Accept: text/event-stream`.

### Admin Command Console

A separate pipeline (`admin_pipeline.py`) for officers. A single question can resolve to one of several **query types** (`query_planner.py`):

| Query type | Example |
|------------|---------|
| `simple` | "Show top 5 performers in BPET" |
| `cross_filter` | "Show top performer in PPT who plays cricket and is currently on leave" |
| `comparison` | "Compare BPET and PPT for Sikh class" |
| `multi_independent` | "Show attendance stats as well as equipment overdue records" |
| `analytics` | "Which section has the highest average score?" |

Supported modules: **Performance, Leave, Medical, Attendance, Verification, Equipment, Distribution, Skills**. Company/platoon names in the question are resolved to IDs against the .NET lookup APIs (`admin_entity_resolver.py`), and follow-up phrasing ("which of them…") is tracked per session (`admin_context.py`).

Reports are generated with strict grounding guards — any number the LLM emits that does not appear in the aggregate data is stripped (`report_generator.py`). Raw .NET responses are **never** forwarded to the frontend (`response_builder.py`).

---

## CLI Commands

| Command | Action |
|---------|--------|
| `/ingest pdf <path>` | Ingest a PDF |
| `/ingest url <url>` | Ingest a webpage |
| `/ingest txt <path>` | Ingest a .txt file |
| `/ingest text <content>` | Ingest raw text |
| `/ingest docx <path>` | Ingest a Word document |
| `/sources` | List all ingested sources |
| `/stats` | Show index vector count |
| `/clear` | Clear conversation memory |
| `/reset` | ⚠ Delete entire knowledge base |
| `/model <name>` | Switch Ollama model mid-session |
| `/help` | Show help |
| `/exit` or `/quit` | Exit |

Answer style is detected automatically: *"briefly"* → SHORT, *"explain"* → ELABORATE, *"in detail"* → DETAIL.

---

## REST API Reference

Start the server with `python app.py`. Interactive docs are available at **`/docs`** (Swagger UI).

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Ollama connectivity, vector/chunk counts, active model |
| GET | `/api/ready` | Whether embedding model + index have warmed up |
| GET | `/metrics`, `/api/metrics` | Prometheus exposition format |
| GET | `/docs`, `/docs/spec` | Swagger UI + raw OpenAPI spec |

### Chat (user RAG)
```
POST /api/chat
Content-Type: application/json

{
  "message": "What is the age limit for Agniveer?",
  "model": "mistral:7b-instruct-q4_K_M",
  "stream": false,
  "session_id": "user-123"
}
```
```json
{ "success": true, "answer": "...", "style": "elaborate", "session_id": "user-123" }
```
Set `"stream": true` with header `Accept: text/event-stream` to receive an SSE token stream.

### Knowledge base

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/ingest` | Ingest by path/URL. `kind`: `pdf` `url` `txt` `text` `docx` `doc` |
| POST | `/api/upload` | Multipart file upload. Allowed: `pdf` `txt` `docx` `doc` |
| GET | `/api/sources` | List ingested sources |
| GET | `/api/stats` | Index vector / chunk count |
| POST | `/api/reset_index` | ⚠ Destructive. Protected by `X-Api-Key` when `API_SECRET_KEY` is set |

```
POST /api/ingest
{ "kind": "pdf", "target": "/path/to/file.pdf" }
```

### Session

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/clear_memory` | Clear sliding-window history for a session |

### Admin Console

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/admin/chat` | Full admin pipeline → formatted answer + analysis + conclusion |
| GET | `/api/admin/health` | Sub-system health (python / dotnet / llm / database) |

```
POST /api/admin/chat
{ "message": "Who are the top 5 performers in BPET?", "session_id": "admin-1" }
```

### WebSocket (Admin, real-time)

Connect to `ws://localhost:5000/socket.io`. Emit a `query` event:
```json
{ "message": "Who is on leave today?", "session_id": "admin-1" }
```
The server streams back `query_received` → `progress` (planner/intent/dotnet/combiner/report) → `intro` → `result` → `analysis` → `conclusion` → `done`.

---

## Configuration

All settings are environment variables — copy `.env.example` to `.env` and edit. Selected keys:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `mistral:7b-instruct-q4_K_M` | LLM to use |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server URL |
| `EMBEDDING_MODEL` | `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` | Embedding model (dim 384) |
| `TOP_K` | `5` | Chunks retrieved per query |
| `USE_HYBRID` | `1` | Enable FAISS + BM25 hybrid retrieval |
| `USE_RERANKER` | `0` | Enable cross-encoder reranking |
| `DOTNET_API_BASE_URL` | — | **Required.** Root URL of the .NET AiCommand API (must differ from port 5000) |
| `DOTNET_SKIP_SSL_VERIFY` | `1` | Skip SSL verification for dev tunnels |
| `API_SECRET_KEY` | _(empty)_ | Protects `/api/reset_index` when set |
| `ALLOWED_ORIGINS` | `*` | CORS origins |
| `RATE_LIMIT_CHAT` | `30 per minute` | Per-IP chat rate limit |
| `ADMIN_RATE_LIMIT` | `20 per minute` | Per-IP admin rate limit |
| `AGNI_LOG_FILE` | `agni.log` | Application log file |

**Feature flags** (`feature_flags.py`, prefix `ENABLE_`): `REPORTS`, `OLLAMA`, `STREAMING`, `WEBSOCKET`, `METRICS`, `HEALTH_ENDPOINT`, `AUDIT_LOGGING`, `OPENTELEMETRY` (default off), `SENTRY` (default off), `PROMETHEUS`. Change a flag in `.env` to toggle a feature at runtime — no code change needed.

> ⚠ **Port rule:** Flask listens on **5000**; the .NET backend runs on a **different** port (default `7257`). Never point `DOTNET_API_BASE_URL` at 5000.

---

## Observability & Reliability

**Metrics** (`metrics.py`) — Prometheus-ready counters and summaries at `/metrics`: requests/successful/failed by query type, per-stage durations (planner, intent, dotnet, combiner, report, pipeline), LLM/dotnet/timeout failures, active WebSocket gauge. Grafana-friendly names.

**Audit logging** (`audit_logger.py`) — one JSON line per admin query to a rotating `audit.log` (10 MB × 30 backups, 90-day retention). Prompts, payloads, raw responses, and secrets are **never** logged.

**Circuit breaker + retries** (`dotnet_executor.py`) — trips after 5 consecutive failures, recovers after a 10s cooldown (HALF-OPEN). Exponential backoff (1s/2s/4s) on 429/502/503/504 and connection/timeout errors; strict `(5, 30)` timeouts.

**Rate limiting** — Flask-Limiter on chat, ingest, and admin routes; returns a clean 429.

**Tracing & error monitoring** (opt-in) — OpenTelemetry spans (`telemetry.py`) and Sentry with PII/secret scrubbing (`sentry_integration.py`); both no-op with zero overhead when disabled.

**Startup safety** (`settings.py`) — the process exits immediately if `DOTNET_API_BASE_URL` is absent, so it never half-starts and fails on the first request.

---

## Project Structure

```
AgniAI/
├── User RAG chatbot
│   ├── main.py                # CLI chat loop + command dispatcher
│   ├── app.py                 # Flask REST API + Socket.IO server
│   ├── rag.py                 # Embeddings, FAISS+BM25, retrieval, deterministic answers
│   ├── ingest.py              # PDF / URL / DOCX / DOC / text ingestion
│   ├── memory.py              # Sliding-window conversation history
│   ├── config.py              # Prompts, intent classifier, style detection, env config
│   ├── ollama_cpu_chat.py     # CPU-optimised Ollama streaming client + fallbacks
│   ├── api_models.py          # Shared JSON response shapes
│   └── runtime_cache.py       # Thread-safe TTL caches
│
├── Admin Command Console
│   ├── admin_pipeline.py      # Single source of truth for admin query execution
│   ├── admin_routes.py        # HTTP transport for /api/admin/*
│   ├── admin_intent.py        # Intent classifier (8 modules, fuzzy vocab, item lists)
│   ├── admin_entity_resolver.py # Company/platoon name → ID via .NET (cached)
│   ├── admin_confidence.py    # Unified confidence scoring
│   ├── admin_context.py       # Per-session follow-up context
│   ├── query_planner.py       # simple / cross_filter / comparison / multi / analytics
│   ├── result_combiner.py     # Intersection, comparison, merge, aggregation
│   ├── report_generator.py    # Grounded LLM analysis + conclusion
│   ├── response_builder.py    # Final payload assembly (no raw backend leakage)
│   └── dotnet_executor.py     # .NET calls with circuit breaker + retries
│
├── Infrastructure
│   ├── websocket_manager.py   # Socket.IO connection registry
│   ├── websocket_routes.py    # WebSocket transport (calls the same pipeline)
│   ├── metrics.py             # Prometheus metrics
│   ├── audit_logger.py        # Rotating JSON audit trail
│   ├── telemetry.py           # OpenTelemetry spans (opt-in)
│   ├── sentry_integration.py  # Sentry with secret scrubbing (opt-in)
│   ├── feature_flags.py       # Pydantic feature flags
│   ├── settings.py            # Validated settings + startup guard
│   ├── swagger_ui.py          # /docs Swagger UI
│   └── static/swagger.json    # OpenAPI 3.0 spec
│
├── Packaging
│   ├── agniai.spec            # PyInstaller build spec
│   ├── app_launcher.py        # Frozen-executable entry point
│   ├── .pyinstaller_hooks/    # torch/transformers freeze hooks
│   └── start.bat              # Windows launcher (Ollama + AgniAI)
│
├── tests/                     # pytest suite (intent, planner, combiner, reliability, …)
├── .github/workflows/         # CI: build, tests, code quality
├── requirements.txt
├── .env.example
├── data/    (auto-created)    # raw data store
└── index/   (auto-created)    # agni.index · docstore.json · bm25.pkl
```

---

## Building a standalone executable (Windows)

AgniAI can be frozen with PyInstaller into a self-contained `.exe`:

```bash
pyinstaller agniai.spec --clean --noconfirm
# Output: dist/agniai/agniai.exe
```

Place your `.env` next to the executable. The launcher (`app_launcher.py`) redirects `data/` and `index/` to writable directories beside the `.exe`. `start.bat` boots Ollama and AgniAI together.

---

## Testing

```bash
pip install pytest pytest-cov
python -m pytest --cov=. -v
```

The suite covers admin intent classification, the query planner, result combiner, response/grounding pipeline, reliability (circuit breaker, retries), metrics & health, observability/log scrubbing, telemetry, feature flags, and settings validation. CI runs build, tests, and code-quality (black / flake8 / isort / mypy) on every push.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Connection refused` on Ollama | Run `ollama serve` in a separate terminal |
| `Model not found` | Run `ollama pull mistral:7b-instruct-q4_K_M` |
| App exits at startup | Set `DOTNET_API_BASE_URL` in `.env` |
| Slow first response | Normal — model loads into RAM on first call |
| "No text extracted" from PDF | PDF is image-based; OCR it first |
| Empty answers | Ingest relevant documents first with `/ingest` |
| Admin returns "Failed to process request" | Check `/api/admin/health`; ensure the .NET backend is reachable |
| High RAM usage | Switch to a smaller model: `ollama pull phi3` |

---

## Privacy

All computation happens **on your machine**. The user RAG chatbot sends no data to any cloud service; the embedding model is downloaded once and cached locally. The Admin Console talks only to your own .NET backend. Raw backend records and internal payloads are never returned to the frontend, and audit/Sentry data is scrubbed of prompts, payloads, and secrets.

```bash
# Two services should always be running in production:
# sudo systemctl status ollama    # Ollama LLM
# sudo systemctl status agniai    # AgniAI Flask API
```
