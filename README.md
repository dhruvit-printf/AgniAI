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
license: mit
---

# AgniAI — Offline Agniveer Chatbot

A fully local, offline-first chatbot that answers **Agniveer / Agnipath recruitment** questions using:

- 🦙 **Ollama** — runs LLMs (Llama 3, Mistral, Phi-3, …) 100% locally
- 🔍 **FAISS + BM25** — hybrid vector + keyword similarity search
- 🧠 **Sentence Transformers** — local embeddings (no API needed)
- 📄 **Dynamic RAG** — ingest PDFs, URLs, or text at any time
- 🌐 **REST API** — Flask server for .NET / React / mobile frontends


## Requirements

| Tool | Minimum Version |
|------|----------------|
| Python | 3.9+ |
| Ollama | 0.1.x+ |
| RAM | 8 GB (16 GB recommended) |
| Disk | ~5 GB for model weights |


## Step-by-step Setup

### 1 — Install Ollama

**Linux / macOS:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Windows:** Download the installer from https://ollama.com/download

### 2 — Pull a local LLM

Choose one (Llama 3 is recommended):
```bash
ollama pull llama3                      # ~4.7 GB — best quality
ollama pull mistral:7b-instruct-q4_K_M # ~4.1 GB — fast & efficient
ollama pull phi3                        # ~2.3 GB — lightest option
```

### 3 — Start Ollama

```bash
ollama serve
```

Keep this terminal open. AgniAI calls it on `http://localhost:11434`.

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

The first run automatically downloads the embedding model (~90 MB). After that, everything runs offline.

### 7 — Ingest your documents

AgniAI starts with an empty knowledge base. Add documents before asking questions:

```bash
# In the CLI (see below), or via the REST API
/ingest pdf /path/to/agniveer_notification.pdf
/ingest url https://joinindianarmy.nic.in/
```

### 8 — Run AgniAI

**CLI mode (interactive terminal):**
```bash
python main.py
```

**REST API mode (for frontends / .NET integration):**
```bash
python app.py
# Server starts at http://0.0.0.0:5000
```


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


## REST API Reference

Start the server with `python app.py` then call these endpoints:

### Health check
```
GET /api/health
```
```json
{ "status": "ok", "vectors": 57, "chunks": 57, "model": "mistral:7b-instruct-q4_K_M" }
```

### Chat
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

Set `"stream": true` to receive a Server-Sent Events stream of tokens.

### Ingest a document
```
POST /api/ingest
Content-Type: application/json

{ "kind": "pdf", "target": "/path/to/file.pdf" }
```
Supported kinds: `pdf`, `url`, `txt`, `text`, `docx`

### Other endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/sources` | List all ingested sources |
| GET | `/api/stats` | Index vector count |
| POST | `/api/clear_memory` | Clear conversation memory for a session |
| POST | `/api/reset_index` | ⚠ Delete entire knowledge base |

> **Security note:** `/api/reset_index` is destructive. Set `API_SECRET_KEY` in your `.env` and protect this endpoint behind authentication in production.


## Configuration

All settings can be overridden with environment variables. Copy `.env.example` to `.env` and edit:

```bash
cp .env.example .env
```

Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `mistral:7b-instruct-q4_K_M` | LLM to use |
| `EMBEDDING_MODEL` | `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` | Embedding model |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server URL |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `USE_HYBRID` | `1` | Enable FAISS + BM25 hybrid retrieval |


## Example Session

```
You: What is the age limit for Agniveer?
AgniAI:
  Agniveer Age Eligibility:
  • Minimum age: 17.5 years
  • Maximum age: 21 years
  (Relaxation may apply for specific categories as per official notification.)

You: What is the salary?
AgniAI:
  Agniveer Monthly Package:
  • Year 1: ₹30,000/month (In hand: ₹21,000)
  • Year 2: ₹33,000/month (In hand: ₹23,100)
  • Year 3: ₹36,500/month (In hand: ₹25,580)
  • Year 4: ₹40,000/month (In hand: ₹28,000)
  • Seva Nidhi corpus after 4 years: ~₹10.04 lakh
```


## Project Structure

```
AgniAI/
├── main.py              # CLI chat loop + command dispatcher
├── app.py               # Flask REST API server
├── rag.py               # Embeddings, FAISS+BM25 search, Ollama LLM calls
├── ingest.py            # PDF / URL / text ingestion pipeline
├── memory.py            # Sliding-window conversation history
├── config.py            # All configuration constants (env-overridable)
├── ollama_cpu_chat.py   # CPU-optimised Ollama streaming client
├── api_models.py        # Shared JSON response shapes for REST API
├── benchmark_test.py    # Latency and throughput benchmarks
├── runtime_cache.py     # Thread-safe TTL cache
├── requirements.txt
├── .env.example         # Template for environment variables
├── data/                # (auto-created) raw data store
│   └── .gitkeep
└── index/
    ├── .gitkeep
    ├── agni.index       # FAISS binary index (auto-created, not committed)
    ├── docstore.json    # Chunk metadata (auto-created, not committed)
    └── bm25.pkl         # BM25 index (auto-created, not committed)
```


## Benchmarking

Run the built-in benchmark to measure startup, retrieval, and generation latency:

```bash
python benchmark_test.py --verbose
# Results saved to benchmark_results/benchmark_results_<timestamp>.json
```


## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Connection refused` on Ollama | Run `ollama serve` in a separate terminal |
| `Model not found` | Run `ollama pull mistral:7b-instruct-q4_K_M` |
| Slow first response | Normal — model loads into RAM on first call |
| "No text extracted" from PDF | PDF is image-based; use OCR tools first |
| Empty answers | Ingest relevant documents first with `/ingest` |
| High RAM usage | Switch to a smaller model: `ollama pull phi3` |


## Privacy

All computation happens **on your machine**. No data is sent to any cloud service. The embedding model is downloaded once and cached locally — subsequent runs are fully offline.