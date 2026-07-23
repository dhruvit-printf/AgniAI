# AgniAI Deployment Guide

This guide covers deploying AgniAI on a new PC or server, either **from
source** (recommended for a dev/server box you'll maintain with `git pull`)
or as the **packaged Windows executable** (recommended for handing off to an
end user who won't touch Python).

AgniAI is a Flask backend with two chat surfaces:

- **General/RAG chatbot** (`/api/chat`) — answers from an ingested knowledge
  base (PDF/DOCX/text) using a local embedding model + FAISS + Ollama.
- **Admin chatbot** (`/api/admin/chat`) — natural-language queries over the
  AgniAI SQL Server database (attendance, performance, leave, medical,
  equipment, etc.), routed through an intent classifier and a read-only SQL
  layer.

The admin chatbot is optional — the app runs fine without a database
connection, it just can't answer admin queries.

> **Deploying to a server with no internet access?** Read §1 and §3 for
> context, then skip straight to **§3a — Offline / air-gapped deployment**.
> Everything else in §2/§3 assumes the target machine can reach the
> internet at least once (to `pip install` or download models).

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| **OS** | Windows (primary target — packaged build, `start.bat`, SQL Server via `Trusted_Connection`/ODBC). Source deployment also works on Linux/macOS if you're not using the packaged `.exe` and adapt the SQL connection string. |
| **Python** | 3.10–3.12 (3.11 recommended). Only needed for source deployment — the packaged `.exe` bundles its own runtime. |
| **Ollama** | Required for narrative/report generation and the general chatbot's answers. Install from [ollama.com](https://ollama.com), then pull the model(s) you configure (default `mistral:7b-instruct-q4_K_M`). |
| **ODBC Driver for SQL Server** | Only needed if enabling the admin/SQL chatbot. Install "ODBC Driver 17" or "18 for SQL Server" from Microsoft. |
| **SQL Server access** | Only needed for the admin chatbot — a **read-only** login (see §5). |
| **Disk space** | ~2–4 GB for the embedding/reranker models plus your knowledge-base documents and FAISS index. |
| **RAM** | 8 GB minimum; 16 GB+ recommended if running Ollama's 7B model and the embedding model concurrently on CPU. |

---

## 2. Option A — Run from source

### 2.1 Get the code

```bash
git clone <your-repo-url> AgniAI
cd AgniAI
```

### 2.2 Create a virtual environment and install dependencies

```bash
python -m venv agniai-env
# Windows:
agniai-env\Scripts\activate
# Linux/macOS:
source agniai-env/bin/activate

pip install -r requirements.txt
```

Use `requirements-lock.txt` instead if you want the exact pinned versions
this project was last verified against:

```bash
pip install -r requirements-lock.txt
```

If you plan to enable the admin/SQL chatbot, also install `pyodbc` (already
listed in `requirements.txt`) and make sure the ODBC Driver for SQL Server
(§1) is installed at the OS level — `pip install pyodbc` alone is not
enough, it needs the native driver.

### 2.3 Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` — see §4 for the full variable reference. At minimum, set:

- `OLLAMA_BASE_URL` (default `http://127.0.0.1:11434` is fine if Ollama runs
  on the same machine)
- `DOTNET_API_BASE_URL` if you have a companion .NET service (optional —
  only used by some admin-pipeline entity resolution paths)
- `ALLOWED_ORIGINS` — set to your actual frontend origin(s) in production
  instead of leaving it wide open
- `API_SECRET_KEY` — set this to require an `X-Api-Key` header on
  sensitive endpoints (index reset, etc.)

### 2.4 Start Ollama and pull the model

```bash
ollama serve
# in another terminal:
ollama pull mistral:7b-instruct-q4_K_M
```

(Match whatever you set in `OLLAMA_MODEL`/`OLLAMA_FALLBACK_MODELS`.)

### 2.5 Run the app

```bash
python app.py
```

On first run, the embedding model (`EMBEDDING_MODEL`, default
`sentence-transformers/multi-qa-MiniLM-L6-cos-v1`) downloads from
HuggingFace Hub and is cached locally — this needs internet access once.
After that, it loads from cache and the app can run fully offline (aside
from Ollama and, if enabled, the SQL Server connection).

You should see:

```
Python / Flask listens on  http://0.0.0.0:5000
Health check  http://localhost:5000/api/health
Chat endpoint http://localhost:5000/api/chat  [POST]
Admin chat    http://localhost:5000/api/admin/chat  [POST]
Swagger UI    http://localhost:5000/docs
```

Verify with:

```bash
curl http://localhost:5000/api/health
```

### 2.6 Load a knowledge base (for the general/RAG chatbot)

Upload documents via `POST /api/upload` (multipart), or place files under
the configured data directory and re-ingest — see `ingest.py`. Until
documents are ingested, `/api/health` will report the knowledge base as
empty and general-chat questions fall back to `REFERENCE_FALLBACK`.

---

## 3. Option B — Run the packaged Windows executable

Use this when handing the app to someone who shouldn't need Python
installed at all.

### 3.1 Build the package (done once, by you, on a dev machine)

```bash
pip install pyinstaller
pyinstaller agniai.spec --clean --noconfirm
```

Output lands in `dist/agniai/` — a one-folder build containing `agniai.exe`
plus all bundled dependencies (Flask, sentence-transformers, torch, FAISS,
etc.). Copy the entire `dist/agniai/` folder to the target machine — it
does not need Python installed there.

### 3.2 On the target machine

1. Copy `dist/agniai/` to somewhere writable, e.g. `C:\AgniAI\`.
2. Place a `.env` file **next to `agniai.exe`** (not inside any subfolder)
   — copy `.env.example` and edit it as in §2.3. The exe looks for `.env`
   in its own directory, not the original project's.
3. Install Ollama on the target machine and pull the configured model (§2.4)
   — the packaged build does not bundle Ollama itself.
4. If using the admin/SQL chatbot, install the ODBC Driver for SQL Server
   on the target machine (§1) and set `SQL_READONLY_CONN` in `.env`.
5. Run `agniai.exe` (a console window opens and stays open — this is
   normal, it's the running server, not a launcher).

`start.bat` in the repo is a convenience script showing the expected
sequence (start Ollama, wait, start the exe) — adapt the paths for your
target machine.

Data and index directories (`data/`, `index/`) are created automatically
next to the exe on first run — that's where uploaded documents and the
FAISS index/docstore live, so back that folder up if you care about the
ingested knowledge base.

### 3.3 Rebuilding after code changes

Re-run step 3.1 and redistribute the new `dist/agniai/` folder. The `.env`,
`data/`, and `index/` you already have on the target machine can be kept —
just don't overwrite them with the fresh build output.

---

## 3a. Offline / air-gapped deployment

The offline server itself never touches the internet. Everything that
would normally be fetched on demand (pip packages, the embedding model,
Ollama's model weights) has to be downloaded once on a separate machine
that **does** have internet — a "staging machine" — and then copied across
by USB drive or internal file share.

Use the packaged `.exe` build (§3), not the from-source route (§2) — it
avoids needing Python, pip, or any PyPI access on the offline machine
entirely. The only remaining internet dependencies to solve are the ML
model weights (embedding model + Ollama) and the ODBC driver installer,
covered below.

### Step 1 — On the staging machine (has internet)

1. **Build the exe** — follow §3.1 (`pyinstaller agniai.spec --clean
   --noconfirm`). This bundles all Python dependencies; nothing further to
   fetch from PyPI.

2. **Pre-download the embedding model into a portable cache folder**
   (don't rely on the default `%USERPROFILE%\.cache\huggingface`, since
   that's tied to a specific Windows user account and not something you'd
   want to depend on existing on the offline machine):

   ```bash
   set HF_HOME=%CD%\hf_cache
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')"
   ```

   If you set `USE_RERANKER=1`, also pre-download that model the same way:

   ```bash
   python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
   ```

   This populates `hf_cache\hub\models--sentence-transformers--multi-qa-MiniLM-L6-cos-v1\...`
   (and the reranker's equivalent folder). That whole `hf_cache` folder is
   what you'll copy to the offline machine.

3. **Install Ollama and pull your model(s)**:

   ```bash
   ollama pull mistral:7b-instruct-q4_K_M
   ```

   Ollama stores model weights as content-addressed blobs under
   `%USERPROFILE%\.ollama\models` (or wherever `OLLAMA_MODELS` points).
   Copying that whole `models` folder to another machine running the
   **same Ollama version** is a supported way to transfer models without
   re-downloading — no `ollama pull` needed on the offline side.

4. **Download the ODBC Driver for SQL Server installer** (MSI) from
   Microsoft's site, if using the admin/SQL chatbot — just the installer
   file, no offline-specific steps needed here.

5. Gather everything to transfer: `dist/agniai/` (the exe folder),
   `hf_cache/`, the Ollama `models/` folder, the Ollama installer itself,
   and the ODBC driver installer.

### Step 2 — On the offline server

1. Copy `dist/agniai/` to e.g. `C:\AgniAI\`.

2. Copy `hf_cache/` to e.g. `C:\AgniAI\hf_cache\`.

3. Install the ODBC driver from the transferred MSI (if using the admin
   chatbot).

4. Install Ollama from the transferred installer, **then stop it**
   (`ollama serve` shouldn't be running yet), and copy the transferred
   `models/` folder over the freshly-installed (empty)
   `%USERPROFILE%\.ollama\models` folder — or point `OLLAMA_MODELS` at
   wherever you placed it instead of overwriting the default location.
   Start Ollama and run `ollama list` to confirm the model shows up
   without needing to pull it.

5. Create `.env` next to `agniai.exe` (§3.2), and add these two blocks on
   top of the normal config:

   ```
   # Point at the pre-populated cache instead of the default user cache,
   # and force pure offline mode so nothing ever tries to reach HuggingFace.
   HF_HOME=C:\AgniAI\hf_cache
   HF_HUB_OFFLINE=1
   TRANSFORMERS_OFFLINE=1
   ```

   Everything else in `.env` (Ollama URL, SQL connection, etc.) stays as
   described in §4 — Ollama and SQL Server are assumed to be reachable on
   the **local network**, just not the internet.

6. Run `agniai.exe` as in §3.2. First startup should now load the
   embedding model straight from `HF_HOME` with no network call — if it
   instead hangs or errors trying to reach `huggingface.co`, see
   troubleshooting below.

### Offline-specific troubleshooting

| Symptom | Cause |
|---|---|
| Startup hangs or fails trying to reach `huggingface.co` | `HF_HOME`/`HF_HUB_OFFLINE` not set, or set *after* import (must be in `.env`, loaded before `rag.py` initializes) — double check the `.env` file is actually being found (§3.2, must sit next to `agniai.exe`) |
| `OSError: ... couldn't connect to 'https://huggingface.co'` even with `HF_HUB_OFFLINE=1` | The model folder name under `hf_cache\hub` doesn't match `EMBEDDING_MODEL` exactly (case/slash-to-double-dash conversion) — re-check step 1.2's output folder name |
| Ollama model not found after copying `models/` | Ollama version mismatch between staging and offline machine, or `OLLAMA_MODELS` env var not pointing at the copied location — run `ollama list` to check what Ollama actually sees |
| Reranker silently never activates | Expected if its model wasn't pre-downloaded in step 1.2 — the code deliberately never auto-downloads it, only checks the local cache (see `_reranker_local_files_available` in `rag.py`) |

---

## 4. Environment variable reference

All variables have sane defaults in `config.py`/`app.py` except where noted
"required". Full list with defaults: `.env.example`.

### Required for the admin/SQL chatbot only

| Variable | Purpose |
|---|---|
| `SQL_READONLY_CONN` | ODBC connection string for a **read-only** SQL Server login (see §5). Admin/SQL queries return `SQL_READONLY_CONN is not configured` errors until this is set. |
| `ENABLE_SQL_EXECUTOR` | Set `true` to route unclassifiable/low-confidence admin queries to the SQL backend instead of failing. Defaults to `false`. |

### Ollama

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama server address |
| `OLLAMA_MODEL` | `mistral:7b-instruct-q4_K_M` | Primary model |
| `OLLAMA_FALLBACK_MODELS` | (list) | Tried in order if the primary isn't installed |
| `OLLAMA_NUM_CTX`, `OLLAMA_MAX_TOKENS`, `OLLAMA_TEMPERATURE` | — | Generation tuning |

### Embeddings / retrieval

| Variable | Default | Purpose |
|---|---|---|
| `EMBEDDING_MODEL` | `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` | Downloaded from HF Hub on first run, cached after |
| `USE_RERANKER` | `0` | Cross-encoder rerank — only activates if the model is already cached locally |
| `TOP_K`, `USE_HYBRID`, `DENSE_WEIGHT`, `BM25_WEIGHT`, `MIN_SCORE` | — | Retrieval tuning, defaults are reasonable |

### Networking / security

| Variable | Default | Purpose |
|---|---|---|
| `ALLOWED_ORIGINS` | (example origins) | CORS allowlist — **set this explicitly in production** |
| `API_SECRET_KEY` | empty (disabled) | If set, requires `X-Api-Key` header on protected endpoints |
| `RATE_LIMIT_CHAT`, `RATE_LIMIT_INGEST`, `RATE_LIMIT_DEFAULT`, `ADMIN_RATE_LIMIT` | see `.env.example` | Flask-Limiter rate limits |
| `DOTNET_API_BASE_URL` | example URL | Companion .NET service, if used |

### Advanced / undocumented-but-supported (defaults are fine for most deployments)

| Variable | Default | Purpose |
|---|---|---|
| `DOTNET_TIMEOUT` | `3` (seconds) | Timeout for company/platoon directory lookups against the .NET service |
| `ENTITY_CACHE_TTL_SECONDS` | `300` | How long company/platoon directory data is cached before re-fetching |
| `SQL_MAX_ROWS` | `500` | Row cap on admin SQL query results |
| `SQL_COMMAND_TIMEOUT_S` | `15` | SQL Server command timeout |
| `AGNI_LOG_FILE` | `agni(ai).log` | Log file path, written next to the app |

---

## 5. Database setup for the admin/SQL chatbot

**Never point `SQL_READONLY_CONN` at a privileged login.** Provision a
dedicated least-privilege login first:

1. Open `db/readonly_login.sql` in the repo.
2. Change the placeholder password.
3. Run it once, as a DBA, against the SQL Server instance hosting the
   AgniAI database. It's idempotent — safe to re-run.

   This creates a `agniai_reporting` login with:
   - `db_datareader` (read-only) on the target database
   - Explicit `DENY` on sensitive columns/tables (`UserMaster.Password`,
     `LoginToken`, `DefaultLog`) even though `db_datareader` would
     otherwise allow reading them
   - Explicit `DENY` on all write/DDL operations, as defense-in-depth

4. Put the resulting connection string in `.env`:

   ```
   SQL_READONLY_CONN=Driver={ODBC Driver 17 for SQL Server};Server=<host>;Database=DB_Agni;UID=agniai_reporting;PWD=<password>;Encrypt=yes;TrustServerCertificate=no;
   ```

   (Local dev commonly uses Windows auth instead —
   `Driver={ODBC Driver 18 for SQL Server};Server=<host>\<instance>;Database=DB_Agni;Trusted_Connection=yes;TrustServerCertificate=yes;`
   — but a dedicated SQL login is what §5's script provisions and what
   production deployments should use.)

5. Verify the login works and is properly restricted using the test
   queries at the bottom of `db/readonly_login.sql`.

### Keeping the schema metadata current

`actual_schema.json` is a cached snapshot of the live DB schema used for
SQL generation/validation. After any schema migration on the AgniAI
database, regenerate it:

```bash
python scripts/regenerate_schema_json.py           # dry run — shows the diff
python scripts/regenerate_schema_json.py --write   # applies it, with an automatic timestamped backup
```

Requires `SQL_READONLY_CONN` to be set and reachable.

---

## 6. First-run verification checklist

- [ ] `GET /api/health` returns 200
- [ ] `GET /docs` loads the Swagger UI
- [ ] `POST /api/chat` with a simple message returns an answer (may say
      "not available in my reference" until documents are ingested —
      that's expected, not an error)
- [ ] `POST /api/admin/chat` with e.g. `"show top performers in bpet"`
      returns real data (requires §5 to be configured) — a "Question is
      not understood" or `SQL_READONLY_CONN is not configured` response
      means the DB isn't wired up yet
- [ ] Logs are being written (`agni.log` / `agniai.log` next to the app)

---

## 7. Running as a persistent service (production)

The app itself is a plain Flask process (`app.run(host="0.0.0.0",
port=5000)`) — it doesn't daemonize or install itself as a service. For a
production server, wrap it with one of:

- **Windows**: [NSSM](https://nssm.cc/) to run `agniai.exe` (or
  `python app.py`) as a Windows service, with auto-restart on crash.
- **Linux**: a `systemd` unit running `python app.py` inside the venv, or a
  process manager like `supervisord`.

Either way, make sure Ollama is also running/auto-starting before AgniAI
starts, since narrative generation and the general chatbot depend on it
being reachable at `OLLAMA_BASE_URL`.

---

## 8. Troubleshooting

| Symptom | Likely cause |
|---|---|
| `SQL_READONLY_CONN is not configured` on admin queries | `.env` doesn't have it set, or the exe is reading a `.env` from the wrong directory (must be next to `agniai.exe`, not the source repo) |
| Admin queries return "Question is not understood" | Either the DB connection is unreachable (check `SQL_READONLY_CONN`, ODBC driver install, firewall/VPN to the SQL Server host) or `ENABLE_SQL_EXECUTOR=false` and the query genuinely wasn't classifiable |
| General chat always returns the "not available in my reference" fallback | No documents ingested yet, Ollama unreachable, or `MIN_SCORE`/`STRICT_MIN_SCORE` too strict for your document set |
| Slow first request after startup | Expected — embedding model (and Ollama's model) are loading/warming up; subsequent requests are fast |
| `Cannot send a request, as the client has been closed` in logs | A transient race in the embedding-model warmup path; self-heals on retry, not a startup failure |
| CORS errors in the browser | `ALLOWED_ORIGINS` doesn't include your frontend's origin |
| 401 `Unauthorized. Provide X-Api-Key header.` | `API_SECRET_KEY` is set — include `X-Api-Key: <value>` on requests to protected endpoints |
| PyInstaller build fails or the exe crashes on an unrelated machine | Usually a missing hidden-import for a transformers/torch submodule — check `.pyinstaller_hooks/` and `agniai.spec`'s `hidden_imports` list before adding new ML dependencies |
