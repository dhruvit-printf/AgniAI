"""
app_launcher.py
===============
PyInstaller entry point for AgniAI.

This wrapper:
1. Fixes sys.path so all AgniAI modules resolve correctly when running
   from inside the PyInstaller bundle (sys._MEIPASS).
2. Sets MEIPASS-aware base paths for data/ and index/ directories so
   FAISS, docstore, and BM25 files are written NEXT TO the executable,
   not inside the read-only bundle.
3. Loads .env from the same directory as the executable.
4. Then hands off to app.py exactly as normal.
"""

import os
import sys
from pathlib import Path


# ── 1. Resolve bundle root ────────────────────────────────────────────────
# When frozen by PyInstaller, sys._MEIPASS is the temp extraction directory.
# When running normally (dev mode), use the script's own directory.
if getattr(sys, "frozen", False):
    BUNDLE_DIR = Path(sys._MEIPASS)          # read-only extracted bundle
    EXE_DIR    = Path(sys.executable).parent # writable dir next to .exe
else:
    BUNDLE_DIR = Path(__file__).resolve().parent
    EXE_DIR    = BUNDLE_DIR

# ── 2. Prepend bundle dir to sys.path ─────────────────────────────────────
# Ensures `import rag`, `import config`, etc. all resolve from the bundle.
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

# ── 3. Load .env from exe directory ───────────────────────────────────────
# Users place their .env next to the executable, not inside the bundle.
env_path = EXE_DIR / ".env"
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=str(env_path), override=False)
    print(f"[AgniAI] Loaded .env from {env_path}")
else:
    print(f"[AgniAI] No .env found at {env_path} — using defaults / environment variables.")

# ── 4. Override data and index paths to use writable exe directory ─────────
# config.py computes DATA_DIR / INDEX_DIR relative to __file__ at import time.
# We override them here BEFORE config is imported so FAISS writes land in
# EXE_DIR/data and EXE_DIR/index instead of the read-only bundle.
os.environ.setdefault("AGNIAI_DATA_DIR",  str(EXE_DIR / "data"))
os.environ.setdefault("AGNIAI_INDEX_DIR", str(EXE_DIR / "index"))

# Ensure directories exist
(EXE_DIR / "data").mkdir(parents=True, exist_ok=True)
(EXE_DIR / "index").mkdir(parents=True, exist_ok=True)

# ── 5. Print startup info ─────────────────────────────────────────────────
print(f"[AgniAI] Bundle dir : {BUNDLE_DIR}")
print(f"[AgniAI] Working dir: {EXE_DIR}")
print(f"[AgniAI] Data dir   : {EXE_DIR / 'data'}")
print(f"[AgniAI] Index dir  : {EXE_DIR / 'index'}")
print()

# ── 6. Patch config.py paths before it is imported ───────────────────────
# config.py uses Path(__file__).resolve().parent — inside the bundle that
# resolves correctly, but DATA_DIR / INDEX_DIR need to be writable.
# We monkey-patch them after import so no source edit is required.
import config as _cfg
_cfg.DATA_DIR         = EXE_DIR / "data"
_cfg.INDEX_DIR        = EXE_DIR / "index"
_cfg.DOCSTORE_PATH    = _cfg.INDEX_DIR / "docstore.json"
_cfg.FAISS_INDEX_PATH = _cfg.INDEX_DIR / "agni.index"
_cfg.BM25_INDEX_PATH  = _cfg.INDEX_DIR / "bm25.pkl"

# Propagate to ingest and rag modules if already partially imported
for _mod_name in ("ingest", "rag"):
    _mod = sys.modules.get(_mod_name)
    if _mod is not None:
        for _attr in ("DATA_DIR", "INDEX_DIR", "DOCSTORE_PATH", "FAISS_INDEX_PATH", "BM25_INDEX_PATH"):
            if hasattr(_mod, _attr):
                setattr(_mod, _attr, getattr(_cfg, _attr))

# ── 7. Hand off to app.py ─────────────────────────────────────────────────
# Import app and run it — identical to `python app.py`.
from app import app, warmup_runtime, index_stats
from ollama_cpu_chat import _start_keepalive_heartbeat
import requests as _requests
import logging
from logging.handlers import RotatingFileHandler

# Set up logging to file next to exe
log_path = EXE_DIR / os.getenv("AGNI_LOG_FILE", "agni.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler(
            str(log_path),
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        ),
    ],
)

logger = logging.getLogger(__name__)

warmup_runtime(async_load=False)

_session = _requests.Session()
_start_keepalive_heartbeat(_session, interval_seconds=300)

stats = index_stats()
logger.info("AgniAI starting — %d vectors in knowledge base", stats["vectors"])
logger.info("Flask listening on http://0.0.0.0:5000")

app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)