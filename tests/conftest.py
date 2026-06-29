import os

# ── Test environment bootstrap ────────────────────────────────────────────
# Must be set BEFORE any AgniAI module is imported, since settings.py
# reads env vars at import time via pydantic-settings.
os.environ.setdefault("ENV", "testing")
os.environ.setdefault("ENVIRONMENT", "testing")
os.environ.setdefault("OLLAMA_BASE_URL", "http://127.0.0.1:9999")
os.environ.setdefault("DOTNET_API_BASE_URL", "http://127.0.0.1:9998")
os.environ.setdefault("BYPASS_CACHE", "true")
os.environ.setdefault("API_SECRET_KEY", "test-secret")
