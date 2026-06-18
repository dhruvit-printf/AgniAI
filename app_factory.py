"""
app_factory.py
==============
AgniAI — Application Factory.

Demonstrates startup validation: the app NEVER reaches runtime
if critical environment variables are absent.

Import path fix: settings is at the root level (settings.py),
not inside a config package. Use `from settings import ...`.
"""

import os
import logging

from settings import (
    AppSettings,
    DotNetAPIConfig,
    APIKeysConfig,
    FeatureFlagConfig,
    TimeoutConfig,
    validate_critical_env,
    get_settings,
    get_dotnet_config,
    get_api_keys,
    get_feature_flags,
    get_timeouts,
)


logger = logging.getLogger(__name__)


def create_app() -> "Flask":  # type annotation avoids hard Flask dependency here
    # ── STEP 1: Critical env guard (sys.exit on failure) ─────────────────────
    validate_critical_env(dict(os.environ))

    # ── STEP 2: Parse & validate all settings (ValidationError → sys.exit) ──
    try:
        settings: AppSettings       = get_settings()
        dotnet:   DotNetAPIConfig   = get_dotnet_config()
        keys:     APIKeysConfig     = get_api_keys()
        flags:    FeatureFlagConfig = get_feature_flags()
        timeouts: TimeoutConfig     = get_timeouts()
    except Exception as exc:
        import sys
        print(f"\n[FATAL] Configuration validation failed:\n{exc}\n", flush=True)
        sys.exit(1)

    # ── STEP 3: Bootstrap logging ─────────────────────────────────────────────
    log_level = logging.DEBUG if flags.ENABLE_DEBUG_LOGGING else logging.INFO
    logging.basicConfig(level=log_level)
    logger.info("Starting %s v%s [%s]", settings.APP_NAME, settings.APP_VERSION, settings.ENV)

    # ── STEP 4: Initialise Flask ──────────────────────────────────────────────
    from flask import Flask
    app = Flask(__name__)

    if flags.ENABLE_ADMIN_CHATBOT:
        logger.info("Admin chatbot routes registered.")

    if flags.ENABLE_WHATSAPP_BOT:
        if not keys.WHATSAPP_TOKEN:
            raise RuntimeError(
                "FEATURE_ENABLE_WHATSAPP_BOT is True but WHATSAPP_TOKEN is not set."
            )
        logger.info("WhatsApp bot routes registered.")

    if flags.ENABLE_SWAGGER_UI and not settings.is_production():
        logger.info("Swagger UI enabled at /api/docs")

    return app


if __name__ == "__main__":
    app = create_app()
    s = get_settings()
    app.run(host=s.HOST, port=s.PORT, debug=s.is_development())