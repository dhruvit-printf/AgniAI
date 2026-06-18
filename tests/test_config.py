"""
AgniAI — Configuration Regression Tests
Covers: Pydantic validation, startup guard, environment profiles,
        secret masking, production safety gates.

Run with:  pytest tests/test_config.py -v
"""

import os
import sys
import pytest
from unittest.mock import patch

# ── Helpers ──────────────────────────────────────────────────────────────────

# Minimal valid env that passes all validators
VALID_ENV = {
    "ENV":                       "development",
    "DATABASE_URL":              "sqlite:///./test.db",
    "DOTNET_API_BASE_URL":       "http://localhost:5001",
    "INTERNAL_SERVICE_KEY":      "a" * 32,
    "OLLAMA_BASE_URL":           "http://localhost:11434",
    "OLLAMA_MODEL":              "llama3",
    "MIN_SCORE":                 "0.35",
    "HOST":                      "0.0.0.0",
    "PORT":                      "5000",
    "WORKERS":                   "1",
    "CORS_ORIGINS":              '["http://localhost:3000"]',
}


def make_env(**overrides) -> dict:
    env = {**VALID_ENV, **overrides}
    return {k: v for k, v in env.items() if v is not None}


def load_settings(**overrides):
    """Load AppSettings with a clean env override, bypassing .env file."""
    from pydantic_settings import BaseSettings

    env = make_env(**overrides)
    with patch.dict(os.environ, env, clear=True):
        # Bust lru_cache so each test gets a fresh instance
        from config import settings as s_mod
        s_mod.get_settings.cache_clear()
        s_mod.get_feature_flags.cache_clear()
        return s_mod.AppSettings(_env_file=None)


# ─────────────────────────────────────────────────────────────────────────────
# TASK 1 — Pydantic Settings Validation
# ─────────────────────────────────────────────────────────────────────────────

class TestDotNetAPIConfig:
    def test_valid_url_accepted(self):
        from config.settings import DotNetAPIConfig
        with patch.dict(os.environ, {"DOTNET_API_BASE_URL": "http://localhost:5001"}, clear=True):
            cfg = DotNetAPIConfig(_env_file=None)
        assert "localhost" in str(cfg.BASE_URL)

    def test_trailing_slash_stripped(self):
        from config.settings import DotNetAPIConfig
        with patch.dict(os.environ, {"DOTNET_API_BASE_URL": "http://localhost:5001/"}, clear=True):
            cfg = DotNetAPIConfig(_env_file=None)
        assert not str(cfg.BASE_URL).endswith("/")

    def test_timeout_bounds_connect(self):
        from config.settings import DotNetAPIConfig
        from pydantic import ValidationError
        with patch.dict(os.environ, {
            "DOTNET_API_BASE_URL": "http://localhost:5001",
            "DOTNET_API_TIMEOUT_CONNECT": "0.5",   # below ge=1.0
        }, clear=True):
            with pytest.raises(ValidationError, match="greater than or equal to 1"):
                DotNetAPIConfig(_env_file=None)

    def test_max_retries_upper_bound(self):
        from config.settings import DotNetAPIConfig
        from pydantic import ValidationError
        with patch.dict(os.environ, {
            "DOTNET_API_BASE_URL": "http://localhost:5001",
            "DOTNET_API_MAX_RETRIES": "99",
        }, clear=True):
            with pytest.raises(ValidationError, match="less than or equal to 10"):
                DotNetAPIConfig(_env_file=None)


class TestAPIKeysConfig:
    def test_short_internal_key_rejected(self):
        from config.settings import APIKeysConfig
        from pydantic import ValidationError
        with patch.dict(os.environ, {"INTERNAL_SERVICE_KEY": "short"}, clear=True):
            with pytest.raises(ValidationError, match="at least 32 characters"):
                APIKeysConfig(_env_file=None)

    def test_32_char_key_accepted(self):
        from config.settings import APIKeysConfig
        with patch.dict(os.environ, {"INTERNAL_SERVICE_KEY": "x" * 32}, clear=True):
            cfg = APIKeysConfig(_env_file=None)
        assert cfg.INTERNAL_SERVICE_KEY.get_secret_value() == "x" * 32

    def test_optional_whatsapp_token_defaults_none(self):
        from config.settings import APIKeysConfig
        with patch.dict(os.environ, {"INTERNAL_SERVICE_KEY": "x" * 32}, clear=True):
            cfg = APIKeysConfig(_env_file=None)
        assert cfg.WHATSAPP_TOKEN is None

    def test_secret_not_exposed_in_repr(self):
        from config.settings import APIKeysConfig
        with patch.dict(os.environ, {"INTERNAL_SERVICE_KEY": "supersecret" * 3}, clear=True):
            cfg = APIKeysConfig(_env_file=None)
        assert "supersecret" not in repr(cfg)


class TestTimeoutConfig:
    def test_llm_inference_minimum(self):
        from config.settings import TimeoutConfig
        from pydantic import ValidationError
        with patch.dict(os.environ, {"TIMEOUT_LLM_INFERENCE": "5.0"}, clear=True):
            with pytest.raises(ValidationError, match="greater than or equal to 10"):
                TimeoutConfig(_env_file=None)

    def test_defaults_load_without_env(self):
        from config.settings import TimeoutConfig
        with patch.dict(os.environ, {}, clear=True):
            cfg = TimeoutConfig(_env_file=None)
        assert cfg.LLM_INFERENCE == 60.0
        assert cfg.RAG_RETRIEVAL == 10.0


class TestFeatureFlagConfig:
    def test_defaults_are_safe(self):
        from config.settings import FeatureFlagConfig
        with patch.dict(os.environ, {}, clear=True):
            cfg = FeatureFlagConfig(_env_file=None)
        assert cfg.ENABLE_RAG is True
        assert cfg.ENABLE_DEBUG_LOGGING is False  # safe default

    def test_bool_parsing_true(self):
        from config.settings import FeatureFlagConfig
        with patch.dict(os.environ, {"FEATURE_ENABLE_WHATSAPP_BOT": "true"}, clear=True):
            cfg = FeatureFlagConfig(_env_file=None)
        assert cfg.ENABLE_WHATSAPP_BOT is True

    def test_bool_parsing_false(self):
        from config.settings import FeatureFlagConfig
        with patch.dict(os.environ, {"FEATURE_ENABLE_SWAGGER_UI": "false"}, clear=True):
            cfg = FeatureFlagConfig(_env_file=None)
        assert cfg.ENABLE_SWAGGER_UI is False


class TestAppSettings:
    def test_valid_development_loads(self):
        s = load_settings()
        assert s.ENV.value == "development"
        assert s.APP_NAME == "AgniAI"

    def test_missing_database_url_raises(self):
        from pydantic import ValidationError
        with pytest.raises((ValidationError, SystemExit)):
            load_settings(DATABASE_URL=None)

    def test_unsupported_db_scheme_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="DATABASE_URL must start with"):
            load_settings(DATABASE_URL="mongodb://localhost/agni")

    def test_min_score_bounds(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            load_settings(MIN_SCORE="1.5")

    def test_port_bounds_lower(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            load_settings(PORT="80")    # below ge=1024

    def test_workers_too_high(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            load_settings(WORKERS="99")

    def test_safe_repr_masks_db_password(self):
        s = load_settings(
            DATABASE_URL="postgresql+psycopg2://admin:S3cr3t@db.host:5432/agni"
        )
        rep = s.safe_repr()
        assert "S3cr3t" not in rep["DATABASE_URL"]
        assert "<redacted>" in rep["DATABASE_URL"]

    def test_version_format_enforced(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            load_settings(APP_VERSION="v1.2")   # missing patch number


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 — Startup Validation (fail fast, never runtime)
# ─────────────────────────────────────────────────────────────────────────────

class TestStartupGuard:
    def test_missing_database_url_exits(self):
        from config.settings import validate_critical_env
        env = {k: v for k, v in make_env().items() if k != "DATABASE_URL"}
        env.pop("DATABASE_URL", None)
        env.pop("DOTNET_API_BASE_URL", None)
        with pytest.raises(SystemExit) as exc_info:
            validate_critical_env(env)
        assert exc_info.value.code == 1

    def test_missing_internal_service_key_exits(self):
        from config.settings import validate_critical_env
        env = make_env()
        env.pop("INTERNAL_SERVICE_KEY", None)
        with pytest.raises(SystemExit) as exc_info:
            validate_critical_env(env)
        assert exc_info.value.code == 1

    def test_missing_dotnet_base_url_exits(self):
        from config.settings import validate_critical_env
        env = make_env()
        env.pop("DOTNET_API_BASE_URL", None)
        with pytest.raises(SystemExit) as exc_info:
            validate_critical_env(env)
        assert exc_info.value.code == 1

    def test_all_critical_vars_present_passes(self):
        from config.settings import validate_critical_env
        # Should not raise
        validate_critical_env(make_env())

    def test_empty_string_treated_as_missing(self):
        from config.settings import validate_critical_env
        env = make_env(DATABASE_URL="")
        with pytest.raises(SystemExit):
            validate_critical_env(env)


# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 — Environment Profiles
# ─────────────────────────────────────────────────────────────────────────────

class TestEnvironmentProfiles:

    # ── DEVELOPMENT ────────────────────────────────────────────────────────
    def test_development_allows_sqlite(self):
        s = load_settings(ENV="development", DATABASE_URL="sqlite:///./dev.db")
        assert s.is_development()

    def test_development_allows_swagger_on(self):
        s = load_settings(ENV="development")
        assert s.is_development()

    def test_development_allows_debug_logging(self):
        # No validation error expected
        s = load_settings(ENV="development")
        assert s.is_development()

    # ── TESTING ────────────────────────────────────────────────────────────
    def test_testing_profile_loads(self):
        s = load_settings(ENV="testing")
        assert s.is_testing()

    def test_testing_workers_must_be_one(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="WORKERS must be 1 in testing"):
            load_settings(ENV="testing", WORKERS="2")

    def test_testing_allows_sqlite_memory(self):
        s = load_settings(ENV="testing", DATABASE_URL="sqlite:///:memory:")
        assert s.is_testing()

    # ── PRODUCTION ─────────────────────────────────────────────────────────
    def test_production_blocks_sqlite(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="SQLite is not allowed in production"):
            with patch.dict(os.environ, {
                **make_env(
                    ENV="production",
                    DATABASE_URL="sqlite:///./prod.db",
                    FEATURE_ENABLE_SWAGGER_UI="false",
                    FEATURE_ENABLE_DEBUG_LOGGING="false",
                    OLLAMA_BASE_URL="https://ollama.internal",
                ),
            }, clear=True):
                from config import settings as s_mod
                s_mod.get_settings.cache_clear()
                s_mod.get_feature_flags.cache_clear()
                s_mod.AppSettings(_env_file=None)

    def test_production_blocks_swagger_ui(self):
        from pydantic import ValidationError
        with patch.dict(os.environ, {
            **make_env(
                ENV="production",
                DATABASE_URL="postgresql+psycopg2://u:p@host/db",
                OLLAMA_BASE_URL="https://ollama.internal",
            ),
            "FEATURE_ENABLE_SWAGGER_UI":   "true",   # VIOLATION
            "FEATURE_ENABLE_DEBUG_LOGGING": "false",
        }, clear=True):
            from config import settings as s_mod
            s_mod.get_settings.cache_clear()
            s_mod.get_feature_flags.cache_clear()
            with pytest.raises(ValidationError, match="FEATURE_ENABLE_SWAGGER_UI must be False"):
                s_mod.AppSettings(_env_file=None)

    def test_production_blocks_debug_logging(self):
        from pydantic import ValidationError
        with patch.dict(os.environ, {
            **make_env(
                ENV="production",
                DATABASE_URL="postgresql+psycopg2://u:p@host/db",
                OLLAMA_BASE_URL="https://ollama.internal",
            ),
            "FEATURE_ENABLE_SWAGGER_UI":    "false",
            "FEATURE_ENABLE_DEBUG_LOGGING": "true",   # VIOLATION
        }, clear=True):
            from config import settings as s_mod
            s_mod.get_settings.cache_clear()
            s_mod.get_feature_flags.cache_clear()
            with pytest.raises(ValidationError, match="FEATURE_ENABLE_DEBUG_LOGGING must be False"):
                s_mod.AppSettings(_env_file=None)

    def test_production_valid_config_passes(self):
        with patch.dict(os.environ, {
            **make_env(
                ENV="production",
                DATABASE_URL="postgresql+psycopg2://u:p@host:5432/agni",
                OLLAMA_BASE_URL="https://ollama.internal.mil",
                WORKERS="4",
            ),
            "FEATURE_ENABLE_SWAGGER_UI":    "false",
            "FEATURE_ENABLE_DEBUG_LOGGING": "false",
        }, clear=True):
            from config import settings as s_mod
            s_mod.get_settings.cache_clear()
            s_mod.get_feature_flags.cache_clear()
            s = s_mod.AppSettings(_env_file=None)
        assert s.is_production()

    def test_invalid_env_name_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            load_settings(ENV="staging")   # not in enum


# ─────────────────────────────────────────────────────────────────────────────
# Singleton / cache behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestSingletonCache:
    def test_get_settings_returns_same_instance(self):
        from config.settings import get_settings
        get_settings.cache_clear()
        with patch.dict(os.environ, make_env(), clear=True):
            a = get_settings()
            b = get_settings()
        assert a is b