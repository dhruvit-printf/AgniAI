"""
tests/test_config.py
Tests for AgniAI settings.py — validates what actually exists.
"""
import os
import sys
import pytest
from unittest.mock import patch


def test_validate_critical_env_passes_with_dotnet_url():
    from settings import validate_critical_env
    env = {"DOTNET_API_BASE_URL": "http://localhost:5001"}
    # Should not raise or sys.exit
    validate_critical_env(env)


def test_validate_critical_env_exits_without_dotnet_url():
    from settings import validate_critical_env
    with pytest.raises(SystemExit) as exc:
        validate_critical_env({})
    assert exc.value.code == 1


def test_validate_critical_env_empty_string_treated_as_missing():
    from settings import validate_critical_env
    with pytest.raises(SystemExit):
        validate_critical_env({"DOTNET_API_BASE_URL": ""})


def test_dotnet_api_config_valid_url():
    from settings import DotNetAPIConfig
    with patch.dict(os.environ, 
                    {"DOTNET_API_BASE_URL": "http://localhost:5001"}, 
                    clear=True):
        cfg = DotNetAPIConfig(_env_file=None)
    assert "localhost" in cfg.BASE_URL


def test_dotnet_api_config_strips_trailing_slash():
    from settings import DotNetAPIConfig
    with patch.dict(os.environ,
                    {"DOTNET_API_BASE_URL": "http://localhost:5001/"},
                    clear=True):
        cfg = DotNetAPIConfig(_env_file=None)
    assert not cfg.BASE_URL.endswith("/")


def test_dotnet_api_config_rejects_non_http_scheme():
    from settings import DotNetAPIConfig
    from pydantic import ValidationError
    with patch.dict(os.environ,
                    {"DOTNET_API_BASE_URL": "ftp://localhost:5001"},
                    clear=True):
        with pytest.raises(ValidationError):
            DotNetAPIConfig(_env_file=None)


def test_feature_flag_defaults_are_safe():
    from settings import FeatureFlagConfig
    with patch.dict(os.environ, {}, clear=True):
        cfg = FeatureFlagConfig(_env_file=None)
    assert cfg.ENABLE_RAG is True
    assert cfg.ENABLE_DEBUG_LOGGING is False
    assert cfg.ENABLE_ADMIN_CHATBOT is True
    assert cfg.ENABLE_WHATSAPP_BOT is False


def test_feature_flag_bool_parsing():
    from settings import FeatureFlagConfig
    with patch.dict(os.environ,
                    {"FEATURE_ENABLE_WHATSAPP_BOT": "true"},
                    clear=True):
        cfg = FeatureFlagConfig(_env_file=None)
    assert cfg.ENABLE_WHATSAPP_BOT is True


def test_timeout_config_defaults():
    from settings import TimeoutConfig
    with patch.dict(os.environ, {}, clear=True):
        cfg = TimeoutConfig(_env_file=None)
    assert cfg.LLM_INFERENCE == 60.0
    assert cfg.RAG_RETRIEVAL == 10.0
    assert cfg.HEALTH_CHECK == 3.0


def test_timeout_config_rejects_too_low_llm():
    from settings import TimeoutConfig
    from pydantic import ValidationError
    with patch.dict(os.environ,
                    {"TIMEOUT_LLM_INFERENCE": "5.0"},
                    clear=True):
        with pytest.raises(ValidationError):
            TimeoutConfig(_env_file=None)


def test_app_settings_loads_with_minimal_env():
    from settings import AppSettings
    with patch.dict(os.environ,
                    {"DOTNET_API_BASE_URL": "http://localhost:5001"},
                    clear=True):
        s = AppSettings(_env_file=None)
    assert s.APP_NAME == "AgniAI"
    assert s.PORT == 5000


def test_get_settings_is_cached():
    from settings import get_settings
    get_settings.cache_clear()
    with patch.dict(os.environ,
                    {"DOTNET_API_BASE_URL": "http://localhost:5001"},
                    clear=True):
        a = get_settings()
        b = get_settings()
    assert a is b
    get_settings.cache_clear()


def test_critical_vars_contains_only_dotnet():
    from settings import CRITICAL_VARS
    assert "DOTNET_API_BASE_URL" in CRITICAL_VARS
    # AgniAI has no database; these must NOT be critical
    assert "DATABASE_URL" not in CRITICAL_VARS
    assert "INTERNAL_SERVICE_KEY" not in CRITICAL_VARS


