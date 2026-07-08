"""
tests/test_feature_flags.py
Unit tests for feature_flags.py — centralized feature flag system.
"""

import os
from unittest.mock import patch

import pytest


class TestFeatureFlagsDefaults:
    def setup_method(self):
        return None

    def teardown_method(self):
        return None

    def test_enable_reports_default_true(self):
        with patch.dict(os.environ, {}, clear=True):
            from feature_flags import FeatureFlags

            f = FeatureFlags(_env_file=None)
        assert f.ENABLE_REPORTS is True

    def test_enable_ollama_default_true(self):
        with patch.dict(os.environ, {}, clear=True):
            from feature_flags import FeatureFlags

            f = FeatureFlags(_env_file=None)
        assert f.ENABLE_OLLAMA is True

    def test_enable_streaming_default_true(self):
        with patch.dict(os.environ, {}, clear=True):
            from feature_flags import FeatureFlags

            f = FeatureFlags(_env_file=None)
        assert f.ENABLE_STREAMING is True

    def test_enable_metrics_default_true(self):
        with patch.dict(os.environ, {}, clear=True):
            from feature_flags import FeatureFlags

            f = FeatureFlags(_env_file=None)
        assert f.ENABLE_METRICS is True

    def test_enable_audit_logging_default_true(self):
        with patch.dict(os.environ, {}, clear=True):
            from feature_flags import FeatureFlags

            f = FeatureFlags(_env_file=None)
        assert f.ENABLE_AUDIT_LOGGING is True

    def test_enable_opentelemetry_default_false(self):
        """OTel is opt-in — default must be False."""
        with patch.dict(os.environ, {}, clear=True):
            from feature_flags import FeatureFlags

            f = FeatureFlags(_env_file=None)
        assert f.ENABLE_OPENTELEMETRY is False

    def test_enable_sentry_default_false(self):
        """Sentry is opt-in — default must be False."""
        with patch.dict(os.environ, {}, clear=True):
            from feature_flags import FeatureFlags

            f = FeatureFlags(_env_file=None)
        assert f.ENABLE_SENTRY is False


class TestFeatureFlagsEnvOverride:
    def test_disable_reports_via_env(self):
        with patch.dict(os.environ, {"ENABLE_REPORTS": "false"}, clear=True):
            from feature_flags import FeatureFlags

            f = FeatureFlags(_env_file=None)
        assert f.ENABLE_REPORTS is False

    def test_enable_sentry_via_env(self):
        with patch.dict(os.environ, {"ENABLE_SENTRY": "true"}, clear=True):
            from feature_flags import FeatureFlags

            f = FeatureFlags(_env_file=None)
        assert f.ENABLE_SENTRY is True

    def test_enable_otel_via_env(self):
        with patch.dict(os.environ, {"ENABLE_OPENTELEMETRY": "1"}, clear=True):
            from feature_flags import FeatureFlags

            f = FeatureFlags(_env_file=None)
        assert f.ENABLE_OPENTELEMETRY is True


class TestGetFlagsSingleton:
    def setup_method(self):
        return None

    def teardown_method(self):
        return None

    def test_get_flags_returns_feature_flags_instance(self):
        from feature_flags import FeatureFlags, get_flags

        f = get_flags()
        assert isinstance(f, FeatureFlags)

    def test_get_flags_is_cached(self):
        from feature_flags import get_flags

        a = get_flags()
        b = get_flags()
        assert a is not b

    def test_module_level_flags_alias(self):
        import importlib

        import feature_flags

        importlib.reload(feature_flags)
        assert feature_flags.flags is not feature_flags.get_flags()


class TestDegradeGracefully:
    def test_degrade_gracefully_true_flag(self):
        from feature_flags import FeatureFlags

        f = FeatureFlags(_env_file=None)
        result = f.degrade_gracefully("ENABLE_REPORTS")
        assert result is True

    def test_degrade_gracefully_false_flag(self):
        with patch.dict(os.environ, {"ENABLE_SENTRY": "false"}, clear=True):
            from feature_flags import FeatureFlags

            f = FeatureFlags(_env_file=None)
        result = f.degrade_gracefully("ENABLE_SENTRY")
        assert result is False

    def test_degrade_gracefully_missing_attr(self):
        from feature_flags import FeatureFlags

        f = FeatureFlags(_env_file=None)
        result = f.degrade_gracefully("NONEXISTENT_FLAG")
        assert result is False
