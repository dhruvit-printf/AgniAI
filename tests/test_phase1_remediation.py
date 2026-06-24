from __future__ import annotations

import importlib
import logging
import os
import sys
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import cache_manager as cache_manager_module
import dotnet_executor
from admin_pipeline import execute_admin_query


@contextmanager
def _without_testing_modules():
    removed = {}
    for name in ("unittest", "pytest"):
        if name in sys.modules:
            removed[name] = sys.modules.pop(name)
    try:
        yield
    finally:
        sys.modules.update(removed)


class TestF1CacheScopeIsolation(unittest.TestCase):
    def setUp(self) -> None:
        cache_manager_module.cache_manager._memory_cache.clear()
        cache_manager_module.cache_manager._redis_client = None

    def test_get_query_hash_includes_scope(self):
        base_query = "show top performers"
        hash_company_1 = cache_manager_module.cache_manager.get_query_hash(
            base_query, scope={"companyId": 1}
        )
        hash_company_2 = cache_manager_module.cache_manager.get_query_hash(
            base_query, scope={"companyId": 2}
        )

        self.assertNotEqual(hash_company_1, hash_company_2)

    @patch("admin_pipeline._call_dotnet")
    @patch("admin_pipeline.generate_report")
    def test_execute_admin_query_does_not_bleed_across_company_scope(
        self, mock_generate_report, mock_call_dotnet
    ):
        mock_generate_report.return_value = {
            "introMessage": "Report generated.",
            "analysis": {"summary": "Summary", "observations": [], "insights": []},
            "conclusion": {"summary": "Conclusion"},
        }
        mock_call_dotnet.side_effect = [
            (
                [
                    {
                        "agniveerNo": "A01",
                        "fullName": "COMPANY ONE",
                        "bestTotal": 95,
                    }
                ],
                None,
            ),
            (
                [
                    {
                        "agniveerNo": "B01",
                        "fullName": "COMPANY TWO",
                        "bestTotal": 88,
                    }
                ],
                None,
            ),
        ]

        with _without_testing_modules():
            first = execute_admin_query(
                "Show top performers in PPT",
                {"companyId": 1, "session_id": "sess-1"},
                trace_id="trace-1",
            )
            second = execute_admin_query(
                "Show top performers in PPT",
                {"companyId": 2, "session_id": "sess-1"},
                trace_id="trace-2",
            )

        first_records = first["response_payload"]["formattedData"]["data"]["rows"]
        second_records = second["response_payload"]["formattedData"]["data"]["rows"]

        self.assertEqual(first_records[0]["fullName"], "COMPANY ONE")   # camelCase after normalisation
        self.assertEqual(second_records[0]["fullName"], "COMPANY TWO")   # camelCase after normalisation
        self.assertEqual(mock_call_dotnet.call_count, 2)


class TestF3SslVerification(unittest.TestCase):
    def test_default_verification_is_enabled(self):
        with patch.dict(
            os.environ,
            {
                "DOTNET_SKIP_SSL_VERIFY": "",
                "DOTNET_VERIFY_SSL": "",
            },
            clear=False,
        ):
            reloaded = importlib.reload(dotnet_executor)
        self.assertTrue(reloaded.DOTNET_VERIFY_SSL)

    def test_skip_flag_disables_verification_and_logs_warning(self):
        with patch.dict(os.environ, {"DOTNET_SKIP_SSL_VERIFY": "1"}, clear=False):
            with self.assertLogs(logging.getLogger("dotnet_executor"), level="WARNING"):
                reloaded = importlib.reload(dotnet_executor)

        self.assertFalse(reloaded.DOTNET_VERIFY_SSL)


if __name__ == "__main__":
    unittest.main()
