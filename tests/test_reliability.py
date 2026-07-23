import time
import unittest
from unittest.mock import MagicMock, patch

import requests

import dotnet_executor
from admin_pipeline import execute_admin_query
from dotnet_executor import CircuitBreaker, _call_dotnet


class TestReliability(unittest.TestCase):
    def setUp(self):
        # Reset the global circuit breaker before each test to have a clean state.
        # We use a short recovery timeout of 0.1s for test speed.
        dotnet_executor._cb = CircuitBreaker(failure_threshold=5, recovery_timeout=0.1)

    @patch("dotnet_executor._dotnet_session.post")
    def test_timeout_protection(self, mock_post):
        # Setup mock success response
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"success": True}
        mock_post.return_value = mock_resp

        data, err = _call_dotnet({"cmd": "test"})
        self.assertIsNone(err)
        self.assertEqual(data, {"success": True})

        # Verify that timeout=(5, 30) is passed
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs.get("timeout"), (5, 30))

    @patch("dotnet_executor._dotnet_session.post")
    @patch("time.sleep")
    def test_retry_policy_transient_http(self, mock_sleep, mock_post):
        # Mock responses: 3 transient failures (503, 502, 429) followed by success
        mock_resp_503 = MagicMock(status_code=503)
        mock_resp_503.json.side_effect = Exception("No JSON")
        mock_resp_503.text = "Service Unavailable"

        mock_resp_502 = MagicMock(status_code=502)
        mock_resp_502.json.side_effect = Exception("No JSON")
        mock_resp_502.text = "Bad Gateway"

        mock_resp_429 = MagicMock(status_code=429)
        mock_resp_429.json.side_effect = Exception("No JSON")
        mock_resp_429.text = "Too Many Requests"

        mock_resp_200 = MagicMock(status_code=200)
        mock_resp_200.json.return_value = {"ok": True}

        mock_post.side_effect = [
            mock_resp_503,
            mock_resp_502,
            mock_resp_429,
            mock_resp_200,
        ]

        data, err = _call_dotnet({"cmd": "test"})
        self.assertIsNone(err)
        self.assertEqual(data, {"ok": True})
        self.assertEqual(mock_post.call_count, 4)

        # Assert sleep is called with 1s, 2s, 4s
        sleep_calls = [c[0][0] for c in mock_sleep.call_args_list]
        self.assertEqual(sleep_calls, [1.0, 2.0, 4.0])

    @patch("dotnet_executor._dotnet_session.post")
    @patch("time.sleep")
    def test_retry_policy_connection_error(self, mock_sleep, mock_post):
        # ConnectionError followed by Timeout followed by success
        mock_resp_200 = MagicMock(status_code=200)
        mock_resp_200.json.return_value = {"ok": True}

        mock_post.side_effect = [
            requests.ConnectionError("Refused"),
            requests.Timeout("Timeout"),
            mock_resp_200,
        ]

        data, err = _call_dotnet({"cmd": "test"})
        self.assertIsNone(err)
        self.assertEqual(data, {"ok": True})
        self.assertEqual(mock_post.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("dotnet_executor._dotnet_session.post")
    @patch("time.sleep")
    def test_no_retry_on_validation_errors(self, mock_sleep, mock_post):
        # 400 Bad Request
        mock_resp_400 = MagicMock(status_code=400)
        mock_resp_400.json.side_effect = Exception("No JSON")
        mock_resp_400.text = "Validation failed"
        mock_post.return_value = mock_resp_400

        data, err = _call_dotnet({"cmd": "invalid"})
        self.assertIsNotNone(err)
        self.assertIn("HTTP 400", err)
        mock_post.assert_called_once()
        mock_sleep.assert_not_called()

    @patch("dotnet_executor._dotnet_session.post")
    def test_circuit_breaker_trip_and_recover(self, mock_post):
        # Setup failure response
        mock_resp_500 = MagicMock(status_code=500)
        mock_resp_500.json.side_effect = Exception("No JSON")
        mock_resp_500.text = "Internal Error"
        mock_post.return_value = mock_resp_500

        # We fail 5 times. Each calls _call_dotnet which runs up to 4 attempts.
        # So we expect 5 calls to _call_dotnet to fail.
        for i in range(5):
            data, err = _call_dotnet({"cmd": "fail"})
            self.assertIsNotNone(err)

        # 6th call should be immediately blocked by the circuit breaker (no mock_post call)
        mock_post.reset_mock()
        data, err = _call_dotnet({"cmd": "blocked"})
        self.assertIsNone(data)
        self.assertEqual(
            err, "Circuit breaker is open. Backend is temporarily unreachable."
        )
        mock_post.assert_not_called()

        # Wait for recovery timeout (0.1s) to enter HALF-OPEN
        time.sleep(0.15)

        # In HALF-OPEN, make a successful request. It should transition to CLOSED.
        mock_resp_200 = MagicMock(status_code=200)
        mock_resp_200.json.return_value = {"recovered": True}
        mock_post.return_value = mock_resp_200

        data, err = _call_dotnet({"cmd": "test"})
        self.assertIsNone(err)
        self.assertEqual(data, {"recovered": True})
        mock_post.assert_called_once()

        # Breaker is CLOSED, next requests succeed normally
        mock_post.reset_mock()
        data, err = _call_dotnet({"cmd": "normal"})
        self.assertIsNone(err)
        mock_post.assert_called_once()

    @patch("dotnet_executor._dotnet_session.post")
    @patch("time.sleep")
    def test_circuit_breaker_validation_errors_dont_trip(self, mock_sleep, mock_post):
        # 400 validation error does NOT count as a system failure
        mock_resp_400 = MagicMock(status_code=400)
        mock_resp_400.json.side_effect = Exception("No JSON")
        mock_resp_400.text = "Validation error"
        mock_post.return_value = mock_resp_400

        for i in range(10):
            data, err = _call_dotnet({"cmd": "invalid"})
            self.assertIsNotNone(err)

        # Circuit breaker should still be CLOSED. Let's verify by checking a successful request gets through
        mock_post.reset_mock()
        mock_resp_200 = MagicMock(status_code=200)
        mock_resp_200.json.return_value = {"ok": True}
        mock_post.return_value = mock_resp_200

        data, err = _call_dotnet({"cmd": "valid"})
        self.assertIsNone(err)
        self.assertEqual(data, {"ok": True})
        mock_post.assert_called_once()

    @patch("admin_pipeline.generate_report")
    @patch("admin_pipeline.fetch_sql_results")
    def test_llm_failure_isolation(self, mock_fetch_sql, mock_gen_report):
        # SQL backend returns successfully
        section = {"success": True, "records": [], "data": [], "count": 0}
        mock_fetch_sql.return_value = ([section], [("Result", section)], None)

        # LLM raises exception
        mock_gen_report.side_effect = Exception("Ollama server down")

        # Execute query
        payload = execute_admin_query("Show me attendance for agniveer 12345", {})

        self.assertEqual(payload["type"], "query")
        response_payload = payload["response_payload"]
        # analysis/conclusion are now root-level strings, not inside widgets
        self.assertIsInstance(response_payload.get("analysis", ""), str)
        self.assertIsInstance(response_payload.get("conclusion", ""), str)
        self.assertEqual(response_payload["status"], True)
        self.assertIn("partial", response_payload["message"].lower())

    @patch("admin_pipeline.fetch_sql_results")
    def test_disqualified_lookup_backend_outage_is_graceful(self, mock_fetch_sql):
        # A SQL backend outage (DB unreachable, query rejected, etc.) has no
        # .NET-style "service_unavailable" path anymore — every SQL failure
        # degrades to the same friendly "couldn't understand" response, and
        # must never crash the pipeline or leak backend detail to the user.
        mock_fetch_sql.return_value = (
            [],
            [],
            "The generated query could not be executed against the database.",
        )

        payload = execute_admin_query("Show disqualified agniveers from Platoon 2.", {})

        self.assertEqual(payload["type"], "unrecognised")
        response_payload = payload["response_payload"]
        message = response_payload["message"].lower()
        self.assertNotIn("database", message)
        self.assertIn("rephrase", message)

    @patch("admin_pipeline.fetch_sql_results")
    def test_disqualified_lookup_does_not_carry_forward_stale_batch(self, mock_fetch_sql):
        section = {"success": True, "records": [], "data": [], "count": 0}
        mock_fetch_sql.return_value = ([section], [("Result", section)], None)

        with patch("admin_pipeline.context_engine.resolve") as mock_resolve:
            mock_resolve.return_value = type(
                "Resolved",
                (),
                {
                    "needs_clarification": False,
                    "clarification_question": None,
                    "context_source": "fresh",
                    "resolved_query": "Show all disqualified agniveers.",
                    "carry_forward_filters": {"batchId": 99},
                    "resolved_entities": {},
                },
            )()
            execute_admin_query("Show all disqualified agniveers.", {})

        primary_intent = mock_fetch_sql.call_args[0][2]
        self.assertNotIn("batchId", primary_intent)

    @patch("admin_pipeline.fetch_sql_results")
    def test_disqualified_lookup_sql_error_is_reported_gracefully(self, mock_fetch_sql):
        mock_fetch_sql.return_value = (
            [],
            [],
            "Statement contains a forbidden keyword.",
        )

        payload = execute_admin_query("Show all disqualified agniveers.", {})

        # The raw validator/backend error text is never shown to the user —
        # only a friendly, conversational message (it's still logged
        # server-side).
        self.assertEqual(payload["type"], "unrecognised")
        response_payload = payload["response_payload"]
        self.assertNotIn("forbidden keyword", response_payload["message"])
        self.assertIn("rephrase", response_payload["message"].lower())

    def test_progress_callback_protection(self):
        # Custom progress callback that raises an exception
        def bad_progress(stage):
            raise RuntimeError("Progress callback failure")

        # Mock fetch_sql_results to succeed immediately
        with patch("admin_pipeline.fetch_sql_results") as mock_fetch_sql:
            section = {"success": True, "records": [], "data": [], "count": 0}
            mock_fetch_sql.return_value = ([section], [("Result", section)], None)

            # This should run to completion and not raise any exceptions
            result = execute_admin_query(
                user_query="Show attendance for agniveer 12345",
                body={},
                progress_callback=bad_progress,
            )
            self.assertEqual(result["type"], "query")


if __name__ == "__main__":
    unittest.main()
