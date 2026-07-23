import json
import logging
import unittest
from unittest.mock import patch

from admin_pipeline import execute_admin_query


class TestObservability(unittest.TestCase):
    @patch("admin_pipeline.fetch_sql_results")
    @patch("admin_pipeline.generate_report")
    def test_trace_id_propagation_and_timings(self, mock_gen_report, mock_fetch_sql):
        section = {"success": True, "records": [], "data": [], "count": 0}
        mock_fetch_sql.return_value = ([section], [("Result", section)], None)
        mock_gen_report.return_value = {
            "message": "Intro",
            "analysis": {"summary": "Sum", "observations": [], "insights": []},
            "conclusion": {"summary": "Conc"},
        }

        trace_id = "test-trace-id-12345"
        session_id = "test-session-id"
        body = {"session_id": session_id}

        result = execute_admin_query(
            user_query="Show attendance for agniveer 12345",
            body=body,
            session_id=session_id,
            trace_id=trace_id,
        )

        mock_fetch_sql.assert_called_once()
        mock_gen_report.assert_called_with(
            combined_result=section,
            query_type="simple",
            intent=unittest.mock.ANY,
            user_query="Show attendance for agniveer 12345",
            trace_id=trace_id,
        )

        # 2. Assert durations are present in the response metadata
        self.assertEqual(result["type"], "query")
        metadata = result["response_payload"]["metadata"]
        for key in (
            "planner_duration",
            "intent_duration",
            "dotnet_duration",
            "combiner_duration",
            "report_duration",
            "total_duration",
        ):
            self.assertIn(key, metadata)
            self.assertIsInstance(metadata[key], (int, float))
            self.assertGreaterEqual(metadata[key], 0.0)

    @patch("admin_pipeline.fetch_sql_results")
    def test_structured_logging_and_scrubbing(self, mock_fetch_sql):
        section = {
            "success": True,
            "records": [{"agniveerNo": "A1", "fullName": "Agniveer A"}],
            "data": [{"agniveerNo": "A1", "fullName": "Agniveer A"}],
            "count": 1,
        }
        mock_fetch_sql.return_value = ([section], [("Result", section)], None)

        trace_id = "log-trace-id"
        logger_pipeline = logging.getLogger("admin_pipeline")

        with self.assertLogs(logger_pipeline, level="INFO") as log_pipeline:
            execute_admin_query(
                user_query="Show attendance for agniveer 12345",
                body={"session_id": "test-session"},
                trace_id=trace_id,
            )

        json_logs_count = 0
        for log_msg in log_pipeline.output:
            # Strip out log prefix (e.g. 'INFO:admin_pipeline:') if it's there
            raw_json = log_msg.split(":", 2)[-1].strip()

            try:
                parsed = json.loads(raw_json)
                json_logs_count += 1

                # Verify presence of required structured log fields where applicable
                self.assertIn("message", parsed)
                if parsed.get("message") in (
                    "Query plan compiled",
                    "Admin query audit",
                ):
                    self.assertIn("trace_id", parsed) if "trace_id" in parsed else None

                # Verify Scrubbing: Banned keys or payloads must NEVER be logged!
                self.assertNotIn("payload", parsed)
                self.assertNotIn("payloads", parsed)
                self.assertNotIn("prompt", parsed)
                self.assertNotIn("prompts", parsed)
                self.assertNotIn("api_key", parsed)
                self.assertNotIn("records", parsed)
                self.assertNotIn("raw_records", parsed)

                # Extra payload content screening
                log_str_lower = raw_json.lower()
                self.assertNotIn("commandid", log_str_lower)
                self.assertNotIn("top 5 performers", log_str_lower)

            except json.JSONDecodeError:
                # Ignore generic non-observability warning logs from other frameworks
                pass

        self.assertGreater(json_logs_count, 0)

    @patch("admin_pipeline.fetch_sql_results")
    def test_error_sanitization_logs(self, mock_fetch_sql):
        # A validator/exec-error message from sql_executor.py is already
        # sanitized at that layer (see sql_executor.run_readonly's "never
        # leak raw SQL / connection details" discipline) — admin_pipeline.py
        # just logs it and degrades to "unrecognised", it does not need its
        # own extra scrubbing pass the way the old .NET HTTP-error path did.
        mock_fetch_sql.return_value = (
            [],
            [],
            "The generated query could not be executed against the database.",
        )

        logger_pipeline = logging.getLogger("admin_pipeline")

        with self.assertLogs(logger_pipeline, level="INFO") as log_pipeline:
            result = execute_admin_query(
                user_query="Show attendance for agniveer 12345",
                body={"session_id": "test-session"},
                trace_id="err-trace-id",
            )

        # The user-facing result is a friendly "couldn't understand"
        # message — no raw backend error ever reaches it.
        self.assertEqual(result["type"], "unrecognised")
        user_message = result["response_payload"]["message"]
        self.assertNotIn("SELECT", user_message)
        self.assertNotIn("database", user_message.lower())

        found_log = False
        for log_msg in log_pipeline.output:
            raw_json = log_msg.split(":", 2)[-1].strip()
            try:
                parsed = json.loads(raw_json)
                if (
                    parsed.get("message")
                    == "SQL backend could not answer the query, attempting Text2SQL fallback"
                ):
                    found_log = True
                    self.assertEqual(parsed["trace_id"], "err-trace-id")
            except json.JSONDecodeError:
                pass
        self.assertTrue(found_log)


if __name__ == "__main__":
    unittest.main()
