import json
import logging
import unittest
from unittest.mock import MagicMock, patch

from admin_pipeline import execute_admin_query
from dotnet_executor import _call_dotnet


class TestObservability(unittest.TestCase):
    @patch('admin_pipeline._call_dotnet')
    @patch('admin_pipeline.generate_report')
    def test_trace_id_propagation_and_timings(self, mock_gen_report, mock_call_dotnet):
        # Mock successful .NET executor and report generator responses
        mock_call_dotnet.return_value = ({"records": []}, None)
        mock_gen_report.return_value = {
            "introMessage": "Intro",
            "analysis": {"summary": "Sum", "observations": [], "insights": []},
            "conclusion": {"summary": "Conc"}
        }

        trace_id = "test-trace-id-12345"
        session_id = "test-session-id"
        body = {"session_id": session_id}

        result = execute_admin_query(
            user_query="Show attendance",
            body=body,
            session_id=session_id,
            trace_id=trace_id
        )

        # 1. Assert trace_id propagated to internal calls
        mock_call_dotnet.assert_called_with({"category": "Attendance", "operation": "Monthly"}, trace_id=trace_id)
        mock_gen_report.assert_called_with(
            combined_result={"records": []},
            query_type="simple",
            intent=unittest.mock.ANY,
            user_query="Show attendance",
            trace_id=trace_id
        )

        # 2. Assert durations are present in the response metadata
        self.assertEqual(result["type"], "query")
        metadata = result["response_payload"]["metadata"]
        for key in ("planner_duration", "intent_duration", "dotnet_duration", "combiner_duration", "report_duration", "total_duration"):
            self.assertIn(key, metadata)
            self.assertIsInstance(metadata[key], (int, float))
            self.assertGreaterEqual(metadata[key], 0.0)

    @patch('dotnet_executor._dotnet_session.post')
    def test_structured_logging_and_scrubbing(self, mock_post):
        # Setup mock success response
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"records": [{"name": "Agniveer A"}]}
        mock_post.return_value = mock_resp

        trace_id = "log-trace-id"
        
        # Capture logs from both pipeline and executor
        logger_pipeline = logging.getLogger("admin_pipeline")
        logger_executor = logging.getLogger("dotnet_executor")
        
        with self.assertLogs(logger_pipeline, level='INFO') as log_pipeline, \
             self.assertLogs(logger_executor, level='INFO') as log_executor:
            
            # Run query
            result = execute_admin_query(
                user_query="Show attendance",
                body={"session_id": "test-session"},
                trace_id=trace_id
            )
            
            # Combine all intercepted log records
            all_log_messages = log_pipeline.output + log_executor.output
            
            # Verify that we generated JSON logs and they are properly structured
            json_logs_count = 0
            for log_msg in all_log_messages:
                # Strip out log prefix (e.g. 'INFO:admin_pipeline:') if it's there
                raw_json = log_msg.split(':', 2)[-1].strip()
                
                try:
                    parsed = json.loads(raw_json)
                    json_logs_count += 1
                    
                    # Verify presence of required structured log fields where applicable
                    self.assertIn("message", parsed)
                    if parsed.get("message") in ("Initiating .NET API call", "Query plan compiled", "Admin pipeline complete"):
                        self.assertIn("trace_id", parsed)
                        self.assertEqual(parsed["trace_id"], trace_id)
                    
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
                    self.assertNotIn("commandId", log_str_lower)
                    self.assertNotIn("agniveerId", log_str_lower)
                    self.assertNotIn("top 5 performers", log_str_lower)
                    
                except json.JSONDecodeError:
                    # Ignore generic non-observability warning logs from other frameworks
                    pass

            self.assertGreater(json_logs_count, 0)

    @patch('admin_pipeline._call_dotnet')
    def test_error_sanitization_logs(self, mock_call_dotnet):
        # Mock a failed .NET call that returns a detailed HTTP error containing a response body
        mock_call_dotnet.return_value = (None, "Backend returned HTTP 400: {'sensitive': 'secret_record_data'}")
        
        logger_pipeline = logging.getLogger("admin_pipeline")
        
        with self.assertLogs(logger_pipeline, level='WARNING') as log_pipeline:
            result = execute_admin_query(
                user_query="Show attendance",
                body={"session_id": "test-session"},
                trace_id="err-trace-id"
            )
            
            # Verify result is error
            self.assertEqual(result["type"], "error")
            
            # Verify logs
            found_log = False
            for log_msg in log_pipeline.output:
                raw_json = log_msg.split(':', 2)[-1].strip()
                try:
                    parsed = json.loads(raw_json)
                    if parsed.get("message") == "Admin .NET call failed":
                        found_log = True
                        self.assertEqual(parsed["trace_id"], "err-trace-id")
                        # Verify the error message is sanitized (contains HTTP 400, but not the body/sensitive info)
                        self.assertIn("Backend returned HTTP 400", parsed["error"])
                        self.assertNotIn("sensitive", parsed["error"])
                        self.assertNotIn("secret_record_data", parsed["error"])
                except json.JSONDecodeError:
                    pass
            self.assertTrue(found_log)


if __name__ == "__main__":
    unittest.main()
