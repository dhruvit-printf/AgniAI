import json
import logging
import unittest
from unittest.mock import MagicMock, patch

from app import app
from admin_pipeline import execute_admin_query
from metrics import metrics_collector


class TestMetricsHealth(unittest.TestCase):
    def setUp(self):
        self.app_client = app.test_client()
        # Reset metrics collector values for testing
        metrics_collector.requests_total.clear()
        metrics_collector.errors_total.clear()
        for name in metrics_collector.durations:
            metrics_collector.durations[name]["sum"] = 0.0
            metrics_collector.durations[name]["count"] = 0.0

    @patch('admin_routes.check_python_health')
    @patch('admin_routes.check_dotnet_health')
    @patch('admin_routes.check_llm_health')
    @patch('admin_routes.check_database_health')
    def test_health_endpoint_all_healthy(self, mock_db, mock_llm, mock_dotnet, mock_py):
        mock_py.return_value = "healthy"
        mock_dotnet.return_value = "healthy"
        mock_llm.return_value = "healthy"
        mock_db.return_value = "healthy"

        resp = self.app_client.get('/api/admin/health')
        self.assertEqual(resp.status_code, 200)
        
        data = json.loads(resp.data.decode('utf-8'))
        self.assertEqual(data["python"], "healthy")
        self.assertEqual(data["dotnet"], "healthy")
        self.assertEqual(data["llm"], "healthy")
        self.assertEqual(data["database"], "healthy")

    @patch('admin_routes.check_python_health')
    @patch('admin_routes.check_dotnet_health')
    @patch('admin_routes.check_llm_health')
    @patch('admin_routes.check_database_health')
    def test_health_endpoint_partial_unhealthy(self, mock_db, mock_llm, mock_dotnet, mock_py):
        mock_py.return_value = "healthy"
        mock_dotnet.return_value = "unhealthy"  # dotnet is down
        mock_llm.return_value = "healthy"
        mock_db.return_value = "healthy"

        resp = self.app_client.get('/api/admin/health')
        self.assertEqual(resp.status_code, 503)
        
        data = json.loads(resp.data.decode('utf-8'))
        self.assertEqual(data["python"], "healthy")
        self.assertEqual(data["dotnet"], "unhealthy")
        self.assertEqual(data["llm"], "healthy")
        self.assertEqual(data["database"], "healthy")

    def test_prometheus_metrics_format_and_export(self):
        # Trigger some metric values
        metrics_collector.inc_requests("simple")
        metrics_collector.inc_errors("comparison")
        metrics_collector.record_duration("pipeline_duration", 150.0)

        resp = self.app_client.get('/metrics')
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/plain", resp.headers.get("Content-Type", ""))

        text = resp.data.decode('utf-8')
        
        # Verify Prometheus HELP and TYPE definitions exist
        self.assertIn("# HELP requests_total", text)
        self.assertIn("# TYPE requests_total counter", text)
        self.assertIn('requests_total{query_type="simple"} 1', text)

        self.assertIn("# HELP errors_total", text)
        self.assertIn("# TYPE errors_total counter", text)
        self.assertIn('errors_total{query_type="comparison"} 1', text)

        self.assertIn("# HELP active_websockets", text)
        self.assertIn("# TYPE active_websockets gauge", text)
        self.assertIn("active_websockets 0", text)

        self.assertIn("# HELP pipeline_duration", text)
        self.assertIn("# TYPE pipeline_duration summary", text)
        self.assertIn("pipeline_duration_sum 150.0", text)
        self.assertIn("pipeline_duration_count 1", text)

    @patch('admin_pipeline._call_dotnet')
    @patch('admin_pipeline.generate_report')
    def test_metrics_updated_on_pipeline_run(self, mock_gen_report, mock_call_dotnet):
        mock_call_dotnet.return_value = ({"records": []}, None)
        mock_gen_report.return_value = {
            "introMessage": "Intro",
            "analysis": {"summary": "Sum", "observations": [], "insights": []},
            "conclusion": {"summary": "Conc"}
        }

        # Clear values first
        metrics_collector.requests_total.clear()
        
        result = execute_admin_query(
            user_query="Show attendance",
            body={"session_id": "test-sess"},
            trace_id="test-trace-id"
        )
        self.assertEqual(result["type"], "query")

        # Requests count for 'simple' should be 1 now
        self.assertEqual(metrics_collector.requests_total.get("simple"), 1)
        self.assertGreater(metrics_collector.durations["pipeline_duration"]["count"], 0)

    @patch('admin_pipeline._call_dotnet')
    @patch('admin_pipeline.generate_report')
    @patch('admin_pipeline.SLOW_QUERY_THRESHOLD', -1.0)  # Set threshold extremely low to trigger log
    def test_slow_query_detection_warning(self, mock_gen_report, mock_call_dotnet):
        mock_call_dotnet.return_value = ({"records": []}, None)
        mock_gen_report.return_value = {
            "introMessage": "Intro",
            "analysis": {"summary": "Sum", "observations": [], "insights": []},
            "conclusion": {"summary": "Conc"}
        }

        logger_pipeline = logging.getLogger("admin_pipeline")
        
        with self.assertLogs(logger_pipeline, level='WARNING') as log_ctx:
            execute_admin_query(
                user_query="Show attendance",
                body={"session_id": "test-sess"},
                trace_id="slow-trace-id"
            )
            
            # Verify the warning message contains "Query exceeded 0 seconds."
            found_slow_log = False
            for log_msg in log_ctx.output:
                if "Query exceeded" in log_msg:
                    found_slow_log = True
                    # Assert it is formatted as JSON containing standard details
                    raw_json = log_msg.split(':', 2)[-1].strip()
                    parsed = json.loads(raw_json)
                    self.assertIn("Query exceeded", parsed["message"])
                    self.assertEqual(parsed["trace_id"], "slow-trace-id")
                    self.assertEqual(parsed["query_type"], "simple")
                    self.assertIn("duration_ms", parsed)
            self.assertTrue(found_slow_log)


if __name__ == "__main__":
    unittest.main()
