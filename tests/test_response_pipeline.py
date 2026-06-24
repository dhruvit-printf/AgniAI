"""
tests/test_response_pipeline.py
================================
Unit tests for the Report Generator and Response Builder layers.
"""

import unittest
from unittest.mock import patch

from admin_pipeline import execute_admin_query
from report_generator import (
    _extract_numbers_from_text,
    _strip_ungrounded_numbers,
    generate_report,
    get_fallback_report,
)
from response_builder import (
    build_response,
)
from response_sanitizer import public_response_view


def build_combined_message(intro="", formatted="", analysis=None, conclusion=None):
    parts = []
    if intro:
        parts.append(intro)
    if formatted:
        parts.append(formatted)
    if analysis:
        if analysis.get("summary"):
            parts.append(analysis["summary"])
        obs = analysis.get("observations") or []
        if obs:
            parts.append("\n".join(f"- {o}" for o in obs))
        ins = analysis.get("insights") or []
        if ins:
            parts.append("\n".join(f"- {i}" for i in ins))
        preds = analysis.get("predictions") or []
        if preds:
            parts.append("Predictions:\n" + "\n".join(f"- {p}" for p in preds))
    if conclusion:
        if conclusion.get("summary"):
            parts.append(f"Conclusion:\n{conclusion['summary']}")
    return "\n\n".join(parts) if parts else ""



class TestGroundingGuard(unittest.TestCase):
    def test_extract_numbers(self):
        self.assertEqual(
            _extract_numbers_from_text("Show 12 items and 3.5 averages"), {"12", "3.5"}
        )
        self.assertEqual(_extract_numbers_from_text("No numbers here"), set())

    def test_strip_ungrounded_numbers(self):
        grounded = "There are 21 completed verifications and 5 pending."

        # Valid sentence with grounded numbers
        llm_text_valid = "We identified 21 completions. 5 cases remain pending."
        self.assertEqual(
            _strip_ungrounded_numbers(llm_text_valid, grounded),
            "We identified 21 completions. 5 cases remain pending.",
        )

        # Invalid sentence with ungrounded number 99
        llm_text_invalid = "We identified 21 completions. 99 cases remain pending."
        self.assertEqual(
            _strip_ungrounded_numbers(llm_text_invalid, grounded),
            "We identified 21 completions.",
        )


class TestFallbackReport(unittest.TestCase):
    def test_fallback_simple(self):
        combined = [{"agniveerId": 1}, {"agniveerId": 2}]
        intent = {"category": "Verification", "subcategory": "CompletedVerification"}
        rep = get_fallback_report(combined, "simple", intent)
        self.assertEqual(
            rep["message"],
            "These records confirm files that have cleared the verification process.",
        )
        self.assertIn("2 matching verification records", rep["analysis"]["summary"])
        self.assertIn("verification", rep["conclusion"]["summary"])

    def test_fallback_cross_filter(self):
        combined = {
            "queryType": "cross_filter",
            "filterDepth": 2,
            "matchCount": 3,
            "totalBeforeFilter": 10,
            "records": [{"agniveerId": 1}, {"agniveerId": 2}, {"agniveerId": 3}],
        }
        intent = {"category": "Performance"}
        rep = get_fallback_report(combined, "cross_filter", intent)
        self.assertEqual(
            rep["message"],
            "I found 3 records that match all of the selected conditions.",
        )
        self.assertIn("exactly 3 records", rep["analysis"]["summary"])
        self.assertEqual(
            rep["conclusion"]["summary"],
            "The cross-filter search is complete and the 3 matching records are ready for review.",
        )


class TestResponseBuilder(unittest.TestCase):
    def test_build_combined_message(self):
        intro = "Hello."
        formatted = "Table content here"
        analysis = {
            "summary": "This is summary.",
            "observations": ["Obs 1", "Obs 2"],
            "insights": ["Insight 1"],
        }
        conclusion = {"summary": "Done."}

        msg = build_combined_message(intro, formatted, analysis, conclusion)
        self.assertIn("Hello.", msg)
        self.assertIn("Table content here", msg)
        self.assertIn("This is summary.", msg)
        self.assertIn("- Obs 1", msg)
        self.assertIn("- Insight 1", msg)
        self.assertIn("Conclusion:\nDone.", msg)

    def test_build_response_schema(self):
        resp = build_response(
            message="Intro",
            formatted_data={"type": "TABLE", "data": {"columns": [], "rows": []}},
            metadata={"sessionId": "session-123", "confidence": 0.95, "queryType": "simple", "operationCount": 1},
            session_id="session-123",
            suggested_questions=["Q1"],
            dotnet_payload={"res": "val"},
        )

        self.assertTrue(resp["status"])
        self.assertEqual(resp["sessionId"], "session-123")
        self.assertEqual(resp["message"], "Intro")
        self.assertEqual(resp["formattedData"]["type"], "TABLE")
        self.assertEqual(resp["dotnetPayload"], {"res": "val"})
        self.assertEqual(resp["suggestedQuestions"], ["Q1"])
        self.assertEqual(resp["metadata"]["operationCount"], 1)

    def test_build_response_uses_real_section_label_and_message(self):
        from message_engine import generate_message
        intent = {
            "category": "Performance",
            "subcategory": "TopPerformers",
            "confidence": "high",
            "section": "PPT",
        }
        combined = [
            {"fullName": "A", "bestTotal": 100, "sectionFilter": "PPT"},
            {"fullName": "B", "bestTotal": 99, "sectionFilter": "PPT"},
        ]
        msg = generate_message(
            user_query="Show top performers in PPT",
            combined_result=combined,
            query_type="simple",
            intent=intent,
        )
        self.assertIn("performance", msg.lower())
        self.assertIn("ppt", msg.lower())

    def test_build_response_uses_overall_label_for_top_performers(self):
        from message_engine import generate_message
        intent = {
            "category": "Performance",
            "subcategory": "TopPerformers",
            "confidence": "high",
        }
        combined = [
            {"fullName": "A", "bestTotal": 100, "platoonName": "PL-05"},
            {"fullName": "B", "bestTotal": 99, "platoonName": "PL-18"},
        ]
        msg = generate_message(
            user_query="Show top performers",
            combined_result=combined,
            query_type="simple",
            intent=intent,
        )
        self.assertIn("performance", msg.lower())

    def test_public_response_view_matches_external_contract(self):
        internal = build_response(
            message="Intro",
            formatted_data={
                "type": "TABLE",
                "title": "Top Performers",
                "data": {"columns": [], "rows": []},
                "analysis": {"summary": "Sum"},
                "prediction": {"trend": "Stable"},
                "conclusion": {"summary": "Conc"},
            },
            metadata={
                "sessionId": "admin-default",
                "confidence": 0.95,
                "queryType": "simple",
                "operationCount": 1,
            },
            session_id="admin-default",
            suggested_questions=[],
            dotnet_payload={},
        )

        public = public_response_view(internal)

        self.assertEqual(
            set(public.keys()),
            {
                "status",
                "sessionId",
                "message",
                "formattedData",
                "suggestedQuestions",
                "dotnetPayload",
                "metadata",
            },
        )
        self.assertNotIn("result", public)
        self.assertNotIn("intent", public)
        self.assertNotIn("answer", public)
        self.assertEqual(public["sessionId"], "admin-default")
        self.assertEqual(public["message"], "Intro")

        # Verify metadata filtering
        metadata = public["metadata"]
        self.assertNotIn("requestId", metadata)
        self.assertNotIn("traceId", metadata)
        self.assertNotIn("timings", metadata)
        self.assertIn("metrics", metadata)
        self.assertEqual(metadata["metrics"]["confidence"], 0.95)
        self.assertIn("analysis", public["formattedData"])
        self.assertIn("prediction", public["formattedData"])
        self.assertIn("conclusion", public["formattedData"])
        self.assertNotIn("answer", public["formattedData"])
        self.assertNotIn("introMessage", public["formattedData"])
        self.assertNotIn("message", public["formattedData"])
        self.assertEqual(public["sessionId"], "admin-default")
        self.assertEqual(public["message"], internal["message"])

        if isinstance(public["formattedData"], dict) and "data" in public["formattedData"]:
            data = public["formattedData"]["data"]
            if isinstance(data, dict) and "rows" in data:
                for row in data["rows"]:
                    self.assertNotIn("dotnetPayload", row)


class TestBuildResponseSecurity:
    def test_dotnet_response_never_in_payload(self):
        """SECURITY: raw backend data must never reach the frontend."""
        from response_builder import build_response

        intent = {
            "category": "Performance",
            "subcategory": "TopPerformers",
            "confidence": "high",
        }
        resp = build_response(
            message="Intro",
            formatted_data={"type": "TABLE", "data": {"columns": [], "rows": []}},
            metadata={
                "sessionId": "session-123",
                "confidence": 0.9,
                "queryType": "simple",
                "operationCount": 1,
            },
            session_id="session-123",
            suggested_questions=[],
            dotnet_payload={"secret": "data"},
        )
        assert "dotnetResponse" not in resp
        assert "rawResponse" not in resp
        assert "raw_results" not in resp
        assert resp["dotnetPayload"] == {"secret": "data"}

    def test_stack_trace_never_in_payload(self):
        from response_builder import build_response

        resp = build_response(
            message="",
            formatted_data={},
            metadata={
                "sessionId": "session-123",
                "confidence": 0.5,
                "queryType": "simple",
                "operationCount": 1,
            },
            session_id="session-123",
        )
        resp_str = str(resp)
        assert "traceback" not in resp_str.lower()
        assert "exception" not in resp_str.lower()


class TestBuildCombinedMessage:
    def test_empty_analysis_no_crash(self):

        result = build_combined_message("Intro", "Data", None, None)
        assert "Intro" in result
        assert "Data" in result

    def test_all_parts_present(self):

        analysis = {"summary": "Sum", "observations": ["O1"], "insights": ["I1"]}
        conclusion = {"summary": "End"}
        result = build_combined_message("Start", "Middle", analysis, conclusion)
        assert "Start" in result
        assert "Middle" in result
        assert "Sum" in result
        assert "O1" in result
        assert "I1" in result
        assert "End" in result

    def test_empty_strings_excluded(self):

        result = build_combined_message("", "", None, None)
        assert result.strip() == ""

    def test_confidence_string_normalized_to_float(self):
        from utils import normalize_confidence
        assert normalize_confidence("high") == 0.95

    def test_confidence_medium_normalized(self):
        from utils import normalize_confidence
        assert normalize_confidence("medium") == 0.70

    def test_confidence_low_normalized(self):
        from utils import normalize_confidence
        assert normalize_confidence("low") == 0.30

    def test_session_id_default_val_when_default(self):
        from response_builder import build_response
        resp = build_response(
            message="",
            formatted_data={},
            metadata={},
            session_id="admin-default",
        )
        assert resp["sessionId"] == "admin-default"

    def test_session_id_included_when_not_default(self):
        from response_builder import build_response
        resp = build_response(
            message="",
            formatted_data={},
            metadata={},
            session_id="real-session-abc",
        )
        assert resp["sessionId"] == "real-session-abc"


class TestResponsePipelinePredictionsAndFallback(unittest.TestCase):

    def test_predictions_combined_message_formatting(self):
        analysis = {
            "summary": "This is summary.",
            "observations": ["Obs 1"],
            "insights": ["Insight 1"],
            "predictions": ["Pred 1", "Pred 2"],
        }
        msg = build_combined_message("Hello.", "", analysis, None)
        self.assertIn("Predictions:\n- Pred 1\n- Pred 2", msg)

    @patch("admin_pipeline._call_dotnet")
    @patch("admin_pipeline.generate_report")
    def test_formatted_data_is_populated_on_data_query(
        self, mock_generate_report, mock_call_dotnet
    ):
        # Verification data query
        mock_call_dotnet.return_value = (
            [{"agniveerNo": "1", "fullName": "John Doe"}],
            None,
        )
        mock_generate_report.return_value = {
            "message": "Intro.",
            "analysis": {
                "summary": "Summary",
                "observations": [],
                "insights": [],
                "predictions": [],
            },
            "conclusion": {"summary": "Conclusion"},
        }

        # Query that invokes a single .NET call
        result = execute_admin_query("Show completed verification records", {})
        self.assertEqual(result["type"], "query")

        response_payload = result["response_payload"]
        self.assertTrue(response_payload["status"])

        # Verify that response_payload contains the record John Doe
        rows = response_payload["formattedData"]["data"]["rows"]
        found_john = any(row.get("fullName") == "John Doe" for row in rows)
        self.assertTrue(found_john)

    @patch("report_generator._call_ollama")
    def test_generate_report_is_honest_without_padding_filler(self, mock_call_ollama):
        # 1. Test when LLM returns short intro/conclusion.
        mock_call_ollama.return_value = (
            '{"message": "Short intro.", "analysis": {"summary": "A summary", '
            '"observations": [], "insights": [], "predictions": []}, "conclusion": "Short conclusion."}'
        )
        combined = {
            "queryType": "cross_filter",
            "records": [{"agniveerNo": "1"}, {"agniveerNo": "2"}],
        }
        intent = {"category": "Performance"}
        report = generate_report(combined, "cross_filter", intent, "query")

        self.assertNotIn(
            "Additional details are saved in the system logs",
            report["message"],
        )
        self.assertNotIn(
            "Additional details are saved in the system logs",
            report["conclusion"]["summary"],
        )
        self.assertEqual(report["message"], "Short intro.")
        self.assertEqual(report["conclusion"]["summary"], "Short conclusion.")

        # 2. Test when LLM fails (fallback path)
        mock_call_ollama.return_value = None
        report_fallback = generate_report(combined, "cross_filter", intent, "query")
        self.assertNotIn(
            "Additional details are saved in the system logs",
            report_fallback["message"],
        )
        self.assertNotIn(
            "Additional details are saved in the system logs",
            report_fallback["conclusion"]["summary"],
        )

    def test_generate_report_returns_none_when_empty_records(self):
        combined = {
            "queryType": "cross_filter",
            "records": [],
        }
        intent = {"category": "Performance"}
        report = generate_report(combined, "cross_filter", intent, "query")
        
        self.assertEqual(report["message"], "No matching records found.")
        self.assertEqual(report["analysis"]["summary"], "No matching records found.")
        self.assertEqual(report["prediction"]["trend"], "Insufficient Data")
        self.assertEqual(report["conclusion"]["bullets"], [])

    @patch("report_generator.generate_analysis")
    @patch("report_generator.generate_predictions")
    @patch("report_generator.generate_conclusion")
    def test_generate_report_fallback_gating(self, mock_conclusion, mock_predictions, mock_analysis):
        mock_analysis.return_value = {
            "summary": "Healthy Analysis Summary",
            "observations": ["Obs"],
            "insights": ["Insight"]
        }
        mock_predictions.return_value = {
            "trend": "Stable",
            "projection": "Stable Projection",
            "heuristicEstimate": "Est",
            "shortTerm": "stable",
            "futureTrends": ["Trend"]
        }
        # Conclusion returns a negative copy marker triggering fallback for conclusion
        mock_conclusion.return_value = {
            "message": "insufficient data to make conclusion"
        }

        combined = {
            "queryType": "simple",
            "records": [{"agniveerNo": "1", "score": 85}],
        }
        intent = {"category": "Performance"}
        report = generate_report(combined, "simple", intent, "query")

        # Analysis and prediction must remain unchanged (not overwritten by fallback)
        self.assertEqual(report["analysis"]["summary"], "Healthy Analysis Summary")
        self.assertEqual(report["prediction"]["projection"], "Stable Projection")

        # Conclusion must be replaced by the fallback conclusion summary
        self.assertNotEqual(report["conclusion"]["summary"], "insufficient data to make conclusion")
        self.assertIn("returned 1 records", report["conclusion"]["summary"])


if __name__ == "__main__":
    unittest.main()
