"""
tests/test_response_pipeline.py
================================
Unit tests for the Report Generator and Response Builder layers.
"""

import unittest
from unittest.mock import patch

from report_generator import (
    _extract_numbers_from_text,
    _strip_ungrounded_numbers,
    get_fallback_report,
    generate_report,
)
from response_builder import (
    build_combined_message,
    build_response,
)


class TestGroundingGuard(unittest.TestCase):
    def test_extract_numbers(self):
        self.assertEqual(_extract_numbers_from_text("Show 12 items and 3.5 averages"), {"12", "3.5"})
        self.assertEqual(_extract_numbers_from_text("No numbers here"), set())

    def test_strip_ungrounded_numbers(self):
        grounded = "There are 21 completed verifications and 5 pending."
        
        # Valid sentence with grounded numbers
        llm_text_valid = "We identified 21 completions. 5 cases remain pending."
        self.assertEqual(_strip_ungrounded_numbers(llm_text_valid, grounded), "We identified 21 completions. 5 cases remain pending.")

        # Invalid sentence with ungrounded number 99
        llm_text_invalid = "We identified 21 completions. 99 cases remain pending."
        self.assertEqual(_strip_ungrounded_numbers(llm_text_invalid, grounded), "We identified 21 completions.")


class TestFallbackReport(unittest.TestCase):
    def test_fallback_simple(self):
        combined = [{"agniveerId": 1}, {"agniveerId": 2}]
        intent = {"category": "Verification", "subcategory": "CompletedVerification"}
        rep = get_fallback_report(combined, "simple", intent)
        self.assertEqual(rep["introMessage"], "These records confirm files that have cleared the verification process.")
        self.assertIn("2 records", rep["analysis"]["summary"])
        self.assertIn("verification", rep["conclusion"]["summary"])

    def test_fallback_cross_filter(self):
        combined = {
            "queryType": "cross_filter",
            "filterDepth": 2,
            "matchCount": 3,
            "totalBeforeFilter": 10,
            "records": [{"agniveerId": 1}, {"agniveerId": 2}, {"agniveerId": 3}]
        }
        intent = {"category": "Performance"}
        rep = get_fallback_report(combined, "cross_filter", intent)
        self.assertEqual(rep["introMessage"], "3 Agniveers matched the requested cross-filter criteria.")
        self.assertIn("3 matches", rep["analysis"]["summary"])
        self.assertEqual(rep["conclusion"]["summary"], "3 trainees have been successfully cross-referenced.")


class TestResponseBuilder(unittest.TestCase):
    def test_build_combined_message(self):
        intro = "Hello."
        formatted = "Table content here"
        analysis = {
            "summary": "This is summary.",
            "observations": ["Obs 1", "Obs 2"],
            "insights": ["Insight 1"]
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
        intent = {"category": "Performance", "subcategory": "TopPerformers", "confidence": "high"}
        analysis = {"summary": "Sum", "observations": ["O"], "insights": ["I"]}
        conclusion = {"summary": "Conc"}
        raw_res = [{"data": 123}]
        
        resp = build_response(
            query_type="simple",
            intro_message="Intro",
            combined_result={"res": "val"},
            analysis=analysis,
            conclusion=conclusion,
            intent=intent,
            raw_results=raw_res,
            confidence=0.95,
            operation_count=1,
            formatted_data="Formatted text",
            session_id="session-123"
        )
        
        self.assertTrue(resp["status"])
        self.assertEqual(resp["queryType"], "simple")
        self.assertEqual(resp["introMessage"], "Intro")
        self.assertEqual(resp["result"]["processedData"], {"res": "val"})
        self.assertEqual(resp["analysis"]["summary"], "Sum")
        self.assertEqual(resp["analysis"]["observations"], ["O"])
        self.assertEqual(resp["conclusion"]["summary"], "Conc")
        self.assertEqual(resp["intent"]["confidence"], 0.95)
        # dotnetResponse is intentionally omitted for security — raw backend
        # data must never reach the frontend. Verify it is absent.
        self.assertNotIn("dotnetResponse", resp)
        self.assertNotIn("rawResponse", resp)
        self.assertEqual(resp["metadata"]["operationCount"], 1)
        self.assertEqual(resp["sessionId"], "session-123")


if __name__ == "__main__":
    unittest.main()
