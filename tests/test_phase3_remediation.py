from __future__ import annotations

import inspect
import os
import unittest
from contextlib import contextmanager
from importlib import reload
from unittest.mock import MagicMock, patch

import app as app_module
import app_factory
import analysis_engine
import conclusion_engine
import prediction_engine
import report_generator
from prediction_engine import generate_predictions
from response_builder import build_response


@contextmanager
def _patched_env(**env):
    old = {k: os.environ.get(k) for k in env}
    os.environ.update({k: v for k, v in env.items() if v is not None})
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class TestF10NoPaddingFiller(unittest.TestCase):
    @patch("report_generator._call_ollama", return_value=None)
    def test_report_generator_does_not_emit_padding_filler(self, _mock_call):
        combined = {
            "queryType": "cross_filter",
            "records": [{"agniveerId": 1}],
            "matchCount": 1,
            "totalBeforeFilter": 1,
        }
        report = report_generator.generate_report(
            combined, "cross_filter", {"category": "Performance"}, "query"
        )
        self.assertNotIn("Additional details are saved in the system logs", report["message"])
        self.assertNotIn(
            "Additional details are saved in the system logs",
            report["conclusion"]["summary"],
        )


class TestF11Grounding(unittest.TestCase):
    def test_grounding_is_single_shared_implementation(self):
        self.assertIs(
            analysis_engine._ground_and_sanitize, conclusion_engine._ground_and_sanitize
        )
        self.assertIs(
            conclusion_engine._ground_and_sanitize, prediction_engine._ground_and_sanitize
        )
        self.assertIs(
            report_generator._strip_ungrounded_numbers,
            analysis_engine._ground_and_sanitize,
        )

    def test_token_boundary_number_matching(self):
        source = "The score was 1985 in the record."
        text = "The score was 85 in the record."
        cleaned = analysis_engine._ground_and_sanitize(text, source)
        self.assertEqual(cleaned, "")

        source_ok = "The score was 85 in the record."
        cleaned_ok = analysis_engine._ground_and_sanitize(text, source_ok)
        self.assertEqual(cleaned_ok, text)


class TestF12OllamaTimeouts(unittest.TestCase):
    @patch("analysis_engine.requests.post")
    @patch("analysis_engine.get_flags")
    def test_analysis_ollama_timeout_is_realistic(self, mock_flags, mock_post):
        mock_flags.return_value = MagicMock(ENABLE_REPORTS=True, ENABLE_OLLAMA=True)
        mock_post.return_value = MagicMock(
            status_code=200, json=lambda: {"message": {"content": "{}"}}
        )
        with patch.dict(
            os.environ,
            {"OLLAMA_CONNECT_TIMEOUT": "", "OLLAMA_READ_TIMEOUT": ""},
            clear=False,
        ):
            analysis_engine.generate_analysis(
                {"sections": [{"label": "Result", "data": [{"id": 1, "score": 10}]}]},
                "simple",
                {"category": "Performance"},
            )
        _, kwargs = mock_post.call_args
        self.assertEqual(kwargs["timeout"], (5.0, 30.0))

    def test_timeout_values_are_configurable(self):
        with _patched_env(OLLAMA_CONNECT_TIMEOUT="2.5", OLLAMA_READ_TIMEOUT="45"):
            reloaded = reload(__import__("ollama_settings"))
            self.assertEqual(reloaded.get_ollama_timeout(), (2.5, 45.0))


class TestF13PredictionLabels(unittest.TestCase):
    def test_prediction_engine_exposes_projection_labels(self):
        result = generate_predictions(
            {
                "sections": [
                    {"label": "Result", "data": [{"bestTotal": 80}, {"bestTotal": 90}]}
                ]
            },
            "simple",
            {"category": "Performance"},
        )
        self.assertIn("projection", result)
        self.assertIn("heuristicEstimate", result)
        self.assertNotIn("forecast", result)

    def test_response_builder_surfaces_projection_labels(self):
        payload = build_response(
            message="Intro",
            formatted_data={
                "type": "TABLE",
                "title": "",
                "data": {},
                "prediction": {
                    "projection": "Projected stable performance.",
                    "heuristicEstimate": "Heuristic estimate remains stable.",
                    "futureTrends": ["Heuristic estimate remains stable."],
                },
            },
            metadata={},
            session_id="session-1",
        )
        prediction = payload["formattedData"][0]["prediction"]
        self.assertEqual("Projected stable performance.", prediction["projection"])
        self.assertEqual("Heuristic estimate remains stable.", prediction["heuristicEstimate"])


class TestF14DeadBranchAndEntryPoint(unittest.TestCase):
    def test_dead_branch_removed(self):
        source = inspect.getsource(prediction_engine)
        self.assertNotIn('globals().get("_get_score")', source)

    def test_single_entry_point(self):
        self.assertIs(app_factory.create_app(), app_module.app)


if __name__ == "__main__":
    unittest.main()
