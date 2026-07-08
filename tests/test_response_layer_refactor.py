import json
import unittest

from response_builder import build_response
from schemas import FinalResponseModel
from widget_engine import build_formatted_data, build_widget_list, validate_payload


class TestResponseLayerRefactor(unittest.TestCase):

    def test_strict_contract_response_builder(self):
        # Verify response builder contract — analysis/prediction/conclusion at root
        resp = build_response(
            message="Test Message",
            formatted_data={
                "type": "TABLE",
                "title": "Test Title",
                "data": {"key": "val"},
            },
            metadata={"queryType": "simple", "sessionId": "123"},
            session_id="123",
            suggested_questions=["question?"],
            dotnet_payload={"dotnet_key": "dotnet_val"},
            analysis={"summary": "Some analysis"},
            prediction={"projection": "Up"},
            conclusion={"summary": "Done"},
        )

        # Root-level narrative strings
        self.assertIsInstance(resp["analysis"], str)
        self.assertIsInstance(resp["prediction"], str)
        self.assertIsInstance(resp["conclusion"], str)
        self.assertEqual(resp["analysis"], "Some analysis.")
        self.assertEqual(resp["prediction"], "Up.")
        self.assertEqual(resp["conclusion"], "Done.")

        # single-widget responses now return a bare object
        self.assertIsInstance(resp["formattedData"], dict)
        fd = resp["formattedData"]
        self.assertEqual(fd["type"], "TABLE")
        self.assertEqual(fd["title"], "Test Title")
        self.assertEqual(fd["data"], {"key": "val"})
        # analysis/prediction/conclusion must NOT be inside the widget
        self.assertNotIn("analysis", fd)
        self.assertNotIn("prediction", fd)
        self.assertNotIn("conclusion", fd)

    def test_multi_independent_preserves_sections(self):
        # Verify multi_independent tables preserve section boundaries (Issue 1)
        combined = {
            "sections": [
                {"label": "Top Performers", "data": [{"name": "Amit", "score": 95}]},
                {"label": "Leave Takers", "data": [{"name": "Kapil", "days": 10}]},
            ]
        }

        fd = build_formatted_data(
            combined,
            query_type="multi_independent",
            intent={"category": "Performance", "subcategory": "Top"},
        )

        # Check that sections is preserved at formattedData.data
        data = fd["data"]
        self.assertIn("sections", data)
        self.assertEqual(len(data["sections"]), 2)

        sec1 = data["sections"][0]
        self.assertEqual(sec1["label"], "Top Performers")
        self.assertEqual(len(sec1["columns"]), 2)  # name and score
        self.assertEqual(sec1["row"][0]["name"], "Amit")  # camelCase key

        sec2 = data["sections"][1]
        self.assertEqual(sec2["label"], "Leave Takers")
        self.assertEqual(len(sec2["columns"]), 2)  # name and days
        self.assertEqual(sec2["row"][0]["name"], "Kapil")  # camelCase key

    def test_comparison_preserves_left_right(self):
        # Verify comparison tables preserve left/right sides without flattening (Issue 2)
        combined = {
            "left": {"label": "Company A", "data": [{"name": "A", "marks": 80}]},
            "right": {"label": "Company B", "data": [{"name": "B", "marks": 90}]},
            "comparison": {"diff": 10},
        }

        fd = build_formatted_data(
            combined,
            query_type="comparison",
            intent={"category": "Performance", "subcategory": "Compare"},
        )

        data = fd["data"]
        self.assertIn("left", data)
        self.assertIn("right", data)
        self.assertIn("comparison", data)

        self.assertEqual(data["left"]["label"], "Company A")
        self.assertEqual(data["left"]["row"][0]["name"], "A")  # camelCase
        self.assertEqual(data["right"]["label"], "Company B")
        self.assertEqual(data["right"]["row"][0]["name"], "B")  # camelCase
        self.assertEqual(data["comparison"], {"diff": 10})

    def test_cross_filter_metadata_propagates(self):
        # Verify cross filter metadata is preserved inside formattedData.data (Issue 3)
        combined = {
            "records": [{"name": "Amit"}],
            "matchCount": 5,
            "totalBeforeFilter": 10,
            "filterDepth": 2,
            "queryType": "cross_filter",
            "degraded": True,
            "failedFilters": ["filter1"],
        }

        fd = build_formatted_data(
            combined,
            query_type="cross_filter",
            intent={"category": "Performance", "subcategory": "Top"},
        )

        data = fd["data"]
        self.assertEqual(data["matchCount"], 5)
        self.assertEqual(data["totalBeforeFilter"], 10)
        self.assertEqual(data["filterDepth"], 2)
        self.assertEqual(data["queryType"], "cross_filter")
        self.assertEqual(data["degraded"], True)
        self.assertEqual(data["failedFilters"], ["filter1"])

    def test_multi_independent_card_sections_use_their_own_records(self):
        combined = {
            "sections": [
                {
                    "label": "Top Performers",
                    "data": [{"name": "Amit", "score": 95}],
                },
                {
                    "label": "Leave Takers",
                    "data": [{"name": "Kapil", "days": 10}],
                },
            ]
        }

        widgets = build_widget_list(
            combined,
            query_type="multi_independent",
            intent={"category": "Performance", "subcategory": "Top"},
            visualization_intent={"widgets": [{"type": "CARD"}, {"type": "CARD"}]},
        )

        self.assertEqual(len(widgets), 2)
        self.assertEqual(widgets[0]["data"]["cards"][0]["title"], "Amit")
        self.assertEqual(widgets[1]["data"]["cards"][0]["title"], "Kapil")
        self.assertNotEqual(widgets[0]["id"], widgets[1]["id"])

    def test_multi_independent_chart_sections_use_their_own_records(self):
        combined = {
            "sections": [
                {
                    "label": "Top Performers",
                    "data": [{"name": "Amit", "score": 95}],
                },
                {
                    "label": "Leave Takers",
                    "data": [{"name": "Kapil", "days": 10}],
                },
            ]
        }

        widgets = build_widget_list(
            combined,
            query_type="multi_independent",
            intent={"category": "Performance", "subcategory": "Top"},
            visualization_intent={"widgets": [{"type": "CHART_PIE"}, {"type": "CHART_PIE"}]},
        )

        self.assertEqual(len(widgets), 2)
        self.assertEqual(widgets[0]["data"].get("rows", [{}])[0].get("label"), "Amit")
        self.assertEqual(widgets[1]["data"].get("rows", [{}])[0].get("label"), "Kapil")

    def test_multi_independent_attendance_calendar_sections_use_their_own_records(self):
        combined = {
            "sections": [
                {
                    "label": "Top Performers",
                    "data": [{"date": "2026-07-01", "present": True}],
                },
                {
                    "label": "Leave Takers",
                    "data": [{"date": "2026-07-02", "present": False}],
                },
            ]
        }

        widgets = build_widget_list(
            combined,
            query_type="multi_independent",
            intent={"category": "Performance", "subcategory": "Top"},
            visualization_intent={"widgets": [{"type": "ATTENDANCE_CALENDAR"}, {"type": "ATTENDANCE_CALENDAR"}]},
        )

        self.assertEqual(len(widgets), 2)
        self.assertEqual(widgets[0]["data"]["days"][0]["date"], "2026-07-01")
        self.assertEqual(widgets[1]["data"]["days"][0]["date"], "2026-07-02")

    def test_row_serialization_preserves_nested(self):
        # Verify nested structures are preserved as JSON strings (Issue 4)
        records = [
            {
                "name": "Amit",
                "attempts": [{"attempt": 1, "score": 90}],
                "metrics": {"avg": 85},
            }
        ]

        from widget_engine import build_table_data

        res = build_table_data(records)
        rows = res["row"]
        self.assertEqual(rows[0]["name"], "Amit")
        # should be JSON strings of the nested dict/list
        self.assertEqual(json.loads(rows[0]["attempts"]), [{"attempt": 1, "score": 90}])
        self.assertEqual(json.loads(rows[0]["metrics"]), {"avg": 85})

    def test_deep_flatten_across_widget_types(self):
        # Verify deep flattening surfaces nested .NET fields for BOTH tables
        # and chart widgets. Previously CARD/CHART_BAR/CHART_LINE/CHART_PIE
        # used deep_flatten=False and silently dropped any field .NET nested
        # under a sub-object (e.g. "nested_info": {"sport": ...}) — those
        # fields never reached the frontend. Charts must now flatten too.
        combined = [
            {
                "name": "Amit",
                "nested_info": {"sport": "Cricket", "detail": {"level": "High"}},
            }
        ]

        # 1. TABLE widget -> deep flatten happens
        fd_table = build_formatted_data(
            combined,
            query_type="simple",
            intent={"category": "Performance", "subcategory": "Top"},
            visualization_intent={
                "requested_widget_type": "TABLE",
                "frontend_override": True,
            },
        )
        # Verify that nested_info keys are flattened (now camelCase: nested_info_Sport -> nested_info_Sport first char lowered)
        self.assertIn("nested_info_Sport", fd_table["data"]["row"][0])
        self.assertIn("nested_info_Detail_Level", fd_table["data"]["row"][0])

        # 2. BAR_CHART widget -> nested fields are ALSO flattened and surfaced
        fd_bar = build_formatted_data(
            combined,
            query_type="simple",
            intent={"category": "Performance", "subcategory": "Top"},
            visualization_intent={
                "requested_widget_type": "BAR_CHART",
                "frontend_override": True,
            },
        )
        # New BAR_CHART schema has rows, not series
        self.assertIn("rows", fd_bar["data"])
        row = fd_bar["data"]["rows"][0]
        self.assertIn("Name", row)
        self.assertIn("Nested_info_Sport", row)
        self.assertIn("Nested_info_Detail_Level", row)

    def test_preserve_dotnet_keys(self):
        # Verify unknown keys pass through to data_payload (Issue 6)
        combined = {
            "records": [{"name": "Amit"}],
            "bestAttempt": {"marks": 95},
            "subItems": ["item1"],
            "customDotnetField": "hello",
        }

        fd = build_formatted_data(
            combined,
            query_type="simple",
            intent={"category": "Performance", "subcategory": "Top"},
        )

        data = fd["data"]
        self.assertEqual(data["bestAttempt"], {"marks": 95})
        self.assertEqual(data["subItems"], ["item1"])
        self.assertEqual(data["customDotnetField"], "hello")

    def test_table_columns_union_generation(self):
        # Verify union of all keys across ALL rows is used (Issue 7)
        combined = [{"name": "A"}, {"name": "B", "attempts_section1_marks": 90}]

        fd = build_formatted_data(
            combined,
            query_type="simple",
            intent={"category": "Performance", "subcategory": "Top"},
        )

        columns = fd["data"]["columns"]
        col_keys = [c["key"] for c in columns]

        # Column keys are now camelCase (first char lowercased)
        self.assertIn("name", col_keys)
        self.assertIn("attempts_section1_marks", col_keys)

        rows = fd["data"]["row"]
        self.assertEqual(rows[0]["name"], "A")
        self.assertIsNone(rows[0]["attempts_section1_marks"])
        self.assertEqual(rows[1]["name"], "B")
        self.assertEqual(rows[1]["attempts_section1_marks"], 90)

    def test_final_response_schema_accepts_single_comparison_widget(self):
        payload = {
            "status": True,
            "message": "Comparison generated",
            "formattedData": {
                "type": "COMPARE_TABLE",
                "title": "Comparison Result",
                "data": {"left": {}, "right": {}},
            },
            "metadata": {
                "sessionId": "admin-default",
                "confidence": 0.9,
                "queryType": "COMPARISON",
                "operationCount": 1,
                "executionTimeMs": 123,
            },
        }

        model = FinalResponseModel.model_validate(payload)
        self.assertEqual(model.formattedData["type"], "COMPARE_TABLE")


if __name__ == "__main__":
    unittest.main()
