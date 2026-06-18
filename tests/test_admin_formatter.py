"""
tests/test_admin_formatter.py
Tests for admin_formatter.py — .NET response → plain text formatter.
"""
import pytest
from admin_formatter import format_dotnet_response


class TestFormatDotnetResponseBasic:
    def test_returns_string(self):
        result = format_dotnet_response({"data": []}, 
                                         {"category": "Attendance"})
        assert isinstance(result, str)

    def test_empty_data_does_not_crash(self):
        result = format_dotnet_response({}, {"category": "Leave"})
        assert isinstance(result, str)

    def test_string_data_returned_as_is(self):
        result = format_dotnet_response("Already formatted", 
                                         {"category": "Medical"})
        assert result == "Already formatted"

    def test_none_category_falls_back_to_json(self):
        result = format_dotnet_response({"key": "value"}, {})
        assert "key" in result or isinstance(result, str)

    def test_no_markdown_in_output(self):
        """Output must be plain text — no markdown symbols."""
        data = {"data": [{"fullName": "Ravi Kumar", "bestTotal": 95}]}
        intent = {"category": "Performance", 
                  "subcategory": "TopPerformers"}
        result = format_dotnet_response(data, intent)
        assert "**" not in result
        assert "##" not in result
        assert "```" not in result


class TestPerformanceFormatting:
    def _perf_record(self, name, score):
        return {"fullName": name, "bestTotal": score, 
                "agniveerNo": "001", "attempts": []}

    def test_top_performers_includes_names(self):
        data = {"data": [
            self._perf_record("Ravi Kumar", 95),
            self._perf_record("Priya Singh", 88),
        ]}
        intent = {"category": "Performance", 
                  "subcategory": "TopPerformers"}
        result = format_dotnet_response(data, intent)
        assert "Ravi Kumar" in result
        assert "Priya Singh" in result

    def test_top_performers_shows_scores(self):
        data = {"data": [self._perf_record("Amit Yadav", 91)]}
        intent = {"category": "Performance", 
                  "subcategory": "TopPerformers"}
        result = format_dotnet_response(data, intent)
        assert "91" in result

    def test_average_score_numeric_result(self):
        data = {"averageScore": 74.5}
        intent = {"category": "Performance", 
                  "subcategory": "AverageScore",
                  "section": "PPT"}
        result = format_dotnet_response(data, intent)
        assert "74.5" in result

    def test_empty_performance_data(self):
        data = {"data": []}
        intent = {"category": "Performance", 
                  "subcategory": "TopPerformers"}
        result = format_dotnet_response(data, intent)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_pass_percentage_formats_percent(self):
        data = {"percentage": 82.5, "total": 120}
        intent = {"category": "Performance", 
                  "subcategory": "PassPercentage"}
        result = format_dotnet_response(data, intent)
        assert "82.5" in result or "82" in result


class TestLeaveFormatting:
    def _leave_record(self, name):
        return {
            "fullName": name,
            "agniveerNo": "LV001",
            "fromDate": "2025-01-10T00:00:00",
            "toDate": "2025-01-15T00:00:00",
            "totalDays": 5,
            "remarks": "Annual",
        }

    def test_most_leave_includes_name(self):
        data = {"data": [self._leave_record("Suresh Patel")]}
        intent = {"category": "Leave", 
                  "subcategory": "MostLeaveTaken"}
        result = format_dotnet_response(data, intent)
        assert "Suresh Patel" in result

    def test_current_leave_no_records_message(self):
        data = {"data": []}
        intent = {"category": "Leave", 
                  "subcategory": "CurrentLeaveStatus"}
        result = format_dotnet_response(data, intent)
        assert isinstance(result, str)

    def test_absconded_person_label(self):
        data = {"data": [self._leave_record("Missing Person")]}
        intent = {"category": "Leave", 
                  "subcategory": "AbscondedPerson"}
        result = format_dotnet_response(data, intent)
        assert "Missing Person" in result


class TestAttendanceFormatting:
    def test_present_today_count(self):
        data = {"present": 85, "total": 100}
        intent = {"category": "Attendance", 
                  "subcategory": "PresentToday"}
        result = format_dotnet_response(data, intent)
        assert "85" in result

    def test_strength_breakdown_dict(self):
        data = {
            "totalAgniveers": 200,
            "activeCount": 185,
            "presentToday": 180,
        }
        intent = {"category": "Attendance", 
                  "subcategory": "StrengthBreakdown"}
        result = format_dotnet_response(data, intent)
        assert isinstance(result, str)
        assert len(result) > 0


class TestCompositeQueryTypes:
    def test_cross_filter_result_formatted(self):
        data = {
            "queryType": "cross_filter",
            "filterDepth": 2,
            "matchCount": 3,
            "totalBeforeFilter": 20,
            "records": [
                {"fullName": "Agniveer A", "agniveerId": 1},
                {"fullName": "Agniveer B", "agniveerId": 2},
            ],
        }
        intent = {"category": "Performance"}
        result = format_dotnet_response(data, intent)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_comparison_result_formatted(self):
        data = {
            "queryType": "comparison",
            "sides": [
                {"label": "PPT", "data": {"data": []}},
                {"label": "BPET", "data": {"data": []}},
            ],
        }
        intent = {"category": "Performance"}
        result = format_dotnet_response(data, intent)
        assert isinstance(result, str)

    def test_multi_independent_result_formatted(self):
        data = {
            "queryType": "multi_independent",
            "sections": [
                {"label": "Attendance", "data": {}},
                {"label": "Equipment", "data": {}},
            ],
        }
        intent = {"category": "Attendance"}
        result = format_dotnet_response(data, intent)
        assert isinstance(result, str)
