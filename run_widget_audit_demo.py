"""
run_widget_audit_demo.py
========================
Exercises widget_field_audit against sample payloads for all 10 widget types,
mimicking exactly what widget_engine.py would build.
"""

from widget_field_audit import audit_widget, print_report

# ── Sample source records (mimics .NET response after extract_records) ──────

PERF_RECORDS = [
    {"agniveerId": 101, "agniveerNo": "AV-101", "fullName": "Rakesh Kumar",
     "grade": "A", "bestTotal": 92, "attempt1_BPET_5km": 40, "remarks": "Good",
     "company": "Alpha", "IsActive": True, "Attempt1_MaxMarks": 50},
    {"agniveerId": 102, "agniveerNo": "AV-102", "fullName": "Suresh Patel",
     "grade": "B", "bestTotal": 78, "attempt1_BPET_5km": 33, "remarks": "Average",
     "company": "Bravo", "IsActive": True, "Attempt1_MaxMarks": 50},
]

TREND_RECORDS = [
    {"agniveerId": 101, "month": "04-2026", "present": 22, "absent": 2, "unit": "Alpha"},
    {"agniveerId": 101, "month": "05-2026", "present": 25, "absent": 1, "unit": "Alpha"},
    {"agniveerId": 101, "month": "06-2026", "present": 24, "absent": 3, "unit": "Alpha"},
]

PIE_RECORDS = [
    {"bloodGroup": "O+", "count": 45},
    {"bloodGroup": "B+", "count": 30},
    {"bloodGroup": "A+", "count": 25},
]

COMBINED_COMPARE = {
    "left": {
        "label": "Alpha Company",
        "data": [
            {"agniveerId": 1, "fullName": "R Kumar", "bestTotal": 92, "month": "05-2026", "score": 92},
            {"agniveerId": 2, "fullName": "S Patel", "bestTotal": 78, "month": "06-2026", "score": 78},
        ],
        "metrics": {"recordCount": 2, "averageScore": 85.0, "topScore": 92,
                    "bottomScore": 78, "customMetric": 42},
    },
    "right": {
        "label": "Bravo Company",
        "data": [
            {"agniveerId": 3, "fullName": "A Singh", "bestTotal": 88, "month": "05-2026", "score": 88},
            {"agniveerId": 4, "fullName": "M Shah", "bestTotal": 70, "month": "06-2026", "score": 70},
        ],
        "metrics": {"recordCount": 2, "averageScore": 79.0, "topScore": 88,
                    "bottomScore": 70, "customMetric": 37},
    },
}

# ── Widget payloads exactly as widget_engine.py builds them ─────────────────

WIDGETS = [
    # 1. TABLE
    ({"type": "TABLE", "title": "Performance", "data": {
        "columns": [{"key": "fullName", "label": "Full Name"},
                    {"key": "agniveerNo", "label": "Agniveer No."},
                    {"key": "bestTotal", "label": "Best Total"},
                    {"key": "grade", "label": "Grade"},
                    {"key": "attempt1_BPET_5km", "label": "Attempt1 BPET 5km"},
                    {"key": "remarks", "label": "Remarks"},
                    {"key": "company", "label": "Company"},
                    {"key": "agniveerId", "label": "Agniveer ID"}],
        "row": [{"fullName": "Rakesh Kumar", "agniveerNo": "AV-101", "bestTotal": 92,
                 "grade": "A", "attempt1_BPET_5km": 40, "remarks": "Good",
                 "company": "Alpha", "agniveerId": 101}],
    }}, PERF_RECORDS),

    # 2. CARD  (built from PERF_RECORDS by build_card_data)
    ({"type": "CARD", "title": "Performance Cards", "data": {
        "cards": [
            {"title": "Rakesh Kumar", "subtitle": "A", "value": "92", "description": ""},
            {"title": "Suresh Patel", "subtitle": "B", "value": "78", "description": ""},
        ]
    }}, PERF_RECORDS),

    # 3. CHART_BAR
    ({"type": "CHART_BAR", "title": "Best Total by Agniveer", "data": {
        "xKey": "fullName", "yKey": "bestTotal",
        "rows": [{"fullName": "Rakesh Kumar", "bestTotal": 92},
                 {"fullName": "Suresh Patel", "bestTotal": 78}],
    }}, PERF_RECORDS),

    # 4. CHART_LINE  (correct series0 mapping, as build_line_chart_data does)
    ({"type": "CHART_LINE", "title": "Attendance Trend", "data": {
        "xKey": "month",
        "series": [{"key": "series0", "label": "Present"},
                   {"key": "series1", "label": "Absent"}],
        "rows": [{"month": "04-2026", "series0": 22, "series1": 2},
                 {"month": "05-2026", "series0": 25, "series1": 1},
                 {"month": "06-2026", "series0": 24, "series1": 3}],
    }}, TREND_RECORDS),

    # 5. CHART_PIE
    ({"type": "CHART_PIE", "title": "Blood Group Distribution", "data": {
        "rows": [{"label": "O+", "value": 45}, {"label": "B+", "value": 30},
                 {"label": "A+", "value": 25}],
    }}, PIE_RECORDS),

    # 6. COMPARE_CARD
    ({"type": "COMPARE_CARD", "title": "Alpha vs Bravo", "data": {
        "left": [{"label": "Total Records", "value": "2"},
                 {"label": "Average Score", "value": "85.0"}],
        "right": [{"label": "Total Records", "value": "2"},
                  {"label": "Average Score", "value": "79.0"}],
    }}, COMBINED_COMPARE),

    # 7. COMPARE_TABLE
    ({"type": "COMPARE_TABLE", "title": "Alpha vs Bravo — Records", "data": {
        "left": {"heading": "Alpha Company",
                 "columns": [{"key": "fullName", "label": "Full Name"},
                             {"key": "bestTotal", "label": "Best Total"},
                             {"key": "month", "label": "Month"},
                             {"key": "score", "label": "Score"},
                             {"key": "agniveerId", "label": "Agniveer ID"}],
                 "row": [{"fullName": "R Kumar", "bestTotal": 92, "month": "05-2026",
                          "score": 92, "agniveerId": 1}]},
        "right": {"heading": "Bravo Company",
                  "columns": [{"key": "fullName", "label": "Full Name"},
                              {"key": "bestTotal", "label": "Best Total"},
                              {"key": "month", "label": "Month"},
                              {"key": "score", "label": "Score"},
                              {"key": "agniveerId", "label": "Agniveer ID"}],
                  "row": [{"fullName": "A Singh", "bestTotal": 88, "month": "05-2026",
                           "score": 88, "agniveerId": 3}]},
    }}, COMBINED_COMPARE),

    # 8. COMPARE_CHART_BAR
    ({"type": "COMPARE_CHART_BAR", "title": "Alpha vs Bravo — Scores", "data": {
        "left": {"heading": "Alpha Company", "xKey": "fullName", "yKey": "bestTotal",
                 "rows": [{"fullName": "R Kumar", "bestTotal": 92},
                          {"fullName": "S Patel", "bestTotal": 78}]},
        "right": {"heading": "Bravo Company", "xKey": "fullName", "yKey": "bestTotal",
                  "rows": [{"fullName": "A Singh", "bestTotal": 88},
                           {"fullName": "M Shah", "bestTotal": 70}]},
    }}, COMBINED_COMPARE),

    # 9. COMPARE_CHART_LINE — reproduces the CURRENT bug in _build_compare_line:
    #    series declares original key names but rows use series0/series1
    ({"type": "COMPARE_CHART_LINE", "title": "Alpha vs Bravo — Trend", "data": {
        "left": {"heading": "Alpha Company", "xKey": "month",
                 "series": [{"key": "bestTotal", "label": "Besttotal"},
                            {"key": "score", "label": "Score"}],
                 "rows": [{"month": "05-2026", "series0": 92, "series1": 92},
                          {"month": "06-2026", "series0": 78, "series1": 78}]},
        "right": {"heading": "Bravo Company", "xKey": "month",
                  "series": [{"key": "bestTotal", "label": "Besttotal"},
                             {"key": "score", "label": "Score"}],
                  "rows": [{"month": "05-2026", "series0": 88, "series1": 88},
                           {"month": "06-2026", "series0": 70, "series1": 70}]},
    }}, COMBINED_COMPARE),

    # 10. COMPARE_CHART_PIE
    ({"type": "COMPARE_CHART_PIE", "title": "Alpha vs Bravo — Distribution", "data": {
        "left": {"heading": "Alpha Company",
                 "rows": [{"label": "R Kumar", "value": 92}, {"label": "S Patel", "value": 78}]},
        "right": {"heading": "Bravo Company",
                  "rows": [{"label": "A Singh", "value": 88}, {"label": "M Shah", "value": 70}]},
    }}, COMBINED_COMPARE),
]


if __name__ == "__main__":
    for widget, source in WIDGETS:
        print_report(audit_widget(widget, source))
    print()