import json
from widget_engine import (
    build_table_data, build_card_data, build_bar_chart_data, 
    build_line_chart_data, build_pie_chart_data,
    _build_compare_bar, _build_compare_line, _build_compare_pie
)
from widget_field_audit import audit_widget, print_report

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
]

PIE_RECORDS = [
    {"bloodGroup": "O+", "count": 45, "disease": "None", "grade": "A", "IsActive": True, "id": 99},
]

COMBINED_COMPARE = {
    "left": {
        "label": "Alpha Company",
        "data": [
            {"agniveerId": 1, "fullName": "R Kumar", "bestTotal": 92, "month": "05-2026", "score": 92, "company": "Alpha"},
        ],
        "metrics": {"recordCount": 2, "averageScore": 85.0}
    },
    "right": {
        "label": "Bravo Company",
        "data": [
            {"agniveerId": 3, "fullName": "A Singh", "bestTotal": 88, "month": "05-2026", "score": 88, "company": "Bravo"},
        ],
        "metrics": {"recordCount": 2, "averageScore": 79.0}
    },
}

bar_data = build_bar_chart_data({"data": PERF_RECORDS})
line_data = build_line_chart_data({"data": TREND_RECORDS}) # Wait, build_line_chart_data uses _extract_records(combined_result)
# build_line_chart_data actually expects combined_result. Let's construct it correctly.
line_data = build_line_chart_data({"data": TREND_RECORDS})
pie_data = build_pie_chart_data({"data": PIE_RECORDS})

print("========== DYNAMIC BAR CHART AUDIT ==========")
print_report(audit_widget({"type": "CHART_BAR", "data": bar_data}, PERF_RECORDS))

print("\n========== DYNAMIC LINE CHART AUDIT ==========")
print_report(audit_widget({"type": "CHART_LINE", "data": line_data}, TREND_RECORDS))

print("\n========== DYNAMIC PIE CHART AUDIT ==========")
print_report(audit_widget({"type": "CHART_PIE", "data": pie_data}, PIE_RECORDS))

print("\n========== DYNAMIC COMPARE CHARTS AUDIT ==========")
print_report(audit_widget({"type": "COMPARE_CHART_BAR", "data": _build_compare_bar(COMBINED_COMPARE)}, COMBINED_COMPARE))
print_report(audit_widget({"type": "COMPARE_CHART_LINE", "data": _build_compare_line(COMBINED_COMPARE)}, COMBINED_COMPARE))
print_report(audit_widget({"type": "COMPARE_CHART_PIE", "data": _build_compare_pie(COMBINED_COMPARE)}, COMBINED_COMPARE))
