# Visualization Defaults

This file documents the default visualization choice when the frontend does **not** send an override.

The backend behavior is split into two layers:

1. `visualization_intent.py` decides the intent shape.
2. `widget_selector.py` / `widget_engine.py` decide the actual widget(s) to show.

## Widget Type Names

The canonical widget `type` values emitted by the backend are:

`TABLE` | `CARD` | `CHART_BAR` | `CHART_LINE` | `CHART_PIE` | `ATTENDANCE_CALENDAR` | `COMPARE_TABLE` | `COMPARE_CARD` | `COMPARE_CHART_BAR` | `COMPARE_CHART_LINE` | `COMPARE_CHART_PIE`

`ATTENDANCE_CALENDAR` is a special-purpose widget used only for `Attendance` / `Daily`. Its `data` shape is fixed:

```json
{
  "year": 2025,
  "month": 7,
  "agniveerNo": "",
  "agniveerName": "",
  "photoPath": "",
  "days": [
    { "date": "", "isPresent": true }
  ]
}
```

Legacy names (`BAR_CHART`, `LINE_CHART`, `PIE_CHART`, `AREA_CHART`, `DONUT_CHART`, `RADIAL_CHART`, `COMPARE_BAR_CHART`, `COMPARE_LINE_CHART`, `COMPARE_PIE_CHART`) are still accepted as **input** aliases (e.g. a frontend override), but are never emitted as output. Donut charts are folded into `CHART_PIE` / `COMPARE_CHART_PIE`; radial and area charts are folded into `CHART_LINE` / `COMPARE_CHART_LINE`.

## Query Type To Default Visualization

| Query type | Default visualization | Notes |
| --- | --- | --- |
| `simple` | `TABLE` | Default fallback for most questions. |
| `compare` / `comparison` | auto-selected compare widget | If both sides infer the same chart type, use that compare chart. If the two sides differ, fall back to `COMPARE_TABLE`. `COMPARE_CARD` is treated as a legacy/internal shape and is rendered as a table. |
| `trend` | `CHART_LINE` | Used for time-based or growth-based questions. |
| `distribution` | `CHART_PIE` | Used for percentage, share, or breakdown questions. |
| `cross_filter` | `CARD` + `TABLE` | Summary card first, then matching records. |
| `multi_independent` | `TABLE` per section | One table per independent result section. |

## Explicit User Requests

When the user explicitly asks for a visual format, the frontend override wins.

| User request | Result |
| --- | --- |
| `pie chart` | `presentation = chart`, `chart_type = pie` |
| `bar chart` | `presentation = chart`, `chart_type = bar` |
| `line chart` | `presentation = chart`, `chart_type = line` |
| `donut chart` | `presentation = chart`, `chart_type = pie` (folded into pie) |
| `radial chart` | `presentation = chart`, `chart_type = line` (folded into line) |
| `area chart` | `presentation = chart`, `chart_type = line` (folded into line) |
| `table` / `tabular` | `presentation = table` |
| `cards` | `presentation = cards` |

## Comparison Overrides

If the question is a comparison and the user also asks for a specific chart style, the override is preserved.

Supported comparison overrides:

- `line`
- `bar`
- `pie`
- `donut` -> treated as `pie`
- `radial` -> treated as `line`
- `area` -> treated as `line`

## Comparison Shape Rule

When there is no frontend override:

- `bar + bar` -> `COMPARE_CHART_BAR`
- `line + line` -> `COMPARE_CHART_LINE`
- `pie + pie` -> `COMPARE_CHART_PIE`
- any mixed pair, like `bar + line` -> `COMPARE_TABLE`

## Practical Rule

If there is no frontend override:

- `simple` questions default to tables.
- `trend` questions default to line charts.
- `distribution` questions default to pie charts.
- `compare` questions default to a shared compare chart only when both sides match; otherwise they default to a compare table.

## Category / Operation Defaults

The backend also applies summary-response defaults per operation. Detailed responses continue to fall back to the detailed widget path unless a more specific override exists.

| Category | Operation | Summary default |
| --- | --- | --- |
| Performance | Top | TABLE |
| Performance | Bottom | TABLE |
| Performance | Improvement | CHART_LINE |
| Performance | Drop | CHART_LINE |
| Performance | Grading | TABLE |
| Performance | GradingSummary | CHART_BAR |
| Performance | Average | CHART_PIE |
| Performance | AttemptWise | TABLE |
| Performance | BestAttempt | TABLE |
| Performance | Trend | CHART_LINE |
| Leave | Most / Highest | TABLE |
| Leave | Least | TABLE |
| Leave | Current | CARD |
| Leave | Absconded | CARD |
| Medical | BMI | CHART_PIE |
| Medical | BloodGroup | CHART_PIE |
| Medical | Disease | CHART_BAR |
| Medical | Individual | CARD |
| Attendance | Monthly | CHART_BAR |
| Attendance | Weekly | CHART_BAR |
| Attendance | Daily | ATTENDANCE_CALENDAR |
| Attendance | Present / On-Campus | CHART_PIE |
| Attendance | Summary | CHART_LINE |
| Verification | Pending | CARD |
| Verification | Sent | CARD |
| Verification | Not Responded | CARD |
| Verification | Completed / Verified | CARD |
| Verification | Rejected | CARD |
| Equipment | Stats / Summary | CHART_BAR |
| Equipment | Search | TABLE |
| Equipment | Returned / Poor Condition | CARD |
| Equipment | Holding / Currently Issued | CARD |
| Equipment | Agniveer-Wise | TABLE |
| Distribution | Latest | TABLE |
| Distribution | By Unit | CHART_BAR |
| Distribution | Unassigned | TABLE |
| Distribution | Top Unit | CHART_BAR |
| Skills | By Sport | TABLE |
| Skills | By Class | CHART_BAR |
| Strength | Strength Breakdown | CHART_LINE |
| Schedule | Today / Company Schedule | TABLE |
| Schedule | Agniveer Schedule | TABLE |
| Personal Details | Info | CARD |
| Disqualified | Removed | CARD |
| Overall | OverallPerformance | CARD |
