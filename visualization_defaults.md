# Visualization Defaults

This file documents the default visualization choice when the frontend does **not** send an override.

The backend behavior is split into two layers:

1. `visualization_intent.py` decides the intent shape.
2. `widget_selector.py` / `widget_engine.py` decide the actual widget(s) to show.

## Query Type To Default Visualization

| Query type | Default visualization | Notes |
| --- | --- | --- |
| `simple` | `TABLE` | Default fallback for most questions. |
| `compare` / `comparison` | auto-selected compare widget | If both sides infer the same chart type, use that compare chart. If the two sides differ, fall back to `COMPARE_TABLE`. `COMPARE_CARD` is treated as a legacy/internal shape and is rendered as a table. |
| `trend` | `LINE_CHART` | Used for time-based or growth-based questions. |
| `distribution` | `PIE_CHART` | Used for percentage, share, or breakdown questions. |
| `cross_filter` | `CARD` + `TABLE` | Summary card first, then matching records. |
| `multi_independent` | `TABLE` per section | One table per independent result section. |

## Explicit User Requests

When the user explicitly asks for a visual format, the frontend override wins.

| User request | Result |
| --- | --- |
| `pie chart` | `presentation = chart`, `chart_type = pie` |
| `bar chart` | `presentation = chart`, `chart_type = bar` |
| `line chart` | `presentation = chart`, `chart_type = line` |
| `donut chart` | `presentation = chart`, `chart_type = donut` |
| `radial chart` | `presentation = chart`, `chart_type = radial` |
| `area chart` | `presentation = chart`, `chart_type = area` |
| `table` / `tabular` | `presentation = table` |
| `cards` | `presentation = cards` |

## Comparison Overrides

If the question is a comparison and the user also asks for a specific chart style, the override is preserved.

Supported comparison overrides:

- `line`
- `bar`
- `pie`
- `donut` -> treated as `pie`
- `radial` -> treated as `pie`
- `area` -> treated as `line`

## Comparison Shape Rule

When there is no frontend override:

- `bar + bar` -> `COMPARE_BAR_CHART`
- `line + line` -> `COMPARE_LINE_CHART`
- `pie + pie` -> `COMPARE_PIE_CHART`
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
| Performance | Improvement | LINE_CHART |
| Performance | Drop | LINE_CHART |
| Performance | Grading | TABLE |
| Performance | GradingSummary | BAR_CHART |
| Performance | Average | PIE_CHART |
| Performance | AttemptWise | TABLE |
| Performance | BestAttempt | TABLE |
| Performance | Trend | LINE_CHART |
| Leave | Most / Highest | TABLE |
| Leave | Least | TABLE |
| Leave | Current | CARD |
| Leave | Absconded | CARD |
| Medical | BMI | DONUT_CHART |
| Medical | BloodGroup | PIE_CHART |
| Medical | Disease | BAR_CHART |
| Medical | Individual | CARD |
| Attendance | Monthly | BAR_CHART |
| Attendance | Weekly | BAR_CHART |
| Attendance | Daily | TABLE |
| Attendance | Present / On-Campus | PIE_CHART |
| Attendance | Summary | RADIAL_CHART |
| Verification | Pending | CARD |
| Verification | Sent | CARD |
| Verification | Not Responded | CARD |
| Verification | Completed / Verified | CARD |
| Verification | Rejected | CARD |
| Equipment | Stats / Summary | BAR_CHART |
| Equipment | Search | TABLE |
| Equipment | Returned / Poor Condition | CARD |
| Equipment | Holding / Currently Issued | CARD |
| Equipment | Agniveer-Wise | TABLE |
| Distribution | Latest | TABLE |
| Distribution | By Unit | BAR_CHART |
| Distribution | Unassigned | TABLE |
| Distribution | Top Unit | BAR_CHART |
| Skills | By Sport | TABLE |
| Skills | By Class | BAR_CHART |
| Strength | Strength Breakdown | RADIAL_CHART |
| Schedule | Today / Company Schedule | TABLE |
| Schedule | Agniveer Schedule | TABLE |
| Personal Details | Info | CARD |
| Disqualified | Removed | CARD |
| Overall | OverallPerformance | CARD |
