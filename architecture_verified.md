# AgniAI Architecture Status

This document reconciles the chatbot system's architectural reality. Features are categorized strictly by their current state to prevent "desired" design from being documented as "implemented" design.

---

## IMPLEMENTED
*The code for these features exists in the repository.*

1. **Unified Query Pipeline Orchestration**:
   - `admin_pipeline.execute_admin_query` is the single source of truth for both HTTP (`admin_routes.py`) and WebSocket (`websocket_routes.py`) transports.
2. **Canonical Query Classification**:
   - Consolidated 5 query types: `FILTER_QUERY`, `CROSS_FILTER`, `COMPARISON`, `ANALYTICS`, `MULTI_OPERATION`.
   - Zombie / redundant types (`SIMPLE`, `MULTI_INDEPENDENT`) deleted.
3. **Python-side Cross-Filtering Intersection**:
   - Python performs multi-set intersection of records by key `agniveerNo` (with ID fallback).
4. **Structured JSON Audit Logging**:
   - Structured JSON logging of query type, durations, combiner strategy, record count, and LLM report strategy is implemented.

---

## VERIFIED
*These features are actively exercised and verified by automated end-to-end tests (`tests/test_pipeline_e2e.py`).*

1. **`FILTER_QUERY` Execution**:
   - Verifies single .NET call + table widget generation.
2. **`CROSS_FILTER` Execution**:
   - Verifies N-way intersections of independent category sets.
3. **`COMPARISON` Execution**:
   - Verifies side-by-side metric comparison for sections.
4. **`MULTI_OPERATION` Execution**:
   - Verifies section merging for multiple independent queries.
5. **`ANALYTICS` Execution**:
   - Verifies analytics and ranking/grading queries.
6. **Observability, Health & Metrics Integration**:
   - Validates metrics collectors, trace ID propagation, circuit breakers, and log scrubbing.

---

## PLANNED
*Future work that is not yet implemented or verified.*

1. **Complex Semantic Entity Relationships**:
   - Deeper reasoning over relative clause syntax that goes beyond vocabulary splits.
2. **Dynamic Live Visualization Customization**:
   - Interactive widget custom filters by operators on the dashboard UI.
