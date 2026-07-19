"""
metrics.py
==========
Centralized Prometheus-compatible metrics registry for AgniAI.
Grafana-ready metric names. Thread-safe. Zero hard dependencies on
prometheus_client — degrades gracefully if not installed.

Metric names (Grafana-ready):
  requests_total          {query_type}
  successful_queries_total {query_type}
  failed_queries_total    {query_type}
  planner_duration        summary (ms)
  intent_duration         summary (ms)
  dotnet_duration         summary (ms)
  combiner_duration       summary (ms)
  report_duration         summary (ms)
  pipeline_duration       summary (ms)
  llm_failures            counter
  dotnet_failures         counter
  timeout_failures        counter

Slow query thresholds:
  > 5000 ms  → WARNING
  > 10000 ms → WARNING (slow_10s)
  > 30000 ms → ERROR   (slow_30s)
"""

from __future__ import annotations

import threading
from typing import Dict


class Metrics:
    """Thread-safe registry for Prometheus instrumentation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

        # Counters — requests by query type
        self.requests_total: Dict[str, int] = {}
        self.successful_queries_total: Dict[str, int] = {}
        self.failed_queries_total: Dict[str, int] = {}
        self.errors_total: Dict[str, int] = {}

        # Failure type counters
        self.llm_failures: int = 0
        self.dotnet_failures: int = 0
        self.timeout_failures: int = 0

        # Cache counters
        self.cache_hits: int = 0
        self.cache_misses: int = 0

        # SQL text-to-SQL backend counters (sql_executor.py)
        self.sql_generated_total: int = 0
        self.sql_validator_rejected_total: int = 0
        self.sql_cannot_answer_total: int = 0
        self.sql_exec_error_total: int = 0
        self.sql_llm_fallback_total: int = 0
        self.sql_capability_gap_fallback_total: int = 0
        self.sql_structural_reject_fallback_total: int = 0
        self.sql_latency_seconds: Dict[str, float] = {"sum": 0.0, "count": 0.0}

        # Duration summaries (milliseconds)
        self.durations: Dict[str, Dict[str, float]] = {
            "entity_resolution_ms": {"sum": 0.0, "count": 0.0},
            "planning_ms": {"sum": 0.0, "count": 0.0},
            "planner_duration": {"sum": 0.0, "count": 0.0},
            "intent_duration": {"sum": 0.0, "count": 0.0},
            "dotnet_duration": {"sum": 0.0, "count": 0.0},
            "combiner_duration": {"sum": 0.0, "count": 0.0},
            "widget_duration": {"sum": 0.0, "count": 0.0},
            "response_assembly_duration": {"sum": 0.0, "count": 0.0},
            "report_duration": {"sum": 0.0, "count": 0.0},
            "pipeline_duration": {"sum": 0.0, "count": 0.0},
        }

    # ── Counters ──────────────────────────────────────────────────────────

    def inc_requests(self, query_type: str) -> None:
        with self._lock:
            self.requests_total[query_type] = self.requests_total.get(query_type, 0) + 1

    def inc_success(self, query_type: str) -> None:
        with self._lock:
            self.successful_queries_total[query_type] = (
                self.successful_queries_total.get(query_type, 0) + 1
            )

    def inc_errors(self, query_type: str) -> None:
        with self._lock:
            self.errors_total[query_type] = self.errors_total.get(query_type, 0) + 1
            self.failed_queries_total[query_type] = (
                self.failed_queries_total.get(query_type, 0) + 1
            )

    def inc_llm_failure(self) -> None:
        with self._lock:
            self.llm_failures += 1

    def inc_dotnet_failure(self) -> None:
        with self._lock:
            self.dotnet_failures += 1

    def inc_timeout_failure(self) -> None:
        with self._lock:
            self.timeout_failures += 1

    def inc_cache_hits(self) -> None:
        with self._lock:
            self.cache_hits += 1

    def inc_cache_misses(self) -> None:
        with self._lock:
            self.cache_misses += 1

    # ── SQL backend counters ─────────────────────────────────────────────

    def inc_sql_generated(self) -> None:
        with self._lock:
            self.sql_generated_total += 1

    def inc_sql_validator_rejected(self) -> None:
        with self._lock:
            self.sql_validator_rejected_total += 1

    def inc_sql_cannot_answer(self) -> None:
        with self._lock:
            self.sql_cannot_answer_total += 1

    def inc_sql_exec_error(self) -> None:
        with self._lock:
            self.sql_exec_error_total += 1

    def inc_sql_llm_fallback(self) -> None:
        with self._lock:
            self.sql_llm_fallback_total += 1

    def inc_sql_capability_gap_fallback(self) -> None:
        with self._lock:
            self.sql_capability_gap_fallback_total += 1

    def inc_sql_structural_reject_fallback(self) -> None:
        with self._lock:
            self.sql_structural_reject_fallback_total += 1
            self.sql_validator_rejected_total += 1

    def record_sql_latency(self, duration_seconds: float) -> None:
        with self._lock:
            self.sql_latency_seconds["sum"] += duration_seconds
            self.sql_latency_seconds["count"] += 1.0

    # ── Durations ──────────────────────────────────────────────────────────

    def record_duration(self, metric_name: str, duration_ms: float) -> None:
        if metric_name in self.durations:
            with self._lock:
                self.durations[metric_name]["sum"] += duration_ms
                self.durations[metric_name]["count"] += 1.0

    # ── Prometheus text export ────────────────────────────────────────────

    def generate_prometheus_text(self) -> str:
        """Format metrics in the Prometheus text exposition format (v0.0.4)."""
        lines = []
        with self._lock:
            # ── requests_total ─────────────────────────────────────────────
            lines.append("# HELP requests_total Total number of queries processed.")
            lines.append("# TYPE requests_total counter")
            if not self.requests_total:
                lines.append('requests_total{query_type="simple"} 0')
            else:
                for qt, val in sorted(self.requests_total.items()):
                    lines.append(f'requests_total{{query_type="{qt}"}} {val}')

            # ── successful_queries_total ───────────────────────────────────
            lines.append("# HELP successful_queries_total Total successful queries.")
            lines.append("# TYPE successful_queries_total counter")
            if not self.successful_queries_total:
                lines.append('successful_queries_total{query_type="simple"} 0')
            else:
                for qt, val in sorted(self.successful_queries_total.items()):
                    lines.append(f'successful_queries_total{{query_type="{qt}"}} {val}')

            # ── failed_queries_total ───────────────────────────────────────
            lines.append("# HELP failed_queries_total Total failed queries.")
            lines.append("# TYPE failed_queries_total counter")
            if not self.failed_queries_total:
                lines.append('failed_queries_total{query_type="simple"} 0')
            else:
                for qt, val in sorted(self.failed_queries_total.items()):
                    lines.append(f'failed_queries_total{{query_type="{qt}"}} {val}')

            # ── errors_total ───────────────────────────────────────────────
            lines.append("# HELP errors_total Total number of failed queries.")
            lines.append("# TYPE errors_total counter")
            if not self.errors_total:
                lines.append('errors_total{query_type="simple"} 0')
            else:
                for qt, val in sorted(self.errors_total.items()):
                    lines.append(f'errors_total{{query_type="{qt}"}} {val}')

            # ── failure type counters ──────────────────────────────────────
            lines.append("# HELP llm_failures Total LLM (Ollama) failures.")
            lines.append("# TYPE llm_failures counter")
            lines.append(f"llm_failures {self.llm_failures}")

            lines.append("# HELP dotnet_failures Total .NET API failures.")
            lines.append("# TYPE dotnet_failures counter")
            lines.append(f"dotnet_failures {self.dotnet_failures}")

            lines.append("# HELP timeout_failures Total timeout failures.")
            lines.append("# TYPE timeout_failures counter")
            lines.append(f"timeout_failures {self.timeout_failures}")

            # ── cache counters ─────────────────────────────────────────────
            lines.append("# HELP cache_hits Total cache hits.")
            lines.append("# TYPE cache_hits counter")
            lines.append(f"cache_hits {self.cache_hits}")

            lines.append("# HELP cache_misses Total cache misses.")
            lines.append("# TYPE cache_misses counter")
            lines.append(f"cache_misses {self.cache_misses}")

            # ── SQL backend counters ────────────────────────────────────────
            lines.append("# HELP sql_generated_total Total LLM-generated SQL queries.")
            lines.append("# TYPE sql_generated_total counter")
            lines.append(f"sql_generated_total {self.sql_generated_total}")

            lines.append(
                "# HELP sql_validator_rejected_total Total SQL rejected by validate_sql."
            )
            lines.append("# TYPE sql_validator_rejected_total counter")
            lines.append(
                f"sql_validator_rejected_total {self.sql_validator_rejected_total}"
            )

            lines.append(
                "# HELP sql_cannot_answer_total Total CANNOT_ANSWER responses from the SQL generator."
            )
            lines.append("# TYPE sql_cannot_answer_total counter")
            lines.append(f"sql_cannot_answer_total {self.sql_cannot_answer_total}")

            lines.append("# HELP sql_exec_error_total Total SQL execution errors.")
            lines.append("# TYPE sql_exec_error_total counter")
            lines.append(f"sql_exec_error_total {self.sql_exec_error_total}")

            lines.append(
                "# HELP sql_llm_fallback_total Total SQL queries that fell back to LLM."
            )
            lines.append("# TYPE sql_llm_fallback_total counter")
            lines.append(f"sql_llm_fallback_total {self.sql_llm_fallback_total}")

            lines.append(
                "# HELP sql_capability_gap_fallback_total Total SQL queries that fell back due to capability gap."
            )
            lines.append("# TYPE sql_capability_gap_fallback_total counter")
            lines.append(
                f"sql_capability_gap_fallback_total {self.sql_capability_gap_fallback_total}"
            )

            lines.append(
                "# HELP sql_structural_reject_fallback_total Total SQL queries that fell back due to structural validator rejection."
            )
            lines.append("# TYPE sql_structural_reject_fallback_total counter")
            lines.append(
                f"sql_structural_reject_fallback_total {self.sql_structural_reject_fallback_total}"
            )

            lines.append(
                "# HELP sql_latency_seconds SQL backend end-to-end latency in seconds."
            )
            lines.append("# TYPE sql_latency_seconds histogram")
            lines.append(f"sql_latency_seconds_sum {self.sql_latency_seconds['sum']}")
            lines.append(
                f"sql_latency_seconds_count {int(self.sql_latency_seconds['count'])}"
            )

            # ── duration summaries ─────────────────────────────────────────
            for name in sorted(self.durations.keys()):
                data = self.durations[name]
                lines.append(
                    f"# HELP {name} Duration of {name.replace('_', ' ')} in milliseconds."
                )
                lines.append(f"# TYPE {name} summary")
                lines.append(f"{name}_sum {data['sum']}")
                lines.append(f"{name}_count {int(data['count'])}")

        return "\n".join(lines) + "\n"


# Global singleton exporter
metrics_collector = Metrics()
