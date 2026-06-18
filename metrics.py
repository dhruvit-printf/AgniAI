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
  active_websockets       gauge
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

        # Duration summaries (milliseconds)
        self.durations: Dict[str, Dict[str, float]] = {
            "planner_duration": {"sum": 0.0, "count": 0.0},
            "intent_duration": {"sum": 0.0, "count": 0.0},
            "dotnet_duration": {"sum": 0.0, "count": 0.0},
            "combiner_duration": {"sum": 0.0, "count": 0.0},
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

    # ── Durations ──────────────────────────────────────────────────────────

    def record_duration(self, metric_name: str, duration_ms: float) -> None:
        if metric_name in self.durations:
            with self._lock:
                self.durations[metric_name]["sum"] += duration_ms
                self.durations[metric_name]["count"] += 1.0

    # ── Active websockets ─────────────────────────────────────────────────

    @property
    def active_websockets(self) -> int:
        """Dynamically query active websockets count from the source of truth."""
        try:
            from websocket_manager import ws_manager

            return ws_manager.active_count
        except Exception:
            return 0

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

            # ── active_websockets ──────────────────────────────────────────
            lines.append(
                "# HELP active_websockets Number of currently active WebSocket connections."
            )
            lines.append("# TYPE active_websockets gauge")
            lines.append(f"active_websockets {self.active_websockets}")

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
