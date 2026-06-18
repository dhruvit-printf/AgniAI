"""
metrics.py
==========
Centralized registry for gathering, tracking, and exposing Prometheus-format
system and execution metrics for AgniAI. Zero dependencies.
"""

from __future__ import annotations

import threading
from typing import Dict

class Metrics:
    """Thread-safe registry for Prometheus instrumentation."""
    
    def __init__(self) -> None:
        self._lock = threading.Lock()
        
        # Counters
        self.requests_total: Dict[str, int] = {}
        self.errors_total: Dict[str, int] = {}
        
        # Durations Summaries
        self.durations: Dict[str, Dict[str, float]] = {
            "planner_duration": {"sum": 0.0, "count": 0.0},
            "intent_duration": {"sum": 0.0, "count": 0.0},
            "dotnet_duration": {"sum": 0.0, "count": 0.0},
            "report_duration": {"sum": 0.0, "count": 0.0},
            "pipeline_duration": {"sum": 0.0, "count": 0.0},
        }

    def inc_requests(self, query_type: str) -> None:
        with self._lock:
            self.requests_total[query_type] = self.requests_total.get(query_type, 0) + 1

    def inc_errors(self, query_type: str) -> None:
        with self._lock:
            self.errors_total[query_type] = self.errors_total.get(query_type, 0) + 1

    def record_duration(self, metric_name: str, duration_ms: float) -> None:
        if metric_name in self.durations:
            with self._lock:
                self.durations[metric_name]["sum"] += duration_ms
                self.durations[metric_name]["count"] += 1.0

    @property
    def active_websockets(self) -> int:
        """Dynamically query active websockets count from the source of truth registry."""
        try:
            from websocket_manager import ws_manager
            return ws_manager.active_count
        except Exception:
            return 0

    def generate_prometheus_text(self) -> str:
        """Format metrics in the standard Prometheus text-based presentation format."""
        lines = []
        with self._lock:
            # requests_total
            lines.append("# HELP requests_total Total number of queries processed.")
            lines.append("# TYPE requests_total counter")
            if not self.requests_total:
                lines.append('requests_total{query_type="simple"} 0')
            else:
                for qt, val in sorted(self.requests_total.items()):
                    lines.append(f'requests_total{{query_type="{qt}"}} {val}')
            
            # errors_total
            lines.append("# HELP errors_total Total number of failed queries.")
            lines.append("# TYPE errors_total counter")
            if not self.errors_total:
                lines.append('errors_total{query_type="simple"} 0')
            else:
                for qt, val in sorted(self.errors_total.items()):
                    lines.append(f'errors_total{{query_type="{qt}"}} {val}')
            
            # active_websockets
            lines.append("# HELP active_websockets Number of currently active WebSocket connections.")
            lines.append("# TYPE active_websockets gauge")
            lines.append(f"active_websockets {self.active_websockets}")
            
            # Summaries
            for name in sorted(self.durations.keys()):
                data = self.durations[name]
                lines.append(f"# HELP {name} Duration of {name.replace('_', ' ')} in milliseconds.")
                lines.append(f"# TYPE {name} summary")
                lines.append(f"{name}_sum {data['sum']}")
                lines.append(f"{name}_count {int(data['count'])}")
                
        return "\n".join(lines) + "\n"

# Global singleton exporter
metrics_collector = Metrics()
