"""
compare_engine.py
=================
Compare engine for comparing N-way datasets side-by-side.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Set

from utils import extract_records as _normalize_records
from utils import get_score as _get_score
from utils import safe_float as _safe_float

logger = logging.getLogger(__name__)

def _extract_records(data: Any) -> List[Dict]:
    """Pull the list of records out of any .NET wrapper shape."""
    return _normalize_records(data)

def _extract_summary_metrics(data: Any) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if isinstance(data, dict):
        inner = data
        for key in ("data", "Data", "result", "Result"):
            val = data.get(key)
            if isinstance(val, dict):
                inner = val
                break
        for k, v in inner.items():
            if isinstance(v, (int, float)):
                metrics[k] = v
            elif isinstance(v, str):
                try:
                    metrics[k] = float(v)
                except ValueError:
                    pass

    records = _extract_records(data)
    if records:
        metrics["recordCount"] = len(records)
        scores = [s for s in (_get_score(r) for r in records) if s is not None]
        if scores:
            metrics["averageScore"] = round(sum(scores) / len(scores), 2)
            metrics["topScore"] = max(scores)
            metrics["bottomScore"] = min(scores)
    return metrics

def compare_datasets(labeled_results: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """
    Compare side-by-side comparison of 2-way and N-way datasets.
    """
    sides: List[Dict[str, Any]] = []
    all_metric_keys: Set[str] = set()

    for label, data in labeled_results:
        is_unavailable = False
        if isinstance(data, dict) and data.get("unavailable") is True:
            is_unavailable = True
            metrics = {}
        else:
            metrics = _extract_summary_metrics(data)
            all_metric_keys.update(metrics.keys())

        side = {
            "label": label,
            "data": _extract_records(data),
            "metrics": metrics,
        }
        if is_unavailable:
            side["unavailable"] = True
        sides.append(side)

    # 2-way values
    left = sides[0] if len(sides) >= 1 else {}
    right = sides[1] if len(sides) >= 2 else {}

    # Calculate comparison for all metrics
    comparison_metrics = {}
    for metric in all_metric_keys:
        valid_sides = []
        for side in sides:
            val = _safe_float(side["metrics"].get(metric))
            if val is not None:
                valid_sides.append((side["label"], val))

        if len(valid_sides) >= 2:
            valid_sides.sort(key=lambda x: x[1])
            lowest_label, lowest_val = valid_sides[0]
            highest_label, highest_val = valid_sides[-1]
            diff = highest_val - lowest_val
            pct = (diff / lowest_val * 100.0) if lowest_val != 0 else 0.0
            comparison_metrics[metric] = {
                "higher": highest_label,
                "lower": lowest_label,
                "difference": round(diff, 2),
                "percentage": round(pct, 2),
            }
        elif len(valid_sides) == 1:
            label, val = valid_sides[0]
            comparison_metrics[metric] = {
                "higher": label,
                "lower": "N/A",
                "difference": 0.0,
                "percentage": 0.0,
            }

    return {
        "left": left,
        "right": right,
        "sides": sides,
        "comparedMetrics": sorted(all_metric_keys),
        "comparison": comparison_metrics,
        "comparisonMetrics": comparison_metrics
    }
