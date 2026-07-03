"""
compare_engine.py
=================
Compare engine for comparing N-way datasets side-by-side.
"""

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from utils import extract_records as _normalize_records
from utils import get_score as _get_score
from utils import safe_float as _safe_float

logger = logging.getLogger(__name__)


def _extract_records(data: Any) -> List[Dict]:
    """Pull the list of records out of any .NET wrapper shape."""
    return _normalize_records(data)


def _is_flat_summary_dict(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    for value in data.values():
        if isinstance(value, (dict, list)):
            return False
    return True


def _extract_chronological_key(record: Dict) -> Any:
    for k in ("date", "Date", "createdDate", "CreatedDate", "timestamp", "Timestamp"):
        val = record.get(k)
        if val:
            return str(val)
    for k in ("attempt", "attemptNo", "Attempt", "AttemptNo"):
        val = _safe_float(record.get(k))
        if val is not None:
            return val
    month = record.get("month") or record.get("Month")
    year = record.get("year") or record.get("Year")
    if year is not None:
        if month is not None:
            return (year, month)
        return year
    return None


def _extract_summary_metrics(data: Any) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    inner: Dict[str, Any] = {}
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
    if records and not (isinstance(data, dict) and _is_flat_summary_dict(inner)):
        metrics["recordCount"] = len(records)
        scores = [s for s in (_get_score(r) for r in records) if s is not None]
        if scores:
            metrics["averageScore"] = round(sum(scores) / len(scores), 2)
            metrics["topScore"] = max(scores)
            metrics["bottomScore"] = min(scores)
            metrics["average"] = metrics["averageScore"]
            metrics["top"] = metrics["topScore"]
            metrics["bottom"] = metrics["bottomScore"]
    return metrics


def _infer_side_visualization_type(side: Dict[str, Any]) -> str:
    records = side.get("data") or []
    if not records:
        return "COMPARE_CARD"

    sample = records[0]
    if not isinstance(sample, dict):
        return "COMPARE_TABLE"

    sample_keys = {k.lower() for k in sample.keys()}

    table_keys = {"fullname", "agniveerno", "rollno", "personnel", "name", "agniveerid"}
    if any(tk in sample_keys for tk in table_keys):
        return "COMPARE_TABLE"

    time_keys = {"date", "month", "year", "attempt", "timestamp", "week", "createddate"}
    if any(tk in sample_keys for tk in time_keys):
        return "COMPARE_LINE_CHART"

    pie_keys = {
        "leavetype",
        "leavestatus",
        "leavecategory",
        "medicalcategory",
        "disease",
        "bloodgroup",
        "blood_group",
        "bmicategory",
        "bmi_category",
        "grading",
        "grade",
    }
    if any(pk in sample_keys for pk in pie_keys):
        return "COMPARE_PIE_CHART"

    bar_keys = {
        "company",
        "platoon",
        "batch",
        "unit",
        "section",
        "group",
        "label",
        "sports",
        "sport",
    }
    if any(bk in sample_keys for bk in bar_keys):
        return "COMPARE_BAR_CHART"

    if len(records) == 1:
        if len(sample.keys()) <= 3:
            return "COMPARE_CARD"
        return "COMPARE_TABLE"

    if len(sample.keys()) > 3:
        return "COMPARE_TABLE"

    return "COMPARE_BAR_CHART"


def select_visualization_type(sides: List[Dict[str, Any]]) -> str:
    inferred = [
        _infer_side_visualization_type(side)
        for side in sides
        if isinstance(side, dict)
    ]
    inferred = [viz for viz in inferred if viz]

    if not inferred:
        return "COMPARE_CARD"

    unique = set(inferred)
    if len(unique) == 1:
        return unique.pop()

    # Mixed visualization shapes should not be forced into a chart.
    return "COMPARE_TABLE"


def compare_datasets(
    labeled_results: List[Tuple[str, Any]],
    comparison_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compare side-by-side comparison of 2-way and N-way datasets.
    """
    if not comparison_context:
        datasets = []
        for idx, (label, data) in enumerate(labeled_results):
            datasets.append(
                {
                    "id": f"dataset_{idx + 1}",
                    "label": label,
                    "intent": {},
                    "dotnetPayload": {},
                    "rawData": data,
                    "metadata": {},
                }
            )
        comparison_context = {"datasets": datasets}

    sides: List[Dict[str, Any]] = []
    all_metric_keys: Set[str] = set()

    for ds in comparison_context["datasets"]:
        data = ds["rawData"]
        is_unavailable = False
        if isinstance(data, dict) and data.get("unavailable") is True:
            is_unavailable = True
            metrics = {}
        else:
            metrics = _extract_summary_metrics(data)
            all_metric_keys.update(metrics.keys())

        side = {
            "id": ds["id"],
            "label": ds["label"],
            "data": _extract_records(data),
            "metrics": metrics,
        }
        if is_unavailable:
            side["unavailable"] = True
        sides.append(side)

    # Auto-resolve recordCount from summary counts metrics if not present
    for ds, side in zip(comparison_context["datasets"], sides):
        if side.get("unavailable"):
            continue
        if "recordCount" not in side["metrics"]:
            intent = ds.get("intent") or {}
            operation = intent.get("operation")
            subcategory = intent.get("subcategory")

            candidates = []
            if operation:
                candidates.extend([operation.lower(), f"{operation.lower()}count", f"{operation.lower()}s", f"{operation.lower()}scount"])
            if subcategory:
                candidates.extend([subcategory.lower(), f"{subcategory.lower()}count", f"{subcategory.lower()}s", f"{subcategory.lower()}scount"])

            # Alias mappings for Completed / Verified
            if operation in ("Completed", "Verified") or subcategory in ("CompletedVerification", "VerifiedVerification"):
                candidates.extend(["verified", "verifiedcount", "completed", "completedcount"])

            found_val = None
            for key, val in side["metrics"].items():
                k_lower = key.lower()
                if k_lower in candidates:
                    found_val = val
                    break

            if found_val is not None:
                side["metrics"]["recordCount"] = found_val
                all_metric_keys.add("recordCount")

    visualization_type = select_visualization_type(sides)

    left = sides[0] if len(sides) >= 1 else {}
    right = sides[1] if len(sides) >= 2 else {}

    comparison_metrics = {}

    comparison_metrics_payload = {
        "recordCount": {},
        "highest": {},
        "lowest": {},
        "difference": {},
        "percentageDifference": {},
        "variance": {},
        "trendDifference": {},
        "distributionDifference": {},
        "categoryDifference": {},
    }

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

            raw_vals = [v[1] for v in valid_sides]
            mean = sum(raw_vals) / len(raw_vals)
            variance = sum((x - mean) ** 2 for x in raw_vals) / len(raw_vals)

            comparison_metrics[metric] = {
                "higher": highest_label,
                "lower": lowest_label,
                "difference": round(diff, 2),
                "percentage": round(pct, 2),
            }

            values_map = {label: val for label, val in valid_sides}
            if metric == "recordCount":
                comparison_metrics_payload["recordCount"] = {
                    "values": values_map,
                    "difference": round(diff, 2),
                    "percentageDifference": round(pct, 2),
                }
            comparison_metrics_payload["highest"][metric] = {
                "label": highest_label,
                "value": round(highest_val, 2),
            }
            comparison_metrics_payload["lowest"][metric] = {
                "label": lowest_label,
                "value": round(lowest_val, 2),
            }
            comparison_metrics_payload["difference"][metric] = round(diff, 2)
            comparison_metrics_payload["percentageDifference"][metric] = round(pct, 2)
            comparison_metrics_payload["variance"][metric] = round(variance, 2)

        elif len(valid_sides) == 1:
            label, val = valid_sides[0]
            comparison_metrics[metric] = {
                "higher": label,
                "lower": "N/A",
                "difference": 0.0,
                "percentage": 0.0,
            }
            values_map = {label: val}
            if metric == "recordCount":
                comparison_metrics_payload["recordCount"] = {
                    "values": values_map,
                    "difference": 0.0,
                    "percentageDifference": 0.0,
                }
            comparison_metrics_payload["highest"][metric] = {
                "label": label,
                "value": round(val, 2),
            }
            comparison_metrics_payload["lowest"][metric] = {
                "label": "N/A",
                "value": 0.0,
            }
            comparison_metrics_payload["difference"][metric] = 0.0
            comparison_metrics_payload["percentageDifference"][metric] = 0.0
            comparison_metrics_payload["variance"][metric] = 0.0

    trend_diff = {}
    if visualization_type == "COMPARE_LINE_CHART":
        for s in sides:
            recs = s.get("data") or []
            if len(recs) >= 2:
                sorted_recs = sorted(
                    recs, key=lambda x: _extract_chronological_key(x) or ""
                )
                first_val = _get_score(sorted_recs[0])
                last_val = _get_score(sorted_recs[-1])
                if first_val is not None and last_val is not None:
                    delta = last_val - first_val
                    trend_diff[s["label"]] = round(delta, 2)
    comparison_metrics_payload["trendDifference"] = trend_diff

    comparison_metrics_payload["distributionDifference"] = {}
    comparison_metrics_payload["categoryDifference"] = {}

    return {
        "left": left,
        "right": right,
        "sides": sides,
        "comparedMetrics": sorted(all_metric_keys),
        "comparison": comparison_metrics,
        "comparisonMetrics": comparison_metrics_payload,
        "visualizationType": visualization_type,
    }
