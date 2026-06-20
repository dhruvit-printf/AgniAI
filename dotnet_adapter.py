"""
dotnet_adapter.py
=================
Canonical .NET response normalization helpers.
"""

from __future__ import annotations

from typing import Any, Dict, List


def normalize_dotnet_response(data: Any) -> Dict[str, Any]:
    """
    Return a canonical response shape with a top-level `records` list.

    The adapter preserves the original object for callers that need it, but
    gives every consumer one shared place for wrapper-unwrapping behavior.
    """
    records = extract_records(data)
    if isinstance(data, dict):
        normalized = dict(data)
        normalized["records"] = records
        return normalized
    return {"records": records}


def extract_records(data: Any) -> List[Dict]:
    if isinstance(data, dict):
        for key in ("data", "Data", "result", "Result", "records", "Records", "persons", "personnel"):
            val = data.get(key)
            if isinstance(val, list):
                return val
            if isinstance(val, dict):
                nested = extract_records(val)
                if nested:
                    return nested

        teams = data.get("teams") or data.get("Teams")
        if isinstance(teams, list):
            members: List[Dict] = []
            for team in teams:
                team_members = team.get("members") or team.get("Members") or []
                if isinstance(team_members, list):
                    members.extend(team_members)
            if members:
                return members

        for value in data.values():
            if isinstance(value, (dict, list)):
                nested = extract_records(value)
                if nested:
                    return nested

    if isinstance(data, list):
        return data
    return []
