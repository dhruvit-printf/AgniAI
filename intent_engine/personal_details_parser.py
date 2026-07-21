import re
from typing import Optional, Dict, Any, List

# All numeric/comparable or text columns in AgniveerMaster that the user might query
AGNIVEER_PERSONAL_COLUMNS = [
    'Address', 'Awards', 'BloodGroup', 'Certificate', 'Class', 'DateOfBirth', 
    'DateOfJoining', 'DisqualifiedDate', 'District', 'Email', 'EroName', 
    'EyeSight', 'FullName', 'Height', 'Hobby', 'HouseNo', 
    'IdMarkI', 'IdMarkI1', 'MainCategory', 'MobileNo', 'NextOfKin', 
    'PhotoPath', 'PinCode', 'PoliceStation', 'PostOffice', 'Qualification', 
    'Remarks', 'Skill', 'Sports', 'State', 'Tehsil', 'Village', 'Weight'
]

# Lowercase mapping for fast lookup
COL_MAP = {col.lower(): col for col in AGNIVEER_PERSONAL_COLUMNS}
# Aliases
COL_MAP["eye sight"] = "EyeSight"
COL_MAP["blood group"] = "BloodGroup"
COL_MAP["contact number"] = "MobileNo"
COL_MAP["contact no"] = "MobileNo"
COL_MAP["phone number"] = "MobileNo"
COL_MAP["mobile number"] = "MobileNo"
COL_MAP["date of birth"] = "DateOfBirth"
COL_MAP["date of joining"] = "DateOfJoining"
COL_MAP["next of kin"] = "NextOfKin"
COL_MAP["house no"] = "HouseNo"

def parse_personal_details(query: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to heuristically match personal detail queries.
    Returns an intent dictionary if matched, else None.
    """
    q_lower = query.lower().strip()

    # 0. Active / inactive status ("how many active Agniveers", "list all
    # inactive/removed Agniveers"). This is the AgniveerMaster.IsActive flag
    # — a distinct concept from IsDisqualified (the "disqualified" category)
    # — so it needs its own match, or "inactive/removed" silently falls into
    # the disqualified-tracking flow, and "active" isn't recognised at all.
    if "agniveer" in q_lower and (
        re.search(r"\b(in)?active\b", q_lower) or "removed" in q_lower
    ):
        is_active = 0 if ("inactive" in q_lower or "removed" in q_lower) else 1
        wants_count = bool(
            re.search(r"\bhow many\b|\bcount of\b|\btotal number\b|\bnumber of\b", q_lower)
        )
        return {
            "category": "personaldetail",
            "operation": "ActiveStatusCount" if wants_count else "ActiveStatusList",
            "is_active": is_active,
            "query_type": "simple",
            "confidence": "high",
            "confidence_score": 1.0,
            "filters": {},
        }

    # 1. Check for aggregators: average, above average, below average, max, min
    agg_match = re.search(r'\b(above average|below average|average|max|maximum|min|minimum)\s+([a-z\s]+)\b', q_lower)
    if agg_match:
        raw_agg = agg_match.group(1).replace("maximum", "max").replace("minimum", "min")
        raw_col = agg_match.group(2).strip()
        
        # Check if the extracted suffix matches a known column
        for col_alias, true_col in COL_MAP.items():
            if raw_col.startswith(col_alias) or col_alias in raw_col:
                op = raw_agg.replace(" ", "_")
                return {
                    "category": "personaldetail",
                    "operation": op,
                    "metric": true_col,
                    "query_type": "simple",
                    "confidence": "high",
                    "confidence_score": 1.0,
                    "filters": {}
                }
                
    # 2. Categorical Match (e.g. "who plays cricket", "eye sight 6/6")
    # Generic catch-all for "plays <sport>"
    sport_match = re.search(r'\bplays?\s+([a-z0-9]+)\b', q_lower)
    if sport_match:
        sport = sport_match.group(1)
        return {
            "category": "personaldetail",
            "operation": "match",
            "metric": "Sports",
            "value": sport,
            "query_type": "simple",
            "confidence": "high",
            "confidence_score": 1.0,
            "filters": {}
        }
        
    # Generic "eye sight <value>"
    eye_match = re.search(r'\beye\s*sight\s*(?:is\s*)?([0-9/]+)\b', q_lower)
    if eye_match:
        return {
            "category": "personaldetail",
            "operation": "match",
            "metric": "EyeSight",
            "value": eye_match.group(1),
            "query_type": "simple",
            "confidence": "high",
            "confidence_score": 1.0,
            "filters": {}
        }
        
    # 3. Generic column lookup ("what is the height", or "height and weight"
    # for more than one field at once — every matched column is kept, not
    # just the first, so a multi-field ask doesn't silently lose the rest).
    # Sort columns by length descending so longer aliases match first and
    # a shorter alias can't steal part of one already matched (e.g. "date of
    # birth" before "date of joining" both containing "date").
    found: List[Any] = []
    seen: set = set()
    for col_alias in sorted(COL_MAP.keys(), key=len, reverse=True):
        m = re.search(rf'\b{re.escape(col_alias)}\b', q_lower)
        if m:
            canonical = COL_MAP[col_alias]
            if canonical not in seen:
                seen.add(canonical)
                found.append((m.start(), canonical))

    if found:
        found.sort(key=lambda pair: pair[0])
        columns = [c for _, c in found]
        return {
            "category": "personaldetail",
            "operation": "lookup",
            "metric": columns[0],
            "metrics": columns,
            "query_type": "simple",
            "confidence": "high",
            "confidence_score": 1.0,
            "filters": {}
        }

    return None
