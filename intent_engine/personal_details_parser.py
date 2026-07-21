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

    # 0b. Attribute filters with no single "show me field X" ask (e.g. "list
    # every Agniveer belonging to Batch 3", "Agniveers taller than 175 cm",
    # "who joined in 2024") — these have no category keyword the main
    # classifier's vocabulary recognises at all, so nothing scores except a
    # stray fuzzy-match point on an unrelated category (observed: Equipment/
    # AgniveerWise winning by default at ~0.2 confidence).
    if "agniveer" in q_lower:
        height_match = re.search(
            r"\b(taller|shorter|height)\b.*?\b(than|above|below|over|under)\s*(\d+(?:\.\d+)?)",
            q_lower,
        )
        if height_match:
            descriptor, comparator, threshold = (
                height_match.group(1),
                height_match.group(2),
                float(height_match.group(3)),
            )
            op = "<" if descriptor == "shorter" or comparator in ("below", "under") else ">"
            return {
                "category": "personaldetail",
                "operation": "lookup",
                "height_filter": {"operator": op, "value": threshold},
                "query_type": "simple",
                "confidence": "high",
                "confidence_score": 0.9,
                "filters": {},
            }

        join_match = re.search(r"\bjoin(?:ed|ing)?\b[^.?]*?\b(19\d{2}|20\d{2})\b", q_lower)
        if join_match:
            return {
                "category": "personaldetail",
                "operation": "lookup",
                "join_year": join_match.group(1),
                "query_type": "simple",
                "confidence": "high",
                "confidence_score": 0.9,
                "filters": {},
            }

        if re.search(r"\bbatch\s+[a-z0-9]+\b", q_lower) and re.search(
            r"\b(belong|belonging|list|show|give|all|every)\b", q_lower
        ):
            return {
                "category": "personaldetail",
                "operation": "lookup",
                "query_type": "simple",
                "confidence": "high",
                "confidence_score": 0.9,
                "filters": {},
            }

    # 0c. Plain total headcount ("how many Agniveers are there in total?").
    # Deliberately narrow — "are/is there" is distinctive phrasing that
    # won't false-fire on "how many Agniveers WERE PRESENT..." or "...ARE
    # HOSPITALIZED..." (those belong to Attendance/Leave, not a total count),
    # and is skipped entirely if "active"/"inactive" already matched above.
    if re.search(r"\bhow many agniveers?\s+(?:are|is)\s+there\b", q_lower) or re.search(
        r"\btotal number of agniveers\b", q_lower
    ):
        return {
            "category": "personaldetail",
            "operation": "ActiveStatusCount",
            "is_active": None,
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
