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

_AGNIVEER_NO_RE = re.compile(r"\b([A-Za-z]\d{5,8}[A-Za-z]?)\b")

# Signals that this query is asking about something beyond a plain personal-
# detail lookup (roster listing, "who plays X") — a Performance ranking, a
# Leave/Medical/Equipment/Verification status, etc. — in which case an
# early-exit heuristic here would answer only a fragment of a more complex
# question. "Show top performer in PPT who plays cricket and is currently on
# leave" contains "plays cricket" (matches the sport pattern below) but is
# really a 3-way cross-filter; matching sport alone silently drops the
# ranking and leave-status halves of the question.
_OTHER_DOMAIN_WORDS_RE = re.compile(
    r"\b(leave|verification|verified|pending|rejected|medical|bmi|"
    r"overweight|underweight|obese|equipment|issued|returned|"
    r"attendance|present|absent|score|scored|top|performer|performers|"
    r"grade|graded|excellent|performance|bpet|ppt|firing|drill|hospital|"
    r"hospitalized|disqualified|status|diagnos\w*|blood|eyesight|"
    r"follow-?up)\b"
)


def parse_personal_details(query: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to heuristically match personal detail queries.
    Returns an intent dictionary if matched, else None.
    """
    q_lower = query.lower().strip()
    agn_match = _AGNIVEER_NO_RE.search(query)

    # -1. "When did A0701943X join?" — DateOfJoining via the verb "join",
    # not the noun phrase "date of joining" the generic column-alias lookup
    # (further below) requires. Checked before anything else needs the
    # literal word "agniveer" since a query naming a specific AgniveerNo
    # usually doesn't say "agniveer" at all.
    if agn_match and re.search(r"\bjoin(?:ed|ing)?\b", q_lower):
        return {
            "category": "personaldetail",
            "operation": "lookup",
            "metric": "DateOfJoining",
            "agniveer_no": agn_match.group(1).upper(),
            "query_type": "simple",
            "confidence": "high",
            "confidence_score": 0.9,
            "filters": {},
        }

    # 0. Active / inactive status — either a plain "how many active
    # Agniveers" / "list all inactive/removed Agniveers" (the whole
    # roster), or "Is A0701948W active?" (one specific Agniveer, via
    # agn_match). AgniveerMaster.IsActive is a distinct concept from
    # IsDisqualified (the "disqualified" category), so it needs its own
    # match or "inactive/removed" silently falls into disqualified-tracking
    # and "active" isn't recognised at all.
    if ("agniveer" in q_lower or agn_match) and (
        re.search(r"\b(in)?active\b", q_lower) or "removed" in q_lower
    ):
        is_active = 0 if ("inactive" in q_lower or "removed" in q_lower) else 1
        wants_count = bool(
            re.search(r"\bhow many\b|\bcount of\b|\btotal number\b|\bnumber of\b", q_lower)
        )
        result: Dict[str, Any] = {
            "category": "personaldetail",
            "operation": "ActiveStatusCount" if wants_count else "ActiveStatusList",
            "is_active": is_active,
            "query_type": "simple",
            "confidence": "high",
            "confidence_score": 1.0,
            "filters": {},
        }
        if agn_match and not wants_count:
            result["agniveer_no"] = agn_match.group(1).upper()
        return result

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

        # A plain roster ask — "give me a list of agniveers in <Company/
        # Platoon/Batch N>" — with no aggregate/ranking language AND no
        # other domain content. Company/platoon NAME -> ID resolution
        # happens separately upstream in admin_entity_resolver.py; this only
        # needs to recognise the question shape. Without the domain-word
        # exclusion, this swallowed queries that only incidentally contain
        # "show"/"give" + "in"/"of"/"from" ANYWHERE in the sentence —
        # "Show Agniveers who are overweight despite scoring Excellent in
        # BPET", "give me all Agniveers who have taken leave from Company
        # X", and "Show verification status of Agniveer X" were all being
        # forced into a plain personal-detail profile listing instead of
        # their real category (cross_filter / Leave / Verification).
        if (
            (
                re.search(r"\bbatch\s+[a-z0-9]+\b", q_lower)
                and re.search(r"\b(belong|belonging|list|show|give|all|every)\b", q_lower)
            )
            or (
                re.search(r"\b(list|show|give)\b", q_lower)
                and re.search(r"\b(in|of|from|belonging to|under)\b", q_lower)
                and not re.search(
                    r"\bwhich\b|\bhow many\b|\bmost\b|\bfewest\b|\bleast\b|\beach\b|\btop\b|\bbest\b|\bworst\b",
                    q_lower,
                )
            )
        ) and not _OTHER_DOMAIN_WORDS_RE.search(q_lower):
            return {
                "category": "personaldetail",
                "operation": "lookup",
                "query_type": "simple",
                "confidence": "high",
                "confidence_score": 0.85,
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
    # Generic catch-all for "plays <sport>" — but not when other domain
    # content is also present ("top performer in PPT who plays cricket and
    # is currently on leave" is a 3-way cross-filter, not a plain sport
    # lookup; matching sport alone here silently dropped the ranking and
    # leave-status halves of the question).
    sport_match = (
        None
        if _OTHER_DOMAIN_WORDS_RE.search(q_lower)
        else re.search(r'\bplays?\s+([a-z0-9]+)\b', q_lower)
    )
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
