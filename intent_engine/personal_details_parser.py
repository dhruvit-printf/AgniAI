import re
from typing import Optional, Dict, Any, List

# All 39 columns in AgniveerMaster
AGNIVEER_PERSONAL_COLUMNS = [
    "FullName",
    "AgniveerNo",
    "DateOfBirth",
    "DateOfJoining",
    "Address",
    "MobileNo",
    "EroName",
    "NextOfKin",
    "Class",
    "BloodGroup",
    "Height",
    "Weight",
    "EyeSight",
    "PlatoonId",
    "IsActive",
    "PhotoPath",
    "District",
    "Email",
    "EnrolledId",
    "HouseNo",
    "IdMarkI",
    "IdMarkI1",
    "MainCategory",
    "PinCode",
    "PoliceStation",
    "PostOffice",
    "Qualification",
    "State",
    "Tehsil",
    "Village",
    "Awards",
    "Certificate",
    "Hobby",
    "Skill",
    "Sports",
    "IsDisqualified",
    "Remarks",
    "SponserUnitId",
    "DisqualifiedDate",
]

# Lowercase mapping for fast lookup
COL_MAP = {col.lower(): col for col in AGNIVEER_PERSONAL_COLUMNS}

# Comprehensive Column Aliases
COLUMN_ALIASES = {
    # Identity
    "fullname": "FullName",
    "name": "FullName",
    "agniveer name": "FullName",
    "agniveername": "FullName",
    # Number
    "agniveer no": "AgniveerNo",
    "agniveerno": "AgniveerNo",
    "agniveer number": "AgniveerNo",
    "enrollment no": "AgniveerNo",
    "enrollment number": "AgniveerNo",
    "enrolled id": "EnrolledId",
    "enrolledid": "EnrolledId",
    # Dates
    "date of birth": "DateOfBirth",
    "dob": "DateOfBirth",
    "birth date": "DateOfBirth",
    "birthday": "DateOfBirth",
    "age": "DateOfBirth",
    "date of joining": "DateOfJoining",
    "doj": "DateOfJoining",
    "joining date": "DateOfJoining",
    "joined date": "DateOfJoining",
    "disqualified date": "DisqualifiedDate",
    "disqualification date": "DisqualifiedDate",
    # Address
    "address": "Address",
    "full address": "Address",
    "home address": "Address",
    "permanent address": "Address",
    "correspondence address": "Address",
    "state": "State",
    "district": "District",
    "tehsil": "Tehsil",
    "village": "Village",
    "city": "District",
    "town": "Tehsil",
    "pin code": "PinCode",
    "pincode": "PinCode",
    "zip code": "PinCode",
    "postal code": "PinCode",
    "zipcode": "PinCode",
    "post office": "PostOffice",
    "police station": "PoliceStation",
    "house no": "HouseNo",
    "houseno": "HouseNo",
    # Contact
    "mobile no": "MobileNo",
    "mobileno": "MobileNo",
    "phone": "MobileNo",
    "phone number": "MobileNo",
    "phone no": "MobileNo",
    "contact": "MobileNo",
    "contact number": "MobileNo",
    "contact no": "MobileNo",
    "mobile": "MobileNo",
    "mobile number": "MobileNo",
    "email": "Email",
    "email id": "Email",
    "mail": "Email",
    # Physical
    "height": "Height",
    "weight": "Weight",
    "eye sight": "EyeSight",
    "eyesight": "EyeSight",
    "vision": "EyeSight",
    # Personal
    "class": "Class",
    "community": "Class",
    "blood group": "BloodGroup",
    "bloodgroup": "BloodGroup",
    "blood type": "BloodGroup",
    "qualification": "Qualification",
    "education": "Qualification",
    "academic": "Qualification",
    "educational qualification": "Qualification",
    "ero name": "EroName",
    "ero": "EroName",
    "next of kin": "NextOfKin",
    "nok": "NextOfKin",
    "kin": "NextOfKin",
    "family": "NextOfKin",
    "main category": "MainCategory",
    "category": "MainCategory",
    # Marks/ID
    "id mark": "IdMarkI",
    "id mark 1": "IdMarkI",
    "id mark i": "IdMarkI",
    "identification mark": "IdMarkI",
    "id mark 2": "IdMarkI1",
    "id mark ii": "IdMarkI1",
    "identification mark 2": "IdMarkI1",
    # Activities
    "hobby": "Hobby",
    "hobbies": "Hobby",
    "skill": "Skill",
    "skills": "Skill",
    "talent": "Skill",
    "sport": "Sports",
    "sports": "Sports",
    "game": "Sports",
    "games": "Sports",
    "award": "Awards",
    "awards": "Awards",
    "achievement": "Awards",
    "certificate": "Certificate",
    "certificates": "Certificate",
    "certification": "Certificate",
    # Status
    "active": "IsActive",
    "status": "IsActive",
    "disqualified": "IsDisqualified",
    "disqualification": "IsDisqualified",
    "remarks": "Remarks",
    "remark": "Remarks",
    "comment": "Remarks",
    "note": "Remarks",
    "photo": "PhotoPath",
    "picture": "PhotoPath",
    "image": "PhotoPath",
    "platoon": "PlatoonId",
    "unit": "PlatoonId",
    "batch": "BatchId",
    "sponsor unit": "SponserUnitId",
    "sponser unit": "SponserUnitId",
}

# Merge all aliases into COL_MAP
COL_MAP.update(COLUMN_ALIASES)

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
            "metrics": ["DateOfJoining"],
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
            re.search(
                r"\bhow many\b|\bcount of\b|\btotal number\b|\bnumber of\b", q_lower
            )
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

    # 1. Check for aggregators: average, above average, below average, max, min
    agg_match = re.search(
        r"\b(above average|below average|average|max|maximum|min|minimum)\s+([a-z\s]+)\b",
        q_lower,
    )
    if agg_match:
        raw_agg = agg_match.group(1).replace("maximum", "max").replace("minimum", "min")
        raw_col = agg_match.group(2).strip()

        # Check if the extracted suffix matches a known column
        for col_alias, true_col in COL_MAP.items():
            if raw_col.startswith(col_alias) or col_alias in raw_col:
                op = raw_agg.replace(" ", "_")
                agg_res = {
                    "category": "personaldetail",
                    "operation": op,
                    "metric": true_col,
                    "metrics": [true_col],
                    "query_type": "simple",
                    "confidence": "high",
                    "confidence_score": 1.0,
                    "filters": {},
                }
                if agn_match:
                    agg_res["agniveer_no"] = agn_match.group(1).upper()
                return agg_res

    # 2. Categorical Match (e.g. "who plays cricket", "playing volleyball", "eye sight 6/6")
    sport_match = None
    if not _OTHER_DOMAIN_WORDS_RE.search(q_lower):
        sport_match = re.search(
            r"\b(?:play|plays|playing)\s+([a-z0-9]+)\b", q_lower
        ) or re.search(r"\b([a-z]+)\s+players?\b", q_lower)
        if not sport_match:
            for known_s in (
                "volleyball",
                "cricket",
                "football",
                "soccer",
                "hockey",
                "basketball",
                "kabaddi",
                "badminton",
                "tennis",
                "swimming",
                "athletics",
                "boxing",
                "wrestling",
                "handball",
                "squash",
            ):
                if known_s in q_lower:
                    sport_match = re.search(rf"\b({known_s})\b", q_lower)
                    break

    if sport_match:
        sport = sport_match.group(1)
        sport_res = {
            "category": "personaldetail",
            "operation": "match",
            "metric": "Sports",
            "metrics": ["Sports"],
            "value": sport.capitalize(),
            "sport": sport.capitalize(),
            "query_type": "simple",
            "confidence": "high",
            "confidence_score": 1.0,
            "filters": {},
        }
        if agn_match:
            sport_res["agniveer_no"] = agn_match.group(1).upper()

        from intent_engine.entity_extractor import extract_entities

        ents = extract_entities(query)
        if ents.get("companyName"):
            sport_res["company_name"] = ents["companyName"]
            sport_res["companyName"] = ents["companyName"]
        if ents.get("companyId"):
            sport_res["company_id"] = ents["companyId"]
            sport_res["companyId"] = ents["companyId"]
        if ents.get("platoonName"):
            sport_res["platoon_name"] = ents["platoonName"]
            sport_res["platoonName"] = ents["platoonName"]
        if ents.get("platoonId"):
            sport_res["platoon_id"] = ents["platoonId"]
            sport_res["platoonId"] = ents["platoonId"]
        if ents.get("batchId"):
            sport_res["batch_id"] = ents["batchId"]
            sport_res["batchId"] = ents["batchId"]
        return sport_res

    # Generic "eye sight <value>"
    eye_match = re.search(r"\beye\s*sight\s*(?:is\s*)?([0-9/]+)\b", q_lower)
    if eye_match:
        eye_res = {
            "category": "personaldetail",
            "operation": "match",
            "metric": "EyeSight",
            "metrics": ["EyeSight"],
            "value": eye_match.group(1),
            "query_type": "simple",
            "confidence": "high",
            "confidence_score": 1.0,
            "filters": {},
        }
        if agn_match:
            eye_res["agniveer_no"] = agn_match.group(1).upper()
        return eye_res

    # 3. Specific column field lookup ("what is the height", "height and weight",
    # "DOB and qualification of Agniveer A0701882L"). Checked BEFORE generic roster
    # matching so specific field queries are never swallowed into returning all 14 profile columns.
    # ONLY applies when no other domain words (e.g. equipment, issued, verification, leave, bpet, etc.)
    # are present in the query (which indicate a multi-domain / cross-filter / multi-independent query).
    if not _OTHER_DOMAIN_WORDS_RE.search(q_lower):
        found: List[Any] = []
        seen: set = set()
        for col_alias in sorted(COL_MAP.keys(), key=len, reverse=True):
            m = re.search(rf"\b{re.escape(col_alias)}\b", q_lower)
            if m:
                canonical = COL_MAP[col_alias]
                if canonical not in seen:
                    seen.add(canonical)
                    found.append((m.start(), canonical))

        if found:
            found.sort(key=lambda pair: pair[0])
            columns = [c for _, c in found]
            spec_res = {
                "category": "personaldetail",
                "operation": "lookup",
                "metric": columns[0],
                "metrics": columns,
                "query_type": "simple",
                "confidence": "high",
                "confidence_score": 1.0,
                "filters": {},
            }
            if agn_match:
                spec_res["agniveer_no"] = agn_match.group(1).upper()
            return spec_res

    # 4. Attribute filters with no single "show me field X" ask (e.g. "Agniveers taller than 175 cm", "who joined in 2024")
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
            op = (
                "<"
                if descriptor == "shorter" or comparator in ("below", "under")
                else ">"
            )
            return {
                "category": "personaldetail",
                "operation": "lookup",
                "height_filter": {"operator": op, "value": threshold},
                "query_type": "simple",
                "confidence": "high",
                "confidence_score": 0.9,
                "filters": {},
            }

        join_match = re.search(
            r"\bjoin(?:ed|ing)?\b[^.?]*?\b(19\d{2}|20\d{2})\b", q_lower
        )
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

        # Plain roster ask — "give me a list of agniveers in Batch 1", "show personal details of Agniveer A0701882L"
        if (
            (
                re.search(r"\bbatch\s+[a-z0-9]+\b", q_lower)
                and re.search(
                    r"\b(belong|belonging|list|show|give|all|every)\b", q_lower
                )
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
            roster_res = {
                "category": "personaldetail",
                "operation": "lookup",
                "query_type": "simple",
                "confidence": "high",
                "confidence_score": 0.85,
                "filters": {},
            }
            if agn_match:
                roster_res["agniveer_no"] = agn_match.group(1).upper()
            return roster_res

    # 5. Plain total headcount ("how many Agniveers are there in total?").
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

    return None
