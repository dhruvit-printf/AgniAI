"""
entity_extractor.py
===================

Single responsibility: extract entities from the user query.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from query_normalizer import clean_query
from query_understanding_engine import understand_query

from .intent_schema import (
    BLOOD_GROUPS,
    BMI_CATEGORIES,
    CLASSES,
    GRADING_CATEGORIES,
    ISSUED_EQUIPMENT_ITEMS,
    LEAVE_TYPES,
    PROCURED_EQUIPMENT_ITEMS,
    RANKING_CONTEXT_PHRASES,
    RELATIVE_DATE_PHRASES,
    SECTION,
    SPORTS,
    SUBSECTIONS_BY_SECTION,
    UNIT_ALIASES,
)

_BLOOD_GROUPS_SORTED = sorted(BLOOD_GROUPS, key=len, reverse=True)

_MONTH_PATTERN = (
    r"\b(January|February|March|April|May|June|July|August|September|October|"
    r"November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b"
)
_ISO_DATE_PATTERN = r"\b\d{4}-\d{2}-\d{2}\b"
_SLASH_DATE_PATTERN = r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b"
_EQUIPMENT_CONTEXT_PHRASES = (
    "equipment",
    "gear",
    "inventory",
    "kit",
    "issued",
    "procured",
    "overdue",
    "returned",
    "holding",
    "item",
    "items",
    "wearing",
    "worn",
    "coat",
    "dress",
    "boot",
    "boots",
    "cap",
    "belt",
    "mug",
    "blanket",
    "bag",
    "sling",
    "shoes",
    "shoe",
    "shirt",
    "trouser",
    "vest",
    "bottle",
    "bucket",
    "mattress",
    "locker",
    "track suit",
    "combat",
)
_NON_EQUIPMENT_DOMAIN_HINTS = (
    "performance",
    "score",
    "marks",
    "grading",
    "leave",
    "medical",
    "attendance",
    "verification",
    "distribution",
    "roster",
    "skills",
    "strength",
    "overall",
    "schedule",
)
_STOPWORDS = {
    "a",
    "all",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "list",
    "of",
    "on",
    "or",
    "show",
    "the",
    "to",
    "with",
    "who",
    "what",
    "which",
    "give",
    "me",
    "please",
    "stats",
    "summary",
    "report",
}


def _normalise(query: str) -> str:
    return clean_query(query).lower()


def _extract_number(query: str) -> Optional[int]:
    text = _normalise(query)
    if not text:
        return None

    blocked_prefixes = (
        "company",
        "coy",
        "batch",
        "platoon",
        "plt",
        "agniveer",
        "attempt",
        "section",
        "unit",
        "date",
        "day",
        "month",
        "year",
    )

    for phrase in RANKING_CONTEXT_PHRASES:
        pattern = rf"\b{re.escape(phrase)}\s+(\d+)\b"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start = max(0, match.start() - 24)
            context = text[start : match.start()].strip()
            if any(context.endswith(prefix) for prefix in blocked_prefixes):
                continue
            return int(match.group(1))

    match = re.search(r"\brank\s+(\d+)\b", text)
    if match:
        return int(match.group(1))
    return None


def _extract_section(query: str) -> Optional[str]:
    query_lower = _normalise(query)
    for section_name, section_data in SECTION.items():
        aliases = section_data.get("aliases", ())
        candidates = (section_name.lower(), *aliases)
        for alias in candidates:
            if len(alias) <= 2:
                # Require case-sensitive match for short acronyms like 'IT' or 'MR'
                orig_pattern = r"\b" + re.escape(alias.upper()).replace(r"\ ", r"\s+") + r"\b"
                if re.search(orig_pattern, query):
                    return section_name
            else:
                pattern = r"\b" + re.escape(alias).replace(r"\ ", r"\s+") + r"\b"
                if re.search(pattern, query_lower, re.IGNORECASE):
                    return section_name
    return None


def _extract_subsection(query: str, section: Optional[str]) -> Optional[str]:
    if not section or section not in SUBSECTIONS_BY_SECTION:
        return None
    query_lower = _normalise(query)
    for subsection in SUBSECTIONS_BY_SECTION[section]:
        if _normalise(subsection) in query_lower:
            return subsection
    return None


def _extract_grading(query: str) -> Optional[str]:
    query_lower = _normalise(query)
    for key, value in GRADING_CATEGORIES.items():
        if _normalise(key) in query_lower:
            return value
    return None


def _extract_leave_type(query: str) -> Optional[str]:
    query_lower = _normalise(query)
    if not any(
        token in query_lower
        for token in (
            "leave",
            "abscond",
            "absent",
            "status",
            "medical leave",
            "hospitalized",
            "threshold",
            "noleave",
            "annual",
            "sick",
        )
    ):
        return None
    for key, value in LEAVE_TYPES.items():
        if _normalise(key) in query_lower:
            if key == "medical" and "medical leave" not in query_lower:
                continue
            return value
    return None


def _extract_bmi_category(query: str) -> Optional[str]:
    query_lower = _normalise(query)
    for key, value in BMI_CATEGORIES.items():
        if _normalise(key) in query_lower:
            return value
    return None


def _extract_blood_group(query: str) -> Optional[str]:
    query_lower = _normalise(query)
    # Sort longest-first so "AB+" is checked before "B+" or "A+" to prevent substring false matches.
    for blood_group in _BLOOD_GROUPS_SORTED:
        variants = (
            blood_group.lower(),
            blood_group.replace("+", " positive").lower(),
            blood_group.replace("-", " negative").lower(),
        )
        if any(variant in query_lower for variant in variants):
            return blood_group
    return None


def _extract_sport(query: str) -> Optional[str]:
    query_lower = _normalise(query)
    for key, value in SPORTS.items():
        if _normalise(key) in query_lower:
            return value
    return None


def _extract_class(query: str) -> Optional[str]:
    query_lower = _normalise(query)
    for key, value in CLASSES.items():
        if _normalise(key) in query_lower:
            return value
    return None


def _tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"\s+", _normalise(text)) if t and t not in _STOPWORDS]


def _score_equipment_match(query_tokens: List[str], item: str) -> Tuple[int, int]:
    item_tokens = _tokenize(item)
    if not item_tokens:
        return 0, 0
    overlap = len(set(query_tokens) & set(item_tokens))
    phrase_hits = 0
    query_text = " ".join(query_tokens)
    item_text = " ".join(item_tokens)
    if item_text in query_text:
        phrase_hits += len(item_tokens)
    if query_text in item_text:
        phrase_hits += len(query_tokens)
    return overlap + phrase_hits, -len(item_tokens)


def _extract_equipment_item(query: str) -> Optional[str]:
    query_lower = _normalise(query)
    query_tokens = _tokenize(query_lower)
    has_equipment_context = any(
        phrase in query_lower for phrase in _EQUIPMENT_CONTEXT_PHRASES
    )
    has_non_equipment_domain = any(
        phrase in query_lower for phrase in _NON_EQUIPMENT_DOMAIN_HINTS
    )

    exact_matches: List[str] = []
    for item in ISSUED_EQUIPMENT_ITEMS + PROCURED_EQUIPMENT_ITEMS:
        item_lower = _normalise(item)
        if item_lower and item_lower in query_lower:
            exact_matches.append(item)
    if exact_matches:
        return exact_matches[0]

    if has_non_equipment_domain and not has_equipment_context:
        return None

    best_item: Optional[str] = None
    best_score: Tuple[int, int] = (0, 0)

    for item in ISSUED_EQUIPMENT_ITEMS + PROCURED_EQUIPMENT_ITEMS:
        score = _score_equipment_match(query_tokens, item)
        if score > best_score and score[0] > 0:
            best_item = item
            best_score = score

    if best_score[0] < 2 and not has_equipment_context:
        return None
    return best_item


def _extract_date_patterns(query: str) -> Optional[str]:
    query_lower = _normalise(query)

    for phrase, canonical in RELATIVE_DATE_PHRASES.items():
        if phrase in query_lower:
            return canonical

    if re.search(r"\bthis\s+week\b", query_lower):
        return "this week"
    if re.search(r"\blast\s+week\b", query_lower):
        return "last week"
    if re.search(r"\bthis\s+month\b", query_lower):
        return "current month"
    if re.search(r"\blast\s+month\b", query_lower):
        return "last month"
    if re.search(r"\bthis\s+year\b", query_lower):
        return "this year"

    match = re.search(_ISO_DATE_PATTERN, query)
    if match:
        return match.group(0)

    match = re.search(_SLASH_DATE_PATTERN, query)
    if match:
        return match.group(0)

    match = re.search(_MONTH_PATTERN, query, re.IGNORECASE)
    if match:
        return match.group(0)

    match = re.search(r"\bdate\s+(\d{1,2})\b", query, re.IGNORECASE)
    if match:
        return match.group(1)

    return None


def _extract_date_range(query: str) -> Tuple[Optional[str], Optional[str]]:
    query_lower = _normalise(query)
    match = re.search(
        r"\bfrom\s+(.+?)\s+(?:to|until)\s+(.+?)(?:$|[,.?])",
        query_lower,
        re.IGNORECASE,
    )
    if not match:
        return None, None
    return match.group(1).strip(), match.group(2).strip()


def _extract_attempt_no(query: str) -> Optional[int]:
    query_lower = _normalise(query)
    match = re.search(r"\battempt\s+(\d+)\b", query_lower)
    if match:
        return int(match.group(1))
    ordinals = {"first": 1, "second": 2, "third": 3}
    for ordinal, num in ordinals.items():
        if f"{ordinal} attempt" in query_lower:
            return num
    return None


def _extract_from_attempt(query: str) -> Optional[int]:
    query_lower = _normalise(query)
    match = re.search(r"(?:from|between)\s+attempt\s+(\d+)", query_lower)
    if match:
        return int(match.group(1))
    match = re.search(r"\battempt\s+(\d+)\s+to\s+attempt", query_lower)
    if match:
        return int(match.group(1))
    return None


def _extract_to_attempt(query: str) -> Optional[int]:
    query_lower = _normalise(query)
    match = re.search(r"(?:to|until)\s+attempt\s+(\d+)", query_lower)
    if match:
        return int(match.group(1))
    match = re.search(r"\battempt\s+\d+\s+to\s+(\d+)", query_lower)
    if match:
        return int(match.group(1))
    return None


def _extract_unit_name(query: str) -> Optional[str]:
    query_lower = _normalise(query)
    match = re.search(
        r"(?:\b(?:in|from|for)\s+unit\s+([A-Za-z]+)\b|\bunit\s+([A-Za-z]+)\b|\b([A-Za-z]+)\s+unit\b)",
        query_lower,
        re.IGNORECASE,
    )
    if match:
        token = (match.group(1) or match.group(2) or match.group(3) or "").lower()
        if token in UNIT_ALIASES:
            return UNIT_ALIASES[token]
        if len(token) == 1:
            return f"Unit {token.upper()}"
        return f"{token.capitalize()} Unit"
    for token, canonical in UNIT_ALIASES.items():
        if re.search(rf"\b{re.escape(token)}\b", query_lower):
            return canonical
    return None


def _extract_numeric_id(query: str, id_pattern: str) -> Optional[int]:
    query_lower = _normalise(query)
    match = re.search(rf"\b(?:{id_pattern})\s+(\d+)\b", query_lower, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _extract_company_id(query: str) -> Optional[int]:
    return _extract_numeric_id(query, "company|coy")


def _extract_platoon_id(query: str) -> Optional[int]:
    return _extract_numeric_id(query, "platoon|plt")


def _extract_batch_id(query: str) -> Optional[int]:
    return _extract_numeric_id(query, "batch")


def _extract_agniveer_no(query: str) -> Optional[str]:
    query_lower = _normalise(query)
    match = re.search(
        r"agniveer\s+(?:no\.?|no\.?\s+)?([a-z]?\d+[a-z]?)", query_lower, re.IGNORECASE
    )
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([A-Z]\d{5,8}[A-Z]?)\b", query)
    if match:
        return match.group(1).upper()
    return None


def _extract_medical_status(query: str) -> Optional[str]:
    query_lower = _normalise(query)
    if any(
        token in query_lower
        for token in (
            "active medical",
            "active case",
            "active cases",
            "under treatment",
            "in hospital",
            "admitted",
        )
    ):
        return "Active"
    return None


def _extract_days(query: str) -> Optional[int]:
    query_lower = _normalise(query)
    match = re.search(r"\b(?:for|last|in|days?:?)\s+(\d+)\s+days?\b", query_lower)
    if match:
        return int(match.group(1))
    match = re.search(r"\bdays?\s+(\d+)\b", query_lower)
    if match:
        return int(match.group(1))
    match = re.search(r"\b(\d+)\s+days?\b", query_lower)
    if match:
        return int(match.group(1))
    return None


CANONICAL_ENTITY_KEYS = frozenset(
    {
        "batchId",
        "platoonId",
        "companyId",
        "agniveerNo",
        "section",
        "subSection",
        "attemptNo",
        "fromAttempt",
        "toAttempt",
        "leaveType",
        "grading",
        "bmiCategory",
        "bloodGroup",
        "equipmentName",
        "sport",
        "class",
        "unitName",
        "n",
        "date",
        "fromDate",
        "toDate",
        "medicalStatus",
        "diagnose",
        "days",
    }
)


def assert_canonical_entity_keys(entities: Dict[str, Any]) -> None:
    for key in entities.keys():
        if key not in CANONICAL_ENTITY_KEYS:
            raise KeyError(f"Non-canonical entity key found: '{key}'")


def _extract_diagnose(query: str) -> Optional[str]:
    query_lower = _normalise(query)
    diseases = (
        "viral fever",
        "covid-19",
        "swine flu",
        "heat stroke",
        "scrub typhus",
        "kidney stone",
        "chicken pox",
        "chickenpox",
        "hepatitis b",
        "hepatitis a",
        "hepatitis c",
        "food poisoning",
        "gastroenteritis",
        "stomach flu",
        "panic attack",
        "bipolar disorder",
        "bone fracture",
        "hairline fracture",
        "bone crack",
        "ligament tear",
        "muscle pull",
        "back pain",
        "joint pain",
        "slipped disc",
        "rheumatoid arthritis",
        "osteoarthritis",
        "fever",
        "cough",
        "cold",
        "dengue",
        "malaria",
        "typhoid",
        "flu",
        "influenza",
        "asthma",
        "bronchitis",
        "pneumonia",
        "tuberculosis",
        "headache",
        "migraine",
        "covid",
        "hypertension",
        "diabetes",
        "cholera",
        "diarrhea",
        "dysentery",
        "sunstroke",
        "dehydration",
        "hepatitis",
        "jaundice",
        "rabies",
        "tetanus",
        "leprosy",
        "leptospirosis",
        "h1n1",
        "cancer",
        "hiv",
        "aids",
        "chikungunya",
        "meningitis",
        "encephalitis",
        "measles",
        "mumps",
        "rubella",
        "polio",
        "allergy",
        "acidity",
        "vomiting",
        "nausea",
        "constipation",
        "ulcer",
        "gastritis",
        "appendicitis",
        "arthritis",
        "hernia",
        "anemia",
        "thyroid",
        "insomnia",
        "depression",
        "anxiety",
        "stress",
        "ptsd",
        "schizophrenia",
        "fracture",
        "dislocation",
        "sprain",
        "concussion",
        "burn",
        "injury",
        "wound",
        "sciatica",
        "spasm",
        "fatigue",
    )
    for d in diseases:
        if d in query_lower:
            return d.title()
    return None


def extract_entities(
    query: str,
    resolved_entities: Optional[Dict[str, Any]] = None,
    semantic: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    raw_query = str(query or "").strip()
    resolved_entities = resolved_entities or {}
    if semantic is None:
        semantic = understand_query(raw_query)

    result: Dict[str, Any] = {
        "batchId": None,
        "platoonId": None,
        "companyId": None,
        "agniveerNo": None,
        "section": None,
        "subSection": None,
        "attemptNo": None,
        "fromAttempt": None,
        "toAttempt": None,
        "leaveType": None,
        "grading": None,
        "bmiCategory": None,
        "bloodGroup": None,
        "equipmentName": None,
        "sport": None,
        "class": None,
        "unitName": None,
        "n": None,
        "date": None,
        "fromDate": None,
        "toDate": None,
        "medicalStatus": None,
        "diagnose": None,
        "days": None,
    }

    result["n"] = _extract_number(raw_query)
    result["section"] = _extract_section(raw_query)
    if result["section"]:
        result["subSection"] = _extract_subsection(raw_query, result["section"])
    result["grading"] = _extract_grading(raw_query)
    result["leaveType"] = _extract_leave_type(raw_query)
    result["bmiCategory"] = _extract_bmi_category(raw_query)
    result["bloodGroup"] = _extract_blood_group(raw_query)
    result["sport"] = _extract_sport(raw_query)
    result["class"] = _extract_class(raw_query)
    result["equipmentName"] = _extract_equipment_item(raw_query)
    result["unitName"] = _extract_unit_name(raw_query)
    result["attemptNo"] = _extract_attempt_no(raw_query)
    result["fromAttempt"] = _extract_from_attempt(raw_query)
    result["toAttempt"] = _extract_to_attempt(raw_query)
    result["date"] = _extract_date_patterns(raw_query)
    result["fromDate"], result["toDate"] = _extract_date_range(raw_query)
    result["medicalStatus"] = _extract_medical_status(raw_query)
    result["diagnose"] = _extract_diagnose(raw_query)
    result["days"] = _extract_days(raw_query)

    result["companyId"] = (
        resolved_entities.get("company_id")
        or resolved_entities.get("companyId")
        or semantic.get("company_id")
        or semantic.get("companyId")
        or _extract_company_id(raw_query)
    )
    result["platoonId"] = (
        resolved_entities.get("platoon_id")
        or resolved_entities.get("platoonId")
        or semantic.get("platoon_id")
        or semantic.get("platoonId")
        or _extract_platoon_id(raw_query)
    )
    result["batchId"] = (
        resolved_entities.get("batch_id")
        or resolved_entities.get("batchId")
        or semantic.get("batch_id")
        or semantic.get("batchId")
        or _extract_batch_id(raw_query)
    )
    result["agniveerNo"] = (
        resolved_entities.get("agniveer_no")
        or resolved_entities.get("agniveerNo")
        or _extract_agniveer_no(raw_query)
    )

    return result
