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
    SUBSECTION_ALIASES,
    SUBSECTIONS_BY_SECTION,
    UNIT_ALIASES,
)

_BLOOD_GROUPS_SORTED = sorted(BLOOD_GROUPS, key=len, reverse=True)

_MONTH_PATTERN = (
    r"\b(January|February|March|April|May|June|July|August|September|October|"
    r"November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b"
)
# Bare month name with no year (e.g. "in June", "for March") — only recognised
# when preceded by a date preposition, so plain English "may" (modal verb) etc.
# isn't misread as a date reference.
_BARE_MONTH_PATTERN = (
    r"\b(?:in|for|during|of|on)\s+"
    r"(January|February|March|April|May|June|July|August|September|October|"
    r"November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b"
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

    blocked_suffixes = (
        "days",
        "day",
        "marks",
        "percent",
        "percentage",
        "score",
        "kg",
        "cm",
        "times",
    )

    # Generic fallback: find any standalone number that isn't preceded by blocked prefixes
    # and isn't a likely year/id.
    for match in re.finditer(r"\b(\d+)\b", text):
        num = int(match.group(1))

        # Skip likely years or large numbers (e.g., Agniveer numbers) if it's > 1000
        if 2000 <= num <= 2100 or num > 1000:
            continue

        start_before = max(0, match.start() - 24)
        context_before = text[start_before : match.start()].strip()

        end_after = min(len(text), match.end() + 24)
        context_after = text[match.end() : end_after].strip()

        if any(context_before.endswith(prefix) for prefix in blocked_prefixes):
            continue

        if any(context_after.startswith(suffix) for suffix in blocked_suffixes):
            continue

        return num

    return None


def detect_query_number_override(raw_query: str) -> Optional[int]:
    """Public: whether the raw query explicitly names a count ("top 5",
    "bottom 10", "rank 3"). Used by query_planner.py to propagate one count
    mentioned once in a multi-part query onto every rankable operation, not
    just whichever fragment's own text happened to contain it.
    """
    return _extract_number(raw_query)


def _extract_section(query: str) -> Optional[str]:
    query_lower = _normalise(query)
    for section_name, section_data in SECTION.items():
        aliases = section_data.get("aliases", ())
        candidates = (section_name.lower(), *aliases)
        for alias in candidates:
            if len(alias) <= 2:
                # Require case-sensitive match for short acronyms like 'IT' or 'MR'
                orig_pattern = (
                    r"\b" + re.escape(alias.upper()).replace(r"\ ", r"\s+") + r"\b"
                )
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
        candidates = (subsection,) + SUBSECTION_ALIASES.get(subsection, ())
        for candidate in candidates:
            if _normalise(candidate) in query_lower:
                return subsection
    return None


_GRADING_CONTEXT_WORDS = frozenset(
    {
        "grade",
        "grading",
        "score",
        "marks",
        "performance",
        "result",
        "rated",
        "scored",
        "achieved",
        "got",
        "obtained",
        "received",
        "classification",
        "percentage",
        "percent",
    }
    | {v.lower() for v in SECTION.keys()}
)
_GRADING_AMBIGUOUS = frozenset(
    {"good", "excellent", "satisfactory", "sat", "fail", "failed", "failing"}
)


def _extract_grading(query: str) -> Optional[str]:
    query_lower = _normalise(query)

    # Grading neutrality: if both pass and fail concepts are mentioned, return None
    # to avoid falsely narrowing to just one of them.
    has_pass = any(w in query_lower for w in ("pass", "passed", "passing"))
    has_fail = any(w in query_lower for w in ("fail", "failed", "failing"))
    if has_pass and has_fail:
        return None

    has_grading_context = any(w in query_lower for w in _GRADING_CONTEXT_WORDS)
    for key, value in GRADING_CATEGORIES.items():
        phrase = _normalise(key)
        if not re.search(rf"\b{re.escape(phrase)}\b", query_lower):
            continue
        # Ambiguous single words need a grading-context word to avoid false positives
        # ("good morning", "satisfactory work", "fail rate" without grading context)
        if phrase in _GRADING_AMBIGUOUS and not has_grading_context:
            continue
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
            "exhausted",
        )
    ):
        return None
    for key, value in LEAVE_TYPES.items():
        if _normalise(key) in query_lower:
            if key == "medical" and "medical leave" not in query_lower:
                continue
            if key == "threshold" and not _has_threshold_filter_signal(query_lower):
                # We used to ignore bare "threshold" mentions, but the user explicitly
                # wants "threshold" to map to the Threshold leave type.
                pass
            return value
    return None


_THRESHOLD_FILTER_SIGNALS = (
    "near",
    "nearing",
    "close to",
    "almost",
    "above",
    "below",
    "reached",
    "crossed",
    "hit ",
    "limit",
    "cap",
    "quota",
    "allowance",
    "warning level",
    "critical level",
    "danger zone",
    "safe limit",
    "ceiling",
    "boundary",
    "cutoff",
    "benchmark",
    "90%",
    "90 %",
    "Threshold",
    "threshold",
    "Threshhold",
    "Thresholds",
)


def _has_threshold_filter_signal(query_lower: str) -> bool:
    return any(signal in query_lower for signal in _THRESHOLD_FILTER_SIGNALS)


_BMI_AMBIGUOUS_TERMS = frozenset({"fit", "normal"})
_BMI_CONTEXT_WORDS = frozenset(
    {"bmi", "weight", "fitness", "medical", "health", "fat", "thin"}
)


def _extract_bmi_category(query: str) -> Optional[str]:
    query_lower = _normalise(query)
    for key, value in BMI_CATEGORIES.items():
        phrase = _normalise(key)
        if phrase in _BMI_AMBIGUOUS_TERMS:
            # Only match highly ambiguous terms (like "fit", "normal") when a medical/BMI context word is present
            has_context = any(w in query_lower for w in _BMI_CONTEXT_WORDS)
            if not has_context:
                continue
        if re.search(rf"\b{re.escape(phrase)}\b", query_lower):
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


_SPORTS_SORTED = sorted(SPORTS.items(), key=lambda kv: len(kv[0]), reverse=True)


def _extract_sport(query: str) -> Optional[str]:
    query_lower = _normalise(query)
    for key, value in _SPORTS_SORTED:
        phrase = _normalise(key)
        if re.search(rf"\b{re.escape(phrase)}\b", query_lower):
            return value
    return None


def _extract_class(query: str) -> Optional[str]:
    query_lower = _normalise(query)
    for key, value in CLASSES.items():
        phrase = _normalise(key)
        if re.search(rf"\b{re.escape(phrase)}\b", query_lower):
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


def _extract_equipment_type(query: str) -> Optional[str]:
    """Extract equipment type (Issued / Procured) from query keywords."""
    query_lower = _normalise(query)
    _ISSUED_HINTS = ("issued", "issue", "currently issued")
    _PROCURED_HINTS = ("procured", "purchased", "bought")
    for hint in _ISSUED_HINTS:
        if hint in query_lower:
            return "Issued"
    for hint in _PROCURED_HINTS:
        if hint in query_lower:
            return "Procured"
    return None


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

    match = re.search(_BARE_MONTH_PATTERN, query, re.IGNORECASE)
    if match:
        return match.group(1)

    match = re.search(r"\bdate\s+(\d{1,2})\b", query, re.IGNORECASE)
    if match:
        return match.group(1)

    return None


def _extract_date_range(query: str) -> Tuple[Optional[str], Optional[str]]:
    query_lower = _normalise(query)

    def _extract_date_fragment(fragment: str) -> Optional[str]:
        """Pull just the date-like substring out of a captured "from X to Y"
        fragment, discarding trailing words the outer regex over-captured
        (e.g. "june for company 2" -> "june")."""
        fragment = fragment.strip()
        if not fragment:
            return None
        match = re.search(_ISO_DATE_PATTERN, fragment)
        if match:
            return match.group(0)
        match = re.search(_SLASH_DATE_PATTERN, fragment)
        if match:
            return match.group(0)
        match = re.search(_MONTH_PATTERN, fragment, re.IGNORECASE)
        if match:
            return match.group(0)
        match = re.search(
            r"\b(January|February|March|April|May|June|July|August|September|"
            r"October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b",
            fragment,
            re.IGNORECASE,
        )
        if match:
            # Bare month name — safe to accept here since it's already inside
            # a "from X to Y" fragment, i.e. date context is established.
            return match.group(0)
        for phrase in (
            "today",
            "yesterday",
            "tomorrow",
            "this week",
            "last week",
            "this month",
            "last month",
            "this year",
            "last year",
        ):
            if phrase in fragment:
                return phrase
        return None

    match = re.search(
        r"\bfrom\s+(.+?)\s+(?:to|until)\s+(.+?)(?:$|[,.?])",
        query_lower,
        re.IGNORECASE,
    )
    if not match:
        return None, None

    from_fragment = _extract_date_fragment(match.group(1))
    to_fragment = _extract_date_fragment(match.group(2))
    if not (from_fragment or to_fragment):
        return None, None
    return from_fragment, to_fragment


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
    # Pattern 1: "unit alpha", "alpha unit", "in unit alpha", "from unit alpha"
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
    # Phonetic-alphabet aliases ONLY match when next to the word "unit"
    # (prevents bare "india", "golf", "delta" etc. from triggering)
    for token, canonical in UNIT_ALIASES.items():
        pattern = rf"\bunit\s+{re.escape(token)}\b|\b{re.escape(token)}\s+unit\b"
        if re.search(pattern, query_lower, re.IGNORECASE):
            return canonical
    return None


def _extract_numeric_id(query: str, id_pattern: str) -> Optional[int]:
    query_lower = _normalise(query)
    match = re.search(rf"\b(?:{id_pattern})\s+(\d+)\b", query_lower, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def _extract_company_id(query: str) -> Optional[int]:
    return _extract_numeric_id(query, "company")


def _extract_platoon_id(query: str) -> Optional[int]:
    return _extract_numeric_id(query, "platoon|plt")


def _extract_batch_id(query: str) -> Optional[int]:
    return _extract_numeric_id(query, "batch")


def _extract_agniveer_no(query: str) -> Optional[str]:
    query_lower = _normalise(query)
    match = re.search(
        r"agniveer\s*(?:no\.?)?\s*([a-z]?\d+[a-z]?)", query_lower, re.IGNORECASE
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
        token in query_lower for token in ("under treatment", "in hospital", "admitted")
    ):
        return "Admitted"
    if any(
        phrase in query_lower
        for phrase in (
            "medically unfit",
            "not fit for duty",
            "unfit for duty",
            "unfit",
        )
    ):
        return "Unfit"
    if any(
        phrase in query_lower
        for phrase in (
            "medically fit",
            "fit for duty",
            "fit",
        )
    ):
        return "Fit"
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


def _extract_return_condition(
    query: str, semantic: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    query_lower = _normalise(query)
    op = ""
    if semantic and semantic.get("module", "").lower() == "equipment":
        op = str(semantic.get("operation") or "").lower()

    for cond in ("good", "fair", "poor", "damaged"):
        if re.search(rf"\b(given|issued)\s+(in\s+)?{cond}\b", query_lower):
            continue
        if re.search(rf"\breturn(ed)?\s+(in\s+)?{cond}\b", query_lower):
            return cond
        if re.search(rf"\b{cond}\b", query_lower):
            if op == "sent":
                continue
            return cond
    return None


def _extract_given_condition(
    query: str, semantic: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    query_lower = _normalise(query)
    op = ""
    if semantic and semantic.get("module", "").lower() == "equipment":
        op = str(semantic.get("operation") or "").lower()

    for cond in ("good", "fair", "poor", "damaged"):
        if re.search(rf"\b(given|issued)\s+(in\s+)?{cond}\b", query_lower):
            return cond
        if re.search(rf"\b{cond}\b", query_lower):
            if op == "sent" and not re.search(
                rf"\breturn(ed)?\s+(in\s+)?{cond}\b", query_lower
            ):
                return cond
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
        "equipmentType",
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
        "returnCondition",
        "givenCondition",
        "Operation",
        "Category",
    }
)


def assert_canonical_entity_keys(entities: Dict[str, Any]) -> None:
    for key in entities.keys():
        if key not in CANONICAL_ENTITY_KEYS:
            raise KeyError(f"Non-canonical entity key found: '{key}'")


_DIAGNOSE_AMBIGUOUS = frozenset(
    {
        "cold",
        "stress",
        "burn",
        "injury",
        "fatigue",
        "wound",
        "allergy",
        "depression",
        "fever",
    }
)
_MEDICAL_CONTEXT_WORDS = frozenset(
    {
        "medical",
        "sick",
        "suffering",
        "diagnosed",
        "hospital",
        "ill",
        "patient",
        "treatment",
    }
)


_KNOWN_DISEASES = (
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


def _extract_diagnose(query: str) -> Optional[str]:
    query_lower = _normalise(query)
    has_medical_context = any(w in query_lower for w in _MEDICAL_CONTEXT_WORDS)
    for d in _KNOWN_DISEASES:
        if not re.search(rf"\b{re.escape(d)}\b", query_lower):
            continue
        # Ambiguous terms require a medical context word
        if d in _DIAGNOSE_AMBIGUOUS and not has_medical_context:
            continue
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
        "equipmentType": None,
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
        "returnCondition": None,
        "givenCondition": None,
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
    eq_type = _extract_equipment_type(raw_query)
    if not eq_type and result["equipmentName"]:
        if result["equipmentName"] in ISSUED_EQUIPMENT_ITEMS:
            eq_type = "Issued"
        elif result["equipmentName"] in PROCURED_EQUIPMENT_ITEMS:
            eq_type = "Procured"
    result["equipmentType"] = eq_type
    result["unitName"] = _extract_unit_name(raw_query)
    result["attemptNo"] = _extract_attempt_no(raw_query)
    result["fromAttempt"] = _extract_from_attempt(raw_query)
    result["toAttempt"] = _extract_to_attempt(raw_query)
    result["date"] = _extract_date_patterns(raw_query)
    result["fromDate"], result["toDate"] = _extract_date_range(raw_query)
    result["medicalStatus"] = _extract_medical_status(raw_query)
    result["diagnose"] = _extract_diagnose(raw_query)
    result["days"] = _extract_days(raw_query)
    result["returnCondition"] = _extract_return_condition(raw_query, semantic)
    result["givenCondition"] = _extract_given_condition(raw_query, semantic)

    # Precedence: explicit value in current query > semantic > stale resolved_entities
    result["companyId"] = (
        _extract_company_id(raw_query)
        or semantic.get("company_id")
        or semantic.get("companyId")
        or resolved_entities.get("company_id")
        or resolved_entities.get("companyId")
    )
    result["platoonId"] = (
        _extract_platoon_id(raw_query)
        or semantic.get("platoon_id")
        or semantic.get("platoonId")
        or resolved_entities.get("platoon_id")
        or resolved_entities.get("platoonId")
    )
    result["batchId"] = (
        _extract_batch_id(raw_query)
        or semantic.get("batch_id")
        or semantic.get("batchId")
        or resolved_entities.get("batch_id")
        or resolved_entities.get("batchId")
    )
    result["agniveerNo"] = (
        _extract_agniveer_no(raw_query)
        or resolved_entities.get("agniveer_no")
        or resolved_entities.get("agniveerNo")
    )

    return result


# Canonical (camelCase) key -> every spelling the frontend payload might use.
# admin_pipeline._extract_frontend_intent() keys its dict in snake_case
# (company_id/platoon_id/batch_id/agniveer_no, per _INTENT_FIELD_ALIASES),
# while extract_entities() above returns camelCase — this bridges the two so
# a frontend-supplied value is recognised under either spelling.
_FRONTEND_OVERRIDE_KEYS: Dict[str, Tuple[str, ...]] = {
    "companyId": ("companyId", "company_id"),
    "platoonId": ("platoonId", "platoon_id"),
    "batchId": ("batchId", "batch_id"),
    "agniveerNo": ("agniveerNo", "agniveer_no"),
    "fromDate": ("fromDate", "from_date"),
    "toDate": ("toDate", "to_date"),
}


def merge_frontend_intent(
    frontend_intent: Dict[str, Any], extracted: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge a frontend-supplied intent payload with free-text extraction.

    Frontend payload wins on ID fields (company/platoon/batch/agniveerNo,
    explicit fromDate/toDate a UI date-picker set) — those are literal user
    selections, not inference. Free-text extraction (`extracted`, the output
    of `extract_entities()`) fills everything the frontend didn't supply
    (section, grading, sport, leaveType, diagnose, ...).
    """
    merged = dict(extracted)
    for canonical, aliases in _FRONTEND_OVERRIDE_KEYS.items():
        for alias in aliases:
            value = frontend_intent.get(alias)
            if value not in (None, "", 0):
                merged[canonical] = value
                break
    return merged
