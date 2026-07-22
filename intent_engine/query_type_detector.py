"""
Word-Based Query Detection Engine for Cross-Filter, Comparison, and Multi-Independent Queries.
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple

# ── 1. CROSS-FILTER MARKERS ──────────────────────────────────────────────────

CROSS_FILTER_PRONOUNS = {
    "who": "WHO",
    "whose": "WHOSE", 
    "whom": "WHOM",
    "which": "WHICH",
    "that": "THAT",
    "whoever": "WHOEVER",
    "whomever": "WHOMEVER",
}

CROSS_FILTER_POSSESSIVE = {
    "with": "WITH",
    "having": "HAVING",
    "has": "HAS",
    "have": "HAVE",
    "had": "HAD",
    "possessing": "POSSESSING",
    "possesses": "POSSESSES",
    "carrying": "CARRYING",
    "carries": "CARRIES",
    "keeping": "KEEPING",
    "keeps": "KEEPS",
    "holding": "HOLDING",
    "holds": "HOLDS",
    "owning": "OWNING",
    "owns": "OWNS",
}

CROSS_FILTER_BELONGING = {
    "belonging to": "BELONGING_TO",
    "belongs to": "BELONGS_TO",
    "belong to": "BELONG_TO",
    "belonged to": "BELONGED_TO",
    "associated with": "ASSOCIATED_WITH",
    "linked to": "LINKED_TO",
    "connected to": "CONNECTED_TO",
    "related to": "RELATED_TO",
    "part of": "PART_OF",
    "member of": "MEMBER_OF",
}

CROSS_FILTER_STATUS = {
    # Medical
    "suffering from": "SUFFERING_FROM",
    "suffered from": "SUFFERED_FROM",
    "diagnosed with": "DIAGNOSED_WITH",
    "diagnosed": "DIAGNOSED",
    "hospitalized": "HOSPITALIZED",
    "hospitalised": "HOSPITALISED",
    "admitted to": "ADMITTED_TO",
    "admitted": "ADMITTED",
    "treated for": "TREATED_FOR",
    "under treatment": "UNDER_TREATMENT",
    "medically unfit": "MEDICALLY_UNFIT",
    "unfit": "UNFIT",
    "fit": "FIT",
    "sick": "SICK",
    "ill": "ILL",
    "injured": "INJURED",
    "wounded": "WOUNDED",
    
    # Leave
    "on leave": "ON_LEAVE",
    "currently on leave": "CURRENTLY_ON_LEAVE",
    "on medical leave": "ON_MEDICAL_LEAVE",
    "on annual leave": "ON_ANNUAL_LEAVE",
    "on sick leave": "ON_SICK_LEAVE",
    "absent": "ABSENT",
    "absconded": "ABSCONDED",
    "away": "AWAY",
    "off duty": "OFF_DUTY",
    
    # Verification
    "pending": "PENDING",
    "verified": "VERIFIED",
    "rejected": "REJECTED",
    "approved": "APPROVED",
    "cleared": "CLEARED",
    "sent": "SENT",
    "not responded": "NOT_RESPONDED",
    "awaiting": "AWAITING",
    
    # Equipment
    "issued": "ISSUED",
    "holding": "HOLDING",
    "returned": "RETURNED",
    "overdue": "OVERDUE",
    "damaged": "DAMAGED",
    "poor condition": "POOR_CONDITION",
    
    # Attendance
    "present": "PRESENT",
    "on campus": "ON_CAMPUS",
    "attending": "ATTENDING",
    
    # Performance
    "excellent": "EXCELLENT",
    "good": "GOOD",
    "sat": "SAT",
    "fail": "FAIL",
    "failed": "FAILED",
    "passing": "PASSING",
    "improved": "IMPROVED",
    "improvement": "IMPROVEMENT",
    "dropped": "DROPPED",
    "declined": "DECLINED",
}

CROSS_FILTER_QUALIFIERS = {
    "only": "ONLY",
    "just": "JUST",
    "specifically": "SPECIFICALLY",
    "exclusively": "EXCLUSIVELY",
    "particularly": "PARTICULARLY",
    "especially": "ESPECIALLY",
    "solely": "SOLELY",
    "merely": "MERELY",
    "strictly": "STRICTLY",
    "exactly": "EXACTLY",
}

CROSS_FILTER_COMPARATIVE = {
    "above": "ABOVE",
    "below": "BELOW",
    "over": "OVER",
    "under": "UNDER",
    "greater than": "GREATER_THAN",
    "less than": "LESS_THAN",
    "more than": "MORE_THAN",
    "higher than": "HIGHER_THAN",
    "lower than": "LOWER_THAN",
    "at least": "AT_LEAST",
    "at most": "AT_MOST",
    "between": "BETWEEN",
    "among": "AMONG",
    "within": "WITHIN",
    "outside": "OUTSIDE",
}

CROSS_FILTER_CONJUNCTIONS = {
    "and": "AND",
    "or": "OR",
    "but": "BUT",
    "yet": "YET",
    "however": "HOWEVER",
    "nevertheless": "NEVERTHELESS",
    "nonetheless": "NONETHELESS",
    "though": "THOUGH",
    "although": "ALTHOUGH",
    "despite": "DESPITE",
    "in spite of": "IN_SPITE_OF",
}

CROSS_FILTER_PREPOSITIONS = {
    "of": "OF",
    "in": "IN",
    "for": "FOR",
    "from": "FROM",
    "to": "TO",
    "by": "BY",
    "on": "ON",
    "at": "AT",
    "with": "WITH",
    "without": "WITHOUT",
    "through": "THROUGH",
    "across": "ACROSS",
}


def _has_cross_filter_marker(text: str) -> bool:
    """Detect ANY cross-filter marker in the text."""
    text_lower = text.lower()
    
    all_markers = (
        list(CROSS_FILTER_PRONOUNS.keys()) +
        list(CROSS_FILTER_POSSESSIVE.keys()) +
        list(CROSS_FILTER_BELONGING.keys()) +
        list(CROSS_FILTER_STATUS.keys()) +
        list(CROSS_FILTER_QUALIFIERS.keys()) +
        list(CROSS_FILTER_COMPARATIVE.keys()) +
        list(CROSS_FILTER_CONJUNCTIONS.keys()) +
        list(CROSS_FILTER_PREPOSITIONS.keys())
    )
    
    all_markers.sort(key=len, reverse=True)
    
    for marker in all_markers:
        pattern = r'\b' + re.escape(marker) + r'\b' if len(marker) <= 4 else re.escape(marker)
        if re.search(pattern, text_lower):
            return True
    
    return False


def _get_cross_filter_markers(text: str) -> List[str]:
    """Get all cross-filter markers found in text."""
    text_lower = text.lower()
    found = []
    
    all_markers = (
        list(CROSS_FILTER_PRONOUNS.keys()) +
        list(CROSS_FILTER_POSSESSIVE.keys()) +
        list(CROSS_FILTER_BELONGING.keys()) +
        list(CROSS_FILTER_STATUS.keys())
    )
    all_markers.sort(key=len, reverse=True)
    
    for marker in all_markers:
        pattern = r'\b' + re.escape(marker) + r'\b' if len(marker) <= 4 else re.escape(marker)
        if re.search(pattern, text_lower):
            found.append(marker)
    
    return found


# ── 2. COMPARISON MARKERS ──────────────────────────────────────────────────

COMPARISON_DIRECT = {
    "compare": "COMPARE",
    "comparison": "COMPARISON",
    "versus": "VERSUS",
    "vs": "VS",
    "vs.": "VS",
    "v/s": "VS",
}

COMPARISON_ADJECTIVES = {
    "better": "BETTER",
    "worse": "WORSE",
    "higher": "HIGHER",
    "lower": "LOWER",
    "greater": "GREATER",
    "smaller": "SMALLER",
    "larger": "LARGER",
    "faster": "FASTER",
    "slower": "SLOWER",
    "stronger": "STRONGER",
    "weaker": "WEAKER",
    "more": "MORE",
    "less": "LESS",
    "fewer": "FEWER",
    "bigger": "BIGGER",
    "taller": "TALLER",
    "shorter": "SHORTER",
    "heavier": "HEAVIER",
    "lighter": "LIGHTER",
    "thicker": "THICKER",
    "thinner": "THINNER",
    "wider": "WIDER",
    "narrower": "NARROWER",
    "deeper": "DEEPER",
    "shallower": "SHALLOWER",
    "older": "OLDER",
    "younger": "YOUNGER",
    "newer": "NEWER",
    "earlier": "EARLIER",
    "later": "LATER",
    "further": "FURTHER",
}

COMPARISON_PHRASES = {
    "compared to": "COMPARED_TO",
    "compared with": "COMPARED_WITH",
    "in comparison to": "IN_COMPARISON_TO",
    "in comparison with": "IN_COMPARISON_WITH",
    "as compared to": "AS_COMPARED_TO",
    "as compared with": "AS_COMPARED_WITH",
    "relative to": "RELATIVE_TO",
    "against": "AGAINST",
    "difference between": "DIFFERENCE_BETWEEN",
    "differ from": "DIFFER_FROM",
    "contrast": "CONTRAST",
    "contrasting": "CONTRASTING",
    "contrasted with": "CONTRASTED_WITH",
    "differentiate": "DIFFERENTIATE",
    "different from": "DIFFERENT_FROM",
    "superior to": "SUPERIOR_TO",
    "inferior to": "INFERIOR_TO",
    "prefer": "PREFER",
    "preferable to": "PREFERABLE_TO",
    "rank against": "RANK_AGAINST",
    "head to head": "HEAD_TO_HEAD",
    "side by side": "SIDE_BY_SIDE",
    "side-by-side": "SIDE_BY_SIDE",
}

COMPARISON_SUPERLATIVES = {
    "best": "BEST",
    "worst": "WORST",
    "highest": "HIGHEST",
    "lowest": "LOWEST",
    "greatest": "GREATEST",
    "smallest": "SMALLEST",
    "largest": "LARGEST",
    "fastest": "FASTEST",
    "slowest": "SLOWEST",
    "strongest": "STRONGEST",
    "weakest": "WEAKEST",
    "biggest": "BIGGEST",
    "tallest": "TALLEST",
    "shortest": "SHORTEST",
    "heaviest": "HEAVIEST",
    "lightest": "LIGHTEST",
    "thickest": "THICKEST",
    "thinnest": "THINNEST",
    "widest": "WIDEST",
    "narrowest": "NARROWEST",
    "deepest": "DEEPEST",
    "oldest": "OLDEST",
    "youngest": "YOUNGEST",
    "newest": "NEWEST",
    "earliest": "EARLIEST",
    "latest": "LATEST",
}

COMPARISON_CONJUNCTIONS = {
    "than": "THAN",
    "as...as": "AS_AS",
    "so...as": "SO_AS",
    "rather than": "RATHER_THAN",
    "instead of": "INSTEAD_OF",
}

COMPARISON_PREPOSITIONS = {
    "between": "BETWEEN",
    "among": "AMONG",
    "amongst": "AMONGST",
    "across": "ACROSS",
    "versus": "VERSUS",
    "vs": "VS",
}


def _has_comparison_marker(text: str) -> bool:
    """Detect ANY comparison marker in the text."""
    text_lower = text.lower()
    
    all_markers = (
        list(COMPARISON_DIRECT.keys()) +
        list(COMPARISON_ADJECTIVES.keys()) +
        list(COMPARISON_PHRASES.keys()) +
        list(COMPARISON_SUPERLATIVES.keys()) +
        list(COMPARISON_CONJUNCTIONS.keys()) +
        list(COMPARISON_PREPOSITIONS.keys())
    )
    all_markers.sort(key=len, reverse=True)
    
    for marker in all_markers:
        pattern = r'\b' + re.escape(marker) + r'\b' if len(marker) <= 4 else re.escape(marker)
        if re.search(pattern, text_lower):
            return True
    
    entity_patterns = [
        r'\b([A-Za-z0-9\s]+)\s+and\s+([A-Za-z0-9\s]+)\s+(?:company|platoon|batch|unit|performance)\b',
        r'\b(platoon|company|batch|coy)\s*\d+\s+and\s+(platoon|company|batch|coy)\s*\d+\b',
        r'\b(BPET|PPT|Firing|Drill)\s+and\s+(BPET|PPT|Firing|Drill)\b',
        r'\b([A-Z]\d{5,8}[A-Z]?)\s+and\s+([A-Z]\d{5,8}[A-Z]?)\b',
    ]
    
    for pattern in entity_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    
    return False


def _get_comparison_markers(text: str) -> List[str]:
    """Get all comparison markers found in text."""
    text_lower = text.lower()
    found = []
    
    all_markers = (
        list(COMPARISON_DIRECT.keys()) +
        list(COMPARISON_ADJECTIVES.keys()) +
        list(COMPARISON_PHRASES.keys()) +
        list(COMPARISON_SUPERLATIVES.keys())
    )
    all_markers.sort(key=len, reverse=True)
    
    for marker in all_markers:
        pattern = r'\b' + re.escape(marker) + r'\b' if len(marker) <= 4 else re.escape(marker)
        if re.search(pattern, text_lower):
            found.append(marker)
    
    return found


def _extract_comparison_entities(text: str) -> List[Tuple[str, str]]:
    """Extract entity pairs being compared."""
    pairs = []
    
    vs_patterns = [
        r'\b(.+?)\s+vs\.?\s+(.+?)(?:\s+|\?|$)',
        r'\b(.+?)\s+versus\s+(.+?)(?:\s+|\?|$)',
        r'\bcompare\s+(.+?)\s+with\s+(.+?)(?:\s+|\?|$)',
        r'\bcompare\s+(.+?)\s+and\s+(.+?)(?:\s+|\?|$)',
        r'\bdifference between\s+(.+?)\s+and\s+(.+?)(?:\s+|\?|$)',
    ]
    
    for pattern in vs_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if len(match) >= 2:
                pairs.append((match[0].strip(), match[1].strip()))
    
    entity_pattern = r'\b([A-Za-z][A-Za-z\s]+)\s+and\s+([A-Za-z][A-Za-z\s]+)\s+(?:company|coy|platoon|batch|unit)\b'
    matches = re.findall(entity_pattern, text, re.IGNORECASE)
    for match in matches:
        pairs.append((match[0].strip(), match[1].strip()))
    
    section_pattern = r'\b(BPET|PPT|Firing|Drill)\s+and\s+(BPET|PPT|Firing|Drill)\b'
    matches = re.findall(section_pattern, text, re.IGNORECASE)
    for match in matches:
        pairs.append((match[0].strip(), match[1].strip()))
    
    agniveer_pattern = r'\b([A-Z]\d{5,8}[A-Z]?)\s+and\s+([A-Z]\d{5,8}[A-Z]?)\b'
    matches = re.findall(agniveer_pattern, text, re.IGNORECASE)
    for match in matches:
        pairs.append((match[0].strip(), match[1].strip()))
    
    return pairs


# ── 3. MULTI-INDEPENDENT MARKERS ───────────────────────────────────────────

MULTI_INDEPENDENT_ADDITIVE = {
    "and": "AND",
    "also": "ALSO",
    "as well as": "AS_WELL_AS",
    "along with": "ALONG_WITH",
    "together with": "TOGETHER_WITH",
    "in addition to": "IN_ADDITION_TO",
    "in addition": "IN_ADDITION",
    "additionally": "ADDITIONALLY",
    "furthermore": "FURTHERMORE",
    "moreover": "MOREOVER",
    "besides": "BESIDES",
    "plus": "PLUS",
    "then": "THEN",
    "next": "NEXT",
    "later": "LATER",
    "afterwards": "AFTERWARDS",
    "subsequently": "SUBSEQUENTLY",
    "following that": "FOLLOWING_THAT",
    "after that": "AFTER_THAT",
    "before that": "BEFORE_THAT",
}

MULTI_INDEPENDENT_SEPARATION = {
    "separately": "SEPARATELY",
    "independently": "INDEPENDENTLY",
    "respectively": "RESPECTIVELY",
    "individually": "INDIVIDUALLY",
    "one by one": "ONE_BY_ONE",
    "at the same time": "AT_THE_SAME_TIME",
    "simultaneously": "SIMULTANEOUSLY",
    "concurrently": "CONCURRENTLY",
    "meanwhile": "MEANWHILE",
    "in the meantime": "IN_THE_MEANTIME",
}

MULTI_INDEPENDENT_LISTING = {
    "first": "FIRST",
    "second": "SECOND",
    "third": "THIRD",
    "fourth": "FOURTH",
    "fifth": "FIFTH",
    "last": "LAST",
    "finally": "FINALLY",
    "lastly": "LASTLY",
    "additionally": "ADDITIONALLY",
    "further": "FURTHER",
    "furthermore": "FURTHERMORE",
    "moreover": "MOREOVER",
    "likewise": "LIKEWISE",
    "similarly": "SIMILARLY",
}

MULTI_INDEPENDENT_BOTH = {
    "both": "BOTH",
    "either": "EITHER",
    "neither": "NEITHER",
}

MULTI_INDEPENDENT_VERBS = {
    "show": "SHOW",
    "show me": "SHOW_ME",
    "give": "GIVE",
    "give me": "GIVE_ME",
    "display": "DISPLAY",
    "provide": "PROVIDE",
    "list": "LIST",
    "get": "GET",
    "fetch": "FETCH",
    "find": "FIND",
    "tell": "TELL",
    "tell me": "TELL_ME",
}


def _has_multi_independent_marker(text: str) -> bool:
    """Detect ANY multi-independent marker in the text."""
    text_lower = text.lower()
    
    all_markers = (
        list(MULTI_INDEPENDENT_ADDITIVE.keys()) +
        list(MULTI_INDEPENDENT_SEPARATION.keys()) +
        list(MULTI_INDEPENDENT_LISTING.keys()) +
        list(MULTI_INDEPENDENT_BOTH.keys()) +
        list(MULTI_INDEPENDENT_VERBS.keys())
    )
    all_markers.sort(key=len, reverse=True)
    
    for marker in all_markers:
        pattern = r'\b' + re.escape(marker) + r'\b' if len(marker) <= 4 else re.escape(marker)
        if re.search(pattern, text_lower):
            return True
    
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) >= 2:
        sections = _extract_sections(text)
        if len(sections) >= 2:
            return True
        part_categories = [_detect_categories(part) for part in parts]
        all_cats = set()
        for cats in part_categories:
            all_cats.update(cats)
        if len(all_cats) >= 2:
            return True
    
    if re.search(r'\b(show|give|list|display)\s+.*? and .*?\b', text_lower):
        return True
    
    return False


def _extract_independent_requests(text: str) -> List[str]:
    """Split text into independent request parts."""
    text = text.strip()
    
    markers = list(MULTI_INDEPENDENT_ADDITIVE.keys()) + list(MULTI_INDEPENDENT_SEPARATION.keys())
    markers.sort(key=len, reverse=True)
    
    for marker in markers:
        if marker in text:
            split_parts = [p.strip() for p in text.split(marker) if p.strip()]
            if len(split_parts) >= 2:
                valid_parts = []
                for part in split_parts:
                    valid_parts.append(part)
                if len(valid_parts) >= 2:
                    return valid_parts
    
    comma_parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(comma_parts) >= 2:
        valid_parts = []
        for part in comma_parts:
            cats = _detect_categories(part)
            if cats:
                valid_parts.append(part)
        if len(valid_parts) >= 2:
            return valid_parts
    
    and_parts = [p.strip() for p in text.split(" and ") if p.strip()]
    if len(and_parts) >= 2:
        valid_parts = []
        for part in and_parts:
            cats = _detect_categories(part)
            if cats:
                valid_parts.append(part)
        if len(valid_parts) >= 2:
            return valid_parts
    
    return [text]


# ── 4. CATEGORY KEYWORDS & QUERY DETECTOR ────────────────────────────────────

_CATEGORY_KEYWORDS = {
    "Performance": ["bpet", "ppt", "firing", "drill", "score", "scores", "marks", "grade", "grading", "performer", "performers", "performance"],
    "Leave": ["leave", "on leave", "absconded", "annual leave", "sick leave", "medical leave"],
    "Medical": ["medical", "bmi", "disease", "diagnosed", "hospital", "hospitalized", "unfit", "blood group", "eyesight"],
    "Verification": ["verification", "police verification", "pending verification", "verified", "rejected"],
    "Equipment": ["equipment", "issued", "holding", "returned", "overdue", "damaged"],
    "Attendance": ["attendance", "present", "absent", "daily attendance", "weekly attendance", "monthly attendance"],
    "Distribution": ["distribution", "unit distribution", "unassigned", "top unit"],
    "PersonalDetail": ["height", "weight", "dob", "qualification", "address", "state", "district", "mobile", "contact", "email"],
}


def _normalize(text: str) -> str:
    """Normalize text for comparison."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _detect_categories(text: str) -> Set[str]:
    """Detect categories mentioned in the text."""
    categories = set()
    text_lower = text.lower()
    
    for category, keywords in _CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                categories.add(category)
                break
    
    return categories


def _extract_sections(text: str) -> List[str]:
    """Extract section names from text."""
    sections = []
    text_lower = text.lower()
    
    section_patterns = [
        (r'\bbpet\b', 'BPET'),
        (r'\bppt\b', 'PPT'),
        (r'\bfiring\b', 'Firing'),
        (r'\bdrill\b', 'Drill'),
    ]
    
    for pattern, section in section_patterns:
        if re.search(pattern, text_lower):
            sections.append(section)
    
    return sections


def detect_query_type(query: str) -> Dict[str, Any]:
    """
    Detect the query type using ALL English words/phrases.
    """
    normalized = _normalize(query)
    categories = _detect_categories(query)
    sections = _extract_sections(query)
    entity_pairs = _extract_comparison_entities(query)
    
    result = {
        "type": "simple",
        "confidence": 0.0,
        "categories": list(categories),
        "sections": sections,
        "entity_pairs": entity_pairs,
        "cross_filter_markers": [],
        "comparison_markers": [],
        "multi_independent_markers": [],
        "reasoning": "",
    }
    
    comp_markers = _get_comparison_markers(query)
    cross_markers = _get_cross_filter_markers(query)
    
    # ── 1. Check for comparison ──────────────────────────────────────────────
    if _has_comparison_marker(query):
        result["comparison_markers"] = comp_markers
        result["type"] = "comparison"
        result["confidence"] = 0.95
        result["reasoning"] = f"Comparison: markers {comp_markers}, {len(entity_pairs)} entity pairs, {len(sections)} sections"
        return result
    
    # ── 2. Check for multi-independent ──────────────────────────────────────
    # "Show attendance and leave", "BPET scores, then PPT scores", "attendance along with verification"
    # Multi-independent phrases (along with, together with, as well as, and, etc.) take precedence when 2+ categories or comma-separated lists are present.
    multi_additive_found = [m for m in MULTI_INDEPENDENT_ADDITIVE if m in normalized]
    if _has_multi_independent_marker(query) and (len(categories) >= 2 or len(sections) >= 2 or "," in query or " then " in normalized or len(multi_additive_found) >= 1):
        if not (any(pronoun in normalized for pronoun in ("who ", "whose ", "which ", "that ")) and "along with" not in normalized):
            result["multi_independent_markers"] = multi_additive_found or ["and"]
            result["type"] = "multi_independent"
            result["confidence"] = 0.90
            result["reasoning"] = f"Multi-independent: {len(categories)} categories, {len(sections)} sections or independent parts"
            return result
    
    # ── 3. Check for cross-filter ───────────────────────────────────────────
    if _has_cross_filter_marker(query) or len(cross_markers) >= 1:
        result["cross_filter_markers"] = cross_markers
        result["type"] = "cross_filter"
        result["confidence"] = 0.95
        result["reasoning"] = f"Cross-filter: markers {cross_markers}, {len(categories)} categories"
        return result
    
    # ── 4. Fallback for multi-category queries without explicit markers ─────
    if len(categories) >= 2:
        if " vs " in normalized or " versus " in normalized:
            result["type"] = "comparison"
            result["confidence"] = 0.85
            result["reasoning"] = "Comparison: 'vs' pattern with multiple categories"
            return result
        
        if " and " in normalized or "," in normalized:
            result["type"] = "multi_independent"
            result["confidence"] = 0.75
            result["reasoning"] = "Multi-independent: multiple categories with connector"
            return result
        
        result["type"] = "cross_filter"
        result["confidence"] = 0.70
        result["reasoning"] = f"Cross-filter: multiple categories ({categories}) without clear connector"
        return result
    
    # ── 5. Final fallback ───────────────────────────────────────────────────
    result["confidence"] = 0.60
    result["reasoning"] = "Defaulting to simple query"
    return result
