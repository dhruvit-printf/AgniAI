"""
org_hierarchy_parser.py
========================
Heuristic early-exit parser for organizational-structure questions (company/
platoon commanders, commanding-officer history, platoon rosters under a
company, headcount-by-company) — mirrors personal_details_parser.py's
pattern of deterministic regex matching that bypasses LLM query planning.

Company/platoon NAME resolution (e.g. "Lakhwinder Company" -> companyId) is
NOT this module's job — admin_pipeline.py already resolves that via
admin_entity_resolver.resolve_entities_from_query() and injects companyId/
platoonId onto every operation before execution. This parser only needs to
recognise the QUESTION shape and pick the right operation.
"""

import re
from typing import Optional, Dict, Any

_AGNIVEER_NO_RE = re.compile(r"\b([A-Za-z]\d{5,8}[A-Za-z]?)\b")


def parse_org_hierarchy(query: str) -> Optional[Dict[str, Any]]:
    q_lower = query.lower().strip()

    if not any(
        kw in q_lower
        for kw in ("command", "platoon", "company", "companies", "posted")
    ):
        return None

    # "Where is X posted?" / "Which platoon is X in?" — same answer shape as
    # WhichCompanyForAgniveer below (it already selects both PlatoonName and
    # CompanyName), just reached by different phrasing that doesn't
    # necessarily say "belong" or "company".
    if re.search(r"\bposted\b", q_lower) or re.search(
        r"\bwhich\s+platoon\b.*\bin\b", q_lower
    ):
        m = _AGNIVEER_NO_RE.search(query)
        if m:
            return {
                "category": "OrgHierarchy",
                "operation": "WhichCompanyForAgniveer",
                "agniveer_no": m.group(1).upper(),
                "query_type": "simple",
                "confidence": "high",
                "confidence_score": 1.0,
                "filters": {},
            }

    target = "Platoon" if "platoon" in q_lower and "company" not in q_lower else "Company"

    # 1. Predecessor — "who commanded X BEFORE the current commander"
    # (checked first: also matches the generic "command" pattern below)
    if "command" in q_lower and "before" in q_lower:
        return {
            "category": "OrgHierarchy",
            "operation": "PredecessorCommander",
            "target": target,
            "query_type": "simple",
            "confidence": "high",
            "confidence_score": 1.0,
            "filters": {},
        }

    # 2. Historical commander/CO — "...last year", "...previously"
    if re.search(r"\bcommand(?:er|ing officer)\b", q_lower) and re.search(
        r"\blast year\b|\bpreviously\b|\blast month\b", q_lower
    ):
        return {
            "category": "OrgHierarchy",
            "operation": "HistoricalOfficer",
            "target": target,
            "query_type": "simple",
            "confidence": "high",
            "confidence_score": 1.0,
            "filters": {},
        }

    # 3. Current commander / commanding officer
    if re.search(r"\bcommand(?:er|ing officer)\b", q_lower):
        return {
            "category": "OrgHierarchy",
            "operation": "CurrentCommander",
            "target": target,
            "query_type": "simple",
            "confidence": "high",
            "confidence_score": 1.0,
            "filters": {},
        }

    # 4. Platoons under a company
    if "platoon" in q_lower and "company" in q_lower and re.search(
        r"\bunder\b|\bin\b|\bof\b|\bbelonging to\b", q_lower
    ):
        return {
            "category": "OrgHierarchy",
            "operation": "PlatoonsUnderCompany",
            "target": "Company",
            "query_type": "simple",
            "confidence": "high",
            "confidence_score": 1.0,
            "filters": {},
        }

    # 5. Headcount per company ("how many Agniveers are in each company")
    if "company" in q_lower and re.search(r"\bhow many\b.*\beach\b", q_lower):
        return {
            "category": "OrgHierarchy",
            "operation": "HeadcountByCompany",
            "query_type": "simple",
            "confidence": "high",
            "confidence_score": 1.0,
            "filters": {},
        }

    # 6. Which company has the most/fewest Agniveers
    most_least = re.search(r"\bwhich compan\w+\b.*\b(most|fewest|least)\s+agniveers?\b", q_lower)
    if most_least:
        return {
            "category": "OrgHierarchy",
            "operation": "TopCompanyByHeadcount",
            "descending": most_least.group(1) == "most",
            "query_type": "simple",
            "confidence": "high",
            "confidence_score": 1.0,
            "filters": {},
        }

    # 7. Which company does a specific Agniveer belong to
    if "company" in q_lower and re.search(r"\bbelong\b", q_lower):
        m = _AGNIVEER_NO_RE.search(query)
        if m:
            return {
                "category": "OrgHierarchy",
                "operation": "WhichCompanyForAgniveer",
                "agniveer_no": m.group(1).upper(),
                "query_type": "simple",
                "confidence": "high",
                "confidence_score": 1.0,
                "filters": {},
            }

    return None
