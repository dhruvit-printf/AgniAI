"""
users_roles_parser.py
======================
Heuristic early-exit parser for the Users & Roles domain (UserMaster /
RoleMaster / UserRole), mirroring personal_details_parser.py's pattern:
deterministic regex matching that bypasses LLM query planning entirely.

Deliberately does NOT cover LoginToken-backed questions ("who last logged in
as X", "list revoked login tokens") — sql_executor.DENIED_TABLES already
hard-blocks the LoginToken table (it holds JWT/refresh tokens), and that is
an existing, intentional security boundary this parser must not route around.
"""

import re
from typing import Optional, Dict, Any

_AGNIVEER_NO_RE = re.compile(r"\b([A-Za-z]\d{5,8}[A-Za-z]?)\b")


def parse_users_roles(query: str) -> Optional[Dict[str, Any]]:
    q_lower = query.lower().strip()

    if "user" not in q_lower and "admin access" not in q_lower:
        return None

    # 1. Which user account is linked to a specific Agniveer
    if "linked to" in q_lower or ("account" in q_lower and "agniveer" in q_lower):
        m = _AGNIVEER_NO_RE.search(query)
        if m:
            return {
                "category": "UsersRoles",
                "operation": "ByAgniveer",
                "agniveer_no": m.group(1).upper(),
                "query_type": "simple",
                "confidence": "high",
                "confidence_score": 1.0,
                "filters": {},
            }

    # 2. Active / inactive user accounts
    if re.search(r"\b(in)?active\b", q_lower) and "user" in q_lower:
        is_active = 0 if "inactive" in q_lower else 1
        return {
            "category": "UsersRoles",
            "operation": "ActiveList",
            "is_active": is_active,
            "query_type": "simple",
            "confidence": "high",
            "confidence_score": 1.0,
            "filters": {},
        }

    # 3. "Who has admin access?" — admin is a role, not a distinct concept
    if "admin access" in q_lower or re.search(r"\badmin\b.*\baccess\b", q_lower):
        return {
            "category": "UsersRoles",
            "operation": "ByRole",
            "role": "Admin",
            "query_type": "simple",
            "confidence": "high",
            "confidence_score": 1.0,
            "filters": {},
        }

    # 4. "List all users with the Doctor role" / "users with the X role"
    role_match = re.search(r"\bwith\s+(?:the\s+)?([a-z][a-z\s]*?)\s+role\b", q_lower)
    if role_match:
        role = role_match.group(1).strip()
        if role:
            return {
                "category": "UsersRoles",
                "operation": "ByRole",
                "role": role.title(),
                "query_type": "simple",
                "confidence": "high",
                "confidence_score": 1.0,
                "filters": {},
            }

    return None
