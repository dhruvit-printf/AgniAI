"""
access_control.py
==================
Hierarchical data isolation for the org chart:
Commanding Officer/Clerk -> Company Commander -> Platoon Commander -> Agniveer.

Each role may only see its own subtree (self + strict descendants), never
peers in the same branch or anything above it.
"""

from typing import Any, Optional

FULL_ACCESS_ROLES = {"Commanding_Officer", "Clerk"}


def filter_user_data(
    current_user: dict[str, Any],
    full_dataset: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return only the records `current_user` is permitted to see."""
    role = current_user.get("role")

    if role in FULL_ACCESS_ROLES:
        return list(full_dataset)

    if role == "Company_Commander":
        company_id = current_user["company_id"]
        return [
            record
            for record in full_dataset
            if record.get("company_id") == company_id
            and record.get("role") != "Commanding_Officer"
        ]

    if role == "Platoon_Commander":
        platoon_id = current_user["platoon_id"]
        return [
            record
            for record in full_dataset
            if record.get("platoon_id") == platoon_id
            and record.get("role") not in FULL_ACCESS_ROLES
            and record.get("role") != "Company_Commander"
        ]

    if role == "Agniveer":
        return [record for record in full_dataset if record.get("id") == current_user.get("id")]

    return []


def check_access(
    current_user: dict[str, Any],
    target_record: dict[str, Any],
    full_user_list: Optional[list[dict[str, Any]]] = None,
) -> tuple[bool, Optional[str]]:
    """Check whether `current_user` may view `target_record`.

    Returns (True, None) when allowed, or (False, denial_message) when not.
    The denial message points the user to whoever owns that data, looked up
    from `full_user_list` by name when available.
    """
    role = current_user.get("role")

    if role in FULL_ACCESS_ROLES:
        return True, None

    if role == "Company_Commander":
        if (
            target_record.get("company_id") == current_user.get("company_id")
            and target_record.get("role") != "Commanding_Officer"
        ):
            return True, None
        contact = _find_commander(full_user_list, "Company_Commander", company_id=target_record.get("company_id"))
        return False, _denial_message(contact, "Company Commander", "company")

    if role == "Platoon_Commander":
        if (
            target_record.get("platoon_id") == current_user.get("platoon_id")
            and target_record.get("role") not in FULL_ACCESS_ROLES
            and target_record.get("role") != "Company_Commander"
        ):
            return True, None
        contact = _find_commander(full_user_list, "Platoon_Commander", platoon_id=target_record.get("platoon_id"))
        return False, _denial_message(contact, "Platoon Commander", "platoon")

    if role == "Agniveer":
        if target_record.get("id") == current_user.get("id"):
            return True, None
        return False, "You are not authorised to see that data. Please contact your Platoon Commander."

    return False, "You are not authorised to see that data."


def _find_commander(
    full_user_list: Optional[list[dict[str, Any]]],
    role: str,
    company_id: Any = None,
    platoon_id: Any = None,
) -> Optional[dict[str, Any]]:
    for user in full_user_list or []:
        if user.get("role") != role:
            continue
        if company_id is not None and user.get("company_id") != company_id:
            continue
        if platoon_id is not None and user.get("platoon_id") != platoon_id:
            continue
        return user
    return None


def _denial_message(contact: Optional[dict[str, Any]], title: str, unit_word: str) -> str:
    who = f"{contact['name']}, the {title} of that {unit_word}" if contact else f"the {title} of that {unit_word}"
    return f"You are not authorised to see that data. Please contact {who}."


def check_batch_access(
    current_user: dict[str, Any],
    target_record: dict[str, Any],
    batch_list: Optional[list[dict[str, Any]]] = None,
) -> tuple[bool, Optional[str]]:
    """Check whether `target_record`'s batch matches the batch `current_user` currently has selected.

    Returns (True, None) when allowed, or (False, message) telling the user to
    switch their active batch. A record with no `batch_id` (not batch-scoped
    data) is always allowed.
    """
    target_batch_id = target_record.get("batch_id")
    if target_batch_id is None or target_batch_id == current_user.get("batch_id"):
        return True, None

    batch = _find_batch(batch_list, target_batch_id)
    batch_name = batch["name"] if batch else f"Batch {target_batch_id}"
    return False, f"You are currently viewing a different batch. Please switch to {batch_name} to see that data."


def _find_batch(batch_list: Optional[list[dict[str, Any]]], batch_id: Any) -> Optional[dict[str, Any]]:
    for batch in batch_list or []:
        if batch.get("id") == batch_id:
            return batch
    return None


def build_user_hierarchy(
    current_user: dict[str, Any],
    full_user_list: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a nested {id, name, role, children} tree rooted per the caller's access level."""
    role = current_user.get("role")

    if role in FULL_ACCESS_ROLES:
        root = next((u for u in full_user_list if u.get("role") == "Commanding_Officer"), current_user)
    else:
        root = current_user

    return _build_node(root, full_user_list)


def _build_node(user: dict[str, Any], full_user_list: list[dict[str, Any]]) -> dict[str, Any]:
    children = [
        _build_node(u, full_user_list)
        for u in full_user_list
        if u.get("parent_id") == user.get("id")
    ]
    return {
        "id": user.get("id"),
        "name": user.get("name"),
        "role": user.get("role"),
        "children": children,
    }
