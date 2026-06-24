"""Entity data models used by the resolver layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BaseEntity:
    id: Optional[int] = None
    name: str = ""
    aliases: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgniveerEntity(BaseEntity):
    agniveer_no: str = ""
    full_name: str = ""


@dataclass
class CompanyEntity(BaseEntity):
    company_id: Optional[int] = None
    company_name: str = ""


@dataclass
class PlatoonEntity(BaseEntity):
    platoon_id: Optional[int] = None
    platoon_no: Optional[int] = None
    company_id: Optional[int] = None
    company_name: str = ""


@dataclass
class EntityMatch:
    value: Any = None
    confidence: float = 0.0
    matched_text: str = ""
    source: str = ""
    match_type: str = ""
    needs_clarification: bool = False
    candidates: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
