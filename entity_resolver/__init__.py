"""Entity resolution package for AgniAI."""

from __future__ import annotations

from .entity_cache import (
    ENTITY_CACHE,
    EntityCache,
    fetch_companies,
    fetch_platoons,
    invalidate_cache,
    preload_entities,
    refresh_all_entities,
)
from .entity_matcher import match_entity, normalize_text

__all__ = [
    "EntityCache",
    "ENTITY_CACHE",
    "fetch_companies",
    "fetch_platoons",
    "invalidate_cache",
    "match_entity",
    "normalize_text",
    "preload_entities",
    "refresh_all_entities",
]
