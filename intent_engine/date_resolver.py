"""
date_resolver.py
================

Single responsibility: resolve the free-form / relative date values produced
by entity_extractor.py ("today", "current month", "June", "June 2026",
"2026-06-01", ...) into ISO 8601 date-time strings.

The .NET AiCommandRequestDto declares `date` / `fromDate` / `toDate` as
`format: date-time`. Sending a raw phrase like "current month" in one of
those fields is not a valid date-time and will fail on the .NET side, so
every value that reaches the payload must pass through here first.
"""

from __future__ import annotations

import calendar
import re
from datetime import datetime, timedelta
from typing import Optional, Tuple

_MONTH_NAMES = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_SLASH_DATE_RE = re.compile(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$")
_MONTH_YEAR_RE = re.compile(
    r"^(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"(?:\s+(\d{4}))?$"
)
_COMMON_DATE_FORMATS = (
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%d.%m.%Y",
    "%d/%m/%y",
    "%d-%m-%y",
    "%d.%m.%y",
    "%d %B %Y",
    "%d %b %Y",
    "%d %B %y",
    "%d %b %y",
    "%d-%B-%Y",
    "%d-%b-%Y",
    "%d/%B/%Y",
    "%d/%b/%Y",
    "%d-%B-%y",
    "%d-%b-%y",
    "%d/%B/%y",
    "%d/%b/%y",
)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%dT00:00:00")


def _month_bounds(year: int, month: int) -> Tuple[datetime, datetime]:
    first = datetime(year, month, 1)
    last_day = calendar.monthrange(year, month)[1]
    last = datetime(year, month, last_day)
    return first, last


def _week_bounds(reference: datetime) -> Tuple[datetime, datetime]:
    monday = datetime(reference.year, reference.month, reference.day) - timedelta(
        days=reference.weekday()
    )
    sunday = monday + timedelta(days=6)
    return monday, sunday


def _phrase_to_dates(
    phrase: str, now: datetime
) -> Tuple[Optional[datetime], Optional[datetime], Optional[datetime]]:
    """Returns (single_day, range_start, range_end) — only one form is populated."""
    text = phrase.strip().lower()
    if not text:
        return None, None, None

    relative_single = {
        "today": now,
        "yesterday": now - timedelta(days=1),
        "tomorrow": now + timedelta(days=1),
    }
    if text in relative_single:
        d = relative_single[text]
        return datetime(d.year, d.month, d.day), None, None

    if text in ("this week", "current week"):
        start, end = _week_bounds(now)
        return None, start, end
    if text == "last week":
        start, end = _week_bounds(now - timedelta(weeks=1))
        return None, start, end

    if text in ("this month", "current month"):
        start, end = _month_bounds(now.year, now.month)
        return None, start, end
    if text == "last month":
        prev = now.replace(day=1) - timedelta(days=1)
        start, end = _month_bounds(prev.year, prev.month)
        return None, start, end

    if text in ("this year", "current year"):
        return None, datetime(now.year, 1, 1), datetime(now.year, 12, 31)

    if _ISO_DATE_RE.match(text):
        return datetime.strptime(text, "%Y-%m-%d"), None, None

    m = _SLASH_DATE_RE.match(text)
    if m:
        day, month, year = (int(g) for g in m.groups())
        try:
            return datetime(year, month, day), None, None
        except ValueError:
            return None, None, None

    if re.fullmatch(r"\d{1,2}[/-]\d{1,2}[/-]\d{2}(?:\d{2})?", text):
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y", "%d-%m-%y"):
            try:
                return datetime.strptime(text, fmt), None, None
            except ValueError:
                continue

    if re.fullmatch(r"\d{1,2}(?:\s|[./-])(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)(?:\s|[./-])\d{2,4}", text):
        for fmt in _COMMON_DATE_FORMATS:
            try:
                return datetime.strptime(text, fmt), None, None
            except ValueError:
                continue

    m = _MONTH_YEAR_RE.match(text)
    if m:
        month = _MONTH_NAMES.get(m.group(1))
        if month is None:
            return None, None, None
        year = int(m.group(2)) if m.group(2) else now.year
        start, end = _month_bounds(year, month)
        return None, start, end

    return None, None, None


def resolve_date_range(
    *,
    operation: Optional[str],
    date: Optional[str],
    from_date: Optional[str],
    to_date: Optional[str],
    now: Optional[datetime] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Resolve raw/relative date entities into ISO 8601 date-time strings ready
    for the .NET payload.

    Returns (date, fromDate, toDate). A single specific day resolves to
    `date`; a period (week/month/year) resolves to `fromDate`/`toDate`.
    When nothing was extracted from the query, defaults to the current
    period for Monthly/Weekly/Daily so those operations always carry a date
    scope.
    """
    now = now or datetime.now()

    if from_date or to_date:
        resolved_from = None
        resolved_to = None
        if from_date:
            single, r_start, _ = _phrase_to_dates(from_date, now)
            resolved_from = single or r_start
        if to_date:
            single, _, r_end = _phrase_to_dates(to_date, now)
            resolved_to = single or r_end
        if resolved_from or resolved_to:
            return None, _iso(resolved_from), _iso(resolved_to)

    if date:
        single, r_start, r_end = _phrase_to_dates(date, now)
        if single:
            return _iso(single), None, None
        if r_start or r_end:
            return None, _iso(r_start), _iso(r_end)

    if operation == "Monthly":
        start, end = _month_bounds(now.year, now.month)
        return None, _iso(start), _iso(end)
    if operation == "Weekly":
        start, end = _week_bounds(now)
        return None, _iso(start), _iso(end)
    if operation == "Daily":
        return _iso(now), None, None

    return None, None, None
