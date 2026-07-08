from datetime import datetime

from intent_engine.date_resolver import resolve_date_range


def test_resolve_date_range_supports_multiple_common_date_formats():
    now = datetime(2026, 7, 8)

    cases = [
        ("08/07/2026", ("2026-07-08T00:00:00", None, None)),
        ("08-07-2026", ("2026-07-08T00:00:00", None, None)),
        ("08 july 2026", ("2026-07-08T00:00:00", None, None)),
        ("8-7-2026", ("2026-07-08T00:00:00", None, None)),
        ("8-7-26", ("2026-07-08T00:00:00", None, None)),
        ("08-Jul-2026", ("2026-07-08T00:00:00", None, None)),
    ]

    for raw_date, expected in cases:
        assert resolve_date_range(
            operation=None,
            date=raw_date,
            from_date=None,
            to_date=None,
            now=now,
        ) == expected
