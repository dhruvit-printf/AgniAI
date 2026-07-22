import logging
from typing import Any, Dict, List, Optional, Tuple

import sql_executor
from sql_executor import _row_cap, _to_section

logger = logging.getLogger("performance_executor")


def _pivot_attemptwise_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transform flat attempt-wise rows into nested structure per Agniveer.

    Input:
    [
        {"agniveerNo": "A1", "fullName": "X", "attemptNo": 1, "sectionName": "BPET", "attemptTotal": 100},
        {"agniveerNo": "A1", "fullName": "X", "attemptNo": 1, "sectionName": "PPT", "attemptTotal": 88},
        ...
    ]

    Output:
    [
        {
            "agniveerNo": "A1",
            "fullName": "X",
            "attempts": {
                "1": {"BPET": 100, "PPT": 88, "FIRING": 28, "DRILL (AMT)": 59},
                "2": {"BPET": 0, "PPT": 0, "FIRING": 0, "DRILL (AMT)": 55},
                "3": {"BPET": 96, "PPT": 100, "FIRING": 17, "DRILL (AMT)": 66}
            }
        }
    ]
    """
    agniveer_map: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        agniveer_no = row.get("agniveerNo") or row.get("AgniveerNo")
        if not agniveer_no:
            continue

        if agniveer_no not in agniveer_map:
            agniveer_map[agniveer_no] = {
                "agniveerNo": agniveer_no,
                "fullName": row.get("fullName") or row.get("FullName") or "",
                "attempts": {},
            }

        agniveer = agniveer_map[agniveer_no]
        attempt_no_val = (
            row.get("attemptNo") if "attemptNo" in row else row.get("AttemptNo")
        )
        attempt_no = str(attempt_no_val if attempt_no_val is not None else "")
        section_name = row.get("sectionName") or row.get("SectionName") or ""
        attempt_total = (
            row.get("attemptTotal")
            if "attemptTotal" in row
            else row.get("AttemptTotal")
        )

        if attempt_no not in agniveer["attempts"]:
            agniveer["attempts"][attempt_no] = {}

        agniveer["attempts"][attempt_no][section_name] = attempt_total

    return list(agniveer_map.values())


def _pivot_best_attempt_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Transform flat (AgniveerNo, SectionName, BestTotal) rows into one row
    per Agniveer with a section-wise breakdown, for the multi-Agniveer
    BestAttempt case (no agniveer_no filter — e.g. "best attempt scores for
    all Agniveers").

    Input:
    [
        {"agniveerNo": "A1", "fullName": "X", "sectionName": "BPET", "bestTotal": 100},
        {"agniveerNo": "A1", "fullName": "X", "sectionName": "PPT", "bestTotal": 88},
        ...
    ]

    Output:
    [
        {"agniveerNo": "A1", "fullName": "X", "sections": {"BPET": 100, "PPT": 88}},
        ...
    ]
    """
    agniveer_map: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        agniveer_no = row.get("agniveerNo") or row.get("AgniveerNo")
        if not agniveer_no:
            continue

        if agniveer_no not in agniveer_map:
            agniveer_map[agniveer_no] = {
                "agniveerNo": agniveer_no,
                "fullName": row.get("fullName") or row.get("FullName") or "",
                "sections": {},
            }

        section_name = row.get("sectionName") or row.get("SectionName") or ""
        best_total = (
            row.get("bestTotal") if "bestTotal" in row else row.get("BestTotal")
        )
        agniveer_map[agniveer_no]["sections"][section_name] = best_total

    return list(agniveer_map.values())


def execute_performance_query(
    intent: Dict,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    category = intent.get("category", "")
    operation = intent.get("operation", "")

    batch_id = intent.get("batch_id") or intent.get("batchId")
    agniveer_class = intent.get("class")
    platoon_id = intent.get("platoon_id") or intent.get("platoonId")
    company_id = intent.get("company_id") or intent.get("companyId")

    agniveer_no = intent.get("agniveer_no") or intent.get("agniveerNo")
    section = str(intent.get("section") or "").strip()
    sub_section = str(intent.get("sub_section") or "").strip()
    attempt_no = intent.get("attemptNo") or intent.get("attempt_no")
    from_attempt = intent.get("from_attempt") or intent.get("fromAttempt")
    to_attempt = intent.get("to_attempt") or intent.get("toAttempt")
    requested_grade = intent.get("grading")
    _top_n_raw = intent.get("number") or intent.get("top_n")
    top_n = int(_top_n_raw) if _top_n_raw is not None else None

    def _scope_clause(alias: str = "a") -> Tuple[str, List[Any]]:
        clauses = [
            f"({alias}.IsActive = 1 AND ISNULL({alias}.IsDisqualified, 0) = 0)"
        ]
        params: List[Any] = []

        if batch_id is not None:
            clauses.append(f"{alias}.BatchId = ?")
            params.append(int(batch_id))
        if agniveer_class is not None:
            clauses.append(f"LOWER({alias}.Class) = LOWER(?)")
            params.append(str(agniveer_class))
        if platoon_id is not None:
            clauses.append(f"{alias}.PlatoonId = ?")
            params.append(int(platoon_id))
        if company_id is not None:
            clauses.append(
                f"EXISTS (SELECT 1 FROM PlatoonMaster p WHERE p.Id = {alias}.PlatoonId AND p.CompanyId = ?)"
            )
            params.append(int(company_id))
        if agniveer_no is not None:
            clauses.append(f"LOWER({alias}.AgniveerNo) LIKE '%' + LOWER(?) + '%'")
            params.append(str(agniveer_no))

        return " AND ".join(clauses), params

    def _section_clause(
        section_alias: str = "sec",
        subitem_alias: Optional[str] = "si",
        is_exceptional_target: Optional[bool] = False,
    ) -> Tuple[str, List[Any]]:
        clauses: List[str] = []
        if is_exceptional_target is not None:
            val = 1 if is_exceptional_target else 0
            clauses.append(f"ISNULL({section_alias}.IsExceptional, 0) = {val}")
        params: List[Any] = []

        if section:
            clauses.append(
                f"LOWER({section_alias}.SectionName) LIKE '%' + LOWER(?) + '%'"
            )
            params.append(section)
        if sub_section and subitem_alias:
            clauses.append(
                f"LOWER({subitem_alias}.Name) LIKE '%' + LOWER(?) + '%'"
            )
            params.append(sub_section)

        return " AND ".join(clauses), params

    def _attempt_clause(require_best_attempt: bool) -> Tuple[str, List[Any]]:
        if attempt_no is not None:
            return "sa.AttemptNo = ?", [int(attempt_no)]
        if require_best_attempt:
            return "sa.IsBestAttempt = 1", []
        if from_attempt is not None and to_attempt is not None:
            return "sa.AttemptNo IN (?, ?)", [int(from_attempt), int(to_attempt)]
        return "1 = 1", []

    def _filtered_attempts_source(
        *, require_best_attempt: bool, include_attempt_window: bool = False
    ) -> Tuple[str, List[Any]]:
        """
        Builds the FilteredAttempts CTE for attempt-wise queries.
        Returns SQL and parameters.
        """
        section_sql, section_params = _section_clause(is_exceptional_target=False)
        scope_sql, scope_params = _scope_clause("a")

        attempt_sql, attempt_params = _attempt_clause(require_best_attempt)
        if include_attempt_window and from_attempt is not None and to_attempt is not None:
            attempt_sql = "sa.AttemptNo IN (?, ?)"
            attempt_params = [int(from_attempt), int(to_attempt)]

        sql = f"""
        SELECT
            a.Id AS AgniveerId,
            a.AgniveerNo,
            a.FullName,
            sa.AttemptNo,
            sa.AttemptedDate,
            sa.MarksObtained,
            sa.IsBestAttempt,
            sec.Id AS SectionId,
            sec.SectionName,
            si.Id AS SubItemId,
            si.Name AS SubItemName,
            si.MaxMarks
        FROM AgniveerScoreAttempt sa
            INNER JOIN AgniveerMaster a ON a.Id = sa.AgniveerId
            INNER JOIN ScoreSubItemMaster si ON si.Id = sa.SubItemId
            INNER JOIN ScoreSectionMaster sec ON sec.Id = si.SectionId
        WHERE sa.MarksObtained IS NOT NULL
            AND {attempt_sql}
            AND {section_sql}
            AND {scope_sql}
        """
        params = attempt_params + section_params + scope_params

        if not sub_section:
            section_exc_sql, section_exc_params = _section_clause(is_exceptional_target=True, subitem_alias=None)
            sr_attempt_sql = ""
            sr_attempt_params: List[Any] = []
            if attempt_no is not None:
                sr_attempt_sql = "AND sr.AttemptNo = ?"
                sr_attempt_params = [str(attempt_no)]

            exc_sql = f"""
            UNION ALL
            SELECT
                a.Id AS AgniveerId,
                a.AgniveerNo,
                a.FullName,
                sr.AttemptNo,
                sr.InsertedDate AS AttemptedDate,
                COALESCE(sr.ExceptionalMarks, sr.OmrInputTotal) AS MarksObtained,
                1 AS IsBestAttempt,
                sec.Id AS SectionId,
                sec.SectionName,
                CAST(0 AS BIGINT) AS SubItemId,
                sec.SectionName AS SubItemName,
                COALESCE(sr.ExceptionalMarks, sr.OmrInputTotal, 100) AS MaxMarks
            FROM AgniveerSectionResult sr
                INNER JOIN AgniveerMaster a ON a.Id = sr.AgniveerId
                INNER JOIN ScoreSectionMaster sec ON sec.Id = sr.SectionId
            WHERE COALESCE(sr.ExceptionalMarks, sr.OmrInputTotal) IS NOT NULL
                {sr_attempt_sql}
                AND {section_exc_sql}
                AND {scope_sql}
            """
            sql += exc_sql
            params += sr_attempt_params + section_exc_params + scope_params

        return sql, params

    def _attemptwise_source(require_best_attempt: bool = False) -> Tuple[str, List[Any]]:
        """
        Dedicated FilteredAttempts CTE for AttemptWise/BestAttempt.
        """
        section_sql, section_params = _section_clause(is_exceptional_target=False)
        scope_sql, scope_params = _scope_clause("a")
        attempt_sql, attempt_params = _attempt_clause(require_best_attempt=require_best_attempt)

        sql = f"""
        SELECT
            a.AgniveerNo,
            a.FullName,
            sa.AttemptNo,
            sec.SectionName,
            sa.MarksObtained
        FROM AgniveerScoreAttempt sa
            INNER JOIN AgniveerMaster a ON a.Id = sa.AgniveerId
            INNER JOIN ScoreSubItemMaster si ON si.Id = sa.SubItemId
            INNER JOIN ScoreSectionMaster sec ON sec.Id = si.SectionId
        WHERE sa.MarksObtained IS NOT NULL
            AND {attempt_sql}
            AND {section_sql}
            AND {scope_sql}
        """
        params = attempt_params + section_params + scope_params

        if not sub_section:
            section_exc_sql, section_exc_params = _section_clause(is_exceptional_target=True, subitem_alias=None)
            sr_attempt_sql = ""
            sr_attempt_params: List[Any] = []
            if attempt_no is not None:
                sr_attempt_sql = "AND sr.AttemptNo = ?"
                sr_attempt_params = [str(attempt_no)]

            exc_sql = f"""
            UNION ALL
            SELECT
                a.AgniveerNo,
                a.FullName,
                sr.AttemptNo,
                sec.SectionName,
                COALESCE(sr.ExceptionalMarks, sr.OmrInputTotal) AS MarksObtained
            FROM AgniveerSectionResult sr
                INNER JOIN AgniveerMaster a ON a.Id = sr.AgniveerId
                INNER JOIN ScoreSectionMaster sec ON sec.Id = sr.SectionId
            WHERE COALESCE(sr.ExceptionalMarks, sr.OmrInputTotal) IS NOT NULL
                {sr_attempt_sql}
                AND {section_exc_sql}
                AND {scope_sql}
            """
            sql += exc_sql
            params += sr_attempt_params + section_exc_params + scope_params

        return sql, params

    def _run(
        sql: str, params: List[Any], max_rows: Optional[int] = None
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        # Call through the sql_executor module attribute (not a bound name
        # captured at import time) so that patching sql_executor.run_readonly
        # in tests always takes effect regardless of module import order —
        # a `from sql_executor import run_readonly` binding here would freeze
        # on whichever function object existed the first time this module
        # was imported, silently ignoring any later monkeypatch.
        # max_rows defaults to top_n's cap; callers that pivot multiple raw
        # rows into one output row per Agniveer (AttemptWise/BestAttempt
        # with no agniveer_no) must pass max_rows=0 (uncapped) — capping at
        # top_n raw rows before pivoting can cut an Agniveer off mid-way
        # and silently return fewer than top_n Agniveers.
        effective_cap = _row_cap(top_n) if max_rows is None else max_rows
        rows, err = sql_executor.run_readonly(sql, tuple(params), max_rows=effective_cap)
        if err:
            return None, err
        return rows or [], None

    def _mark_generated() -> None:
        try:
            from sql_executor import metrics_hook

            metrics_hook("generated")
        except Exception:
            pass

    try:
        _raw_q = str(intent.get("raw_query") or "").lower()
        if any(
            phrase in _raw_q
            for phrase in ("full marks", "maximum marks", "perfect score", "full score")
        ):
            # "scored full marks in any subject/section" — a genuine 100%
            # (MarksObtained == that sub-item's MaxMarks) check across every
            # section (no `section` filter unless one was explicitly named),
            # reusing the same DynamicMax/Percentage CTE the Grading branch
            # below builds. This used to fall through with no dedicated
            # handling and land on whatever operation the classifier's
            # keyword scoring happened to guess (observed: "Drop").
            source_sql, source_params = _filtered_attempts_source(
                require_best_attempt=True
            )
            sql = f"""
            WITH FilteredAttempts AS (
                {source_sql}
            ),
            AttemptedMax AS (
                SELECT DISTINCT AgniveerId, SectionId, SubItemId, MaxMarks
                FROM FilteredAttempts
            ),
            DynamicMax AS (
                SELECT AgniveerId, SectionId, SUM(COALESCE(MaxMarks, 0)) AS DynamicMax
                FROM AttemptedMax
                GROUP BY AgniveerId, SectionId
            ),
            BestTotals AS (
                SELECT
                    AgniveerId, AgniveerNo, FullName, SectionId, SectionName,
                    SUM(MarksObtained) AS BestTotal
                FROM FilteredAttempts
                GROUP BY AgniveerId, AgniveerNo, FullName, SectionId, SectionName
            )
            SELECT
                bt.AgniveerNo,
                bt.FullName,
                bt.SectionName,
                bt.BestTotal AS MarksObtained,
                dm.DynamicMax AS MaxMarks
            FROM BestTotals bt
                INNER JOIN DynamicMax dm
                    ON dm.AgniveerId = bt.AgniveerId AND dm.SectionId = bt.SectionId
            WHERE dm.DynamicMax > 0 AND bt.BestTotal = dm.DynamicMax
            ORDER BY bt.SectionName ASC, bt.AgniveerNo ASC
            """
            rows, err = _run(sql, source_params)
            if err:
                return None, err
            rows = rows if top_n is None else rows[:top_n]
            _mark_generated()
            return _to_section(rows=rows, intent=intent, sql=sql), None

        if operation in ("Top", "Bottom", "OverallPerformance"):
            descending = operation != "Bottom"
            order_dir = "DESC" if descending else "ASC"
            source_sql, source_params = _filtered_attempts_source(
                require_best_attempt=(attempt_no is None)
            )
            sql = f"""
            WITH FilteredAttempts AS (
                {source_sql}
            )
            SELECT
                AgniveerNo,
                FullName,
                SUM(MarksObtained) AS BestTotal
            FROM FilteredAttempts
            GROUP BY AgniveerNo, FullName
            ORDER BY BestTotal {order_dir}, AgniveerNo ASC
            """
            rows, err = _run(sql, source_params)
            if err:
                return None, err
            rows = rows if top_n is None else rows[:top_n]
            _mark_generated()
            return _to_section(rows=rows, intent=intent, sql=sql), None

        if operation == "BestAttempt":
            # A grand total across every section (the old behaviour) hides
            # exactly the section-level detail "best attempt" is supposed to
            # answer — an Agniveer's best BPET score, best PPT score, etc.,
            # not one number that blends BPET+PPT+FIRING+DRILL together.
            # Mirrors AttemptWise's shape but collapsed to each section's
            # single best attempt instead of every attempt.
            source_sql, source_params = _attemptwise_source(
                require_best_attempt=(attempt_no is None)
            )
            sql = f"""
            WITH FilteredAttempts AS (
                {source_sql}
            )
            SELECT
                AgniveerNo,
                FullName,
                SectionName,
                SUM(MarksObtained) AS BestTotal
            FROM FilteredAttempts
            GROUP BY AgniveerNo, FullName, SectionName
            ORDER BY AgniveerNo ASC, SectionName ASC
            """

            # Uncapped fetch when we'll pivot below — a top_n row cap here
            # would apply to raw (Agniveer, Section) rows, not Agniveers,
            # and could cut an Agniveer off mid-way through their sections.
            rows, err = _run(sql, source_params, max_rows=0 if not agniveer_no else None)
            if err:
                return None, err

            _mark_generated()

            if rows:
                if agniveer_no:
                    if top_n is not None and len(rows) > top_n:
                        rows = rows[:top_n]
                    return _to_section(rows, intent, sql=sql), None
                else:
                    # Limit AFTER pivoting, not before — slicing raw rows
                    # first can cut off mid-Agniveer (its later sections
                    # land past the row cutoff), so "top 3" would silently
                    # return fewer than 3 Agniveers.
                    pivoted = _pivot_best_attempt_rows(rows)
                    if top_n is not None and len(pivoted) > top_n:
                        pivoted = pivoted[:top_n]
                    return _to_section(pivoted, intent, sql=sql), None

            return _to_section(rows or [], intent, sql=sql), None

        if operation == "Average":
            source_sql, source_params = _filtered_attempts_source(
                require_best_attempt=(attempt_no is None)
            )
            sql = f"""
            WITH FilteredAttempts AS (
                {source_sql}
            )
            SELECT
                AgniveerNo,
                FullName,
                SectionName,
                AVG(CAST(MarksObtained AS DECIMAL(18, 2))) AS AverageMarks
            FROM FilteredAttempts
            GROUP BY AgniveerNo, FullName, SectionName
            ORDER BY AverageMarks DESC, AgniveerNo ASC
            """
            rows, err = _run(sql, source_params)
            if err:
                return None, err
            rows = rows if top_n is None else rows[:top_n]
            _mark_generated()
            return _to_section(rows=rows, intent=intent, sql=sql), None

        if operation == "AttemptWise":
            # Get the source CTE
            source_sql, source_params = _attemptwise_source()

            # Build the final aggregated query
            sql = f"""
            WITH FilteredAttempts AS (
                {source_sql}
            )
            SELECT
                AgniveerNo,
                FullName,
                SectionName,
                AttemptNo,
                SUM(MarksObtained) AS AttemptTotal
            FROM FilteredAttempts
            GROUP BY AgniveerNo, FullName, SectionName, AttemptNo
            ORDER BY AgniveerNo ASC, SectionName ASC, AttemptNo ASC
            """

            # Uncapped fetch when we'll pivot below — a top_n row cap here
            # would apply to raw (Agniveer, Section, Attempt) rows, not
            # Agniveers, and could cut an Agniveer off mid-way through
            # their sections/attempts.
            rows, err = _run(sql, source_params, max_rows=0 if not agniveer_no else None)
            if err:
                return None, err

            _mark_generated()

            # If AgniveerNo is provided, return flat (single Agniveer)
            # If no AgniveerNo, pivot to nested structure per Agniveer
            if rows:
                if agniveer_no:
                    # Single Agniveer — return as-is or limit
                    if top_n is not None and len(rows) > top_n:
                        rows = rows[:top_n]
                    return _to_section(rows, intent, sql=sql), None
                else:
                    # Multiple Agniveers — pivot for better display. Limit
                    # AFTER pivoting: slicing raw rows first can cut off
                    # mid-Agniveer (its later sections/attempts land past
                    # the row cutoff), so "top 3" would silently return
                    # fewer than 3 Agniveers.
                    pivoted = _pivot_attemptwise_rows(rows)
                    if top_n is not None and len(pivoted) > top_n:
                        pivoted = pivoted[:top_n]
                    return _to_section(pivoted, intent, sql=sql), None

            return _to_section(rows or [], intent, sql=sql), None

        if operation == "Trend":
            source_sql, source_params = _filtered_attempts_source(
                require_best_attempt=False
            )
            sql = f"""
            WITH FilteredAttempts AS (
                {source_sql}
            )
            SELECT
                SectionName,
                AttemptNo,
                COUNT(DISTINCT AgniveerId) AS AgniveerCount,
                AVG(CAST(MarksObtained AS DECIMAL(18, 2))) AS AverageMarks,
                SUM(MarksObtained) AS TotalMarks
            FROM FilteredAttempts
            GROUP BY SectionName, AttemptNo
            ORDER BY SectionName ASC, AttemptNo ASC
            """
            rows, err = _run(sql, source_params)
            if err:
                return None, err
            rows = rows if top_n is None else rows[:top_n]
            _mark_generated()
            return _to_section(rows=rows, intent=intent, sql=sql), None

        if operation in ("Improvement", "Drop"):
            _raw_q = str(intent.get("raw_query") or "").lower()
            _wants_first_vs_best = (
                from_attempt is None
                and to_attempt is None
                and "first" in _raw_q
                and "best" in _raw_q
            )
            if _wants_first_vs_best:
                # "improved between first and best attempt" has no numeric
                # from/to attempt (entity extraction only recognises literal
                # "attempt N" windows), so without this branch it silently
                # fell through to the no-window default below, which compares
                # the two MOST RECENT attempts (rn=1 vs rn=2) — answering a
                # different question than the one asked whenever an
                # Agniveer's best attempt isn't also their latest one.
                source_sql, source_params = _filtered_attempts_source(
                    require_best_attempt=False
                )
                delta_filter = "Delta > 0" if operation == "Improvement" else "Delta < 0"
                order_column = "[Improvement]" if operation == "Improvement" else "[Drop]"

                sql = f"""
                WITH FilteredAttempts AS (
                    {source_sql}
                ),
                AttemptTotals AS (
                    SELECT
                        AgniveerId,
                        AgniveerNo,
                        FullName,
                        SectionName,
                        AttemptNo,
                        SUM(MarksObtained) AS AttemptTotal
                    FROM FilteredAttempts
                    GROUP BY AgniveerId, AgniveerNo, FullName, SectionName, AttemptNo
                ),
                FirstAttempt AS (
                    SELECT * FROM AttemptTotals WHERE AttemptNo = 1
                ),
                BestAttempt AS (
                    SELECT t.*
                    FROM AttemptTotals t
                    WHERE t.AttemptTotal = (
                        SELECT MAX(t2.AttemptTotal)
                        FROM AttemptTotals t2
                        WHERE t2.AgniveerId = t.AgniveerId
                            AND t2.SectionName = t.SectionName
                    )
                ),
                Compared AS (
                    SELECT
                        cur.AgniveerNo,
                        cur.FullName,
                        cur.SectionName,
                        first.AttemptNo AS FirstAttemptNo,
                        cur.AttemptNo AS BestAttemptNo,
                        first.AttemptTotal AS FirstTotal,
                        cur.AttemptTotal AS BestTotal,
                        cur.AttemptTotal - first.AttemptTotal AS Delta
                    FROM BestAttempt cur
                        INNER JOIN FirstAttempt first
                            ON first.AgniveerId = cur.AgniveerId
                            AND first.SectionName = cur.SectionName
                )
                SELECT
                    AgniveerNo,
                    FullName,
                    SectionName,
                    FirstAttemptNo,
                    BestAttemptNo,
                    FirstTotal,
                    BestTotal,
                    CASE WHEN Delta > 0 THEN Delta ELSE 0 END AS Improvement,
                    CASE WHEN Delta < 0 THEN -Delta ELSE 0 END AS [Drop]
                FROM Compared
                WHERE {delta_filter}
                ORDER BY {order_column} DESC, AgniveerNo ASC
                """
                rows, err = _run(sql, source_params)
                if err:
                    return None, err
                rows = rows if top_n is None else rows[:top_n]
                _mark_generated()
                return _to_section(rows=rows, intent=intent, sql=sql), None

            if from_attempt is not None and to_attempt is not None:
                current_attempt = int(to_attempt)
                previous_attempt = int(from_attempt)
                source_sql, source_params = _filtered_attempts_source(
                    require_best_attempt=False, include_attempt_window=True
                )
                delta_filter = "Delta > 0" if operation == "Improvement" else "Delta < 0"
                order_column = "[Improvement]" if operation == "Improvement" else "[Drop]"

                sql = f"""
                WITH FilteredAttempts AS (
                    {source_sql}
                ),
                AttemptTotals AS (
                    SELECT
                        AgniveerId,
                        AgniveerNo,
                        FullName,
                        SectionName,
                        AttemptNo,
                        SUM(MarksObtained) AS AttemptTotal,
                        MAX(AttemptedDate) AS AttemptedDate
                    FROM FilteredAttempts
                    GROUP BY AgniveerId, AgniveerNo, FullName, SectionName, AttemptNo
                ),
                Compared AS (
                    SELECT
                        cur.AgniveerNo,
                        cur.FullName,
                        cur.SectionName,
                        cur.AttemptNo AS CurrentAttemptNo,
                        prev.AttemptNo AS PreviousAttemptNo,
                        cur.AttemptTotal AS CurrentTotal,
                        prev.AttemptTotal AS PreviousTotal,
                        cur.AttemptTotal - prev.AttemptTotal AS Delta
                    FROM AttemptTotals cur
                        INNER JOIN AttemptTotals prev
                            ON prev.AgniveerId = cur.AgniveerId
                            AND prev.SectionName = cur.SectionName
                    WHERE cur.AttemptNo = ? AND prev.AttemptNo = ?
                )
                SELECT
                    AgniveerNo,
                    FullName,
                    SectionName,
                    CurrentAttemptNo,
                    PreviousAttemptNo,
                    CurrentTotal,
                    PreviousTotal,
                    CASE WHEN Delta > 0 THEN Delta ELSE 0 END AS Improvement,
                    CASE WHEN Delta < 0 THEN -Delta ELSE 0 END AS [Drop]

                FROM Compared
                WHERE {delta_filter}
                ORDER BY {order_column} DESC, AgniveerNo ASC
                """
                params = source_params + [current_attempt, previous_attempt]
            else:
                source_sql, source_params = _filtered_attempts_source(
                    require_best_attempt=False
                )
                delta_filter = "Delta > 0" if operation == "Improvement" else "Delta < 0"
                order_column = "[Improvement]" if operation == "Improvement" else "[Drop]"

                sql = f"""
                WITH FilteredAttempts AS (
                    {source_sql}
                ),
                AttemptTotals AS (
                    SELECT
                        AgniveerId,
                        AgniveerNo,
                        FullName,
                        SectionName,
                        AttemptNo,
                        SUM(MarksObtained) AS AttemptTotal,
                        MAX(AttemptedDate) AS AttemptedDate
                    FROM FilteredAttempts
                    GROUP BY AgniveerId, AgniveerNo, FullName, SectionName, AttemptNo
                ),
                Ranked AS (
                    SELECT
                        *,
                        ROW_NUMBER() OVER (
                            PARTITION BY AgniveerId, SectionName
                            ORDER BY AttemptNo DESC, AttemptedDate DESC
                        ) AS rn
                    FROM AttemptTotals
                ),
                Compared AS (
                    SELECT
                        cur.AgniveerNo,
                        cur.FullName,
                        cur.SectionName,
                        cur.AttemptNo AS CurrentAttemptNo,
                        prev.AttemptNo AS PreviousAttemptNo,
                        cur.AttemptTotal AS CurrentTotal,
                        prev.AttemptTotal AS PreviousTotal,
                        cur.AttemptTotal - prev.AttemptTotal AS Delta
                    FROM Ranked cur
                        INNER JOIN Ranked prev
                            ON prev.AgniveerId = cur.AgniveerId
                            AND prev.SectionName = cur.SectionName
                    WHERE cur.rn = 1 AND prev.rn = 2
                )
                SELECT
                    AgniveerNo,
                    FullName,
                    SectionName,
                    CurrentAttemptNo,
                    PreviousAttemptNo,
                    CurrentTotal,
                    PreviousTotal,
                    CASE WHEN Delta > 0 THEN Delta ELSE 0 END AS Improvement,
                    CASE WHEN Delta < 0 THEN -Delta ELSE 0 END AS [Drop]

                FROM Compared
                WHERE {delta_filter}
                ORDER BY {order_column} DESC, AgniveerNo ASC
                """
                params = source_params

            rows, err = _run(sql, params)
            if err:
                return None, err
            rows = rows if top_n is None else rows[:top_n]
            _mark_generated()
            return _to_section(rows=rows, intent=intent, sql=sql), None

        if operation in ("Grading", "GradingSummary"):
            source_sql, source_params = _filtered_attempts_source(
                require_best_attempt=True
            )
            sql = f"""
            WITH FilteredAttempts AS (
                {source_sql}
            ),
            AttemptedMax AS (
                SELECT DISTINCT
                    AgniveerId,
                    SectionId,
                    SubItemId,
                    MaxMarks
                FROM FilteredAttempts
            ),
            DynamicMax AS (
                SELECT
                    AgniveerId,
                    SectionId,
                    SUM(COALESCE(MaxMarks, 0)) AS DynamicMax
                FROM AttemptedMax
                GROUP BY AgniveerId, SectionId
            ),
            BestTotals AS (
                SELECT
                    AgniveerId,
                    AgniveerNo,
                    FullName,
                    SectionId,
                    SectionName,
                    SUM(MarksObtained) AS BestTotal
                FROM FilteredAttempts
                GROUP BY AgniveerId, AgniveerNo, FullName, SectionId, SectionName
            ),
            Scored AS (
                SELECT
                    bt.AgniveerId,
                    bt.AgniveerNo,
                    bt.FullName,
                    bt.SectionName,
                    bt.BestTotal,
                    dm.DynamicMax,
                    CASE
                        WHEN dm.DynamicMax IS NULL OR dm.DynamicMax = 0 THEN NULL
                        WHEN 100.0 * bt.BestTotal / dm.DynamicMax >= 90 THEN 'Exceptionally Well'
                        WHEN 100.0 * bt.BestTotal / dm.DynamicMax >= 75 THEN 'Excellent'
                        WHEN 100.0 * bt.BestTotal / dm.DynamicMax >= 60 THEN 'Good'
                        WHEN 100.0 * bt.BestTotal / dm.DynamicMax >= 45 THEN 'SAT'
                        ELSE 'Fail'
                    END AS Grade,
                    CASE
                        WHEN dm.DynamicMax > 0 THEN ROUND(100.0 * bt.BestTotal / dm.DynamicMax, 2)
                        ELSE NULL
                    END AS Percentage
                FROM BestTotals bt
                    LEFT JOIN DynamicMax dm
                        ON dm.AgniveerId = bt.AgniveerId
                        AND dm.SectionId = bt.SectionId
            )
            """
            params = source_params

            if operation == "GradingSummary":
                sql += """
                SELECT
                    SectionName,
                    Grade,
                    COUNT(*) AS Count
                FROM Scored
                GROUP BY SectionName, Grade
                ORDER BY SectionName ASC, Grade ASC
                """
            else:
                if section:
                    sql += """
                    SELECT
                        AgniveerNo,
                        FullName,
                        SectionName,
                        Percentage,
                        Grade
                    FROM Scored
                    """
                else:
                    sql += """
                    SELECT
                        AgniveerNo,
                        FullName,
                        SectionName,
                        Percentage AS OverallPercentage,
                        Grade
                    FROM Scored
                    """

                if requested_grade:
                    sql += " WHERE LOWER(Grade) = LOWER(?)"
                    params = params + [str(requested_grade)]
                if section:
                    sql += (
                        " ORDER BY Percentage DESC, AgniveerNo ASC"
                    )
                else:
                    sql += (
                        " ORDER BY OverallPercentage DESC, AgniveerNo ASC"
                    )
                if top_n:
                    sql += ""

            rows, err = _run(sql, params)
            if err:
                return None, err
            if operation != "GradingSummary":
                rows = rows if top_n is None else rows[:top_n]
            _mark_generated()
            return _to_section(rows=rows, intent=intent, sql=sql), None

        return None, f"Operation {operation} logic not yet fully translated in executor."
    except Exception as exc:
        logger.error("Performance executor failed: %s", exc, exc_info=True)
        return None, str(exc)
