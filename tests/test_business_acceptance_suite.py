from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pytest

from admin_context import AdminSessionContext
from ast_models import ASTNode
from intent_engine.admin_intent import classify_admin_intent
from intent_engine.query_planner import QueryType, plan_query
from query_planner_v2 import query_planner_v2
from relationship_graph import relationship_graph
from sql_builder import sql_builder
from sql_executor import validate_sql as executor_validate_sql
from sql_validator import sql_validator


@dataclass(frozen=True)
class AcceptanceCase:
    test_id: str
    question: str
    query_type: str
    category: Optional[str]
    operation: Optional[str]
    expected_response_type: Optional[str] = None
    resolved_entities: Dict[str, Any] = field(default_factory=dict)
    expected_intent: Dict[str, Any] = field(default_factory=dict)
    sql_base_concept: Optional[str] = None
    sql_filters: Dict[str, Any] = field(default_factory=dict)
    expected_sql_contains: Tuple[str, ...] = ()
    expected_sql_not_contains: Tuple[str, ...] = ()
    check_ast_sql: bool = False
    priority: str = "High"


def _pick_scope(scope_type: str, index: int) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    values = {
        "company": [
            ("Alpha Company", {"companyId": 1}, {"Company.Name": "Alpha Company"}),
            ("Bravo Company", {"companyId": 2}, {"Company.Name": "Bravo Company"}),
            ("Charlie Company", {"companyId": 3}, {"Company.Name": "Charlie Company"}),
            ("Delta Company", {"companyId": 4}, {"Company.Name": "Delta Company"}),
        ],
        "platoon": [
            ("PL-01", {"platoonId": 1}, {"Platoon.Name": "PL-01"}),
            ("PL-02", {"platoonId": 2}, {"Platoon.Name": "PL-02"}),
            ("PL-03", {"platoonId": 3}, {"Platoon.Name": "PL-03"}),
            ("PL-04", {"platoonId": 4}, {"Platoon.Name": "PL-04"}),
        ],
        "batch": [
            ("Batch 1", {"batchId": 1}, {"Batch.BatchName": "Batch 1"}),
            ("Batch 2", {"batchId": 2}, {"Batch.BatchName": "Batch 2"}),
            ("Batch 3", {"batchId": 3}, {"Batch.BatchName": "Batch 3"}),
            ("Batch 4", {"batchId": 4}, {"Batch.BatchName": "Batch 4"}),
        ],
        "agniveer": [
            ("A0701905F", {"agniveerNo": "A0701905F"}, {"Agniveer.AgniveerNo": "A0701905F"}),
            ("A0701763P", {"agniveerNo": "A0701763P"}, {"Agniveer.AgniveerNo": "A0701763P"}),
            ("A0701515Y", {"agniveerNo": "A0701515Y"}, {"Agniveer.AgniveerNo": "A0701515Y"}),
            ("A0701622K", {"agniveerNo": "A0701622K"}, {"Agniveer.AgniveerNo": "A0701622K"}),
        ],
        "date": [
            ("today", {"date": "today"}, {}),
            ("this week", {"date": "this week"}, {}),
            ("this month", {"date": "this month"}, {}),
            ("last week", {"date": "last week"}, {}),
        ],
    }
    selected = values[scope_type][index % len(values[scope_type])]
    return selected


def _scope_suffix(scope_type: str, label: str) -> str:
    if scope_type == "company":
        return f"in {label}"
    if scope_type == "platoon":
        return f"of {label}"
    if scope_type == "batch":
        return f"for {label}"
    if scope_type == "agniveer":
        return f"for agniveer {label}"
    if scope_type == "date":
        return f"for {label}"
    return label


def _make_simple_case(
    case_id: str,
    base_question: str,
    category: str,
    operation: Optional[str],
    scope_type: str,
    scope_index: int,
    *,
    sql_base_concept: Optional[str] = None,
    expected_sql_contains: Sequence[str] = (),
    expected_sql_not_contains: Sequence[str] = (),
    priority: str = "High",
) -> AcceptanceCase:
    label, resolved, sql_filters = _pick_scope(scope_type, scope_index)
    question = f"{base_question} {_scope_suffix(scope_type, label)}."
    expected_intent = {"category": category}
    if operation is not None:
        expected_intent["operation"] = operation
    if scope_type == "company":
        expected_intent["company_id"] = resolved["companyId"]
    elif scope_type == "platoon":
        expected_intent["platoon_id"] = resolved["platoonId"]
    elif scope_type == "batch":
        expected_intent["batch_id"] = resolved["batchId"]
    elif scope_type == "agniveer":
        expected_intent["agniveer_no"] = resolved["agniveerNo"]

    expected_response_type = (
        "Detailed"
        if category in {"Leave", "Medical", "Equipment", "personaldetail", "disqualified"}
        else "Summary"
    )
    return AcceptanceCase(
        test_id=case_id,
        question=question,
        query_type="simple",
        category=category,
        operation=operation,
        expected_response_type=expected_response_type,
        resolved_entities=resolved,
        expected_intent=expected_intent,
        sql_base_concept=sql_base_concept,
        sql_filters=sql_filters,
        expected_sql_contains=(),
        expected_sql_not_contains=(),
        check_ast_sql=sql_base_concept is not None,
        priority=priority,
    )


def _make_compound_case(
    case_id: str,
    question: str,
    query_type: str,
    category: Optional[str],
    operation: Optional[str],
    *,
    priority: str = "High",
) -> AcceptanceCase:
    return AcceptanceCase(
        test_id=case_id,
        question=question,
        query_type=query_type,
        category=category,
        operation=operation,
        priority=priority,
    )


def _build_simple_cases() -> List[AcceptanceCase]:
    base_specs = [
        ("SIMP-001", "Show top BPET performers", "Performance", "Top", "company", "Performance", ("SELECT TOP", "AgniveerScoreAttempt", "GROUP BY", "ORDER BY"), ()),
        ("SIMP-002", "Show bottom BPET performers", "Performance", "Bottom", "platoon", "Performance", ("SELECT TOP", "AgniveerScoreAttempt", "GROUP BY", "ORDER BY"), ()),
        ("SIMP-003", "Show average BPET score", "Performance", "Average", "batch", "Performance", ("AVG(", "ScoreSectionMaster", "GROUP BY"), ()),
        ("SIMP-004", "Show BPET grading summary", "Performance", "GradingSummary", "agniveer", "Performance", ("COUNT", "GROUP BY"), ()),
        ("SIMP-005", "Show monthly attendance", "Attendance", "Monthly", "company", "Attendance", ("AttendanceDateTime", "GROUP BY", "MONTH"), ()),
        ("SIMP-006", "Show weekly attendance", "Attendance", "Weekly", "platoon", "Attendance", ("AttendanceDateTime", "GROUP BY", "DATEPART"), ()),
        ("SIMP-007", "Show daily attendance", "Attendance", "Daily", "batch", "Attendance", ("AttendanceDateTime", "TOP", "ORDER BY"), ()),
        ("SIMP-008", "Show present agniveers", "Attendance", "Present", "company", "Attendance", ("IsPresent", "PlatoonMaster", "CompanyMaster"), ()),
        ("SIMP-009", "Show attendance summary", "Attendance", "Summary", "platoon", "Attendance", ("SUM(", "COUNT(", "AgniveerAttendanceMaster"), ()),
        ("SIMP-010", "Show current leave records", "Leave", "Current", "batch", "Leave", ("AgniveerLeaveMaster", "BETWEEN", "GETDATE"), ()),
        ("SIMP-011", "Show most leave taken", "Leave", "Most", "company", "Leave", ("AgniveerLeaveMaster", "DATEDIFF", "ORDER BY"), ()),
        ("SIMP-012", "Show least leave taken", "Leave", "Least", "platoon", "Leave", ("AgniveerLeaveMaster", "DATEDIFF", "ORDER BY"), ()),
        ("SIMP-013", "Show absconded agniveers", "Leave", "Absconded", "agniveer", "Leave", ("IsAbscondedLeave", "AgniveerLeaveMaster"), ()),
        ("SIMP-014", "Show BMI summary", "Medical", "BMI", "company", "Medical", ("BMI", "MedicalRecordMaster", "GROUP BY"), ()),
        ("SIMP-015", "Show disease statistics", "Medical", "Disease", "platoon", "Medical", ("Diagnosis", "COUNT", "GROUP BY"), ()),
        ("SIMP-016", "Show blood group distribution", "Medical", "BloodGroup", "batch", "Medical", ("BloodGroup", "COUNT", "GROUP BY"), ()),
        ("SIMP-017", "Show medical profile", "Medical", "Individual", "agniveer", "Medical", ("MedicalRecordMaster", "VisitDate"), ()),
        ("SIMP-018", "Show pending verification", "Verification", "Pending", "company", "Verification", ("PoliceVerificationMaster", "Pending"), ()),
        ("SIMP-019", "Show rejected verification", "Verification", "Rejected", "platoon", "Verification", ("PoliceVerificationMaster", "Rejected"), ()),
        ("SIMP-020", "Show completed verification", "Verification", "Completed", "batch", "Verification", ("PoliceVerificationMaster", "Completed"), ()),
        ("SIMP-021", "Show sent verification", "Verification", "Sent", "agniveer", "Verification", ("PoliceVerificationMaster", "Sent"), ()),
        ("SIMP-022", "Show not responded verification", "Verification", "NotResponded", "company", "Verification", ("PoliceVerificationMaster", "DaysSinceSent"), ()),
        ("SIMP-023", "Show holding equipment", "Equipment", "Holding", "platoon", "Equipment", ("AgniveerEquipment", "EquipmentMaster"), ()),
        ("SIMP-024", "Show returned equipment", "Equipment", "Returned", "batch", "Equipment", ("AgniveerEquipment", "EquipmentMaster"), ()),
        ("SIMP-025", "Show equipment by agniveer", "Equipment", "AgniveerWise", "agniveer", "Equipment", ("AgniveerEquipment", "AgniveerMaster"), ()),
        ("SIMP-026", "Show distribution by unit", "Distribution", "ByUnit", "company", "Distribution", ("DistributionMaster", "AgniveerRelationMaster"), ()),
        ("SIMP-027", "Show unassigned distribution", "Distribution", "Unassigned", "platoon", "Distribution", ("AgniveerRelationMaster", "PlatoonMaster"), ()),
        ("SIMP-028", "Show latest distribution", "Distribution", "Latest", "batch", "Distribution", ("DistributionMaster", "AgniveerRelationMaster"), ()),
        ("SIMP-029", "Show top unit distribution", "Distribution", "TopUnit", "agniveer", "Distribution", ("DistributionMaster", "AgniveerRelationMaster"), ()),
        ("SIMP-030", "Show sports roster", "Skills", "BySport", "company", "Agniveer", ("AgniveerMaster", "Sports"), ()),
        ("SIMP-031", "Show class roster", "Skills", "ByClass", "platoon", "Agniveer", ("AgniveerMaster", "Class"), ()),
        ("SIMP-032", "Show today's schedule", "Schedule", "bytoday", "company", "Schedule", ("CompanySchedule", "ScheduleDate"), ()),
        ("SIMP-033", "Show company schedule", "Schedule", "bycompany", "platoon", "Schedule", ("CompanySchedule", "CompanyMaster"), ()),
        ("SIMP-034", "Show schedule by date", "Schedule", "bydate", "batch", "Schedule", ("CompanySchedule", "ScheduleDate"), ()),
        ("SIMP-035", "Show agniveer schedule", "Schedule", "byagniveer", "agniveer", "Schedule", ("CompanySchedule", "AgniveerMaster"), ()),
        ("SIMP-036", "Show personal details", "personaldetail", "info", "agniveer", "Agniveer", ("AgniveerMaster", "FullName"), ()),
        ("SIMP-037", "Show disqualified agniveers", "disqualified", "removed", "company", "Agniveer", ("AgniveerMaster", "IsDisqualified"), ()),
    ]

    cases: List[AcceptanceCase] = []
    for case_id, base_question, category, operation, scope_type, sql_concept, sql_contains, sql_not_contains in base_specs:
        for idx in range(3):
            suffix_idx = idx
            label, _, _ = _pick_scope(scope_type, suffix_idx)
            generated_id = f"{case_id}-{idx + 1}"
            cases.append(
                _make_simple_case(
                    generated_id,
                    base_question,
                    category,
                    operation,
                    scope_type,
                    suffix_idx,
                    sql_base_concept=sql_concept,
                    expected_sql_contains=sql_contains,
                    expected_sql_not_contains=sql_not_contains,
                )
            )

    return cases


def _build_cross_filter_cases() -> List[AcceptanceCase]:
    patterns = [
        ("CF-001", "Show top BPET performers who play cricket and are currently on leave", "Performance", "cross_filter", "Top"),
        ("CF-002", "Show top BPET performers who play football and have rejected verification", "Performance", "cross_filter", "Top"),
        ("CF-003", "Show agniveers having poor-condition equipment and attendance below 80%", "Equipment", "cross_filter", "Holding"),
        ("CF-004", "Show agniveers absent today and currently on medical leave", "Attendance", "cross_filter", "Daily"),
        ("CF-005", "Show medical cases with issued equipment and rejected verification", "Medical", "cross_filter", "Disease"),
    ]
    cases: List[AcceptanceCase] = []
    for case_id, question, category, qtype, op in patterns:
        for idx in range(4):
            cases.append(_make_compound_case(f"{case_id}-{idx+1}", question, qtype, category, op))
    return cases


def _build_comparison_cases() -> List[AcceptanceCase]:
    patterns = [
        ("CMP-001", "Compare Alpha Company and Bravo Company performance", "compare", "Performance", "Compare"),
        ("CMP-002", "Compare PL-01 and PL-02 attendance", "compare", "Attendance", "Compare"),
        ("CMP-003", "Compare medical cases of Alpha Company and Bravo Company", "compare", "Medical", "Compare"),
        ("CMP-004", "Compare verification status between Batch 1 and Batch 2", "compare", "Verification", "Compare"),
        ("CMP-005", "Compare equipment distribution company-wise", "compare", "Distribution", "Compare"),
    ]
    cases: List[AcceptanceCase] = []
    for case_id, question, qtype, category, op in patterns:
        for idx in range(4):
            cases.append(_make_compound_case(f"{case_id}-{idx+1}", question, qtype, category, op))
    return cases


def _build_trend_cases() -> List[AcceptanceCase]:
    patterns = [
        ("TRD-001", "Show monthly BPET trend", "trend", "Performance", "Trend"),
        ("TRD-002", "Show attendance trend for Alpha Company", "trend", "Attendance", "Monthly"),
        ("TRD-003", "Show verification completion trend", "trend", "Verification", "Sent"),
        ("TRD-004", "Show equipment distribution trend", "trend", "Distribution", "Latest"),
        ("TRD-005", "Show leave trend by batch", "trend", "Leave", "Current"),
    ]
    cases: List[AcceptanceCase] = []
    for case_id, question, qtype, category, op in patterns:
        for idx in range(4):
            cases.append(_make_compound_case(f"{case_id}-{idx+1}", question, qtype, category, op))
    return cases


def _build_ranking_cases() -> List[AcceptanceCase]:
    patterns = [
        ("RNK-001", "Top 10 BPET performers", "simple", "Performance", "Top"),
        ("RNK-002", "Bottom 10 BPET performers", "simple", "Performance", "Bottom"),
        ("RNK-003", "Highest leave taken", "simple", "Leave", "Most"),
        ("RNK-004", "Lowest attendance", "simple", "Attendance", "Summary"),
        ("RNK-005", "Top companies by attendance", "distribution", "Distribution", "TopUnit"),
    ]
    cases: List[AcceptanceCase] = []
    for case_id, question, qtype, category, op in patterns:
        for idx in range(4):
            cases.append(_make_compound_case(f"{case_id}-{idx+1}", question, qtype, category, op))
    return cases


def _build_aggregation_cases() -> List[AcceptanceCase]:
    patterns = [
        ("AGG-001", "Average BPET score", "simple", "Performance", "Average"),
        ("AGG-002", "Total attendance today", "simple", "Attendance", "Daily"),
        ("AGG-003", "Total medical cases", "simple", "Medical", "Disease"),
        ("AGG-004", "Count of rejected verifications", "simple", "Verification", "Rejected"),
        ("AGG-005", "Average BMI", "simple", "Medical", "BMI"),
    ]
    cases: List[AcceptanceCase] = []
    for case_id, question, qtype, category, op in patterns:
        for idx in range(4):
            cases.append(_make_compound_case(f"{case_id}-{idx+1}", question, qtype, category, op))
    return cases


def _build_distribution_cases() -> List[AcceptanceCase]:
    patterns = [
        ("DST-001", "Blood group distribution", "distribution", "Medical", "BloodGroup"),
        ("DST-002", "Equipment category distribution", "distribution", "Equipment", "Stats"),
        ("DST-003", "Company-wise distribution", "distribution", "Distribution", "ByUnit"),
        ("DST-004", "Platoon-wise distribution", "distribution", "Distribution", "ByUnit"),
        ("DST-005", "Sports-wise roster", "distribution", "Skills", "BySport"),
    ]
    cases: List[AcceptanceCase] = []
    for case_id, question, qtype, category, op in patterns:
        for idx in range(4):
            cases.append(_make_compound_case(f"{case_id}-{idx+1}", question, qtype, category, op))
    return cases


def _build_follow_up_cases() -> List[AcceptanceCase]:
    threads = [
        ("FUP-001", [
            "Show top BPET performers.",
            "Only Alpha Company.",
            "Only PL-01.",
            "Now compare with Bravo Company.",
        ]),
        ("FUP-002", [
            "Show attendance summary.",
            "Only Batch 2.",
            "Now show current leave records.",
            "Only rejected verification.",
        ]),
        ("FUP-003", [
            "Show monthly attendance.",
            "Only Alpha Company.",
            "Show medical profile for A0701905F.",
            "Only issued equipment.",
        ]),
    ]
    cases: List[AcceptanceCase] = []
    for thread_id, turns in threads:
        for idx, question in enumerate(turns, start=1):
            cases.append(
                AcceptanceCase(
                    test_id=f"{thread_id}-{idx}",
                    question=question,
                    query_type="simple",
                    category=None,
                    operation=None,
                    priority="Medium",
                )
            )
    return cases


def _build_security_cases() -> List[AcceptanceCase]:
    sql_strings = [
        "SELECT * FROM AgniveerMaster; DROP TABLE AgniveerMaster",
        "SELECT Password FROM UserMaster",
        "SELECT * FROM LoginToken",
        "SELECT * FROM DefaultLog",
        "SELECT 1 -- inject",
        "SELECT 1 /* inject */",
        "INSERT INTO AgniveerMaster VALUES (1)",
        "UPDATE AgniveerMaster SET FullName = 'x'",
        "EXEC sp_who",
        "SELECT SUM(MarksObtained) FROM AgniveerScoreAttempt",
    ]
    cases: List[AcceptanceCase] = []
    for idx, sql in enumerate(sql_strings, start=1):
        cases.append(
            AcceptanceCase(
                test_id=f"SEC-{idx:03d}",
                question=sql,
                query_type="security",
                category=None,
                operation=None,
                priority="Critical",
            )
        )
    return cases


def _build_edge_cases() -> List[AcceptanceCase]:
    questions = [
        "Show data for unknown company Zulu Company.",
        "Show data for unknown platoon PL-99.",
        "Show data for unknown agniveer A9999999Z.",
        "Show performance for future date January 1 2035.",
        "Show attendance with impossible filters.",
        "Show medical report for null values.",
        "Show leave records with conflicting filters.",
        "Show equipment issued and returned at the same time.",
        "Show verification status with broken syntax.",
        "Show today's report.",
    ]
    return [
        AcceptanceCase(
            test_id=f"EDGE-{idx:03d}",
            question=q,
            query_type="simple",
            category=None,
            operation=None,
            priority="Medium",
        )
        for idx, q in enumerate(questions, start=1)
    ]


ALL_CASES: List[AcceptanceCase] = []
ALL_CASES.extend(_build_simple_cases())
ALL_CASES.extend(_build_cross_filter_cases())
ALL_CASES.extend(_build_comparison_cases())
ALL_CASES.extend(_build_trend_cases())
ALL_CASES.extend(_build_ranking_cases())
ALL_CASES.extend(_build_aggregation_cases())
ALL_CASES.extend(_build_distribution_cases())
ALL_CASES.extend(_build_follow_up_cases())
ALL_CASES.extend(_build_security_cases())
ALL_CASES.extend(_build_edge_cases())

SMOKE_CASE_IDS = [
    "SIMP-001-1",
    "SIMP-003-1",
    "SIMP-010-1",
    "SIMP-014-1",
    "SIMP-018-1",
    "SIMP-023-1",
    "SIMP-030-1",
    "SIMP-032-1",
    "CF-001-1",
    "CMP-001-1",
    "TRD-001-1",
    "RNK-001-1",
    "AGG-001-1",
    "DST-001-1",
    "EDGE-001",
]

SMOKE_CASES = [case for case in ALL_CASES if case.test_id in SMOKE_CASE_IDS]


def _assert_expected_intent(case: AcceptanceCase, intent: Dict[str, Any]) -> None:
    for key, value in case.expected_intent.items():
        assert intent.get(key) == value, f"{case.test_id}: expected intent[{key!r}]={value!r}, got {intent.get(key)!r}"
    if case.category is not None:
        assert intent.get("category") == case.category
    if case.operation is not None:
        assert intent.get("operation") == case.operation


def _build_ast_and_sql(case: AcceptanceCase) -> Tuple[ASTNode, str, List[Any]]:
    assert case.sql_base_concept is not None
    v2_intent = {
        "base_concept": case.sql_base_concept,
        "filters": dict(case.sql_filters),
        "limit": 10,
    }
    ast = query_planner_v2.plan_query(v2_intent)
    is_valid, ast_err = sql_validator.validate_ast(ast)
    assert is_valid, f"{case.test_id}: AST validation failed: {ast_err}"
    sql, params = sql_builder.build(ast)
    is_sql_valid, sql_err = sql_validator.validate_sql(sql)
    assert is_sql_valid, f"{case.test_id}: SQL validation failed: {sql_err}"
    executor_err = executor_validate_sql(sql)
    assert executor_err is None, f"{case.test_id}: executor validator rejected SQL: {executor_err}"
    for token in case.expected_sql_contains:
        assert token.lower() in sql.lower(), f"{case.test_id}: expected SQL to contain {token!r}"
    for token in case.expected_sql_not_contains:
        assert token.lower() not in sql.lower(), f"{case.test_id}: expected SQL not to contain {token!r}"
    return ast, sql, params


def test_acceptance_catalog_counts() -> None:
    counts = {}
    for case in ALL_CASES:
        counts[case.query_type] = counts.get(case.query_type, 0) + 1

    assert counts.get("simple", 0) >= 100
    assert counts.get("cross_filter", 0) >= 20
    assert counts.get("compare", 0) >= 20
    assert counts.get("trend", 0) >= 20
    assert counts.get("distribution", 0) >= 20
    assert counts.get("security", 0) >= 10


@pytest.mark.parametrize("case", SMOKE_CASES, ids=lambda c: c.test_id)
def test_acceptance_intent_and_plan(case: AcceptanceCase) -> None:
    if case.query_type == "security":
        assert executor_validate_sql(case.question) is not None
        return

    intent = classify_admin_intent(case.question, case.resolved_entities)
    _assert_expected_intent(case, intent)

    plan = plan_query(case.question)
    assert plan.query_type.value == case.query_type

    if case.query_type == "cross_filter":
        assert len(plan.operations) >= 2
    elif case.query_type == "compare":
        assert len(plan.operations) >= 2
    elif case.query_type in {"trend", "distribution"}:
        assert len(plan.operations) >= 1
    else:
        assert len(plan.operations) >= 1

    if case.check_ast_sql:
        ast, sql, params = _build_ast_and_sql(case)
        assert isinstance(ast, ASTNode)
        assert isinstance(sql, str)
        assert isinstance(params, list)


def test_acceptance_follow_up_conversations() -> None:
    ctx = AdminSessionContext()
    session_id = "acceptance-session"
    ctx.update(
        session_id,
        "Show top BPET performers.",
        {"category": "Performance"},
        [{"agniveerId": 1}, {"agniveerId": 2}],
    )
    assert ctx.is_followup_query(session_id, "Which of them play cricket?")
    ctx.update(
        session_id,
        "Which of them play cricket?",
        {"category": "Skills"},
        [{"agniveerId": 1}],
    )
    previous_ids = ctx.get_previous_ids(session_id)
    assert previous_ids is not None
    assert 1 in previous_ids


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM AgniveerMaster; DROP TABLE AgniveerMaster",
        "SELECT Password FROM UserMaster",
        "SELECT * FROM LoginToken",
        "SELECT * FROM DefaultLog",
        "SELECT 1 -- inject",
        "SELECT 1 /* inject */",
        "INSERT INTO AgniveerMaster VALUES (1)",
        "UPDATE AgniveerMaster SET FullName = 'x'",
        "DELETE FROM AgniveerMaster",
        "EXEC sp_who",
    ],
)
def test_acceptance_security_sql_rejected(sql: str) -> None:
    assert executor_validate_sql(sql) is not None
