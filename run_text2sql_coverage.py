from __future__ import annotations

import builtins
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from intent_engine.admin_intent import classify_admin_intent
import sql_executor


QUESTION_GROUPS = [
    (
        "PERFORMANCE",
        [
            "Show the top 10 performers.",
            "Who are the top performers in BPET?",
            "Show the bottom 10 performers.",
            "Who scored the lowest in Firing?",
            "Show the most improved Agniveers.",
            "Who improved the most in BPET?",
            "Show the Agniveers whose performance dropped the most.",
            "Show the biggest performance drop in PPT.",
            "Show the BPET grades.",
            "What is the grade of Agniveer A0701882L in Firing?",
            "Show the BPET grading summary.",
            "How many Agniveers received an Excellent grade?",
            "Show the average BPET marks.",
            "What is the average Firing score?",
            "Show the attempt-wise performance of Agniveer A0701882L.",
            "Show the attempt-wise BPET marks.",
            "Show the best attempt of Agniveer A0701882L.",
            "Which is the best BPET attempt?",
            "Show the BPET performance trend.",
            "Show the performance trend of Agniveer A0701882L.",
        ],
    ),
    (
        "ATTENDANCE",
        [
            "Show today's attendance.",
            "Who is absent today?",
            "Show this week's attendance.",
            "Give me the weekly attendance report.",
            "Show this month's attendance.",
            "Give me the monthly attendance summary.",
            "Who is present today?",
            "Show all Agniveers currently on campus.",
            "Show the current strength.",
            "What is the total manpower strength?",
            "Show the attendance summary.",
            "Give me the overall attendance report.",
        ],
    ),
    (
        "LEAVE",
        [
            "Who has taken the most leave?",
            "Show the highest leave records.",
            "Who has taken the least leave?",
            "Show the lowest leave records.",
            "Who is currently on leave?",
            "Show all active leave records.",
            "Show the absconded Agniveers.",
            "Who is marked as absconded?",
        ],
    ),
    (
        "MEDICAL",
        [
            "Show the BMI report.",
            "What is the BMI of Agniveer A0701882L?",
            "Show the blood group of Agniveer A0701882L.",
            "Show the blood group report.",
            "Show all diagnosed Agniveers.",
            "Show the disease report.",
            "Show the medical report of Agniveer A0701882L.",
            "Display the complete health record of Agniveer A0701882L.",
        ],
    ),
    (
        "EQUIPMENT",
        [
            "Show the equipment summary.",
            "Give me the equipment statistics.",
            "Show equipment by category.",
            "List all rifle equipment.",
            "Show all returned equipment.",
            "Which equipment has been returned?",
            "Show the currently issued equipment.",
            "Which equipment is currently with Agniveers?",
            "Show the equipment of Agniveer A0701882L.",
            "List the issued items of Agniveer A0701882L.",
        ],
    ),
    (
        "VERIFICATION",
        [
            "Show pending police verification.",
            "Which Agniveers have pending verification?",
            "Show all sent verification records.",
            "Which police verifications have been sent?",
            "Show the verification cases with no response.",
            "Which verification requests have not received a response?",
            "Show all verified Agniveers.",
            "Show completed police verification records.",
            "Show rejected police verifications.",
            "Which Agniveers have rejected verification?",
        ],
    ),
    (
        "DISTRIBUTION",
        [
            "Show the latest distribution.",
            "Display the latest unit allocation.",
            "Show the distribution by unit.",
            "Give me the unit-wise distribution report.",
            "Show all unassigned Agniveers.",
            "Which Agniveers have not been assigned to any unit?",
            "Which unit has the highest number of Agniveers?",
            "Show the unit with the maximum strength.",
        ],
    ),
    (
        "PERSONAL DETAILS",
        [
            "Show the personal details of Agniveer A0701882L.",
            "Show the date of birth of Agniveer A0701882L.",
            "Show the height of Agniveer A0701882L.",
            "Show the weight of Agniveer A0701882L.",
            "Show the father's name of Agniveer A0701882L.",
            "Show the mother's name of Agniveer A0701882L.",
            "Show the Aadhaar number of Agniveer A0701882L.",
            "Show the mobile number of Agniveer A0701882L.",
            "Show the address of Agniveer A0701882L.",
            "Show the educational qualification of Agniveer A0701882L.",
        ],
    ),
]


@contextmanager
def _offline_execution() -> Any:
    original_print = builtins.print

    def _quiet_print(*_args: Any, **_kwargs: Any) -> None:
        return None

    def _fake_generate_sql(question: str, intent: Optional[Dict] = None):
        _fake_generate_sql.called = True
        _fake_generate_sql.last_question = question
        _fake_generate_sql.last_intent = intent or {}
        return (
            "SELECT TOP 1 a.AgniveerNo, a.FullName FROM AgniveerMaster a "
            "WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)",
            None,
        )

    _fake_generate_sql.called = False
    _fake_generate_sql.last_question = ""
    _fake_generate_sql.last_intent = {}

    def _fake_run_readonly(sql: str, params: Optional[List[Any]] = None):
        _fake_run_readonly.last_sql = sql
        _fake_run_readonly.last_params = params or []
        return [], None

    _fake_run_readonly.last_sql = ""
    _fake_run_readonly.last_params: List[Any] = []

    generate_sql_patch = patch.object(
        sql_executor, "generate_sql", side_effect=_fake_generate_sql
    )
    run_readonly_patch = patch.object(
        sql_executor, "run_readonly", side_effect=_fake_run_readonly
    )
    validate_sql_patch = patch.object(
        sql_executor.sql_validator, "validate_sql", return_value=(True, None)
    )
    validate_ast_patch = patch.object(
        sql_executor.sql_validator, "validate_ast", return_value=(True, None)
    )
    print_patch = patch("builtins.print", side_effect=_quiet_print)

    with (
        validate_sql_patch,
        validate_ast_patch,
        generate_sql_patch as generate_sql_mock,
        run_readonly_patch as run_readonly_mock,
        print_patch,
    ):
        yield {
            "generate_sql_mock": generate_sql_mock,
            "run_readonly_mock": run_readonly_mock,
            "original_print": original_print,
        }


def _flatten_cases() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for section, questions in QUESTION_GROUPS:
        for question in questions:
            rows.append({"section": section, "question": question})
    return rows


def _run_question(question: str, state: Dict[str, Any]) -> Dict[str, Any]:
    intent = classify_admin_intent(question)
    query_type = str(intent.get("query_type") or "unknown")
    operations = [intent]

    leg_results: List[Dict[str, Any]] = []
    any_text2sql = False

    for leg in operations:
        state["generate_sql_mock"].reset_mock()

        result, error = sql_executor.execute_sql_query(question=question, intent=leg)

        leg_results.append(
            {
                "category": (leg or {}).get("category"),
                "operation": (leg or {}).get("operation"),
                "query_type": (leg or {}).get("query_type"),
                "route": "text2sql"
                if state["generate_sql_mock"].call_count > 0
                else "deterministic",
                "error": error,
                "sql": (result or {}).get("sql") if isinstance(result, dict) else None,
            }
        )
        if state["generate_sql_mock"].call_count > 0:
            any_text2sql = True

    return {
        "query_type": query_type,
        "legs": leg_results,
        "any_text2sql": any_text2sql,
    }


def main() -> None:
    rows = []
    with _offline_execution() as state:
        for case in _flatten_cases():
            outcome = _run_question(case["question"], state)
            legs = outcome["legs"]
            fallback_legs = [leg for leg in legs if leg["route"] == "text2sql"]
            rows.append(
                {
                    "section": case["section"],
                    "question": case["question"],
                    "query_type": outcome["query_type"],
                    "route": "text2sql" if outcome["any_text2sql"] else "deterministic",
                    "fallback_legs": fallback_legs,
                    "legs": legs,
                }
            )

    report_path = Path("text2sql_coverage_report.md")
    text2sql_questions = [r for r in rows if r["route"] == "text2sql"]

    lines = [
        "# Text2SQL Coverage Report",
        "",
        "This report was generated offline by running the real planner and executor with the database and LLM calls stubbed out.",
        "It shows which questions still fall back to text2sql in the current codebase.",
        "",
        f"Total questions tested: {len(rows)}",
        f"Questions that fell back to text2sql: {len(text2sql_questions)}",
        "",
        "## Fallback Questions",
    ]

    if text2sql_questions:
        for row in text2sql_questions:
            lines.append(f"- [{row['section']}] {row['question']}")
    else:
        lines.append("- None")

    lines.extend(["", "## Full Matrix", "", "| Section | Question | Query Type | Route |", "| --- | --- | --- | --- |"])

    for row in rows:
        lines.append(
            f"| {row['section']} | {row['question']} | {row['query_type']} | {row['route']} |"
        )

    lines.extend(["", "## Per-Leg Details", ""])
    for row in rows:
        lines.append(f"### {row['section']}: {row['question']}")
        for idx, leg in enumerate(row["legs"], start=1):
            lines.append(
                f"- Leg {idx}: category={leg['category']} operation={leg['operation']} route={leg['route']}"
            )
            if leg["error"]:
                lines.append(f"  - error: {leg['error']}")
            if leg["sql"]:
                lines.append(f"  - sql: {leg['sql']}")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_path.resolve()}")
    print(f"Text2SQL fallbacks: {len(text2sql_questions)} / {len(rows)}")


if __name__ == "__main__":
    main()
