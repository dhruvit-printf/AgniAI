from __future__ import annotations

from unittest.mock import patch
import pytest

from intent_engine.personal_details_parser import parse_personal_details
from sql_executor import execute_sql_query


def test_single_field_personal_detail_query():
    query = "Show height of Agniveer A0701882L"
    intent = parse_personal_details(query)
    assert intent is not None
    assert intent["category"] == "personaldetail"
    assert intent["operation"] == "lookup"
    assert intent.get("agniveer_no") == "A0701882L"
    assert intent.get("metrics") == ["Height"]

    intent["raw_query"] = query
    with patch("sql_executor.run_readonly") as mock_run:
        mock_run.return_value = (
            [{"agniveerNo": "A0701882L", "fullName": "HARMAN SINGH", "height": 175}],
            None,
        )
        res, err = execute_sql_query(intent=intent)

    assert err is None
    assert res["success"] is True
    sql_executed = mock_run.call_args[0][0]
    assert "m.Height" in sql_executed
    assert "m.Address" not in sql_executed
    assert "m.Qualification" not in sql_executed


def test_multi_field_personal_detail_query():
    query = "Give me dob and qualification of Agniveer A0701882L"
    intent = parse_personal_details(query)
    assert intent is not None
    assert intent["category"] == "personaldetail"
    assert intent["operation"] == "lookup"
    assert intent.get("agniveer_no") == "A0701882L"
    assert "DateOfBirth" in intent.get("metrics", [])
    assert "Qualification" in intent.get("metrics", [])

    intent["raw_query"] = query
    with patch("sql_executor.run_readonly") as mock_run:
        mock_run.return_value = (
            [
                {
                    "agniveerNo": "A0701882L",
                    "fullName": "HARMAN SINGH",
                    "dateOfBirth": "2001-05-14",
                    "qualification": "12th",
                }
            ],
            None,
        )
        res, err = execute_sql_query(intent=intent)

    assert err is None
    assert res["success"] is True
    sql_executed = mock_run.call_args[0][0]
    assert "m.DateOfBirth" in sql_executed
    assert "m.Qualification" in sql_executed
    assert "m.Address" not in sql_executed
    assert "m.MobileNo" not in sql_executed


def test_full_personal_details_query():
    query = "Show personal details of Agniveer A0701882L"
    intent = parse_personal_details(query)
    assert intent is not None
    assert intent["category"] == "personaldetail"
    assert intent["operation"] == "lookup"
    assert intent.get("agniveer_no") == "A0701882L"
    # General personal detail request has no specific metric filter
    assert not intent.get("metrics")

    intent["raw_query"] = query
    with patch("sql_executor.run_readonly") as mock_run:
        mock_run.return_value = ([{}], None)
        res, err = execute_sql_query(intent=intent)

    assert err is None
    assert res["success"] is True
    sql_executed = mock_run.call_args[0][0]
    assert "m.Address" in sql_executed
    assert "m.Qualification" in sql_executed
    assert "m.Height" in sql_executed
    assert "m.MobileNo" in sql_executed


def test_contact_and_email_aliases():
    query = "What is the contact number and email of Agniveer A0701882L"
    intent = parse_personal_details(query)
    assert intent is not None
    assert intent.get("agniveer_no") == "A0701882L"
    assert "MobileNo" in intent.get("metrics", [])
    assert "Email" in intent.get("metrics", [])

    intent["raw_query"] = query
    with patch("sql_executor.run_readonly") as mock_run:
        mock_run.return_value = ([{}], None)
        res, err = execute_sql_query(intent=intent)

    assert err is None
    assert res["success"] is True
    sql_executed = mock_run.call_args[0][0]
    assert "m.MobileNo" in sql_executed
    assert "m.Email" in sql_executed
    assert "m.Address" not in sql_executed


def test_volleyball_playing_query_with_company_scope():
    for query in (
        "List agniveers who plays volleyball in Jas - Jaswant company",
        "List agniveers playing volleyball in Jas - Jaswant company",
    ):
        intent = parse_personal_details(query)
        assert intent is not None
        assert intent.get("sport") == "Volleyball"
        # CompanyMaster stores this row as "Jas - Jaswant", not bare
        # "Jaswant" — the raw "jaswant" token extracted from the query text
        # is mapped to the canonical stored name so downstream SQL's
        # LOWER(c.Name) = LOWER(?) match actually finds the row.
        assert intent.get("company_name") == "Jas - Jaswant"

        intent["raw_query"] = query
        with patch("sql_executor.run_readonly") as mock_run:
            mock_run.return_value = (
                [{"AgniveerNo": "A0701882L", "FullName": "HARMAN SINGH", "Sports": "Volleyball"}],
                None,
            )
            res, err = execute_sql_query(intent=intent)

        assert err is None
        assert res["success"] is True
        sql_executed = mock_run.call_args[0][0]
        params = mock_run.call_args[0][1]
        assert "LOWER(m.Sports) LIKE '%' + LOWER(?) + '%'" in sql_executed
        assert "LOWER(c.Name) LIKE '%' + LOWER(?) + '%'" in sql_executed
        assert "Jas - Jaswant" in params
        assert "Volleyball" in params

