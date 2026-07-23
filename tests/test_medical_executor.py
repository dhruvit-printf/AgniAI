from __future__ import annotations

from unittest.mock import patch
import pytest

from sql_executor import (
    execute_sql_query,
    execute_medical_query,
    _build_medical_base_scope,
)


def test_medical_base_scope_builder():
    intent = {"agniveer_no": "A0701882L", "company_id": 1, "class_": "Gorkha"}
    where_sql, params = _build_medical_base_scope(intent)
    assert "(a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)" in where_sql
    assert "a.IsActive = 1" in where_sql
    assert "LOWER(a.AgniveerNo) LIKE" in where_sql
    assert (
        "EXISTS (SELECT 1 FROM PlatoonMaster p WHERE p.Id = a.PlatoonId AND p.CompanyId = ?)"
        in where_sql
    )
    assert "LOWER(a.Class) = LOWER(?)" in where_sql
    assert params == ["A0701882L", 1, "Gorkha"]


def test_execute_medical_bmi_summary():
    def mock_run(sql, params=(), max_rows=None):
        assert "BmiCategory" in sql
        assert "LatestMedical" in sql
        return (
            [
                {"BmiCategory": "Normal", "AgniveerCount": 120},
                {"BmiCategory": "Overweight", "AgniveerCount": 15},
            ],
            None,
        )

    intent = {"category": "Medical", "operation": "BMI", "responseType": "Summary"}

    with patch("sql_executor.run_readonly", side_effect=mock_run):
        res, err = execute_sql_query(intent=intent)

    assert err is None
    assert res["success"] is True
    assert len(res["records"]) == 2


def test_execute_medical_blood_group():
    def mock_run(sql, params=(), max_rows=None):
        assert "BloodGroup" in sql
        return (
            [
                {"BloodGroup": "O+", "AgniveerCount": 45},
                {"BloodGroup": "B+", "AgniveerCount": 30},
            ],
            None,
        )

    intent = {
        "category": "Medical",
        "operation": "BloodGroup",
        "responseType": "Summary",
    }

    with patch("sql_executor.run_readonly", side_effect=mock_run):
        res, err = execute_sql_query(intent=intent)

    assert err is None
    assert res["success"] is True
    assert res["records"][0]["bloodGroup"] == "O+"


def test_execute_medical_disease_specific():
    def mock_run(sql, params=(), max_rows=None):
        assert "LIKE '%' + LOWER(?) + '%'" in sql
        return (
            [
                {
                    "AgniveerNo": "A0701882L",
                    "FullName": "HARMAN SINGH",
                    "Diagnosis": "Malaria",
                }
            ],
            None,
        )

    intent = {
        "category": "Medical",
        "operation": "Disease",
        "diagnose": "Malaria",
        "responseType": "Detailed",
    }

    with patch("sql_executor.run_readonly", side_effect=mock_run):
        res, err = execute_sql_query(intent=intent)

    assert err is None
    assert res["success"] is True
    assert res["records"][0]["diagnosis"] == "Malaria"


def test_execute_medical_individual():
    def mock_run(sql, params=(), max_rows=None):
        assert "INNER JOIN MedicalRecordMaster mr" in sql
        return (
            [
                {
                    "AgniveerNo": "A0701882L",
                    "FullName": "HARMAN SINGH",
                    "Diagnosis": "Fever",
                    "DoctorName": "Dr. Sharma",
                }
            ],
            None,
        )

    intent = {
        "category": "Medical",
        "operation": "Individual",
        "agniveer_no": "A0701882L",
    }

    with patch("sql_executor.run_readonly", side_effect=mock_run):
        res, err = execute_sql_query(intent=intent)

    assert err is None
    assert res["success"] is True
    assert res["records"][0]["doctorName"] == "Dr. Sharma"


def test_execute_medical_followup():
    def mock_run(sql, params=(), max_rows=None):
        assert "mr.FollowUpDate IS NOT NULL" in sql
        return ([{"AgniveerNo": "A0701882L", "FollowUpDate": "2026-08-01"}], None)

    intent = {"category": "Medical", "operation": "FollowUp"}

    with patch("sql_executor.run_readonly", side_effect=mock_run):
        res, err = execute_sql_query(intent=intent)

    assert err is None
    assert res["success"] is True
    assert res["records"][0]["followUpDate"] == "2026-08-01"


def test_execute_medical_hospital_stats():
    def mock_run(sql, params=(), max_rows=None):
        assert "mr.HospitalNameLocation" in sql
        return (
            [{"HospitalNameLocation": "Base Hospital Delhi", "AgniveerCount": 25}],
            None,
        )

    intent = {"category": "Medical", "operation": "HospitalStats"}

    with patch("sql_executor.run_readonly", side_effect=mock_run):
        res, err = execute_sql_query(intent=intent)

    assert err is None
    assert res["success"] is True
    assert res["records"][0]["agniveerCount"] == 25
