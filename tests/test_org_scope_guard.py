"""
tests/test_org_scope_guard.py
==============================
Covers the company/platoon access-boundary short-circuit in
execute_admin_query(): a Company Commander (frontend-scoped via
id_filters["companyId"]) or Platoon Commander (id_filters["platoonId"])
whose query text names a different company/platoon must be denied outright,
never silently answered against their own scope or the named one.

Resolving a company/platoon mention to an ID normally requires the live
.NET company/platoon directory (admin_entity_resolver._fetch_companies/
_fetch_platoons) — unlike batch, which resolves locally. Every test here
patches those two so resolution is deterministic and doesn't depend on that
service being reachable.

Queries below name the target company/platoon by NAME ("Bravo", "PL-02"),
not by bare number ("company 6") — a bare-digit mention hits an unrelated,
pre-existing collision in resolve_entities_from_query's directory-scan
tie-break (every "Company N" name partially matches the generic word
"company", so ties get silently broken by list order instead of the digit).
"""

from __future__ import annotations

import os
import sys
import types
import unittest
from unittest.mock import patch

_STUB_MODS = [
    "flask",
    "flask_cors",
    "flask_limiter",
    "flask_limiter.util",
    "dotenv",
    "requests",
]
for mod in _STUB_MODS:
    try:
        __import__(mod)
    except ImportError:
        if mod not in sys.modules:
            stub = types.ModuleType(mod)
            if mod == "dotenv":
                stub.load_dotenv = lambda *a, **kw: None  # type: ignore[attr-defined]
            sys.modules[mod] = stub

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from admin_pipeline import execute_admin_query

_COMPANIES = [
    {"companyId": 5, "companyName": "Alpha"},
    {"companyId": 6, "companyName": "Bravo"},
]
_PLATOONS = [
    {"platoonId": 1, "platoonName": "PL-01", "companyId": 5},
    {"platoonId": 2, "platoonName": "PL-02", "companyId": 5},
]


def _patch_directory():
    return patch.multiple(
        "admin_entity_resolver",
        _fetch_companies=lambda **kw: _COMPANIES,
        _fetch_platoons=lambda **kw: _PLATOONS,
    )


class TestOrgScopeGuard(unittest.TestCase):

    @patch("sql_executor.get_company_commander_contact")
    @patch("admin_pipeline.fetch_sql_results")
    @patch("admin_pipeline.generate_report")
    def test_company_commander_denied_other_company_names_its_commander(
        self, mock_generate_report, mock_fetch_sql, mock_get_contact
    ):
        mock_get_contact.return_value = {
            "CommanderId": 9,
            "CommanderName": "Bravo Commander",
            "CompanyId": 6,
            "CompanyName": "Bravo",
        }

        with _patch_directory():
            result = execute_admin_query(
                "Give me top 10 performers from Bravo company",
                {"companyId": 5, "session_id": "test-org-guard-1"},
            )

        self.assertEqual(result["type"], "clarification")
        message = result["combined_message"]
        self.assertIn("not authorised", message)
        self.assertIn("Bravo Commander", message)
        mock_fetch_sql.assert_not_called()
        mock_generate_report.assert_not_called()

    @patch("sql_executor.get_commanding_officer_contact", return_value=None)
    @patch("sql_executor.get_company_commander_contact", return_value=None)
    @patch("admin_pipeline.fetch_sql_results")
    @patch("admin_pipeline.generate_report")
    def test_company_commander_denied_falls_back_without_any_contact(
        self, mock_generate_report, mock_fetch_sql, mock_get_contact, mock_get_co
    ):
        with _patch_directory():
            result = execute_admin_query(
                "Give me top 10 performers from Bravo company",
                {"companyId": 5, "session_id": "test-org-guard-2"},
            )

        self.assertEqual(result["type"], "clarification")
        message = result["combined_message"]
        self.assertIn("not authorised", message)
        self.assertIn("Company Commander of that company", message)
        mock_fetch_sql.assert_not_called()

    @patch("sql_executor.get_commanding_officer_contact")
    @patch("sql_executor.get_company_commander_contact", return_value=None)
    @patch("admin_pipeline.fetch_sql_results")
    @patch("admin_pipeline.generate_report")
    def test_company_commander_denied_escalates_to_commanding_officer(
        self, mock_generate_report, mock_fetch_sql, mock_get_contact, mock_get_co
    ):
        mock_get_co.return_value = {
            "OfficerId": 21,
            "OfficerName": "Commandant Rao",
            "CompanyId": 6,
            "CompanyName": "Bravo",
        }

        with _patch_directory():
            result = execute_admin_query(
                "Give me top 10 performers from Bravo company",
                {"companyId": 5, "session_id": "test-org-guard-2b"},
            )

        self.assertEqual(result["type"], "clarification")
        message = result["combined_message"]
        self.assertIn("not authorised", message)
        self.assertIn("Commandant Rao, the Commanding Officer of Bravo", message)
        mock_fetch_sql.assert_not_called()

    @patch("sql_executor.get_platoon_commander_contact")
    @patch("admin_pipeline.fetch_sql_results")
    @patch("admin_pipeline.generate_report")
    def test_platoon_commander_denied_other_platoon_names_its_commander(
        self, mock_generate_report, mock_fetch_sql, mock_get_contact
    ):
        mock_get_contact.return_value = {
            "CommanderId": 11,
            "CommanderName": "PL-02 Commander",
            "PlatoonId": 2,
            "PlatoonName": "PL-02",
            "CompanyId": 5,
            "CompanyName": "Alpha",
        }

        with _patch_directory():
            result = execute_admin_query(
                "Give me top 10 performers from PL-02",
                {"platoonId": 1, "session_id": "test-org-guard-3"},
            )

        self.assertEqual(result["type"], "clarification")
        message = result["combined_message"]
        self.assertIn("not authorised", message)
        self.assertIn("PL-02 Commander", message)
        mock_fetch_sql.assert_not_called()
        mock_generate_report.assert_not_called()

    @patch("admin_pipeline.fetch_sql_results")
    @patch("admin_pipeline.generate_report")
    def test_no_frontend_company_scope_does_not_short_circuit(
        self, mock_generate_report, mock_fetch_sql
    ):
        section = {"success": True, "records": [], "data": [], "count": 0}
        mock_fetch_sql.return_value = ([section], [("Result", section)], None)
        mock_generate_report.return_value = {
            "message": "Report generated.",
            "analysis": {"summary": "s", "observations": [], "insights": []},
            "conclusion": {"summary": "c"},
        }

        with _patch_directory():
            result = execute_admin_query(
                "Give me top 10 performers from Bravo company",
                {"session_id": "test-org-guard-4"},
            )

        self.assertNotEqual(result["type"], "clarification")
        mock_fetch_sql.assert_called_once()

    @patch("admin_pipeline.fetch_sql_results")
    @patch("admin_pipeline.generate_report")
    def test_company_commander_query_naming_own_company_proceeds_normally(
        self, mock_generate_report, mock_fetch_sql
    ):
        section = {"success": True, "records": [], "data": [], "count": 0}
        mock_fetch_sql.return_value = ([section], [("Result", section)], None)
        mock_generate_report.return_value = {
            "message": "Report generated.",
            "analysis": {"summary": "s", "observations": [], "insights": []},
            "conclusion": {"summary": "c"},
        }

        with _patch_directory():
            result = execute_admin_query(
                "Give me top 10 performers from Alpha company",
                {"companyId": 5, "session_id": "test-org-guard-5"},
            )

        self.assertNotEqual(result["type"], "clarification")
        mock_fetch_sql.assert_called_once()


if __name__ == "__main__":
    unittest.main()
