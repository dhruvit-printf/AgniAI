"""
tests/test_admin_entity_resolver.py
Tests for entity extraction (no .NET API calls — extraction only).
"""

import pytest

from admin_entity_resolver import (
    _name_matches,
    _normalise_name,
    extract_company_mention,
    extract_platoon_mention,
    resolve_entities_from_query,
)


class TestExtractCompanyMention:
    def test_alpha_company(self):
        assert extract_company_mention("show alpha company stats") == "alpha"

    def test_company_alpha(self):
        assert extract_company_mention("company alpha attendance") == "alpha"

    def test_bravo_coy(self):
        assert extract_company_mention("bravo coy performance") == "bravo"

    def test_coy_charlie(self):
        result = extract_company_mention("coy charlie leave")
        assert result == "charlie"

    def test_no_company_mention(self):
        result = extract_company_mention("show top performers")
        assert result is None

    def test_multiword_company(self):
        result = extract_company_mention("14 punjab company attendance")
        assert result is not None
        assert "punjab" in result.lower()

    def test_empty_string(self):
        assert extract_company_mention("") is None

    def test_single_letter_company(self):
        assert extract_company_mention("show A company performance") == "a"
        assert extract_company_mention("show B coy stats") == "b"


class TestExtractPlatoonMention:
    def test_platoon_3(self):
        assert extract_platoon_mention("platoon 3 attendance") == "3"

    def test_platoon_no_5(self):
        assert extract_platoon_mention("platoon no. 5 details") == "5"

    def test_pl_dash_01(self):
        assert extract_platoon_mention("PL-01 performance") == "01"

    def test_pl_space_2(self):
        assert extract_platoon_mention("pl 2 leave records") == "2"

    def test_3_platoon(self):
        assert extract_platoon_mention("3 platoon") == "3"

    def test_no_platoon_mention(self):
        result = extract_platoon_mention("show top performers")
        assert result is None

    def test_empty_string(self):
        assert extract_platoon_mention("") is None


class TestNameMatches:
    def test_exact_match(self):
        assert _name_matches("Alpha", "alpha") is True

    def test_normalised_match_with_spaces(self):
        assert _name_matches("Alpha Company", "alphacompany") is True

    def test_substring_match(self):
        assert _name_matches("PL-01", "01") is True

    def test_no_match(self):
        assert _name_matches("Bravo", "charlie") is False

    def test_case_insensitive(self):
        assert _name_matches("ALPHA", "alpha") is True


class TestResolveEntitiesFromQuery:
    def test_returns_dict_with_required_keys(self):
        result = resolve_entities_from_query("show attendance")
        for key in ("companyId", "platoonId", "companyName", "platoonName"):
            assert key in result

    def test_no_mention_returns_none_ids(self):
        result = resolve_entities_from_query("show top performers")
        assert result["companyId"] is None
        assert result["platoonId"] is None

    def test_existing_ids_not_overwritten(self):
        result = resolve_entities_from_query(
            "show attendance",
            existing_company_id=42,
            existing_platoon_id=7,
        )
        assert result["companyId"] == 42
        assert result["platoonId"] == 7

    def test_company_name_extracted(self):
        result = resolve_entities_from_query("alpha company performance")
        assert result["companyName"] == "alpha"

    def test_platoon_name_extracted(self):
        result = resolve_entities_from_query("platoon 3 leave status")
        assert result["platoonName"] == "3"

    def test_fallback_lookup_from_api(self):
        from unittest.mock import patch

        with (
            patch("admin_entity_resolver._fetch_companies") as mock_fetch_companies,
            patch("admin_entity_resolver._fetch_platoons") as mock_fetch_platoons,
        ):

            mock_fetch_companies.return_value = [
                {"companyId": 10, "companyName": "Alpha Company"},
                {"companyId": 20, "companyName": "Bravo"},
            ]
            mock_fetch_platoons.return_value = [
                {"platoonId": 101, "platoonName": "PL-01", "companyId": 10},
                {"platoonId": 102, "platoonName": "PL-02", "companyId": 10},
                {"platoonId": 201, "platoonName": "Vanguard Platoon", "companyId": 20},
            ]

            # 1. Matches "PL-02" directly anywhere in the query
            r1 = resolve_entities_from_query("Show stats for PL-02")
            assert r1["platoonId"] == 102
            assert r1["platoonName"] == "PL-02"

            # 2. Matches company name "Alpha" directly without the "company" keyword
            r2 = resolve_entities_from_query("Is Alpha performing well?")
            assert r2["companyId"] == 10
            assert r2["companyName"] == "Alpha Company"

            # 3. Matches company "Bravo" directly
            r3 = resolve_entities_from_query("Compare Bravo and Alpha Company")
            assert r3["companyId"] == 20
            assert r3["companyName"] == "Bravo"
