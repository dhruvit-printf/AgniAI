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

    def test_typo_two_substitutions_on_longer_name(self):
        """Names > 4 chars tolerate 2 edits (here: two substitutions)."""
        assert _name_matches("Charlie", "charlyy") is True

    def test_no_fuzzy_beyond_distance_budget(self):
        # 4-char words only tolerate 1 edit — "alfa" is 2 edits from "alpha"
        # (this specific case is instead caught upstream by query_normalizer's
        # FUZZY_VOCAB alias "alfa" -> "Alpha Unit", not by this function).
        assert _name_matches("Alpha", "alfa") is False

    def test_typo_extra_letter(self):
        assert _name_matches("Bravo", "bravoo") is True

    def test_typo_single_substitution(self):
        assert _name_matches("Charlie", "charlle") is True

    def test_no_match_for_unrelated_real_names(self):
        assert _name_matches("Bravo", "charlie") is False

    def test_no_fuzzy_match_for_short_names(self):
        # Both sides must be >= 4 chars — short names/IDs must never
        # fuzzy-collide with each other.
        assert _name_matches("Pl3", "pl4") is False

    def test_no_fuzzy_match_for_domain_keyword(self):
        # "compare" is a recognized application keyword, not a mistyped
        # company name, even though it's edit-distance 2 from "Company".
        assert _name_matches("Company", "compare") is False


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

            # 3. Both companies are named — a single global resolution can
            # only pick one, so it prefers the longer/more complete verified
            # match ("Alpha Company", the full real name) over the shorter
            # one ("Bravo"). Per-fragment resolution (used for comparison
            # queries) is what correctly resolves each side independently.
            r3 = resolve_entities_from_query("Compare Bravo and Alpha Company")
            assert r3["companyId"] == 10
            assert r3["companyName"] == "Alpha Company"

            # 4. Misspelled company name (typo tolerance) — no exact/prefix
            # hit exists for "Alfa", so the authoritative scan must fall
            # back to fuzzy matching against the real "Alpha Company".
            r4 = resolve_entities_from_query("schedule for alfa company")
            assert r4["companyId"] == 10

            # 5. "Compare" must never be mistaken for a typo of "Company" —
            # it's a recognized application keyword, not a name.
            r5 = resolve_entities_from_query("compare attendance this week")
            assert r5["companyId"] is None
