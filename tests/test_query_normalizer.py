"""
tests/test_query_normalizer.py
===============================
Unit tests for the query normalization preprocessing layer
(query_normalizer.py) — duplicate-char collapsing, missing-space splitting,
generic fuzzy typo correction, and punctuation cleanup.
"""

import unittest

from query_normalizer import clean_query, normalize_query_detailed


class TestDuplicateCharCollapsing(unittest.TestCase):
    def test_collapses_triple_repeats(self):
        self.assertIn("firing", clean_query("ffffiring result"))
        self.assertIn("attendance", clean_query("attttendance report"))

    def test_preserves_legitimate_double_letters(self):
        # "attendance" has a real double 't' — must not be destroyed.
        self.assertEqual(clean_query("attendance report"), "attendance report")


class TestConcatenatedWordSplitting(unittest.TestCase):
    def test_splits_two_domain_words(self):
        self.assertEqual(clean_query("showattendance"), "show attendance")
        self.assertEqual(clean_query("medicalreport"), "medical report")
        self.assertEqual(clean_query("bpetresult"), "bpet result")

    def test_leaves_short_or_unknown_tokens_alone(self):
        # Below the length floor / no valid full segmentation exists.
        self.assertEqual(clean_query("bpet"), "bpet")


class TestFuzzyTypoCorrection(unittest.TestCase):
    def test_corrects_typo_not_in_curated_map(self):
        self.assertEqual(clean_query("shwo attendance"), "show attendance")

    def test_corrects_curated_map_entries_unchanged(self):
        self.assertEqual(clean_query("attandnce"), "attendance")
        self.assertEqual(clean_query("firng result"), "firing result")
        self.assertEqual(clean_query("medcal report"), "Medical report")

    def test_does_not_guess_on_ambiguous_short_words(self):
        # "PT" is genuinely ambiguous (Physical Training vs Platoon) —
        # too short to run through fuzzy correction at all, so it must
        # pass through completely untouched rather than being guessed at.
        self.assertEqual(clean_query("PT"), "PT")

    def test_leaves_unrelated_words_unchanged(self):
        # A word that isn't close to any domain vocabulary term shouldn't
        # be forced into a correction.
        self.assertIn("disqualify", clean_query("give me disqualify performers"))

    def test_protects_common_words_from_false_correction(self):
        # Regression: "food" is one substitution away from "foot" (a BPET
        # subsection term), which previously caused "who is suffering from
        # food poisoning?" to be misclassified entirely.
        self.assertEqual(
            clean_query("Who is suffering from food poisoning?"),
            "Who is suffering from food poisoning?",
        )


class TestPunctuationAndWhitespaceCleanup(unittest.TestCase):
    def test_collapses_repeated_punctuation(self):
        self.assertEqual(clean_query("attendance???"), "attendance?")
        self.assertEqual(clean_query("BPET!!!!"), "bpet!")
        self.assertEqual(clean_query("firing....."), "firing.")

    def test_collapses_extra_whitespace(self):
        self.assertEqual(clean_query("need   attendance"), "need attendance")

    def test_strips_invisible_unicode(self):
        self.assertEqual(clean_query("attendance​ report"), "attendance report")


class TestNormalizeQueryDetailed(unittest.TestCase):
    def test_confidence_is_lower_when_corrections_applied(self):
        clean = normalize_query_detailed("show attendance")
        typo = normalize_query_detailed("shwo attandnce")
        self.assertEqual(clean.confidence, 1.0)
        self.assertLess(typo.confidence, clean.confidence)

    def test_corrections_trace_is_populated(self):
        result = normalize_query_detailed("shwo attandnce")
        self.assertTrue(result.corrections)

    def test_empty_query_is_safe(self):
        result = normalize_query_detailed("")
        self.assertEqual(result.normalized, "")
        self.assertEqual(result.confidence, 1.0)

    def test_original_is_preserved(self):
        result = normalize_query_detailed("shwo attandnce")
        self.assertEqual(result.original, "shwo attandnce")


if __name__ == "__main__":
    unittest.main()
