"""
tests/test_system_messages.py
==============================
Unit tests for system_messages module verifying proper fallback messages.
"""

import unittest

from system_messages import (
    DATABASE_CONNECTION_ISSUES,
    DUPLICATE_INFORMATION,
    ENCOURAGING_AND_KIND_WORDS,
    FILE_ATTACHMENT_ISSUES,
    FINAL_KIND_CLOSING_MESSAGES,
    GENERAL_HELPFUL_SUGGESTIONS,
    INCORRECT_FORMAT,
    MISSING_INFORMATION,
    PERMISSION_ISSUES,
    QUERY_PROCESSING_ISSUES,
    SUGGESTING_NEXT_STEPS,
    WHEN_AGNIVEERS_NOT_FOUND,
    WHEN_BATCHES_NOT_FOUND,
    WHEN_CANNOT_FIND_WHAT_LOOKING_FOR,
    WHEN_COMMANDERS_NOT_FOUND,
    WHEN_COMPANIES_NOT_FOUND,
    WHEN_PLATOONS_NOT_FOUND,
    WHEN_SPECIFIC_RECORD_NOT_FOUND,
    WHEN_USERS_NOT_FOUND,
    get_database_error_message,
    get_entity_not_found_message,
    get_not_understood_message,
    get_specific_record_not_found_message,
)


class TestSystemMessages(unittest.TestCase):
    def test_query_processing_not_understood_messages(self):
        main_msg = get_not_understood_message(use_alternative=False)
        self.assertEqual(
            main_msg, "I didn't quite understand that - could you please rephrase?"
        )
        alt_msg = get_not_understood_message(use_alternative=True)
        self.assertIn(alt_msg, QUERY_PROCESSING_ISSUES["alternatives"])

    def test_database_connection_messages(self):
        main_msg = get_database_error_message(use_alternative=False)
        self.assertEqual(
            main_msg, "I'm having trouble reaching the database right now."
        )
        alt_msg = get_database_error_message(use_alternative=True)
        self.assertIn(alt_msg, DATABASE_CONNECTION_ISSUES["alternatives"])

    def test_entity_not_found_messages(self):
        self.assertEqual(
            get_entity_not_found_message("agniveer"),
            "I searched for Agniveers, but couldn't find any in our system.",
        )
        self.assertEqual(
            get_entity_not_found_message("commander"),
            "I looked for commanders but couldn't find any assigned yet.",
        )
        self.assertEqual(
            get_entity_not_found_message("company"),
            "I couldn't find any companies in the system.",
        )
        self.assertEqual(
            get_entity_not_found_message("platoon"),
            "I couldn't find any platoons in the system.",
        )
        self.assertEqual(
            get_entity_not_found_message("user"),
            "I couldn't find any users in the system.",
        )
        self.assertEqual(
            get_entity_not_found_message("batch"),
            "I couldn't find any batches in the system.",
        )
        self.assertEqual(
            get_entity_not_found_message("unknown_entity"),
            "I searched everywhere, but I couldn't find what you're looking for.",
        )

    def test_specific_record_not_found_messages(self):
        main_msg = get_specific_record_not_found_message(use_alternative=False)
        self.assertEqual(
            main_msg, "I searched for that specific record but couldn't find it."
        )
        alt_msg = get_specific_record_not_found_message(use_alternative=True)
        self.assertIn(alt_msg, WHEN_SPECIFIC_RECORD_NOT_FOUND["alternatives"])


if __name__ == "__main__":
    unittest.main()
