import unittest

from utils import extract_records


class TestUtils(unittest.TestCase):
    def test_extract_records_ignores_empty_lists_when_alternative_exists(self):
        # Verify that an empty list under 'data' does not block extraction of records from 'records'
        data = {"data": [], "records": [{"agniveerNo": "1"}]}
        res = extract_records(data)
        self.assertEqual(res, [{"agniveerNo": "1"}])

    def test_extract_records_returns_empty_when_all_are_empty(self):
        # Verify that when all candidate lists are empty, it returns an empty list
        data = {"data": [], "records": []}
        res = extract_records(data)
        self.assertEqual(res, [])

    def test_extract_records_merges_single_entity_sections(self):
        # Medical/Individual-style shape: one ID-bearing section ("profile")
        # plus scalar-only sibling sections must all merge into one row.
        data = {
            "data": {
                "profile": {"agniveerNo": "A001", "fullName": "Alice"},
                "bmi": {"bmiValue": 24.5},
                "stats": {"height": 170},
                "latestVitals": {"bp": "120/80"},
                "medicalHistory": [],
            }
        }
        res = extract_records(data)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["agniveerNo"], "A001")
        self.assertEqual(res[0]["bmiValue"], 24.5)
        self.assertEqual(res[0]["height"], 170)
        self.assertEqual(res[0]["bp"], "120/80")

    def test_extract_records_summarizes_unrelated_sibling_list(self):
        # A non-empty sibling list of a different cardinality (one row per
        # visit, not per person) must not steal the whole extraction and
        # discard the entity's own sections.
        data = {
            "data": {
                "profile": {"agniveerNo": "A001", "fullName": "Alice"},
                "bmi": {"bmiValue": 24.5},
                "medicalHistory": [{"date": "2025-01-01", "diagnosis": "Flu"}],
            }
        }
        res = extract_records(data)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["agniveerNo"], "A001")
        self.assertEqual(res[0]["bmiValue"], 24.5)
        self.assertEqual(res[0]["medicalHistoryCount"], 1)

    def test_extract_records_schedule_by_company_unaffected(self):
        # A genuine multi-row list with no ID-bearing sibling dict must
        # still be returned as-is.
        data = {
            "byCompany": [
                {"companyName": "Alpha", "count": 5},
                {"companyName": "Bravo", "count": 3},
            ]
        }
        res = extract_records(data)
        self.assertEqual(res, data["byCompany"])

    def test_extract_records_equipment_stats_unaffected(self):
        # Scalar-only sibling groups with no ID field anywhere must still
        # collapse to a single aggregate row, not be merged/altered.
        data = {
            "issued": {"issuedTotal": 254},
            "procured": {"procuredTotal": 250},
            "totalAssigned": 505,
        }
        res = extract_records(data)
        self.assertEqual(res, [data])
