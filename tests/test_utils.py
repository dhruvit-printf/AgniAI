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
