import unittest

from rag import answer_is_grounded


class TestRagGrounding(unittest.TestCase):
    def test_answer_is_grounded_number_boundaries(self):
        # Verify that substring numbers do not cause false positives in grounding check
        self.assertFalse(answer_is_grounded("The score is 5", "The score is 25"))
        self.assertTrue(answer_is_grounded("The score is 5", "The score is 5"))
        self.assertFalse(answer_is_grounded("The score is 5%", "The score is 25%"))
        self.assertTrue(answer_is_grounded("The score is 5%", "The score is 5%"))
