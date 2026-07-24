import unittest

from access_control import build_user_hierarchy, check_access, filter_user_data

FULL_DATASET = [
    {"id": 1, "name": "CO", "role": "Commanding_Officer", "company_id": None, "platoon_id": None, "parent_id": None},
    {"id": 2, "name": "Company 1 Commander", "role": "Company_Commander", "company_id": 1, "platoon_id": None, "parent_id": 1},
    {"id": 3, "name": "Platoon 1 Commander", "role": "Platoon_Commander", "company_id": 1, "platoon_id": 1, "parent_id": 2},
    {"id": 4, "name": "Agniveer X", "role": "Agniveer", "company_id": 1, "platoon_id": 1, "parent_id": 3},
    {"id": 5, "name": "Company 2 Commander", "role": "Company_Commander", "company_id": 2, "platoon_id": None, "parent_id": 1},
]


class TestFilterUserData(unittest.TestCase):
    def test_commanding_officer_sees_everything(self):
        co = FULL_DATASET[0]
        self.assertEqual(filter_user_data(co, FULL_DATASET), FULL_DATASET)

    def test_company_commander_excludes_other_companies_and_co(self):
        company_1_commander = FULL_DATASET[1]
        result = filter_user_data(company_1_commander, FULL_DATASET)
        result_ids = {record["id"] for record in result}
        self.assertNotIn(5, result_ids)  # Company 2 Commander
        self.assertNotIn(1, result_ids)  # Commanding Officer
        self.assertEqual(result_ids, {2, 3, 4})

    def test_platoon_commander_excludes_other_platoons(self):
        platoon_1_commander = FULL_DATASET[2]
        agniveer_other_platoon = {
            "id": 6, "name": "Agniveer X", "role": "Agniveer",
            "company_id": 1, "platoon_id": 2, "parent_id": 99,
        }
        dataset = FULL_DATASET + [agniveer_other_platoon]
        result = filter_user_data(platoon_1_commander, dataset)
        result_ids = {record["id"] for record in result}
        self.assertNotIn(6, result_ids)
        self.assertEqual(result_ids, {3, 4})

    def test_agniveer_sees_only_self(self):
        agniveer_x = FULL_DATASET[3]
        result = filter_user_data(agniveer_x, FULL_DATASET)
        self.assertEqual(result, [agniveer_x])


class TestCheckAccess(unittest.TestCase):
    def test_commanding_officer_always_allowed(self):
        co = FULL_DATASET[0]
        allowed, message = check_access(co, FULL_DATASET[4], FULL_DATASET)
        self.assertTrue(allowed)
        self.assertIsNone(message)

    def test_company_commander_denied_other_company_names_its_commander(self):
        company_1_commander = FULL_DATASET[1]
        company_2_commander = FULL_DATASET[4]
        allowed, message = check_access(company_1_commander, company_2_commander, FULL_DATASET)
        self.assertFalse(allowed)
        self.assertIn("not authorised", message)
        self.assertIn("Company 2 Commander", message)

    def test_company_commander_allowed_own_company(self):
        company_1_commander = FULL_DATASET[1]
        agniveer_x = FULL_DATASET[3]
        allowed, message = check_access(company_1_commander, agniveer_x, FULL_DATASET)
        self.assertTrue(allowed)
        self.assertIsNone(message)

    def test_platoon_commander_denied_other_platoon_names_its_commander(self):
        platoon_1_commander = FULL_DATASET[2]
        platoon_2_commander = {
            "id": 7, "name": "Platoon 2 Commander", "role": "Platoon_Commander",
            "company_id": 1, "platoon_id": 2, "parent_id": 2,
        }
        agniveer_other_platoon = {
            "id": 6, "name": "Agniveer Y", "role": "Agniveer",
            "company_id": 1, "platoon_id": 2, "parent_id": 7,
        }
        dataset = FULL_DATASET + [platoon_2_commander, agniveer_other_platoon]
        allowed, message = check_access(platoon_1_commander, agniveer_other_platoon, dataset)
        self.assertFalse(allowed)
        self.assertIn("Platoon 2 Commander", message)

    def test_agniveer_denied_other_agniveer_data(self):
        agniveer_x = FULL_DATASET[3]
        other_agniveer = {
            "id": 6, "name": "Agniveer Y", "role": "Agniveer",
            "company_id": 1, "platoon_id": 1, "parent_id": 3,
        }
        allowed, message = check_access(agniveer_x, other_agniveer, FULL_DATASET)
        self.assertFalse(allowed)
        self.assertIn("Platoon Commander", message)


class TestBuildUserHierarchy(unittest.TestCase):
    def test_commanding_officer_tree_includes_all_subordinates(self):
        co = FULL_DATASET[0]
        tree = build_user_hierarchy(co, FULL_DATASET)
        self.assertEqual(tree["id"], 1)
        company_ids = {child["id"] for child in tree["children"]}
        self.assertEqual(company_ids, {2, 5})

    def test_company_commander_tree_excludes_other_companies_and_co(self):
        company_1_commander = FULL_DATASET[1]
        tree = build_user_hierarchy(company_1_commander, FULL_DATASET)
        self.assertEqual(tree["id"], 2)
        self.assertEqual([child["id"] for child in tree["children"]], [3])
        self.assertEqual([grandchild["id"] for grandchild in tree["children"][0]["children"]], [4])

    def test_platoon_commander_tree_includes_only_own_agniveers(self):
        platoon_1_commander = FULL_DATASET[2]
        tree = build_user_hierarchy(platoon_1_commander, FULL_DATASET)
        self.assertEqual(tree["id"], 3)
        self.assertEqual([child["id"] for child in tree["children"]], [4])

    def test_agniveer_tree_has_empty_children(self):
        agniveer_x = FULL_DATASET[3]
        tree = build_user_hierarchy(agniveer_x, FULL_DATASET)
        self.assertEqual(tree["children"], [])


if __name__ == "__main__":
    unittest.main()
