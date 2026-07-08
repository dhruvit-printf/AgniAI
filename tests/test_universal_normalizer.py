import pytest
from universal_normalizer import normalize_response


def test_flat_array_extraction():
    data = [
        {"agniveerNo": "A001", "fullName": "Alice", "score": 90},
        {"agniveerNo": "A002", "fullName": "Bob", "score": 85},
    ]
    records = normalize_response(data)
    assert len(records) == 2
    assert records[0]["agniveerNo"] == "A001"
    assert records[1]["agniveerNo"] == "A002"


def test_nested_object_inheritance():
    data = {
        "equipmentName": "Rifle Sling",
        "equipmentCategory": "Rifle",
        "count": 187,
        "agniveers": [
            {"agniveerNo": "A001", "fullName": "Alice"},
            {"agniveerNo": "A002", "fullName": "Bob"},
        ],
    }
    records = normalize_response(data)
    assert len(records) == 2
    
    # Check inheritance
    assert records[0]["equipmentName"] == "Rifle Sling"
    assert records[0]["equipmentCategory"] == "Rifle"
    
    # Check that aggregate field 'count' was NOT inherited
    assert "count" not in records[0]


def test_deeply_nested_tree():
    data = {
        "data": {
            "companies": [
                {
                    "companyName": "Alpha",
                    "platoons": [
                        {
                            "platoonName": "Platoon 1",
                            "agniveers": [
                                {"agniveerNo": "A001", "fullName": "Alice"}
                            ],
                        }
                    ],
                }
            ]
        }
    }
    records = normalize_response(data)
    assert len(records) == 1
    assert records[0]["agniveerNo"] == "A001"
    assert records[0]["companyName"] == "Alpha"
    assert records[0]["platoonName"] == "Platoon 1"


def test_duplicate_resolution():
    data = [
        {"agniveerNo": "A001", "fullName": "Alice", "status": "Passed"},
        {"agniveerNo": "A001", "bmi": 24.5}, # Missing fullName, but has bmi
    ]
    records = normalize_response(data)
    assert len(records) == 1
    
    # First record's fields are preserved
    assert records[0]["fullName"] == "Alice"
    assert records[0]["status"] == "Passed"
    
    # Second record's unique fields are merged
    assert records[0]["bmi"] == 24.5


def test_child_wins_over_parent():
    data = {
        "status": "Failed",
        "agniveers": [
            {"agniveerNo": "A001", "status": "Passed"}
        ]
    }
    records = normalize_response(data)
    assert len(records) == 1
    # Child's status should override the inherited parent status
    assert records[0]["status"] == "Passed"


def test_metadata_stamping():
    data = [{"agniveerNo": "A001"}]
    records = normalize_response(data, base_metadata={"category": "Equipment", "operation": "Holding"})
    assert len(records) == 1
    assert records[0]["__category"] == "Equipment"
    assert records[0]["__operation"] == "Holding"


def test_ignore_empty_or_no_id_objects():
    data = {
        "summary": "This is a summary",
        "some_list": [
            {"name": "Not an agniveer"}, # No canonical ID
            {"agniveerNo": "A001", "fullName": "Alice"}
        ]
    }
    records = normalize_response(data)
    assert len(records) == 1
    assert records[0]["agniveerNo"] == "A001"
    # summary keyword should be ignored because it's an aggregate keyword
    assert "summary" not in records[0]

