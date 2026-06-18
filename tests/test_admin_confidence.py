"""
tests/test_admin_confidence.py  
Tests for admin_confidence.py — unified confidence scoring.
"""
import pytest
from admin_confidence import (
    AdminConfidence,
    compute_confidence,
    normalise_intent_confidence,
)


class TestAdminConfidenceProperties:
    def test_is_high_above_threshold(self):
        c = AdminConfidence(score=0.80, label="high")
        assert c.is_high is True
        assert c.is_medium is False
        assert c.is_low is False

    def test_is_medium_in_range(self):
        c = AdminConfidence(score=0.60, label="medium")
        assert c.is_high is False
        assert c.is_medium is True
        assert c.is_low is False

    def test_is_low_below_threshold(self):
        c = AdminConfidence(score=0.30, label="low")
        assert c.is_high is False
        assert c.is_medium is False
        assert c.is_low is True

    def test_to_dict_has_score_and_label(self):
        c = AdminConfidence(score=0.75, label="high")
        d = c.to_dict()
        assert "score" in d
        assert "label" in d
        assert d["score"] == 0.75
        assert d["label"] == "high"

    def test_to_dict_score_is_rounded(self):
        c = AdminConfidence(score=0.756789, label="high")
        d = c.to_dict()
        assert d["score"] == round(0.756789, 3)

    def test_boundary_exactly_0_75_is_high(self):
        c = AdminConfidence(score=0.75, label="high")
        assert c.is_high is True

    def test_boundary_exactly_0_50_is_medium(self):
        c = AdminConfidence(score=0.50, label="medium")
        assert c.is_medium is True


class TestComputeConfidence:
    def test_high_intent_gives_high_confidence(self):
        intent = {"category": "Performance", 
                  "subcategory": "TopPerformers",
                  "confidence": "high"}
        conf = compute_confidence(intent_result=intent)
        assert conf.label == "high"
        assert conf.score >= 0.75

    def test_low_intent_gives_low_confidence(self):
        intent = {"category": None, "confidence": "low"}
        conf = compute_confidence(intent_result=intent)
        assert conf.label == "low"
        assert conf.score < 0.75

    def test_medium_intent_label_in_expected_range(self):
        intent = {"category": "Leave", "confidence": "medium"}
        conf = compute_confidence(intent_result=intent)
        assert conf.label in ("medium", "high")

    def test_plan_only_high(self):
        class Plan:
            confidence = 0.90
        conf = compute_confidence(plan=Plan())
        assert conf.label == "high"

    def test_plan_only_low(self):
        class Plan:
            confidence = 0.20
        conf = compute_confidence(plan=Plan())
        assert conf.label == "low"

    def test_combined_intent_and_plan(self):
        intent = {"category": "Medical", "confidence": "high"}
        class Plan:
            confidence = 0.85
        conf = compute_confidence(intent_result=intent, plan=Plan())
        assert conf.label == "high"
        assert conf.score > 0.75

    def test_no_inputs_returns_low(self):
        conf = compute_confidence()
        assert conf.label == "low"
        assert conf.score < 0.5

    def test_score_clamped_0_to_1(self):
        intent = {"category": "Performance", "confidence": "high"}
        class Plan:
            confidence = 1.0
        conf = compute_confidence(intent_result=intent, plan=Plan())
        assert 0.0 <= conf.score <= 1.0

    def test_category_and_subcategory_boost(self):
        """Having both category and subcategory adds +0.05 to score."""
        intent_with_sub = {
            "category": "Attendance", 
            "subcategory": "PresentToday",
            "confidence": "medium"
        }
        intent_without_sub = {
            "category": "Attendance",
            "confidence": "medium"
        }
        conf_with = compute_confidence(intent_result=intent_with_sub)
        conf_without = compute_confidence(intent_result=intent_without_sub)
        assert conf_with.score >= conf_without.score


class TestNormaliseIntentConfidence:
    def test_returns_dict(self):
        intent = {"category": "Leave", "confidence": "high"}
        result = normalise_intent_confidence(intent)
        assert isinstance(result, dict)

    def test_confidence_replaced_with_dict(self):
        intent = {"category": "Skills", "subcategory": "BySport",
                  "confidence": "medium"}
        result = normalise_intent_confidence(intent)
        assert isinstance(result["confidence"], dict)
        assert "score" in result["confidence"]
        assert "label" in result["confidence"]

    def test_original_fields_preserved(self):
        intent = {"category": "Equipment", "subcategory": "OverdueEquipment",
                  "confidence": "high", "section": "PPT"}
        result = normalise_intent_confidence(intent)
        assert result["category"] == "Equipment"
        assert result["subcategory"] == "OverdueEquipment"
        assert result["section"] == "PPT"

    def test_original_intent_not_mutated(self):
        intent = {"category": "Medical", "confidence": "low"}
        original_confidence = intent["confidence"]
        normalise_intent_confidence(intent)
        assert intent["confidence"] == original_confidence
