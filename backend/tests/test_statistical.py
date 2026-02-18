"""
Tests for StatisticalService.

Covers bootstrap CIs, McNemar's test, and edge cases.
"""

import pytest
from app.services.statistical_service import StatisticalService


class TestBootstrapCI:
    def test_basic_ci(self):
        values = [0.8, 0.7, 0.9, 0.85, 0.75, 0.82, 0.88, 0.91, 0.78, 0.86]
        result = StatisticalService.bootstrap_confidence_interval(values)
        
        assert "mean" in result
        assert "lower" in result
        assert "upper" in result
        assert "std" in result
        
        # CI should contain the mean
        assert result["lower"] <= result["mean"] <= result["upper"]
        
        # CI should be narrower than [0, 1]
        assert result["upper"] - result["lower"] < 1.0
    
    def test_all_same_values(self):
        values = [0.5] * 20
        result = StatisticalService.bootstrap_confidence_interval(values)
        
        assert result["mean"] == 0.5
        assert result["lower"] == 0.5
        assert result["upper"] == 0.5
        assert result["std"] == 0.0
    
    def test_empty_values(self):
        result = StatisticalService.bootstrap_confidence_interval([])
        assert result["mean"] == 0.0
        assert result["lower"] == 0.0
    
    def test_single_value(self):
        result = StatisticalService.bootstrap_confidence_interval([0.75])
        assert result["mean"] == 0.75
    
    def test_reproducibility(self):
        values = [0.1, 0.9, 0.5, 0.7, 0.3]
        r1 = StatisticalService.bootstrap_confidence_interval(values, seed=42)
        r2 = StatisticalService.bootstrap_confidence_interval(values, seed=42)
        assert r1 == r2


class TestMcNemarTest:
    def test_identical_results(self):
        correct = [True, False, True, True, False, True, False, True, True, False]
        result = StatisticalService.mcnemar_test(correct, correct)
        
        assert result["p_value"] == 1.0
        assert not result["is_significant"]
        assert result["b"] == 0
        assert result["c"] == 0
    
    def test_different_results(self):
        # Method A is clearly better
        correct_a = [True] * 30 + [False] * 10
        correct_b = [True] * 10 + [False] * 30
        
        result = StatisticalService.mcnemar_test(correct_a, correct_b)
        
        assert result["p_value"] < 0.05
        assert result["is_significant"]
    
    def test_empty_results(self):
        result = StatisticalService.mcnemar_test([], [])
        assert result["p_value"] == 1.0
        assert result["n"] == 0
    
    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            StatisticalService.mcnemar_test([True, False], [True])
    
    def test_small_sample(self):
        # With very few samples, should not be significant
        correct_a = [True, False, True]
        correct_b = [False, True, True]
        result = StatisticalService.mcnemar_test(correct_a, correct_b)
        
        assert "p_value" in result
        assert "is_significant" in result
    
    def test_contingency_counts(self):
        # A: [T, T, F, F]
        # B: [T, F, T, F]
        # b (A right, B wrong) = 1
        # c (A wrong, B right) = 1
        correct_a = [True, True, False, False]
        correct_b = [True, False, True, False]
        
        result = StatisticalService.mcnemar_test(correct_a, correct_b)
        assert result["b"] == 1
        assert result["c"] == 1
        assert result["n"] == 4
