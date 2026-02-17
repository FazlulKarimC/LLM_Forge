"""
Metrics Service Unit Tests

Tests for accuracy, latency, and cost metric computations.
"""

from app.services.metrics_service import MetricsService


class TestF1Score:
    """Tests for F1 token overlap computation."""

    def test_exact_match_gives_perfect_f1(self):
        assert MetricsService.compute_f1("Paris", "Paris") == 1.0

    def test_case_insensitive_f1(self):
        assert MetricsService.compute_f1("paris", "Paris") == 1.0

    def test_partial_overlap(self):
        """'capital city Paris' vs 'Paris France' -> common = {'paris'}"""
        f1 = MetricsService.compute_f1("capital city Paris", "Paris France")
        assert 0.0 < f1 < 1.0
        # pred_tokens = ['capital', 'city', 'paris'], truth_tokens = ['paris', 'france']
        # common = {'paris'}, precision = 1/3, recall = 1/2, f1 = 2*(1/3)*(1/2) / (1/3 + 1/2)
        expected = 2 * (1 / 3) * (1 / 2) / (1 / 3 + 1 / 2)
        assert abs(f1 - expected) < 0.001

    def test_no_overlap_gives_zero(self):
        assert MetricsService.compute_f1("hello world", "goodbye moon") == 0.0

    def test_empty_prediction(self):
        assert MetricsService.compute_f1("", "Paris") == 0.0

    def test_empty_ground_truth(self):
        assert MetricsService.compute_f1("Paris", "") == 0.0


class TestExactMatch:
    """Tests for case-insensitive exact match."""

    def test_identical(self):
        assert MetricsService.check_exact_match("Paris", "Paris") is True

    def test_case_insensitive(self):
        assert MetricsService.check_exact_match("paris", "Paris") is True

    def test_with_whitespace(self):
        assert MetricsService.check_exact_match("  Paris  ", "Paris") is True

    def test_with_punctuation(self):
        assert MetricsService.check_exact_match("Paris.", "Paris") is True

    def test_not_matching(self):
        assert MetricsService.check_exact_match("London", "Paris") is False


class TestSubstring:
    """Tests for substring containment."""

    def test_exact_is_substring(self):
        assert MetricsService.check_substring("Paris", "Paris") is True

    def test_contained_in_longer(self):
        assert MetricsService.check_substring("The answer is Paris", "Paris") is True

    def test_case_insensitive(self):
        assert MetricsService.check_substring("the answer is paris", "Paris") is True

    def test_not_contained(self):
        assert MetricsService.check_substring("London", "Paris") is False


class TestAliasMatch:
    """Tests for multi-alias matching."""

    def test_exact_alias_match(self):
        exact, sub, f1 = MetricsService.check_any_alias_match(
            "Da Vinci", ["Leonardo da Vinci", "Da Vinci", "Leonardo"]
        )
        assert exact is True

    def test_substring_alias_match(self):
        exact, sub, f1 = MetricsService.check_any_alias_match(
            "I think it was painted by Leonardo da Vinci",
            ["Leonardo da Vinci", "Da Vinci"],
        )
        assert sub is True

    def test_best_f1_selected(self):
        _, _, f1 = MetricsService.check_any_alias_match(
            "Leonardo",
            ["Leonardo da Vinci", "Leonardo"],
        )
        # "Leonardo" exactly matches alias "Leonardo" -> F1 = 1.0
        assert f1 == 1.0

    def test_no_match(self):
        exact, sub, f1 = MetricsService.check_any_alias_match(
            "Raphael", ["Leonardo da Vinci", "Leonardo"]
        )
        assert exact is False
        assert sub is False
        assert f1 < 1.0
