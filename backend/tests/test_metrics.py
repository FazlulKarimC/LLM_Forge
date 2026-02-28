"""
Metrics Service Unit Tests

Tests for accuracy, latency, cost, faithfulness,
and semantic similarity metric computations.
Updated for P0/P1/P2 evaluation improvements.
"""

from unittest.mock import MagicMock
from app.services.metrics_service import MetricsService, _normalize


class TestNormalize:
    """Tests for shared normalization function (P0 #2)."""

    def test_lowercase(self):
        assert _normalize("PARIS") == "paris"

    def test_strip_whitespace(self):
        assert _normalize("  Paris  ") == "paris"

    def test_strip_trailing_punctuation(self):
        assert _normalize("Paris.") == "paris"
        assert _normalize("Paris!") == "paris"
        assert _normalize("Paris,") == "paris"

    def test_collapse_internal_whitespace(self):
        assert _normalize("New   York  City") == "new york city"

    def test_empty_string(self):
        assert _normalize("") == ""


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
        # common = {'paris'}, precision = 1/3, recall = 1/2
        expected = 2 * (1 / 3) * (1 / 2) / (1 / 3 + 1 / 2)
        assert abs(f1 - expected) < 0.001

    def test_no_overlap_gives_zero(self):
        assert MetricsService.compute_f1("hello world", "goodbye moon") == 0.0

    def test_empty_prediction(self):
        assert MetricsService.compute_f1("", "Paris") == 0.0

    def test_empty_ground_truth(self):
        assert MetricsService.compute_f1("Paris", "") == 0.0

    def test_punctuation_consistency_with_exact(self):
        """P0 #2: F1 should normalize same as exact match."""
        # 'Paris.' should normalize to 'paris' for both methods
        f1 = MetricsService.compute_f1("Paris.", "Paris")
        assert f1 == 1.0, "F1 should be 1.0 after punctuation normalization"

        exact = MetricsService.check_exact_match("Paris.", "Paris")
        assert exact is True, "Exact match should also be True"


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

    def test_word_boundary(self):
        """P0 #2: Should not match 'paris' inside 'comparison'."""
        # This tests the word-boundary improvement
        assert MetricsService.check_substring("comparison", "paris") is False


class TestAliasMatch:
    """Tests for multi-alias matching (now returns matched_alias)."""

    def test_exact_alias_match(self):
        exact, sub, f1, alias = MetricsService.check_any_alias_match(
            "Da Vinci", ["Leonardo da Vinci", "Da Vinci", "Leonardo"]
        )
        assert exact is True
        assert alias == "Da Vinci"

    def test_substring_alias_match(self):
        exact, sub, f1, alias = MetricsService.check_any_alias_match(
            "I think it was painted by Leonardo da Vinci",
            ["Leonardo da Vinci", "Da Vinci"],
        )
        assert sub is True
        assert alias != ""

    def test_best_f1_selected(self):
        _, _, f1, alias = MetricsService.check_any_alias_match(
            "Leonardo",
            ["Leonardo da Vinci", "Leonardo"],
        )
        # "Leonardo" exactly matches alias "Leonardo" -> F1 = 1.0
        assert f1 == 1.0

    def test_no_match(self):
        exact, sub, f1, alias = MetricsService.check_any_alias_match(
            "Raphael", ["Leonardo da Vinci", "Leonardo"]
        )
        assert exact is False
        assert sub is False
        assert f1 < 1.0


class TestAccuracyFromBooleans:
    """P0 #1: Tests that accuracy uses stored booleans, not score heuristic."""

    def _make_mock_run(self, is_exact=None, is_substring=None, is_correct=None, score=None):
        run = MagicMock()
        run.is_exact_match = is_exact
        run.is_substring_match = is_substring
        run.is_correct = is_correct
        run.score = score
        run.attempt = 1
        return run

    def test_exact_match_from_boolean(self):
        """P0 #1: Exact match should come from is_exact_match, not score."""
        runs = [
            self._make_mock_run(is_exact=True, is_substring=False, score=1.0),
            self._make_mock_run(is_exact=False, is_substring=True, score=0.5),
            self._make_mock_run(is_exact=False, is_substring=False, score=0.0),
        ]
        svc = MetricsService.__new__(MetricsService)
        acc = svc._compute_accuracy(runs)
        assert acc["exact_match"] == 1 / 3
        assert acc["substring"] == 1 / 3

    def test_legacy_fallback(self):
        """Legacy runs without is_exact_match should fall back to score heuristic."""
        runs = [
            self._make_mock_run(is_exact=None, is_substring=None, is_correct=True, score=1.0),
            self._make_mock_run(is_exact=None, is_substring=None, is_correct=True, score=0.7),
        ]
        svc = MetricsService.__new__(MetricsService)
        acc = svc._compute_accuracy(runs)
        # score=1.0 -> exact, score=0.7 -> substring (legacy fallback)
        assert acc["exact_match"] == 0.5
        assert acc["substring"] == 0.5

    def test_f1_bug_case(self):
        """
        P0 #1 bug case: substring match yields F1=1.0 for single-word answer.
        Old code would count this as exact match because score==1.0.
        New code uses is_exact_match boolean.
        """
        run = self._make_mock_run(is_exact=False, is_substring=True, score=1.0)
        svc = MetricsService.__new__(MetricsService)
        acc = svc._compute_accuracy([run])
        # Despite score=1.0, this should NOT count as exact match
        assert acc["exact_match"] == 0.0
        assert acc["substring"] == 1.0


class TestThroughput:
    """P0 #3: Tests that throughput uses wall-clock time."""

    def _make_mock_run(self, latency_ms):
        run = MagicMock()
        run.latency_ms = latency_ms
        return run

    def test_wall_clock_throughput(self):
        """P0 #3: Throughput from wall-clock, not sum of latencies."""
        runs = [self._make_mock_run(100), self._make_mock_run(100)]
        svc = MetricsService.__new__(MetricsService)

        # 2 runs in 150ms wall-clock = 2/0.15 = ~13.33 req/s
        latency = svc._compute_latency(runs, wall_clock_ms=150)
        assert abs(latency["throughput"] - (2 / 0.15)) < 0.1
        assert latency["throughput_source"] == "wall_clock"

    def test_fallback_throughput(self):
        """Without wall-clock, falls back to sum of latencies."""
        runs = [self._make_mock_run(100), self._make_mock_run(100)]
        svc = MetricsService.__new__(MetricsService)

        latency = svc._compute_latency(runs, wall_clock_ms=None)
        assert latency["throughput_source"] == "sum_latency_fallback"


class TestFaithfulnessAggregation:
    """P0 #4: Tests faithfulness aggregation from run-level scores."""

    def _make_mock_run(self, faithfulness=None):
        run = MagicMock()
        run.faithfulness_score = faithfulness
        return run

    def test_mean_faithfulness(self):
        runs = [self._make_mock_run(0.8), self._make_mock_run(0.6)]
        svc = MetricsService.__new__(MetricsService)
        f = svc._compute_faithfulness(runs)
        assert abs(f["mean"] - 0.7) < 0.001

    def test_hallucination_rate(self):
        # 2 below 0.5, 1 above -> hallucination_rate = 2/3
        runs = [
            self._make_mock_run(0.3),
            self._make_mock_run(0.4),
            self._make_mock_run(0.8),
        ]
        svc = MetricsService.__new__(MetricsService)
        f = svc._compute_faithfulness(runs)
        assert abs(f["hallucination_rate"] - 2 / 3) < 0.001

    def test_no_faithfulness_scores(self):
        runs = [self._make_mock_run(None)]
        svc = MetricsService.__new__(MetricsService)
        f = svc._compute_faithfulness(runs)
        assert f["mean"] is None
        assert f["count"] == 0
