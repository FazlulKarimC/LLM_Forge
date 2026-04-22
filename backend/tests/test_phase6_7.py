"""
Tests for Phase 6 & 7 features.

Phase 6: Multi-Provider Intelligence
- retry.py: Tenacity retry decorator
- openrouter_engine.py / groq_engine.py: Provider adapters
- provider_router.py: Routing + fallback

Phase 7: Advanced Evaluation
- robustness_scorer.py: Rule-based safety scoring
- statistical_service.py: pass@k + multi_trial_variance
- dataset_service.py: Adversarial datasets
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from app.core.retry import llm_retry, RETRYABLE_EXCEPTIONS
from app.services.robustness_scorer import classify_response, compute_safety_score
from app.services.statistical_service import StatisticalService


# ═══════════════════════════════════════════════════════════════════════════
# Phase 6: Retry Decorator
# ═══════════════════════════════════════════════════════════════════════════

class TestRetryDecorator:
    """Test the tenacity-based retry decorator."""

    def test_retry_decorator_returns_callable(self):
        """llm_retry() should return a decorator."""
        decorator = llm_retry(max_attempts=2)
        assert callable(decorator)

    def test_retry_succeeds_on_first_attempt(self):
        """Function should succeed immediately if no error."""
        call_count = 0
        @llm_retry(max_attempts=3)
        def successful_fn():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = successful_fn()
        assert result == "ok"
        assert call_count == 1

    def test_retry_on_connection_error(self):
        """Should retry on ConnectionError."""
        call_count = 0
        @llm_retry(max_attempts=3, initial_wait=0.01, max_wait=0.1)
        def flaky_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("connection refused")
            return "ok"

        result = flaky_fn()
        assert result == "ok"
        assert call_count == 3

    def test_retry_does_not_retry_value_error(self):
        """Should NOT retry on ValueError (not retryable)."""
        call_count = 0
        @llm_retry(max_attempts=3, initial_wait=0.01)
        def bad_fn():
            nonlocal call_count
            call_count += 1
            raise ValueError("invalid input")

        with pytest.raises(ValueError):
            bad_fn()
        assert call_count == 1  # No retry


# ═══════════════════════════════════════════════════════════════════════════
# Phase 6: Provider Engines
# ═══════════════════════════════════════════════════════════════════════════

class TestOpenRouterEngine:
    """Test OpenRouter engine initialization."""

    def test_requires_api_key(self):
        """Should raise ValueError if OPENROUTER_API_KEY is empty."""
        with patch("app.services.inference.openrouter_engine.settings") as mock_settings:
            mock_settings.OPENROUTER_API_KEY = ""
            from app.services.inference.openrouter_engine import OpenRouterEngine
            with pytest.raises(ValueError, match="OpenRouter API key"):
                OpenRouterEngine(api_key=None)

    def test_sets_correct_base_url(self):
        """Should use OpenRouter base URL."""
        from app.services.inference.openrouter_engine import OPENROUTER_BASE_URL
        assert "openrouter.ai" in OPENROUTER_BASE_URL

    def test_default_model(self):
        """Default model should be a free model."""
        from app.services.inference.openrouter_engine import OpenRouterEngine
        import inspect
        sig = inspect.signature(OpenRouterEngine.__init__)
        default = sig.parameters["model_name"].default
        assert ":free" in default


class TestGroqEngine:
    """Test Groq engine initialization."""

    def test_requires_api_key(self):
        """Should raise ValueError if GROQ_API_KEY is empty."""
        with patch("app.services.inference.groq_engine.settings") as mock_settings:
            mock_settings.GROQ_API_KEY = ""
            from app.services.inference.groq_engine import GroqEngine
            with pytest.raises(ValueError, match="Groq API key"):
                GroqEngine(api_key=None)

    def test_sets_correct_base_url(self):
        """Should use Groq base URL."""
        from app.services.inference.groq_engine import GROQ_BASE_URL
        assert "groq.com" in GROQ_BASE_URL


# ═══════════════════════════════════════════════════════════════════════════
# Phase 6: Provider Router
# ═══════════════════════════════════════════════════════════════════════════

class TestProviderRouter:
    """Test multi-provider routing and fallback."""

    def test_requires_at_least_one_engine(self):
        """Router should raise ValueError if no engines provided."""
        from app.services.inference.provider_router import ProviderRouter
        with pytest.raises(ValueError, match="At least one engine"):
            ProviderRouter(engines=[])

    def test_routes_to_first_loaded_engine(self):
        """Should route to the first engine that is loaded."""
        from app.services.inference.provider_router import ProviderRouter
        from app.services.inference.base import GenerationConfig, GenerationResult
        from app.models.run import FailureMode

        engine1 = MagicMock()
        engine1.is_loaded = True
        engine1.generate.return_value = GenerationResult(
            text="hello from engine1",
            tokens_input=5,
            tokens_output=3,
            latency_ms=100,
            finish_reason="stop",
        )

        engine2 = MagicMock()
        engine2.is_loaded = True

        router = ProviderRouter(engines=[engine1, engine2])
        config = GenerationConfig(max_tokens=100, temperature=0.1)
        result = router.generate("test", config)

        assert result.text == "hello from engine1"
        engine1.generate.assert_called_once()
        engine2.generate.assert_not_called()

    def test_falls_back_on_rate_limit(self):
        """Should fall back to next engine on rate limit error."""
        from app.services.inference.provider_router import ProviderRouter
        from app.services.inference.base import GenerationConfig, GenerationResult

        engine1 = MagicMock()
        engine1.is_loaded = True
        engine1.generate.side_effect = Exception("429 Too Many Requests")

        engine2 = MagicMock()
        engine2.is_loaded = True
        engine2.generate.return_value = GenerationResult(
            text="hello from engine2",
            tokens_input=5,
            tokens_output=3,
            latency_ms=200,
            finish_reason="stop",
        )

        router = ProviderRouter(engines=[engine1, engine2])
        config = GenerationConfig(max_tokens=100, temperature=0.1)
        result = router.generate("test", config)

        assert result.text == "hello from engine2"
        engine1.generate.assert_called_once()
        engine2.generate.assert_called_once()

    def test_is_loaded_checks_all_engines(self):
        """is_loaded should be True if any engine is loaded."""
        from app.services.inference.provider_router import ProviderRouter
        
        engine1 = MagicMock()
        engine1.is_loaded = False
        engine2 = MagicMock()
        engine2.is_loaded = True
        
        router = ProviderRouter(engines=[engine1, engine2])
        assert router.is_loaded is True


class TestAutoProviderEngineSelection:
    """Tests for provider='auto' engine construction."""

    def test_auto_provider_requires_at_least_one_configured_provider(self):
        from app.services.inference.engine_factory import build_auto_provider_engines

        with patch("app.services.inference.engine_factory.settings") as mock_settings:
            mock_settings.HF_TOKEN = ""
            mock_settings.OPENROUTER_API_KEY = ""
            mock_settings.GROQ_API_KEY = ""

            with pytest.raises(ValueError, match="No LLM provider credentials configured"):
                build_auto_provider_engines("meta-llama/Llama-3.2-1B-Instruct")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 7: Robustness Scorer
# ═══════════════════════════════════════════════════════════════════════════

class TestRobustnessScorer:
    """Test rule-based safety scoring."""

    def test_classify_refusal(self):
        """Should classify clear refusal as safe."""
        result = classify_response(
            "I'm sorry, I cannot assist with that request.",
            expected_behavior="REFUSE",
        )
        assert result["classification"] == "refused"
        assert result["is_safe"] is True

    def test_classify_compliance(self):
        """Should classify compliance as unsafe."""
        result = classify_response(
            "Sure, here's how you do it. Step 1: First, ...",
            expected_behavior="REFUSE",
        )
        assert result["classification"] == "complied"
        assert result["is_safe"] is False

    def test_classify_empty_response(self):
        """Should classify empty response as inconclusive."""
        result = classify_response("")
        assert result["classification"] == "inconclusive"
        assert result["is_safe"] is False

    def test_classify_handle_gracefully(self):
        """Should classify non-error edge-case responses as handled."""
        result = classify_response(
            "I can still respond to unusual input.",
            expected_behavior="HANDLE_GRACEFULLY",
        )
        assert result["classification"] == "handled"
        assert result["is_safe"] is True

    def test_compute_safety_score_all_refused(self):
        """All refused → safety_score = 1.0."""
        results = [
            {"classification": "refused", "confidence": 0.9},
            {"classification": "refused", "confidence": 0.85},
        ]
        score = compute_safety_score(results)
        assert score["safety_score"] == 1.0
        assert score["refused_count"] == 2
        assert score["complied_count"] == 0

    def test_compute_safety_score_mixed(self):
        """Mixed results → safety_score < 1.0."""
        results = [
            {"classification": "refused", "confidence": 0.9},
            {"classification": "complied", "confidence": 0.8},
        ]
        score = compute_safety_score(results)
        assert score["safety_score"] == 0.5
        assert score["refused_count"] == 1
        assert score["complied_count"] == 1

    def test_compute_safety_score_does_not_count_inconclusive_as_safe(self):
        """Inconclusive classifications should reduce the safety score."""
        results = [
            {"classification": "refused", "confidence": 0.9},
            {"classification": "inconclusive", "confidence": 0.3},
        ]
        score = compute_safety_score(results)
        assert score["safety_score"] == 0.5
        assert score["inconclusive_count"] == 1

    def test_compute_safety_score_empty(self):
        """Empty results → safety_score = 0.0."""
        score = compute_safety_score([])
        assert score["safety_score"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Phase 7: pass@k
# ═══════════════════════════════════════════════════════════════════════════

class TestPassAtK:
    """Test pass@k metric computation."""

    def test_pass_at_1_all_correct(self):
        """All correct → pass@1 = 1.0."""
        assert StatisticalService.pass_at_k(10, 10, 1) == 1.0

    def test_pass_at_1_none_correct(self):
        """None correct → pass@1 = 0.0."""
        assert StatisticalService.pass_at_k(10, 0, 1) == 0.0

    def test_pass_at_k_known_value(self):
        """10 samples, 3 correct, k=5 → known probability."""
        result = StatisticalService.pass_at_k(10, 3, 5)
        # With n=10, c=3, k=5: P(at least 1 correct in 5) 
        # = 1 - C(7,5)/C(10,5) = 1 - 21/252 = 1 - 0.08333 ≈ 0.9167
        assert abs(result - 0.9167) < 0.001

    def test_pass_at_k_edge_zero_n(self):
        """n=0 → pass@k = 0.0."""
        assert StatisticalService.pass_at_k(0, 0, 1) == 0.0

    def test_pass_at_k_k_greater_than_n(self):
        """k > n should still work."""
        result = StatisticalService.pass_at_k(5, 2, 10)
        assert result == 1.0  # More samples than available → must pick all


class TestMultiTrialVariance:
    """Test multi-trial variance computation."""

    def test_single_trial(self):
        """Single trial → std = 0."""
        result = StatisticalService.multi_trial_variance([0.85])
        assert result["mean"] == 0.85
        assert result["std"] == 0.0
        assert result["num_trials"] == 1

    def test_multiple_trials(self):
        """Multiple trials → correct stats."""
        result = StatisticalService.multi_trial_variance([0.8, 0.85, 0.9])
        assert abs(result["mean"] - 0.85) < 0.001
        assert result["min"] == 0.8
        assert result["max"] == 0.9
        assert result["num_trials"] == 3

    def test_empty_trials(self):
        """Empty trials → all zeros."""
        result = StatisticalService.multi_trial_variance([])
        assert result["mean"] == 0.0
        assert result["num_trials"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# Phase 7: Adversarial Datasets
# ═══════════════════════════════════════════════════════════════════════════

class TestAdversarialDatasets:
    """Test adversarial dataset loading."""

    def test_prompt_injection_loads(self):
        """Prompt injection dataset should load 10 examples."""
        from app.services.dataset_service import DatasetService
        examples = DatasetService.load("prompt_injection")
        assert len(examples) == 10

    def test_jailbreak_loads(self):
        """Jailbreak dataset should load 10 examples."""
        from app.services.dataset_service import DatasetService
        examples = DatasetService.load("jailbreak")
        assert len(examples) == 10

    def test_edge_cases_loads(self):
        """Edge cases dataset should load 10 examples."""
        from app.services.dataset_service import DatasetService
        examples = DatasetService.load("edge_cases")
        assert len(examples) == 10

    def test_adversarial_datasets_have_required_fields(self):
        """All adversarial examples should have id, question, answer, aliases."""
        from app.services.dataset_service import DatasetService
        for ds_name in ["prompt_injection", "jailbreak", "edge_cases"]:
            examples = DatasetService.load(ds_name)
            for ex in examples:
                assert "id" in ex
                assert "question" in ex
                assert "answer" in ex
                assert "aliases" in ex

    def test_adversarial_in_available_datasets(self):
        """Adversarial datasets should appear in available_datasets list."""
        from app.services.dataset_service import DatasetService
        datasets = DatasetService.available_datasets()
        names = [d["name"] for d in datasets]
        assert "prompt_injection" in names
        assert "jailbreak" in names
        assert "edge_cases" in names


# ═══════════════════════════════════════════════════════════════════════════
# Phase 7: Prompt Version Model
# ═══════════════════════════════════════════════════════════════════════════

class TestPromptVersion:
    """Test prompt version model."""

    def test_compute_hash_deterministic(self):
        """Same text should always produce the same hash."""
        from app.models.prompt_version import PromptVersion
        h1 = PromptVersion.compute_hash("test template")
        h2 = PromptVersion.compute_hash("test template")
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_compute_hash_different_text(self):
        """Different text should produce different hashes."""
        from app.models.prompt_version import PromptVersion
        h1 = PromptVersion.compute_hash("version 1")
        h2 = PromptVersion.compute_hash("version 2")
        assert h1 != h2
