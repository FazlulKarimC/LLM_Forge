"""Factory helpers for inference-engine selection."""

from __future__ import annotations

import logging
from typing import Optional, Tuple

from app.core.config import settings

logger = logging.getLogger(__name__)


def build_auto_provider_engines(model_name: str) -> list:
    """
    Build the provider fallback chain for provider="auto".

    Fails fast when no usable provider credentials are configured so the caller
    gets a clear operational error instead of a late initialization failure.
    """
    from app.services.inference.hf_api_engine import HFAPIEngine

    engines = []
    provider_errors: list[str] = []

    if settings.HF_TOKEN:
        engines.append(HFAPIEngine(model_name=model_name))

    if settings.OPENROUTER_API_KEY:
        try:
            from app.services.inference.openrouter_engine import OpenRouterEngine

            engines.append(OpenRouterEngine(model_name=model_name))
        except Exception as exc:
            logger.warning("Could not init OpenRouter: %s", exc)
            provider_errors.append(f"openrouter={exc}")

    if settings.GROQ_API_KEY:
        try:
            from app.services.inference.groq_engine import GroqEngine

            engines.append(GroqEngine(model_name=model_name))
        except Exception as exc:
            logger.warning("Could not init Groq: %s", exc)
            provider_errors.append(f"groq={exc}")

    if engines:
        return engines

    if provider_errors:
        raise ValueError(
            "No usable auto-routed providers could be initialized. "
            + "; ".join(provider_errors)
        )

    raise ValueError(
        "No LLM provider credentials configured for provider='auto'. "
        "Set HF_TOKEN, OPENROUTER_API_KEY, or GROQ_API_KEY, "
        "or use a custom provider in development."
    )


def create_inference_engine(
    *,
    model_name: str,
    provider_value: str,
    routing_config=None,
    custom_base_url: Optional[str] = None,
    custom_api_key: Optional[str] = None,
    default_engine: Optional[str] = None,
) -> Tuple[object, str]:
    """Create the configured inference engine and a human-readable engine label."""
    from app.services.inference.mock_engine import MockEngine

    if provider_value == "custom" and not custom_base_url:
        raise ValueError(
            "Custom provider runs require stored custom endpoint credentials for the configured model."
        )

    if "mock" in model_name.lower():
        return MockEngine(), "mock"

    if provider_value == "custom":
        from app.services.inference.openai_engine import OpenAIEngine

        return OpenAIEngine(
            base_url=custom_base_url,
            api_key=custom_api_key,
            model_name=model_name,
        ), "custom"

    if provider_value == "openrouter":
        from app.services.inference.openrouter_engine import OpenRouterEngine

        return OpenRouterEngine(model_name=model_name), "openrouter"

    if provider_value == "groq":
        from app.services.inference.groq_engine import GroqEngine

        return GroqEngine(model_name=model_name), "groq"

    if provider_value == "auto":
        from app.services.inference.provider_router import ProviderRouter
        from app.services.inference.provider_stats import RoutingPolicy

        engines = build_auto_provider_engines(model_name)
        router_policy = RoutingPolicy.FALLBACK_CHAIN
        epsilon = 0.15
        exploration_window = 10

        if routing_config:
            try:
                router_policy = RoutingPolicy(routing_config.policy)
            except ValueError:
                router_policy = RoutingPolicy.FALLBACK_CHAIN
            epsilon = routing_config.epsilon
            exploration_window = routing_config.exploration_window

        return ProviderRouter(
            engines=engines,
            policy=router_policy,
            epsilon=epsilon,
            exploration_window=exploration_window,
        ), "auto (router)"

    if default_engine == "hf_api":
        from app.services.inference.hf_api_engine import HFAPIEngine

        return HFAPIEngine(model_name=model_name), "hf_api"

    return MockEngine(), default_engine or "mock"
