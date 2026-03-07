"""
Model Pricing Lookup Table

Provides estimated pricing per 1K tokens for various LLM providers
and models. Pricing is best-effort and used for cost estimation only.

Many free-tier providers (HF Inference API, Groq free tier) do not
charge per-token, but we still track cost to demonstrate the pattern.
"""

from typing import Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    """Pricing per 1K tokens in USD."""
    input_per_1k: float
    output_per_1k: float
    provider: str


# Pricing table: model_name -> ModelPricing
# Prices are approximate and may change. Last updated: 2025-01.
PRICING_TABLE: dict[str, ModelPricing] = {
    # --- HuggingFace Inference API (free tier — $0/token) ---
    "mistralai/Mistral-7B-Instruct-v0.3":       ModelPricing(0.0, 0.0, "hf_api"),
    "meta-llama/Meta-Llama-3-8B-Instruct":       ModelPricing(0.0, 0.0, "hf_api"),
    "meta-llama/Llama-3.2-3B-Instruct":          ModelPricing(0.0, 0.0, "hf_api"),
    "Qwen/Qwen2.5-72B-Instruct":                ModelPricing(0.0, 0.0, "hf_api"),
    "google/gemma-2-9b-it":                      ModelPricing(0.0, 0.0, "hf_api"),
    "microsoft/Phi-3.5-mini-instruct":           ModelPricing(0.0, 0.0, "hf_api"),
    "NousResearch/Hermes-3-Llama-3.1-8B":        ModelPricing(0.0, 0.0, "hf_api"),

    # --- OpenAI models (for custom-hosted / proxy scenarios) ---
    "gpt-4o":                                    ModelPricing(0.005, 0.015, "openai"),
    "gpt-4o-mini":                               ModelPricing(0.00015, 0.0006, "openai"),
    "gpt-3.5-turbo":                             ModelPricing(0.0005, 0.0015, "openai"),

    # --- Groq (free tier — $0/token) ---
    "llama-3.1-70b-versatile":                   ModelPricing(0.0, 0.0, "groq"),
    "llama-3.1-8b-instant":                      ModelPricing(0.0, 0.0, "groq"),
    "gemma2-9b-it":                              ModelPricing(0.0, 0.0, "groq"),
    "mixtral-8x7b-32768":                        ModelPricing(0.0, 0.0, "groq"),

    # --- OpenRouter (free tier — $0/token for :free models) ---
    "meta-llama/llama-3.1-8b-instruct:free":     ModelPricing(0.0, 0.0, "openrouter"),
    "qwen/qwen3-8b:free":                        ModelPricing(0.0, 0.0, "openrouter"),
    "google/gemma-3-1b-it:free":                 ModelPricing(0.0, 0.0, "openrouter"),
}

# Default pricing for unknown models
DEFAULT_PRICING = ModelPricing(0.00015, 0.0006, "unknown")


def get_model_pricing(model_name: str) -> ModelPricing:
    """
    Look up pricing for a model name.

    Falls back to DEFAULT_PRICING for unknown models.
    Performs substring matching for partial model name matches.
    """
    # Exact match
    if model_name in PRICING_TABLE:
        return PRICING_TABLE[model_name]

    # Substring match (e.g., "gpt-4o" matches "gpt-4o-mini")
    for key, pricing in PRICING_TABLE.items():
        if model_name.lower() in key.lower() or key.lower() in model_name.lower():
            return pricing

    return DEFAULT_PRICING


def estimate_cost(
    model_name: str,
    tokens_input: int = 0,
    tokens_output: int = 0,
) -> dict:
    """
    Estimate cost for a given model and token counts.

    Returns:
        dict with input_cost, output_cost, total_cost (all in USD),
        and pricing metadata.
    """
    pricing = get_model_pricing(model_name)

    input_cost = (tokens_input / 1_000) * pricing.input_per_1k
    output_cost = (tokens_output / 1_000) * pricing.output_per_1k
    total_cost = input_cost + output_cost

    return {
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(total_cost, 6),
        "provider": pricing.provider,
        "input_per_1k": pricing.input_per_1k,
        "output_per_1k": pricing.output_per_1k,
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
    }
