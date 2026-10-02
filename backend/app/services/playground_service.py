"""Bounded, request-only playground calls. No runs or credentials are persisted."""
import asyncio
import time
from fastapi import HTTPException
from openai import APIConnectionError, APIStatusError, APITimeoutError
from app.services.prompt_templates import compile_prompt, prompt_preview
from app.services.inference.base import GenerationConfig
from app.services.inference.openai_engine import OpenAIEngine

PROVIDERS = {
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


async def generate_playground(data):
    try:
        compiled = compile_prompt(data.template_text, data.messages, data.prompt_type, data.variables, data.template_format)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    start = time.perf_counter()
    preview = prompt_preview(compiled)
    if data.provider == "mock":
        # A reliable demo, explicitly distinguished from actual model inference.
        return {"output": "[Demo output — no model was called]\n" + preview[:2000],
            "compiled_prompt": preview, "compiled_messages": compiled if isinstance(compiled, list) else None,
            "provider": "mock", "model": data.model, "is_mock": True,
            "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            "tokens_input": None, "tokens_output": None, "finish_reason": "demo"}
    key = data.api_key.get_secret_value().strip() if data.api_key else ""
    if not key:
        raise HTTPException(422, "Provide your API key for the selected provider. It is used for this request only.")
    engine = OpenAIEngine(base_url=PROVIDERS[data.provider], api_key=key, model_name=data.model, provider_id=data.provider, async_only=True)
    try:
        result = await asyncio.wait_for(engine.generate_async(compiled,
            GenerationConfig(max_tokens=data.max_tokens, temperature=data.temperature)), timeout=30)
    except (APITimeoutError, asyncio.TimeoutError) as exc:
        raise HTTPException(504, "The provider did not respond within 30 seconds. Try a smaller output limit.") from exc
    except APIStatusError as exc:
        if exc.status_code in (401, 403):
            message = "The provider rejected your API key or model access. Check both and try again."
        elif exc.status_code == 429:
            message = "The provider rate or credit limit was reached. Try again later."
        elif exc.status_code == 404:
            message = "This model is not available to your provider account. Enter a model ID from its current catalog."
        else:
            message = "The provider rejected the request. Check the model ID and supported generation settings."
        raise HTTPException(502, message) from exc
    except APIConnectionError as exc:
        raise HTTPException(502, "Could not connect to the selected provider. Try again.") from exc
    except RuntimeError as exc:
        raise HTTPException(502, "The provider returned an invalid completion response.") from exc
    finally:
        engine.unload_model()
    return {"output": result.text, "compiled_prompt": preview, "compiled_messages": compiled if isinstance(compiled, list) else None, "provider": data.provider,
        "model": data.model, "is_mock": False, "latency_ms": round(result.latency_ms, 2),
        "tokens_input": result.tokens_input, "tokens_output": result.tokens_output, "finish_reason": result.finish_reason}
