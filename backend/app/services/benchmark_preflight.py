"""Provider-free validation for the small hosted benchmark profile."""

from app.core.config import settings
from app.services.dataset_service import DATASET_REGISTRY, DatasetService


def validate_benchmark(config, examples=None):
    if config.dataset_name not in DATASET_REGISTRY and config.dataset_name != "sample":
        raise ValueError("Choose a registered bundled diagnostic dataset")
    if examples is None:
        examples = DatasetService.load(
            config.dataset_name, config.num_samples, config.hyperparameters.seed
        )
    ids = [item.get("id") for item in examples]
    if not ids or any(not identity for identity in ids) or len(ids) != len(set(ids)):
        raise ValueError("Benchmark cases need unique, non-empty IDs")
    if config.num_samples > settings.BENCHMARK_MAX_SAMPLES:
        raise ValueError(
            f"Hosted benchmarks allow at most {settings.BENCHMARK_MAX_SAMPLES} cases"
        )
    if config.rag and config.rag.retrieval_method.value != "none":
        raise ValueError(
            "Hosted RAG execution is paused until retrieval and enrichment have explicit call budgets; existing RAG results remain readable"
        )
    if config.routing and not config.routing.strict_comparison:
        raise ValueError(
            "Hosted benchmarks require strict comparison routing; cross-provider fallback is disabled"
        )
    calls = len(examples)
    if config.reasoning_method.value == "react":
        if config.prompt_version_id:
            raise ValueError(
                "Saved prompts are not supported for ReAct; the agent builds its own prompt"
            )
        iterations = config.agent.max_iterations if config.agent else 3
        tools = (
            config.agent.tools if config.agent else ["wikipedia_search", "calculator"]
        )
        if (
            len(examples) > settings.BENCHMARK_REACT_MAX_SAMPLES
            or iterations > settings.BENCHMARK_REACT_MAX_ITERATIONS
        ):
            raise ValueError(
                f"Hosted ReAct allows at most {settings.BENCHMARK_REACT_MAX_SAMPLES} cases and {settings.BENCHMARK_REACT_MAX_ITERATIONS} iterations"
            )
        if not tools or set(tools) - {"wikipedia_search", "calculator"}:
            raise ValueError(
                "Hosted ReAct tools are wikipedia_search and calculator; retrieval is not enabled"
            )
        calls *= iterations
    if calls > settings.BENCHMARK_MAX_CALLS:
        raise ValueError(
            f"This run exceeds the {settings.BENCHMARK_MAX_CALLS}-call hosted budget"
        )
    return {
        "selected_cases": len(examples),
        "max_provider_calls": calls,
        "max_seconds": settings.BENCHMARK_MAX_SECONDS,
        "call_timeout_seconds": settings.BENCHMARK_CALL_TIMEOUT_SECONDS,
        "provider_concurrency": 1,
        "optional_scoring": False,
    }
