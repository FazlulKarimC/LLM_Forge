"""Public, provider-free discovery with no database access or credential state."""

from functools import lru_cache
from typing import get_args

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.build_info import build_identity
from app.core.config import settings
from app.schemas.evaluation import EvaluationProvider, JudgeProvider

router = APIRouter()


class HostedBenchmarkProfile(BaseModel):
    max_samples: int
    max_provider_calls: int
    max_seconds: int
    call_timeout_seconds: int
    react_max_samples: int
    react_max_iterations: int
    global_concurrency: int = 1
    provider_concurrency: int = 1
    case_page_size: int = 25
    max_input_bytes: int = 8192
    max_output_bytes: int = 8192
    max_trace_bytes: int = 16384
    max_case_bytes: int = 65536
    max_attempt_bytes: int = 2097152
    max_project_bytes: int = 104857600
    max_project_attempts: int = 100
    rag_enabled: bool = False
    enrichment_enabled: bool = False
    dataset_counts: dict[str, int]


@lru_cache(maxsize=1)
def benchmark_dataset_counts():
    from app.services.dataset_service import DATASET_REGISTRY, DatasetService
    return {name: len(DatasetService.load(name)) for name in [*DATASET_REGISTRY, "sample"]}


class EvaluationCapabilities(BaseModel):
    generation_providers: list[EvaluationProvider]
    judge_providers: list[JudgeProvider]
    external_submissions: bool
    benchmarks: HostedBenchmarkProfile


class VersionMetadata(BaseModel):
    service: str
    api_version: str
    commit: str | None
    build_id: str | None


@router.get("/capabilities", response_model=EvaluationCapabilities)
async def capabilities():
    return EvaluationCapabilities(
        generation_providers=list(get_args(EvaluationProvider)),
        judge_providers=list(get_args(JudgeProvider)),
        external_submissions=True,
        benchmarks=HostedBenchmarkProfile(
            max_samples=settings.BENCHMARK_MAX_SAMPLES,
            max_provider_calls=settings.BENCHMARK_MAX_CALLS,
            max_seconds=settings.BENCHMARK_MAX_SECONDS,
            call_timeout_seconds=settings.BENCHMARK_CALL_TIMEOUT_SECONDS,
            react_max_samples=settings.BENCHMARK_REACT_MAX_SAMPLES,
            react_max_iterations=settings.BENCHMARK_REACT_MAX_ITERATIONS,
            dataset_counts=benchmark_dataset_counts(),
        ),
    )


@router.get("/version", response_model=VersionMetadata)
async def version():
    return VersionMetadata(service="llmforge-backend", api_version=settings.VERSION, **build_identity())
