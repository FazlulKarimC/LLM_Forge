import json
import math
from typing import Literal
from uuid import UUID

from pydantic import Field, SecretStr, field_validator, model_validator

from app.schemas.prompt import CompileRequest, PromptCreate, StrictModel


class DatasetCase(StrictModel):
    inputs: dict[str, str] = Field(max_length=100)
    expected_output: str | None = Field(default=None, max_length=10_000)
    name: str = Field(default="", max_length=200)

    @field_validator("inputs")
    @classmethod
    def bounded(cls, value):
        return CompileRequest.bounded_variables(value)


class RevisionCreate(StrictModel):
    cases: list[DatasetCase] = Field(min_length=1, max_length=100)
    base_version: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def bounded_size(self):
        if (
            len(json.dumps([case.model_dump() for case in self.cases]).encode())
            > 1_000_000
        ):
            raise ValueError("Dataset revision is limited to 1 MB")
        return self


class DatasetCreate(RevisionCreate):
    name: str = Field(min_length=1, max_length=255)
    description: str = Field(default="", max_length=4000)

    @field_validator("name")
    @classmethod
    def clean_name(cls, value):
        if "/" in value:
            raise ValueError("Dataset names cannot include /")
        return PromptCreate.clean_name(value)


class DatasetUpdate(StrictModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=4000)
    archived: bool | None = None

    @field_validator("name")
    @classmethod
    def clean_name(cls, value):
        if value is not None and "/" in value:
            raise ValueError("Dataset names cannot include /")
        return PromptCreate.clean_name(value) if value is not None else None


class DatasetImport(StrictModel):
    format: Literal["csv", "json"]
    content: str = Field(min_length=1, max_length=1_000_000)


class Assertion(StrictModel):
    kind: Literal[
        "exact_match",
        "contains",
        "regex",
        "json_valid",
        "json_equals",
        "json_reference",
        "json_path",
    ]
    value: str = Field(default="", max_length=2000)
    path: str = Field(default="", max_length=200)


class JudgeConfig(StrictModel):
    provider: Literal["groq", "openrouter", "openai"]
    model: str = Field(min_length=1, max_length=255)
    api_key: SecretStr = Field(max_length=512)
    rubric: str = Field(min_length=1, max_length=4000)
    threshold: float = Field(default=0.7, ge=0, le=1, allow_inf_nan=False)


class EvaluatorSelection(StrictModel):
    version_id: UUID
    required: bool = True
    api_key: SecretStr | None = Field(default=None, max_length=512)


class EvaluationCreate(StrictModel):
    prompt_version_id: UUID
    dataset_revision_id: UUID
    provider: Literal["mock", "groq", "openrouter", "openai"] = "mock"
    model: str = Field(default="demo-model", min_length=1, max_length=255)
    api_key: SecretStr | None = Field(default=None, max_length=512)
    temperature: float = Field(default=0, ge=0, le=2, allow_inf_nan=False)
    max_tokens: int = Field(default=256, ge=1, le=2048)
    assertions: list[Assertion] = Field(default_factory=list, max_length=10)
    judge: JudgeConfig | None = None
    evaluators: list[EvaluatorSelection] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def requires_checks_and_key(self):
        if (
            not self.assertions
            and self.judge is None
            and not any(item.required for item in self.evaluators)
        ):
            raise ValueError("Choose at least one assertion or an LLM judge")
        if self.provider != "mock" and (
            not self.api_key or not self.api_key.get_secret_value().strip()
        ):
            raise ValueError("An API key is required for live generation")
        if self.judge and not self.judge.api_key.get_secret_value().strip():
            raise ValueError("An API key is required for judging")
        if len({item.version_id for item in self.evaluators}) != len(self.evaluators):
            raise ValueError("Select each evaluator version once")
        return self


class SubmittedCheck(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    passed: bool
    score: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    reason: str = Field(default="", max_length=2000)


class SubmittedResult(StrictModel):
    case_index: int = Field(ge=0, le=99)
    output: str | None = Field(default=None, max_length=32_000)
    error: str | None = Field(default=None, min_length=1, max_length=2000)
    checks: list[SubmittedCheck] = Field(default_factory=list, max_length=10)
    latency_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    tokens_input: int | None = Field(default=None, ge=0, le=1_000_000)
    tokens_output: int | None = Field(default=None, ge=0, le=1_000_000)


class EvaluationSubmission(StrictModel):
    prompt_version_id: UUID
    dataset_revision_id: UUID
    model: str = Field(min_length=1, max_length=255)
    results: list[SubmittedResult] = Field(min_length=1, max_length=100)
    metrics: dict[str, float] = Field(default_factory=dict, max_length=20)

    @model_validator(mode="after")
    def bounded_payload(self):
        if len(self.model_dump_json().encode()) > 1_000_000:
            raise ValueError("Evaluation submissions are limited to 1 MB")
        return self

    @field_validator("metrics")
    @classmethod
    def bounded_metrics(cls, value):
        if any(
            not key.strip() or len(key) > 100 or not math.isfinite(score)
            for key, score in value.items()
        ):
            raise ValueError(
                "Metrics require names up to 100 characters and finite numbers"
            )
        return value


def run_response(run):
    return {
        key: getattr(run, key)
        for key in (
            "id",
            "prompt_version_id",
            "dataset_revision_id",
            "source_run_id",
            "config",
            "status",
            "total",
            "completed",
            "passed",
            "errors",
            "error",
            "created_at",
            "updated_at",
        )
    }


def dataset_response(item):
    return {
        key: getattr(item, key)
        for key in (
            "id",
            "name",
            "description",
            "archived",
            "latest_version",
            "created_at",
        )
    }


def revision_response(item):
    return {
        key: getattr(item, key)
        for key in ("id", "dataset_id", "version", "cases", "created_at")
    }
