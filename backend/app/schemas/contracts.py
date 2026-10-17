"""Canonical response contracts exported by FastAPI to SDK/frontend tooling."""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict
from app.schemas.prompt import PromptResponse, VersionResponse, ChatMessage
from app.schemas.evaluation import DatasetCase
from app.schemas.evaluator import EvaluatorDefinition


class ProjectResponse(BaseModel):
    id: UUID
    name: str
    slug: str
    organization_id: UUID


class OrganizationResponse(BaseModel):
    id: UUID
    name: str
    slug: str
    role: Literal["owner", "member"]
    projects: list[ProjectResponse]


class WorkspaceUser(BaseModel):
    id: UUID
    display_name: str


class WorkspaceResponse(BaseModel):
    user: WorkspaceUser
    organizations: list[OrganizationResponse]


class OrganizationCreated(BaseModel):
    id: UUID
    name: str
    project_id: UUID


class KeyResponse(BaseModel):
    id: UUID
    name: str
    prefix: str
    scope: str
    scopes: list[str]
    created_at: datetime
    revoked_at: datetime | None


class KeyCreated(KeyResponse):
    secret: str


class PromptDetail(BaseModel):
    prompt: PromptResponse
    version: VersionResponse


class PromptPage(BaseModel):
    items: list[PromptResponse]
    total: int


class CompiledPrompt(BaseModel):
    compiled_prompt: str
    compiled_messages: list[ChatMessage] | None


class PlaygroundResponse(CompiledPrompt):
    output: str
    provider: str
    model: str
    is_mock: bool
    latency_ms: float
    tokens_input: int | None
    tokens_output: int | None
    finish_reason: str | None


class DatasetResponse(BaseModel):
    id: UUID
    name: str
    description: str
    archived: bool
    latest_version: int
    created_at: datetime


class RevisionResponse(BaseModel):
    id: UUID
    dataset_id: UUID
    version: int
    cases: list[DatasetCase]
    created_at: datetime


class DatasetDetail(BaseModel):
    dataset: DatasetResponse
    revision: RevisionResponse


class DatasetPage(BaseModel):
    items: list[DatasetResponse]
    total: int


class CaseImport(BaseModel):
    cases: list[DatasetCase]


class EvaluatorVersionResponse(BaseModel):
    id: UUID
    evaluator_id: UUID
    version: int
    definition: EvaluatorDefinition
    notes: str
    created_at: datetime


class EvaluatorResponse(BaseModel):
    id: UUID
    name: str
    description: str
    archived: bool
    latest_version: int
    latest: EvaluatorVersionResponse
    created_at: datetime


class EvaluatorPage(BaseModel):
    items: list[EvaluatorResponse]
    total: int


class CheckResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    kind: str
    passed: bool
    reason: str


class EvaluatorPreview(BaseModel):
    checks: list[CheckResponse]
    preview: bool


class EvaluationRunResponse(BaseModel):
    id: UUID
    prompt_version_id: UUID
    dataset_revision_id: UUID | None
    source_run_id: UUID | None
    config: dict[str, Any]
    status: Literal["queued", "running", "completed", "failed", "cancelled"]
    total: int
    completed: int
    passed: int
    errors: int
    error: str | None
    created_at: datetime
    updated_at: datetime


class EvaluationPage(BaseModel):
    items: list[EvaluationRunResponse]
    total: int


class CaseResponse(DatasetCase):
    case_index: int
    output: str | None
    passed: bool
    error: str | None
    checks: list[CheckResponse]
    latency_ms: float | None
    tokens_input: int | None
    tokens_output: int | None


class ScoreSummaryResponse(BaseModel):
    assignment: str
    name: str
    evaluator_name: str | None
    evaluator_version_id: str | None
    data_type: str
    required: bool
    count: int
    errors: int
    passed: int
    missing: int
    mean: float | None
    categories: dict[str, int]


class EvaluationDetailResponse(BaseModel):
    run: EvaluationRunResponse
    results: list[CaseResponse]
    score_summary: list[ScoreSummaryResponse]
    recovery: "RecoveryResponse | None" = None


class AttemptResponse(BaseModel):
    attempt: int
    status: str
    failure_code: str | None
    request_id: str | None
    created_at: datetime
    updated_at: datetime


class RemainingCalls(BaseModel):
    generation: int
    judging: int
    total: int


class RecoveryResponse(BaseModel):
    available: bool
    remaining_calls: RemainingCalls
    failure_code: str | None
    attempts: list[AttemptResponse]


class TokenUsage(BaseModel):
    tokens_input: int
    tokens_output: int


class EvaluationUsage(BaseModel):
    generation: TokenUsage
    judging: TokenUsage


class EvaluationSummaryResponse(EvaluationDetailResponse):
    recovery: RecoveryResponse
    usage: EvaluationUsage


class CaseRowResponse(BaseModel):
    case_index: int
    name: str
    passed: bool
    error: str | None
    output_preview: str | None
    latency_ms: float | None


class CasePageResponse(BaseModel):
    items: list[CaseRowResponse]
    total: int
    offset: int
    limit: int
    comparison_totals: dict[str, int] | None = None


EvaluationDetailResponse.model_rebuild()
EvaluationSummaryResponse.model_rebuild()
