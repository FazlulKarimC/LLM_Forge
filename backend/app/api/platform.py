"""Public, provider-free discovery with no database access or credential state."""

from typing import get_args

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.build_info import build_identity
from app.core.config import settings
from app.schemas.evaluation import EvaluationProvider, JudgeProvider

router = APIRouter()


class EvaluationCapabilities(BaseModel):
    generation_providers: list[EvaluationProvider]
    judge_providers: list[JudgeProvider]
    external_submissions: bool


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
    )


@router.get("/version", response_model=VersionMetadata)
async def version():
    return VersionMetadata(service="llmforge-backend", api_version=settings.VERSION, **build_identity())
