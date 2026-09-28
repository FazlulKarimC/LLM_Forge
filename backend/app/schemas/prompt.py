from datetime import datetime
from typing import Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from app.services.prompt_templates import template_variables

TemplateFormat = Literal["mustache", "fstring"]
ReleaseLabel = Literal["staging", "production"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VersionCreate(StrictModel):
    template_text: str = Field(min_length=1, max_length=50_000)
    template_format: TemplateFormat = "mustache"
    description: str = Field(default="", max_length=4000)
    base_version: int | None = Field(default=None, ge=1)

    @field_validator("template_text")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Template cannot be blank")
        return value


class PromptCreate(VersionCreate):
    name: str = Field(min_length=1, max_length=255)

    @field_validator("name")
    @classmethod
    def clean_name(cls, value):
        value = value.strip()
        if not value or value in {".", ".."} or any(ord(char) < 32 for char in value) or "/" in value:
            raise ValueError("Name cannot be blank, . or .., contain control characters or include /")
        return value


class PromptUpdate(StrictModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=4000)
    archived: bool | None = None

    @field_validator("name")
    @classmethod
    def clean_name(cls, value):
        return PromptCreate.clean_name(value) if value is not None else None


class VersionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    prompt_id: UUID
    name: str
    template_text: str
    template_format: TemplateFormat
    version: int
    sha256_hash: str
    parent_id: UUID | None
    description: str | None
    created_at: datetime
    variables: list[str]


class LabelResponse(BaseModel):
    label: ReleaseLabel
    version_id: UUID
    version: int
    updated_at: datetime


class PromptResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: UUID
    name: str
    description: str
    archived: bool
    latest_version: int
    created_at: datetime
    updated_at: datetime
    labels: list[LabelResponse]


class Promotion(StrictModel):
    version_id: UUID


class CompileRequest(VersionCreate):
    variables: dict[str, str] = Field(default_factory=dict, max_length=100)

    @field_validator("variables")
    @classmethod
    def bounded_variables(cls, value):
        if any(len(key) > 100 or len(text) > 10_000 for key, text in value.items()):
            raise ValueError("Variable names are limited to 100 characters and values to 10,000")
        return value


class PlaygroundRequest(CompileRequest):
    provider: Literal["mock", "groq", "openrouter", "openai"] = "mock"
    model: str = Field(default="demo-model", min_length=1, max_length=255)
    api_key: SecretStr | None = Field(default=None, max_length=512)
    temperature: float = Field(default=0.7, ge=0, le=2, allow_inf_nan=False)
    max_tokens: int = Field(default=256, ge=1, le=2048)


def version_response(version):
    return VersionResponse.model_validate({**{field: getattr(version, field) for field in VersionResponse.model_fields if field != "variables"},
        "variables": template_variables(version.template_text, version.template_format)})
