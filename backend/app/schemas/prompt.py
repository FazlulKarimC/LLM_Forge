from datetime import datetime
import json
from typing import Annotated, Any, Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, SecretStr, StringConstraints, field_validator, model_validator
from app.services.prompt_templates import prompt_variables

TemplateFormat = Literal["mustache", "fstring"]
ReleaseLabel = Annotated[str, StringConstraints(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")]
PromptType = Literal["text", "chat"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatMessage(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=50_000)


class VersionCreate(StrictModel):
    template_text: str = Field(default="", max_length=50_000)
    prompt_type: PromptType = "text"
    messages: list[ChatMessage] = Field(default_factory=list, max_length=100)
    config: dict[str, Any] = Field(default_factory=dict)
    labels: list[ReleaseLabel] = Field(default_factory=list, max_length=20)
    template_format: TemplateFormat = "mustache"
    description: str = Field(default="", max_length=4000)
    base_version: int | None = Field(default=None, ge=1)

    @field_validator("labels")
    @classmethod
    def mutable_labels(cls, value):
        if "latest" in value:
            raise ValueError("latest is managed automatically and cannot be assigned")
        return sorted(set(value))

    @field_validator("config")
    @classmethod
    def bounded_config(cls, value):
        try:
            encoded = json.dumps(value, allow_nan=False)
        except (ValueError, TypeError, RecursionError) as exc:
            raise ValueError("Config must be a finite JSON object") from exc
        if len(encoded.encode()) > 20_000:
            raise ValueError("Config is limited to 20,000 bytes")
        return value

    @model_validator(mode="after")
    def content_contract(self):
        if self.prompt_type == "text":
            if not self.template_text.strip() or self.messages:
                raise ValueError("Text prompts require a template and cannot contain chat messages")
        elif self.template_text or not self.messages or not any(message.content.strip() for message in self.messages):
            raise ValueError("Chat prompts require messages and an empty text template")
        if sum(len(message.content) for message in self.messages) > 50_000:
            raise ValueError("Chat content is limited to 50,000 characters")
        prompt_variables(self.template_text, self.messages, self.prompt_type, self.template_format)
        return self


def clean_tags(value):
    tags = sorted(set(tag.strip() for tag in value))
    if len(tags) > 20 or any(not tag or len(tag) > 64 or any(ord(char) < 32 for char in tag) for tag in tags):
        raise ValueError("Use up to 20 nonblank tags, each at most 64 characters")
    return tags


class PromptCreate(VersionCreate):
    name: str = Field(min_length=1, max_length=255)
    tags: list[str] = Field(default_factory=list, max_length=20)

    _tags = field_validator("tags")(clean_tags)

    @field_validator("name")
    @classmethod
    def clean_name(cls, value):
        value = value.strip()
        if not value or any(segment in {"", ".", ".."} for segment in value.split("/")) or any(ord(char) < 32 for char in value):
            raise ValueError("Use a nonblank name or folder/name path without empty, . or .. segments or control characters")
        return value


class PromptUpdate(StrictModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=4000)
    archived: bool | None = None
    tags: list[str] | None = Field(default=None, max_length=20)

    @field_validator("tags")
    @classmethod
    def bounded_tags(cls, value):
        return clean_tags(value) if value is not None else None

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
    prompt_type: PromptType = "text"
    messages: list[ChatMessage] = Field(default_factory=list)
    config: dict[str, Any] = Field(default_factory=dict)
    created_by: str = "UI"
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
    prompt_type: PromptType = "text"
    tags: list[str] = Field(default_factory=list)
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
        "variables": prompt_variables(version.template_text, version.messages, version.prompt_type, version.template_format)})
