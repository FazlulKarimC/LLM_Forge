"""Request-only credentials for an explicit interrupted-attempt recovery."""

from pydantic import Field, SecretStr
from app.schemas.prompt import StrictModel


class ResumeEvaluation(StrictModel):
    api_key: SecretStr | None = Field(default=None, max_length=512)
    judge_key: SecretStr | None = Field(default=None, max_length=512)
    evaluator_keys: dict[str, SecretStr] = Field(default_factory=dict, max_length=10)
