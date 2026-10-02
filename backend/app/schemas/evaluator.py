"""Strict reusable evaluator and score contracts. Definitions contain no secrets."""

from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator

from app.schemas.evaluation import Assertion, EvaluatorSelection
from app.schemas.prompt import StrictModel


class ScoreDefinition(StrictModel):
    name: str = Field(min_length=1, max_length=80, pattern=r"^[a-zA-Z][a-zA-Z0-9_.-]*$")
    data_type: Literal["boolean", "numeric", "categorical"] = "boolean"
    minimum: float = Field(default=0, allow_inf_nan=False)
    maximum: float = Field(default=1, allow_inf_nan=False)
    threshold: float = Field(default=0.7, allow_inf_nan=False)
    categories: list[str] = Field(default_factory=list, max_length=20)
    passing_categories: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="after")
    def valid_range(self):
        if (
            self.data_type == "numeric"
            and not self.minimum <= self.threshold <= self.maximum
        ):
            raise ValueError("Numeric threshold must be inside the score range")
        if self.data_type == "categorical":
            if not self.categories or len(set(self.categories)) != len(self.categories):
                raise ValueError("Categorical scores require unique categories")
            if any(not item.strip() or len(item) > 80 for item in self.categories):
                raise ValueError("Categories must contain 1–80 characters")
            if not self.passing_categories or not set(self.passing_categories) <= set(
                self.categories
            ):
                raise ValueError(
                    "Choose passing categories from the declared categories"
                )
        return self


class EvaluatorDefinition(StrictModel):
    kind: Literal["builtin", "llm_judge"] = "builtin"
    assertion: Assertion | None = None
    provider: Literal["groq", "openrouter", "openai"] = "groq"
    model: str = Field(default="", max_length=255)
    rubric: str = Field(default="", max_length=4000)
    outputs: list[ScoreDefinition] = Field(min_length=1, max_length=5)
    # A deliberately small mapping vocabulary; no expressions or code execution.
    mapping: dict[str, Literal["inputs", "output", "expected_output"]] = Field(
        default_factory=lambda: {
            "inputs": "inputs",
            "candidate": "output",
            "reference": "expected_output",
        },
        max_length=10,
    )

    @model_validator(mode="after")
    def valid_definition(self):
        if len({item.name for item in self.outputs}) != len(self.outputs):
            raise ValueError("Score names must be unique in an evaluator")
        if not self.mapping or any(
            not key.isidentifier() or len(key) > 80 for key in self.mapping
        ):
            raise ValueError("Mappings require simple named inputs")
        if self.kind == "builtin":
            if (
                self.assertion is None
                or len(self.outputs) != 1
                or self.outputs[0].data_type != "boolean"
            ):
                raise ValueError(
                    "Built-in evaluators require one assertion and one boolean output"
                )
            # Dataset reference requirements are checked at dispatch; validate static fields now.
            from app.services.evaluation_service import validate_assertions
            from fastapi import HTTPException

            try:
                validate_assertions([self.assertion], [{"expected_output": "null"}])
            except HTTPException as exc:
                raise ValueError(str(exc.detail)) from exc
        elif (
            self.assertion is not None
            or not self.model.strip()
            or not self.rubric.strip()
        ):
            raise ValueError(
                "Judges require a model and rubric, without a built-in assertion"
            )
        return self


class EvaluatorVersionCreate(StrictModel):
    definition: EvaluatorDefinition
    notes: str = Field(default="", max_length=2000)
    base_version: int | None = Field(default=None, ge=1)


class EvaluatorCreate(EvaluatorVersionCreate):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)

    @field_validator("name")
    @classmethod
    def clean_name(cls, value):
        value = value.strip()
        if not value or any(ord(char) < 32 for char in value):
            raise ValueError(
                "Evaluator name cannot be blank or contain control characters"
            )
        return value


class EvaluatorUpdate(StrictModel):
    archived: bool


class ScoreOnlyCreate(StrictModel):
    # Import EvaluationCreate's legacy fields through a separate selection contract.
    evaluators: list[EvaluatorSelection] = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def unique_versions(self):
        if len({item.version_id for item in self.evaluators}) != len(self.evaluators):
            raise ValueError("Select each evaluator version once")
        if not any(item.required for item in self.evaluators):
            raise ValueError("Select at least one required evaluator")
        return self


class EvaluatorTest(StrictModel):
    inputs: dict[str, str] = Field(default_factory=dict, max_length=100)
    output: str = Field(max_length=32_000)
    expected_output: str | None = Field(default=None, max_length=10_000)
    api_key: SecretStr | None = Field(default=None, max_length=512)

    @field_validator("inputs")
    @classmethod
    def bound_inputs(cls, value):
        from app.schemas.prompt import CompileRequest

        return CompileRequest.bounded_variables(value)
