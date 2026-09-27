from dataclasses import dataclass
from typing import Any, Mapping

from .templates import compile_template


@dataclass(frozen=True)
class Prompt:
    id: str
    prompt_id: str
    name: str
    version: int
    template_text: str
    template_format: str
    variables: tuple[str, ...]

    def compile(self, variables: Mapping[str, str] | None = None, **values: str) -> str:
        supplied = dict(variables or {})
        if supplied.keys() & values.keys():
            raise ValueError("Variable provided twice")
        supplied.update(values)
        return compile_template(self.template_text, supplied, self.template_format)


@dataclass(frozen=True)
class Evaluation:
    run: dict[str, Any]
    results: list[dict[str, Any]]

    def __post_init__(self):
        if (
            not isinstance(self.run, dict)
            or not isinstance(self.results, list)
            or not isinstance(self.run.get("id"), str)
        ):
            raise ValueError("Invalid evaluation response")
        counts = [
            self.run.get(key) for key in ("total", "completed", "passed", "errors")
        ]
        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("Invalid evaluation counts")
        total, completed, passed, errors = counts
        if total < 1 or not passed + errors <= completed <= total:
            raise ValueError("Invalid evaluation counts")
        if self.run.get("status") not in (
            "queued",
            "running",
            "completed",
            "failed",
            "cancelled",
        ):
            raise ValueError("Invalid evaluation status")

    @property
    def id(self) -> str:
        return self.run["id"]

    @property
    def status(self) -> str:
        return self.run["status"]

    @property
    def pass_rate(self) -> float:
        return self.run["passed"] / self.run["total"] if self.run["total"] else 0.0

    def meets_threshold(self, minimum: float = 1.0) -> bool:
        if not 0 <= minimum <= 1:
            raise ValueError("Minimum pass rate must be between 0 and 1")
        return (
            self.status == "completed"
            and self.run["completed"] == self.run["total"]
            and self.run["errors"] == 0
            and self.pass_rate >= minimum
        )

    def to_dict(self) -> dict[str, Any]:
        return {"run": self.run, "results": self.results}
