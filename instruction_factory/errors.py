from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Failure:
    code: str
    message: str
    stage: str
    file: str | None = None
    line: int | None = None
    field: str | None = None
    path: str | None = None
    candidate_id: str | None = None
    task_type: str | None = None
    batch_id: str | None = None
    fatal: bool = False

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "stage": self.stage,
            "fatal": self.fatal,
        }
        for key in ("file", "line", "field", "path", "candidate_id", "task_type", "batch_id"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result


@dataclass
class FailureCollector:
    items: list[Failure] = field(default_factory=list)

    def add(
        self,
        code: str,
        message: str,
        *,
        stage: str,
        file: str | None = None,
        line: int | None = None,
        field: str | None = None,
        path: str | None = None,
        candidate_id: str | None = None,
        task_type: str | None = None,
        batch_id: str | None = None,
        fatal: bool = False,
    ) -> Failure:
        failure = Failure(
            code=code,
            message=message,
            stage=stage,
            file=str(file) if file is not None else None,
            line=line,
            field=field,
            path=path,
            candidate_id=candidate_id,
            task_type=task_type,
            batch_id=batch_id,
            fatal=fatal,
        )
        self.items.append(failure)
        return failure

    def extend(self, failures: list[Failure]) -> None:
        self.items.extend(failures)

    @property
    def fatal(self) -> bool:
        return any(item.fatal for item in self.items)

    def has_code(self, code: str) -> bool:
        return any(item.code == code for item in self.items)

    def to_list(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self.items]


class PlanApplyError(RuntimeError):
    def __init__(self, failures: list[Failure]):
        super().__init__("effective plan could not be applied")
        self.failures = failures

