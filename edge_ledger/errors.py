from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


STAGE_ORDER = {
    "injection": 0,
    "load_templates": 10,
    "load_samples": 20,
    "load_config": 30,
    "resolve": 40,
    "schedule": 45,
    "render": 50,
    "score": 60,
    "dedupe": 70,
    "stats": 80,
    "snapshot": 90,
}


@dataclass(frozen=True)
class Failure:
    code: str
    stage: str
    message: str
    target_kind: str = "sample"
    target_id: str = ""
    revision: int | None = None
    injection_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def sort_key(self) -> tuple[Any, ...]:
        return (
            STAGE_ORDER.get(self.stage, 999),
            self.code,
            self.target_kind,
            self.target_id,
            -1 if self.revision is None else self.revision,
            self.injection_id or "",
            self.message,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "stage": self.stage,
            "message": self.message,
            "target_kind": self.target_kind,
            "target_id": self.target_id,
            "revision": self.revision,
            "injection_id": self.injection_id,
            "details": self.details,
        }


def sorted_failure_dicts(failures: list[Failure]) -> list[dict[str, Any]]:
    return [item.as_dict() for item in sorted(failures, key=Failure.sort_key)]


class BusinessError(Exception):
    def __init__(self, code: str, message: str, *, stage: str = "engine") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage = stage

    def as_failure(self, target_kind: str = "engine", target_id: str = "") -> Failure:
        return Failure(
            code=self.code,
            stage=self.stage,
            message=self.message,
            target_kind=target_kind,
            target_id=target_id,
        )


class SnapshotError(BusinessError):
    pass


class SimulatedCrash(RuntimeError):
    """Raised only by the explicit crash-recovery test path."""

