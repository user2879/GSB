from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


STAGE_ORDER = {
    "load": 0,
    "schedule": 1,
    "inference": 2,
    "merge": 3,
    "evaluation": 4,
    "snapshot": 5,
}


@dataclass(frozen=True)
class Failure:
    code: str
    item: str
    stage: str
    message: str
    details: dict[str, Any] = field(default_factory=dict, compare=False)

    def sort_key(self) -> tuple[int, str, str, str]:
        return (STAGE_ORDER.get(self.stage, 99), self.item, self.code, self.message)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "details": _sort_mapping(self.details),
            "item": self.item,
            "message": self.message,
            "stage": self.stage,
        }


def sort_failures(failures: list[Failure] | tuple[Failure, ...]) -> list[Failure]:
    return sorted(failures, key=Failure.sort_key)


def failures_as_dicts(failures: list[Failure] | tuple[Failure, ...]) -> list[dict[str, Any]]:
    return [failure.as_dict() for failure in sort_failures(list(failures))]


def _sort_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sort_mapping(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_sort_mapping(item) for item in value]
    return value
