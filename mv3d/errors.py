"""Business failures and deterministic ordering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Failure:
    code: str
    view_id: int = 0
    artifact: str = ""
    detail: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sort_key(self) -> tuple[int, str, str, str]:
        return (self.view_id, self.code, self.artifact, self.detail)

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "code": self.code,
            "view_id": self.view_id,
            "artifact": self.artifact,
            "detail": self.detail,
        }
        if self.metadata:
            value["metadata"] = self.metadata
        return value


class WorkspaceLockedError(RuntimeError):
    pass


def sorted_failures(failures: list[Failure]) -> list[Failure]:
    return sorted(failures, key=lambda item: item.sort_key)

