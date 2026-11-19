from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any

EXIT_CODES: dict[str, int] = {
    "E_SPEC": 2,
    "E_CYCLE": 3,
    "E_PORT": 4,
    "E_GRID": 5,
    "E_SNAPSHOT": 6,
    "E_REPLAY": 7,
    "E_STALE": 8,
    "E_ORIGIN": 9,
    "E_PROTOCOL": 10,
    "E_STATE": 11,
    "E_IO": 12,
}


@dataclass
class PipelineError(Exception):
    code: str
    message: str
    path: str | None = None
    field: str | None = None
    details: dict[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "message": self.message,
        }
        if self.path is not None:
            payload["path"] = self.path
        if self.field is not None:
            payload["field"] = self.field
        if self.details:
            payload["details"] = self.details
        return payload

    @property
    def exit_code(self) -> int:
        return EXIT_CODES.get(self.code, 1)


def e_spec(message: str, path: str, field: str, **details: Any) -> PipelineError:
    return PipelineError("E_SPEC", message, path=path, field=field, details=details)
