from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AuditError(Exception):
    code: str
    message: str
    line: int | None = None
    field: str | None = None
    sample_id: str | None = None
    scheme_id: str | None = None
    stage: str | None = None

    def __str__(self) -> str:
        location = []
        if self.line is not None:
            location.append(f"line={self.line}")
        if self.field:
            location.append(f"field={self.field}")
        if self.sample_id:
            location.append(f"sample={self.sample_id}")
        if self.stage:
            location.append(f"stage={self.stage}")
        suffix = f" ({', '.join(location)})" if location else ""
        return f"{self.code}: {self.message}{suffix}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "line": self.line,
            "field": self.field,
            "sample_id": self.sample_id,
            "scheme_id": self.scheme_id,
            "stage": self.stage,
        }


class ConfigInvalid(AuditError):
    def __init__(
        self,
        message: str,
        line: int | None = None,
        field: str | None = None,
        sample_id: str | None = None,
    ) -> None:
        super().__init__(
            code="E_CONFIG_INVALID",
            message=message,
            line=line,
            field=field,
            sample_id=sample_id,
        )


def failure_record(error: AuditError | Exception, **extra: Any) -> dict[str, Any]:
    if isinstance(error, AuditError):
        record = error.to_dict()
    else:
        record = {
            "code": "E_SAMPLE_FAILED",
            "message": f"{type(error).__name__}: {error}",
            "line": None,
            "field": None,
        }
    record.update(extra)
    record.setdefault("code", "E_SAMPLE_FAILED")
    return record
