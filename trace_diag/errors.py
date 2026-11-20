from __future__ import annotations

from typing import Any


class AppError(Exception):
    code = "E_APP"

    def __init__(self, path: str, message: str, context: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.path = path
        self.message = message
        self.context = context or {}

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "path": self.path,
            "message": self.message,
        }
        if self.context:
            result["context"] = self.context
        return result

    def __str__(self) -> str:
        return f"{self.code} at {self.path}: {self.message}"


class SpecError(AppError):
    code = "E_SPEC"


class RecordError(AppError):
    code = "E_RECORD"


class BucketError(AppError):
    code = "E_BUCKET"


class SnapshotError(AppError):
    code = "E_SNAPSHOT"


class ReplayError(AppError):
    code = "E_REPLAY"

    def __init__(
        self,
        path: str,
        message: str,
        *,
        service_id: str,
        field: str,
        context: dict[str, Any] | None = None,
    ) -> None:
        merged = dict(context or {})
        merged.update({"service_id": service_id, "field": field})
        super().__init__(path, message, merged)
        self.service_id = service_id
        self.field = field
