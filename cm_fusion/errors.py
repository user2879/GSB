from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ValidationError(Exception):
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


class PipelineError(Exception):
    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.details = details or {}


@dataclass(frozen=True)
class FailureRecord:
    source: str
    line: int
    code: str
    item_id: str
    message: str

    def sort_key(self) -> tuple:
        return (self.source, self.line, self.code, self.item_id, self.message)

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "line": self.line,
            "code": self.code,
            "item_id": self.item_id,
            "message": self.message,
        }


def sort_failures(failures: list[FailureRecord]) -> list[FailureRecord]:
    return sorted(failures, key=FailureRecord.sort_key)

