"""Shared immutable-ish domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Issue:
    code: str
    sample_id: str | None
    field: str
    message: str
    line: int | None = None
    severity: str = "error"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "sample_id": self.sample_id,
            "field": self.field,
            "message": self.message,
            "line": self.line,
            "severity": self.severity,
        }

    @property
    def sort_key(self) -> tuple[str, int, str, str, str]:
        return (
            self.sample_id or "",
            self.line if self.line is not None else -1,
            self.code,
            self.field,
            self.message,
        )


@dataclass(frozen=True)
class Sample:
    sample_id: str
    query_position: int
    key_positions: tuple[int, ...]
    query_vector: tuple[float, ...]
    key_vectors: tuple[tuple[float, ...], ...]
    content_fingerprint: str
    line: int

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "query_position": self.query_position,
            "key_positions": list(self.key_positions),
            "query_vector": list(self.query_vector),
            "key_vectors": [list(vector) for vector in self.key_vectors],
        }

    @property
    def point_cost(self) -> int:
        return 1 + len(self.key_positions)


@dataclass
class LoadedSamples:
    samples: list[Sample]
    issues: list[Issue]
    line_count: int = 0
    duplicate_count: int = 0

    @property
    def by_id(self) -> dict[str, Sample]:
        return {sample.sample_id: sample for sample in self.samples}


@dataclass
class LoadedCommands:
    commands: list[dict[str, Any]]
    issues: list[Issue]
    line_count: int = 0


@dataclass
class StageCache:
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get(self, key: str) -> dict[str, Any] | None:
        return self.entries.get(key)

    def put(self, key: str, value: dict[str, Any]) -> None:
        self.entries[key] = value
