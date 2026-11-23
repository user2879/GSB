"""Stable error codes and failure-record collection for the VC pipeline."""
from __future__ import annotations

from dataclasses import dataclass, asdict


E_FIELD_TRUNCATED = "E_FIELD_TRUNCATED"
E_UNKNOWN_TIMBRE = "E_UNKNOWN_TIMBRE"
E_V2_SCHEMA = "E_V2_SCHEMA"
E_SEGMENT_TOO_LONG = "E_SEGMENT_TOO_LONG"
E_SNAPSHOT_MISMATCH = "E_SNAPSHOT_MISMATCH"


class PipelineError(Exception):
    """Fatal pipeline error carrying a stable code."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Failure:
    """A non-fatal, order-stable failure attached to a segment or a file."""

    code: str
    stage: str
    segment: int
    line: int
    field: str
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


class FailureLog:
    """Collects failures and emits them in a deterministic order.

    Ordering key: (segment index, stage name, line, field, code).  This makes the
    emitted order independent of thread scheduling and the order in which worker
    threads observed the failure.
    """

    def __init__(self) -> None:
        self._items: list[Failure] = []

    def add(
        self,
        code: str,
        stage: str,
        segment: int = -1,
        line: int = -1,
        field: str = "",
        message: str = "",
    ) -> None:
        self._items.append(
            Failure(code=code, stage=stage, segment=segment, line=line,
                    field=field, message=message)
        )

    def sorted(self) -> list[Failure]:
        return sorted(
            self._items,
            key=lambda f: (f.segment, f.stage, f.line, f.field, f.code),
        )

    def to_list(self) -> list[dict]:
        return [f.to_dict() for f in self.sorted()]

    def __len__(self) -> int:
        return len(self._items)
