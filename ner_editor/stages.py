"""Stage result and failure helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import Failure


class StageFailure(Exception):
    def __init__(self, failure: Failure):
        super().__init__(failure.message)
        self.failure = failure


@dataclass
class PipelineContext:
    feature: dict[str, Any] | None = None
    emission: dict[str, Any] | None = None
    decode: dict[str, Any] | None = None
    merge: dict[str, Any] | None = None
    ledger: dict[str, str] = field(default_factory=dict)
    failures: list[Failure] = field(default_factory=list)
    code: str | None = None

