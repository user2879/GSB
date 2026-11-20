"""Business error codes for the GPU quota and shard scheduling component.

Every validation or runtime failure is reported through :class:`SchedulerError`
with a stable business code, a human readable message and a list of structured
problem descriptors carrying the offending JSON field path and the expected
constraint.  Codes are plain strings so they can be embedded in structured
logs and SQLite records without further mapping.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional

# Business codes required by the specification.
E_SPEC = "E_SPEC"          # declaration / strong validation failure
E_MEMORY = "E_MEMORY"      # per-task memory budget violation
E_QUOTA = "E_QUOTA"        # quota share total or global quota violation
E_CYCLE = "E_CYCLE"        # dependency cycle
E_DEPEND = "E_DEPEND"      # missing reference or unrecoverable dependency issue
E_LEDGER = "E_LEDGER"      # ledger digest / format mismatch
E_REPLAY = "E_REPLAY"      # crash recovery replay divergence


@dataclass(frozen=True)
class Problem:
    """A single validation problem anchored at a JSON field path."""

    path: str
    expected: str
    actual: Any = None
    code: str = E_SPEC

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "path": self.path,
            "expected": self.expected,
            "actual": self.actual,
        }


@dataclass
class SchedulerError(Exception):
    """Raised for every predictable business failure.

    ``problems`` keeps one entry per rejected field so that a single load can
    report every issue instead of failing at the first one.  The loader still
    rejects the whole declaration atomically (nothing is persisted).
    """

    code: str
    message: str
    problems: List[Problem] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "problems": [problem.to_dict() for problem in self.problems],
        }

    def first_path(self) -> Optional[str]:
        return self.problems[0].path if self.problems else None
