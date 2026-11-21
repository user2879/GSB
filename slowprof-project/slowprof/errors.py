"""Domain errors for the slow-request profiler.

Error codes are stable identifiers used by the CLI, the Tkinter diagnostics
view and the test-suite:

* ``E_SAMPLE`` - a malformed or semantically invalid NDJSON sample row.
* ``E_RANGE``  - an invalid observation interval / session parameter.
* ``E_LEDGER`` - a local SQLite ledger integrity problem.
* ``E_REPLAY`` - a re-computation diverges from a persisted snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class ProfilerError(Exception):
    """Base class for all profiler errors carrying a stable ``code``."""

    code = "E_INTERNAL"

    def __init__(self, message: str, *, field: Optional[str] = None,
                 line_number: Optional[int] = None) -> None:
        super().__init__(message)
        self.message = message
        self.field = field
        self.line_number = line_number

    def __str__(self) -> str:  # pragma: no cover - trivial
        location = ""
        if self.line_number is not None:
            location = f" line={self.line_number}"
        field_part = f" field={self.field}" if self.field else ""
        return f"[{self.code}]{location}{field_part}: {self.message}"


class SampleError(ProfilerError):
    """A single NDJSON row failed parsing or validation."""

    code = "E_SAMPLE"


class RangeError(ProfilerError):
    """An observation interval or session parameter is invalid."""

    code = "E_RANGE"


class LedgerError(ProfilerError):
    """The SQLite ledger is missing, corrupt or internally inconsistent."""

    code = "E_LEDGER"


@dataclass(frozen=True)
class ReplayMismatch:
    """First divergent item discovered while replaying a historical profile."""

    frame_name: str
    field: str
    expected: object
    actual: object

    def describe(self) -> str:
        return (
            f"frame={self.frame_name!r} field={self.field!r} "
            f"expected={self.expected!r} actual={self.actual!r}"
        )


class ReplayError(ProfilerError):
    """Re-computed results do not match the persisted snapshot."""

    code = "E_REPLAY"

    def __init__(self, mismatch: ReplayMismatch) -> None:
        super().__init__("replay mismatch: " + mismatch.describe())
        self.mismatch = mismatch

    @classmethod
    def from_mismatch(cls, frame_name: str, field: str,
                      expected: object, actual: object) -> "ReplayError":
        return cls(ReplayMismatch(frame_name, field, expected, actual))
