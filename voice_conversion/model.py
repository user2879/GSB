"""Unified intermediate representation shared by v1 and v2 exchange formats."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass
class Segment:
    """One conversion segment in the canonical representation.

    Frame boundaries are kept as floats (the v2 native form).  For v1 inputs the
    integer boundaries simply have zero fractional residuals.  ``intensity`` is
    the 0..1 floating point timbre strength; ``intensity_permille`` preserves
    the original v1 integer so an upgrade/downgrade round trip is exact.
    """

    index: int
    start_frame: float
    end_frame: float
    intensity: float
    timbre: str
    source_version: str
    intensity_permille: int | None = None
    precision_tag: str = "exact"
    start_residual: float = 0.0
    end_residual: float = 0.0
    missing: bool = False
    source_line: int = -1
    timbre_code: int | None = None
    passthrough: dict[str, Any] = field(default_factory=dict)

    def frame_count(self) -> int:
        return int(round(self.end_frame)) - int(round(self.start_frame))

    def copy(self) -> "Segment":
        return replace(self, passthrough=dict(self.passthrough))
