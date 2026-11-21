from __future__ import annotations

from dataclasses import dataclass, field

from .errors import BusinessError
from .models import RawCandidate


@dataclass
class ParsedExchange:
    version: str
    path: str
    frames: dict[tuple[str, int], list[RawCandidate]] = field(default_factory=dict)
    failures: list[BusinessError] = field(default_factory=list)
    file_sha256: str = ""
    format_fingerprint: str = ""

    def add_candidate(self, candidate: RawCandidate) -> None:
        key = (candidate.series_name, candidate.frame_index)
        self.frames.setdefault(key, []).append(candidate)

    @property
    def series_names(self) -> list[str]:
        return sorted({series for series, _ in self.frames})

    @property
    def frame_indices(self) -> dict[str, list[int]]:
        result: dict[str, list[int]] = {}
        for series, frame_index in self.frames:
            result.setdefault(series, []).append(frame_index)
        for values in result.values():
            values.sort()
        return result
