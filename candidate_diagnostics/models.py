from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    raw_score: float
    channel: str
    aux_feature: int = 0
    content_key: str | None = None
    source_line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Request:
    request_id: str
    candidates: list[Candidate] = field(default_factory=list)
    source_line: int | None = None

    @property
    def channels(self) -> set[str]:
        return {candidate.channel for candidate in self.candidates}

    def candidate(self, candidate_id: int) -> Candidate | None:
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "source_line": self.source_line,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def candidates_sorted(candidates: list[Candidate]) -> list[Candidate]:
    return sorted(candidates, key=lambda item: (item.channel, item.candidate_id))
