from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ResultCache:
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)
    corrupt_reason: str = ""

    def get(self, key: str) -> dict[str, Any] | None:
        value = self.entries.get(key)
        return copy.deepcopy(value) if value is not None else None

    def contains(self, key: str) -> bool:
        return key in self.entries

    def put(self, key: str, value: dict[str, Any]) -> None:
        self.entries[key] = copy.deepcopy(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "rope-diag-cache-v1",
            "entries": {key: self.entries[key] for key in sorted(self.entries)},
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "ResultCache":
        if not isinstance(payload, dict) or not isinstance(payload.get("entries"), dict):
            return cls(corrupt_reason="cache payload is not a rope-diag-cache-v1 object")
        entries: dict[str, dict[str, Any]] = {}
        for key, value in payload["entries"].items():
            if isinstance(key, str) and isinstance(value, dict):
                entries[key] = value
        return cls(entries=entries)


@dataclass(frozen=True)
class CacheDecision:
    scheme: str
    sample_id: str
    position_key: str
    scoring_key: str
    position_hit: bool
    scoring_hit: bool
    action: str
    points: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme,
            "sample_id": self.sample_id,
            "position_key": self.position_key,
            "scoring_key": self.scoring_key,
            "position_hit": self.position_hit,
            "scoring_hit": self.scoring_hit,
            "action": self.action,
            "points": self.points,
        }


def derived_stage_status(
    decisions: list[CacheDecision], stage: str
) -> dict[str, Any]:
    if not decisions:
        return {
            "status": "not_run",
            "hit_count": 0,
            "computed_count": 0,
            "failed_count": 0,
        }
    hit_attr = "position_hit" if stage == "position" else "scoring_hit"
    hit_count = sum(1 for decision in decisions if getattr(decision, hit_attr))
    failed_count = sum(1 for decision in decisions if decision.action == "failed")
    computed_count = len(decisions) - hit_count - failed_count
    if failed_count == 0 and hit_count == len(decisions):
        status = "reused"
    elif hit_count == 0 and computed_count == len(decisions):
        status = "recomputed"
    else:
        status = "partial"
    return {
        "status": status,
        "hit_count": hit_count,
        "computed_count": computed_count,
        "failed_count": failed_count,
    }

