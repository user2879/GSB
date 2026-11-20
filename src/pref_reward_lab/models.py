from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RawPair:
    pair_id: str
    prompt: str
    chosen: str
    rejected: str
    annotator_id: str
    timestamp: str

    def content_dict(self) -> dict[str, str]:
        return {
            "pair_id": self.pair_id,
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "annotator_id": self.annotator_id,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True)
class V1Record:
    pair_id: str
    score_milli: int
    delta_milli: int

    def content_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "score_milli": self.score_milli,
            "delta_milli": self.delta_milli,
            "source_version": "v1",
        }


@dataclass(frozen=True)
class V2Record:
    pair_id: str
    reward_score: float
    reward_delta: float
    reward_label: str
    precision_label: str

    def content_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "reward_score": self.reward_score,
            "reward_delta": self.reward_delta,
            "reward_label": self.reward_label,
            "precision_label": self.precision_label,
            "source_version": "v2",
        }


@dataclass
class ParseResult:
    records: dict[str, Any]
    failures: list[dict[str, Any]]


@dataclass
class FeatureResult:
    pair_id: str
    buckets: dict[int, int]
    prompt_buckets: dict[int, int]
    chosen_buckets: dict[int, int]
    rejected_buckets: dict[int, int]
    prompt_tokens: int
    chosen_tokens: int
    rejected_tokens: int
    precision_fingerprint: str

    def content_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "buckets": {str(key): self.buckets[key] for key in sorted(self.buckets)},
            "prompt_buckets": {
                str(key): self.prompt_buckets[key] for key in sorted(self.prompt_buckets)
            },
            "chosen_buckets": {
                str(key): self.chosen_buckets[key] for key in sorted(self.chosen_buckets)
            },
            "rejected_buckets": {
                str(key): self.rejected_buckets[key] for key in sorted(self.rejected_buckets)
            },
            "prompt_tokens": self.prompt_tokens,
            "chosen_tokens": self.chosen_tokens,
            "rejected_tokens": self.rejected_tokens,
            "precision_fingerprint": self.precision_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FeatureResult":
        return cls(
            pair_id=str(value["pair_id"]),
            buckets={int(key): int(count) for key, count in value["buckets"].items()},
            prompt_buckets={
                int(key): int(count) for key, count in value.get("prompt_buckets", {}).items()
            },
            chosen_buckets={
                int(key): int(count) for key, count in value.get("chosen_buckets", {}).items()
            },
            rejected_buckets={
                int(key): int(count) for key, count in value.get("rejected_buckets", {}).items()
            },
            prompt_tokens=int(value.get("prompt_tokens", 0)),
            chosen_tokens=int(value.get("chosen_tokens", 0)),
            rejected_tokens=int(value.get("rejected_tokens", 0)),
            precision_fingerprint=str(value.get("precision_fingerprint", "")),
        )


@dataclass
class ScoreResult:
    pair_id: str
    chosen_score: float
    rejected_score: float
    reward_delta: float
    effective_threshold: float
    boundary: bool

    def content_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "chosen_score": self.chosen_score,
            "rejected_score": self.rejected_score,
            "reward_delta": self.reward_delta,
            "effective_threshold": self.effective_threshold,
            "boundary": self.boundary,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ScoreResult":
        return cls(
            pair_id=str(value["pair_id"]),
            chosen_score=float(value["chosen_score"]),
            rejected_score=float(value["rejected_score"]),
            reward_delta=float(value["reward_delta"]),
            effective_threshold=float(value["effective_threshold"]),
            boundary=bool(value["boundary"]),
        )

