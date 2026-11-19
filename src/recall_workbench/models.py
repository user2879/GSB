"""Typed configuration and in-memory data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .fixed import FixedFormat


CHANNEL_ORDER = ("lexical", "collaborative", "popularity")


@dataclass(frozen=True)
class ChannelConfig:
    enabled: bool
    weight: float
    candidate_limit: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "weight": self.weight,
            "candidate_limit": self.candidate_limit,
        }


@dataclass(frozen=True)
class FeatureConfig:
    name: str
    weight: float

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "weight": self.weight}


@dataclass(frozen=True)
class AppConfig:
    raw: dict[str, Any]
    config_hash: str
    schema_version: int
    seed: int
    event_types: tuple[str, ...]
    event_weights: dict[str, float]
    channels: dict[str, ChannelConfig]
    fusion_strategy: str
    temperature: float
    normalization: str
    zero_channel_policy: str
    ranking_features: tuple[FeatureConfig, ...]
    primary_bits: int
    reference_bits: int
    tolerance: float
    rounding: str
    score_min: float
    score_max: float
    half_life_ms: float
    popularity_floor: float
    max_workers: int
    snapshot_commit_interval: int
    top_k: int

    @property
    def primary_format(self) -> FixedFormat:
        return FixedFormat(self.primary_bits, self.rounding)

    @property
    def reference_format(self) -> FixedFormat:
        return FixedFormat(self.reference_bits, self.rounding)

    def to_summary(self) -> dict[str, Any]:
        return {
            "config_hash": self.config_hash,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "event_types": list(self.event_types),
            "channels": {name: self.channels[name].to_dict() for name in CHANNEL_ORDER},
            "fusion": {
                "strategy": self.fusion_strategy,
                "temperature": self.temperature,
                "normalization": self.normalization,
                "zero_channel_policy": self.zero_channel_policy,
            },
            "ranking_features": [feature.to_dict() for feature in self.ranking_features],
            "fixed_point": {
                "primary_bits": self.primary_bits,
                "reference_bits": self.reference_bits,
                "primary_quantum": self.primary_format.quantum,
                "reference_quantum": self.reference_format.quantum,
                "tolerance": self.tolerance,
                "rounding": self.rounding,
            },
            "score_clip": {"min": self.score_min, "max": self.score_max},
            "retrieval": {
                "half_life_ms": self.half_life_ms,
                "event_weights": dict(self.event_weights),
                "popularity_floor": self.popularity_floor,
            },
            "execution": {
                "max_workers": self.max_workers,
                "snapshot_commit_interval": self.snapshot_commit_interval,
                "top_k": self.top_k,
            },
        }


@dataclass(frozen=True)
class Interaction:
    user_id: str
    item_id: str
    event_ms: int
    event_type: str
    line: int


@dataclass(frozen=True)
class Item:
    item_id: str
    terms: tuple[str, ...]
    line: int


@dataclass(frozen=True)
class TruthEntry:
    candidates: tuple[str, ...]
    ranking: tuple[str, ...]


@dataclass
class Dataset:
    interactions: list[Interaction] = field(default_factory=list)
    items: dict[str, Item] = field(default_factory=dict)
    truth: dict[str, TruthEntry] = field(default_factory=dict)
    failures: list[Any] = field(default_factory=list)
    file_hashes: dict[str, str] = field(default_factory=dict)

    @property
    def user_ids(self) -> list[str]:
        users = {record.user_id for record in self.interactions}
        users.update(self.truth)
        return sorted(users)

    @property
    def histories(self) -> dict[str, list[Interaction]]:
        histories: dict[str, list[Interaction]] = {user_id: [] for user_id in self.user_ids}
        for record in self.interactions:
            histories.setdefault(record.user_id, []).append(record)
        for records in histories.values():
            records.sort(key=lambda record: (record.event_ms, record.item_id, record.event_type, record.line))
        return histories
