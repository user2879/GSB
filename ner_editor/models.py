"""Small data objects shared by the editor stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from .constants import TYPE_CODES


@dataclass(frozen=True)
class VocabEntry:
    entry_id: str
    text: str
    type_code: str

    def as_dict(self) -> dict[str, str]:
        return {"id": self.entry_id, "text": self.text, "type": self.type_code}


@dataclass(frozen=True)
class CorpusSample:
    sample_id: str
    chars: tuple[str, ...]
    labels: tuple[str, ...]
    types: tuple[str, ...]

    @property
    def text(self) -> str:
        return "".join(self.chars)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.sample_id,
            "chars": list(self.chars),
            "labels": list(self.labels),
            "types": list(self.types),
        }


@dataclass
class ModelParams:
    version: str
    feature_dim: int
    hash_buckets: int
    labels: tuple[str, ...]
    char_embeddings: dict[str, list[Decimal]]
    emission_weights: list[list[Decimal]]
    transition_weights: list[list[Decimal]]
    start_weights: list[Decimal]
    default_entity_type: str = "ORG"
    vocabulary: dict[str, VocabEntry] = field(default_factory=dict)

    def clone(self) -> "ModelParams":
        return ModelParams(
            version=self.version,
            feature_dim=self.feature_dim,
            hash_buckets=self.hash_buckets,
            labels=self.labels,
            char_embeddings={
                key: list(values) for key, values in self.char_embeddings.items()
            },
            emission_weights=[list(row) for row in self.emission_weights],
            transition_weights=[list(row) for row in self.transition_weights],
            start_weights=list(self.start_weights),
            default_entity_type=self.default_entity_type,
            vocabulary=dict(self.vocabulary),
        )


@dataclass
class InteractionConfig:
    latency_budget_ms: float
    fixed_decimals: int
    quantization_level: int
    quantization_levels: tuple[int, ...]
    thread_pool_workers: int
    snapshot_commit_interval_commands: int
    boundary_score_tolerance: Decimal
    temp_dir: str
    default_entity_type: str = "ORG"

    def clone(self) -> "InteractionConfig":
        return InteractionConfig(
            latency_budget_ms=self.latency_budget_ms,
            fixed_decimals=self.fixed_decimals,
            quantization_level=self.quantization_level,
            quantization_levels=self.quantization_levels,
            thread_pool_workers=self.thread_pool_workers,
            snapshot_commit_interval_commands=self.snapshot_commit_interval_commands,
            boundary_score_tolerance=self.boundary_score_tolerance,
            temp_dir=self.temp_dir,
            default_entity_type=self.default_entity_type,
        )

    @property
    def scale(self) -> int:
        return 10**self.fixed_decimals


@dataclass
class CommandSpec:
    position: int
    command_type: str
    payload: dict[str, Any]
    sequence: int | None = None


@dataclass
class Failure:
    code: str
    stage: str
    message: str
    sample_id: str | None = None
    command_position: int | None = None
    locator: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def sort_key(self) -> tuple[Any, ...]:
        label_order = {
            "load": 0,
            "command": 1,
            "feature": 2,
            "emission": 3,
            "decode": 4,
            "merge": 5,
        }
        return (
            label_order.get(self.stage, 99),
            self.command_position if self.command_position is not None else -1,
            self.sample_id or "",
            self.code,
            self.locator or "",
            self.message,
        )

    def identity(self) -> tuple[Any, ...]:
        return (
            self.code,
            self.stage,
            self.sample_id,
            self.command_position,
            self.locator,
            self.message,
            repr(sorted(self.details.items())),
        )

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "stage": self.stage,
            "message": self.message,
        }
        if self.sample_id is not None:
            result["sample_id"] = self.sample_id
        if self.command_position is not None:
            result["command_position"] = self.command_position
        if self.locator is not None:
            result["locator"] = self.locator
        if self.details:
            result["details"] = self.details
        return result


@dataclass
class StageArtifact:
    stage: str
    sample_id: str
    cache_key: str
    input_fingerprint: str
    output_fingerprint: str
    payload: dict[str, Any]
    refs: dict[str, Any]
    version: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "sample_id": self.sample_id,
            "cache_key": self.cache_key,
            "input_fingerprint": self.input_fingerprint,
            "output_fingerprint": self.output_fingerprint,
            "version": self.version,
            "refs": self.refs,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StageArtifact":
        return cls(
            stage=str(value["stage"]),
            sample_id=str(value["sample_id"]),
            cache_key=str(value["cache_key"]),
            input_fingerprint=str(value["input_fingerprint"]),
            output_fingerprint=str(value["output_fingerprint"]),
            payload=dict(value.get("payload", {})),
            refs=dict(value.get("refs", {})),
            version=str(value["version"]),
        )


def type_sort_key(type_code: str) -> tuple[int, str]:
    try:
        return TYPE_CODES.index(type_code), type_code
    except ValueError:
        return len(TYPE_CODES), type_code
