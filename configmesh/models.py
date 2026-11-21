"""Typed data models for registry, manifest, compiled artifacts and receipts."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    port: int
    host: str = "127.0.0.1"
    timeout_ms: int = 2000
    retry_policy: str = "none"          # none | fixed
    max_retries: int = 0
    bias: int = 0                       # optional node-side derived-state bias (demos)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "NodeSpec":
        return NodeSpec(
            node_id=str(data["node_id"]),
            host=str(data.get("host", "127.0.0.1")),
            port=int(data["port"]),
            timeout_ms=int(data.get("timeout_ms", 2000)),
            retry_policy=str(data.get("retry_policy", "none")),
            max_retries=int(data.get("max_retries", 0)),
            bias=int(data.get("bias", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Strategy:
    mode: str                           # full | shard
    shard_count: int = 1

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Strategy":
        return Strategy(
            mode=str(data["mode"]),
            shard_count=int(data.get("shard_count", 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConfigItem:
    key: str
    value: float | None                 # None for pure derived items
    derived_from: dict[str, float] = field(default_factory=dict)

    def is_derived(self) -> bool:
        return bool(self.derived_from)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"key": self.key}
        if self.value is not None:
            out["value"] = self.value
        if self.derived_from:
            out["derived_from"] = dict(self.derived_from)
        return out


@dataclass(frozen=True)
class ManifestVersion:
    version: int
    items: dict[str, ConfigItem]
    targets: list[str]
    strategy: Strategy

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "items": [item.to_dict() for item in self.items.values()],
            "targets": list(self.targets),
            "strategy": self.strategy.to_dict(),
        }


@dataclass(frozen=True)
class CompiledItem:
    """Per-key immutable compilation artifact.

    The artifact intentionally excludes the manifest version label so that an
    unchanged item compiled in a later version stays byte-identical.
    """
    key: str
    qvalue: int | None
    is_derived: bool
    derived_from_q: dict[str, int]
    targets: tuple[str, ...]
    strategy: dict[str, Any]
    error_bound: dict[str, Any] | None
    artifact_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "qvalue": self.qvalue,
            "is_derived": self.is_derived,
            "derived_from_q": dict(self.derived_from_q),
            "targets": list(self.targets),
            "strategy": dict(self.strategy),
            "error_bound": self.error_bound,
            "artifact_digest": self.artifact_digest,
        }


@dataclass
class Receipt:
    node_id: str
    version: int
    shard_index: int
    shard_count: int
    derived_state: dict[str, int]
    aggregate_digest: str
    compute_ms: int
    attempt: int = 1

    @staticmethod
    def required_fields() -> tuple[str, ...]:
        return (
            "node_id",
            "version",
            "shard_index",
            "shard_count",
            "derived_state",
            "aggregate_digest",
            "compute_ms",
        )

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Receipt":
        return Receipt(
            node_id=str(data["node_id"]),
            version=int(data["version"]),
            shard_index=int(data["shard_index"]),
            shard_count=int(data["shard_count"]),
            derived_state={str(k): int(v) for k, v in data["derived_state"].items()},
            aggregate_digest=str(data["aggregate_digest"]),
            compute_ms=int(data["compute_ms"]),
            attempt=int(data.get("attempt", 1)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
