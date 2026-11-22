"""Plain data models shared by loading, compilation, dispatch and ledger."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

STRATEGIES = ("full", "shard")


@dataclass(frozen=True)
class NodeDef:
    node_id: str
    host: str
    port: int
    per_node_quota: int
    dispatch_timeout_ms: int
    retry_policy: str  # "none" | "retry"
    max_retries: int
    fail_mode: str = "none"  # none | timeout | exit | bad_reply | quant_mismatch

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShardDecl:
    path: str
    length: int
    shards: Tuple[Dict[str, int], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "length": self.length,
            "shards": [dict(s) for s in self.shards],
        }


@dataclass(frozen=True)
class DerivedSpec:
    # Ordered list of (upstream path, decimal coefficient as written).
    sources: Tuple[Tuple[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {"sources": [[p, c] for p, c in self.sources]}


@dataclass(frozen=True)
class PathEntry:
    path: str
    length: int
    weight: Any
    shards: Tuple[Dict[str, int], ...]
    targets: Tuple[str, ...]
    strategy: str
    priority: int
    derived: Optional[DerivedSpec]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "length": self.length,
            "weight": self.weight,
            "shards": [dict(s) for s in self.shards],
            "targets": list(self.targets),
            "strategy": self.strategy,
            "priority": self.priority,
            "derived": None if self.derived is None else self.derived.to_dict(),
        }


@dataclass(frozen=True)
class VersionManifest:
    version: int
    entries: Tuple[PathEntry, ...]

    def entry_map(self) -> Dict[str, PathEntry]:
        return {entry.path: entry for entry in self.entries}


@dataclass(frozen=True)
class CompiledPath:
    """Byte-stable compilation artifact for one path."""

    path: str
    version: int
    strategy: str
    priority: int
    targets: Tuple[str, ...]
    shards: Tuple[Dict[str, int], ...]
    weight_q: int
    weight_single: int
    coeff_q: Tuple[Tuple[str, int, int], ...]  # (source, coeff_q, single)
    derived_from: Tuple[str, ...]
    content_hash: int
    digest: int
    error: Dict[str, int]
    saturated: bool
    artifact_digest: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlanItem:
    item_id: str  # stable "{version}:{path}:{node}"
    version: int
    path: str
    node_id: str
    strategy: str
    priority: int
    expected_digest: int
    shard_hashes: Tuple[int, ...]
    shard_hit_digest: int
    expected_error: Dict[str, int]
    quant_version: str
    compiled_artifact_digest: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Receipt:
    item_id: str
    node_id: str
    version: int
    path: str
    derived_digest: int
    shard_hit_digest: int
    cost_ms: int
    quant_version: str
    attempt: int
    status: str  # ok | timeout | error
    error_code: Optional[str] = None
    waited_ms: int = 0
    service_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NodeOutcome:
    node_id: str
    status: str  # success | timeout | failed | not_dispatched | unavailable
    ok: int = 0
    timeout: int = 0
    failed: int = 0
    not_dispatched: int = 0
    attempts: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
