"""Frozen, hashable data models shared by every layer."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .fixedpoint import FixedConfig


@dataclass(frozen=True)
class Sample:
    sample_id: str
    features: Tuple[float, ...]
    label: Optional[int]

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "features": list(self.features),
            "label": self.label,
        }


@dataclass(frozen=True)
class StageDef:
    name: str
    kind: str
    upstream: Tuple[str, ...]
    params: Dict[str, Any]
    order: int

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "upstream": list(self.upstream),
            "params": self.params,
            "order": self.order,
        }


@dataclass(frozen=True)
class Pipeline:
    name: str
    feature_count: int
    stages: Tuple[StageDef, ...]

    def stage(self, name: str) -> StageDef:
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(name)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "feature_count": self.feature_count,
            "stages": [stage.to_dict() for stage in self.stages],
        }


@dataclass(frozen=True)
class Budget:
    batch_points: int
    stage_quotas: Dict[str, int]

    def to_dict(self) -> dict:
        return {"batch_points": self.batch_points, "stage_quotas": dict(self.stage_quotas)}


@dataclass(frozen=True)
class Policy:
    # When receipts lack digests, never reuse (strict) or compare param versions.
    degraded_reuse: str

    def to_dict(self) -> dict:
        return {"degraded_reuse": self.degraded_reuse}


@dataclass(frozen=True)
class Config:
    fixed: FixedConfig
    budget: Budget
    policy: Policy
    seed: int

    def to_dict(self) -> dict:
        return {
            "fixed": {"frac_bits": self.fixed.frac_bits, "gear": self.fixed.gear},
            "budget": self.budget.to_dict(),
            "policy": self.policy.to_dict(),
            "seed": self.seed,
        }


@dataclass(frozen=True)
class ReceiptRecord:
    """Normalized representation of one cache receipt line."""

    source_version: int
    wire_version: int
    line_no: int
    artifact_id: str
    hit_permille: int
    present: Dict[str, bool]
    param_version: Optional[int] = None
    elapsed_ms: Optional[int] = None
    digest: Optional[str] = None
    duplicate_refs: Tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "source_version": self.source_version,
            "wire_version": self.wire_version,
            "line_no": self.line_no,
            "artifact_id": self.artifact_id,
            "hit_permille": self.hit_permille,
            "param_version": self.param_version,
            "elapsed_ms": self.elapsed_ms,
            "digest": self.digest,
            "duplicate_refs": list(self.duplicate_refs),
            "present": dict(self.present),
        }


@dataclass(frozen=True)
class Artifact:
    artifact_id: str
    stage: str
    scope: str  # "sample:<id>" or "global"
    payload: Any
    digest: str
    upstream: Tuple[str, ...]
    params: Dict[str, Any]
    param_versions: Dict[str, int]
    quant_step: int
    residual: int

    def to_dict(self) -> dict:
        return {
            "artifact_id": self.artifact_id,
            "stage": self.stage,
            "scope": self.scope,
            "payload": _jsonable(self.payload),
            "digest": self.digest,
            "upstream": list(self.upstream),
            "params": self.params,
            "param_versions": dict(self.param_versions),
            "quant_step": self.quant_step,
            "residual": self.residual,
        }


@dataclass(frozen=True)
class LedgerEntry:
    """One reviewable reuse/recompute decision."""

    order: int
    sort_key: str
    artifact_id: str
    stage: str
    scope: str
    decision: str  # REUSE / RECOMPUTE / PENDING / STAGE_ABORTED
    reason: str
    receipt_version: Optional[int]
    digest_match: Optional[bool]
    degraded: bool
    degradation_basis: Optional[str]
    upstream: Tuple[str, ...]
    params: Dict[str, Any]
    points: int
    points_after: int
    stage_points_after: int

    def to_dict(self) -> dict:
        data = self.__dict__.copy()
        data["upstream"] = list(self.upstream)
        data["params"] = dict(self.params)
        return data


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite values cannot be serialized")
    return value
