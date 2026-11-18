"""Immutable domain models for tiers, objects, rules and manifests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class Tier:
    tier_id: str
    role: str  # "hot" or "cold"
    capacity_bytes: int
    unit_cost: int
    read_weight: int
    write_weight: int
    evict_priority: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier_id": self.tier_id,
            "role": self.role,
            "capacity_bytes": self.capacity_bytes,
            "unit_cost": self.unit_cost,
            "read_weight": self.read_weight,
            "write_weight": self.write_weight,
            "evict_priority": self.evict_priority,
        }


@dataclass(frozen=True)
class Obj:
    key: str
    size_bytes: int
    last_access_ms: int
    access_count: int
    digest: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "size_bytes": self.size_bytes,
            "last_access_ms": self.last_access_ms,
            "access_count": self.access_count,
            "digest": self.digest,
        }


@dataclass(frozen=True)
class Rule:
    hot_watermark_bytes: int
    cold_reclaim_bytes: int
    batch_size: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hot_watermark_bytes": self.hot_watermark_bytes,
            "cold_reclaim_bytes": self.cold_reclaim_bytes,
            "batch_size": self.batch_size,
        }


@dataclass(frozen=True)
class Manifest:
    policy_version: str
    tier_param_version: str
    tiers: List[Tier]
    objects: List[Obj]
    rule: Rule
    injection_sequence: List[str]

    def tier_by_id(self, tier_id: str) -> Tier:
        for tier in self.tiers:
            if tier.tier_id == tier_id:
                return tier
        raise KeyError(tier_id)

    def object_by_key(self, key: str) -> Obj:
        for obj in self.objects:
            if obj.key == key:
                return obj
        raise KeyError(key)

    def hot(self) -> Tier:
        return next(tier for tier in self.tiers if tier.role == "hot")

    def cold(self) -> Tier:
        return next(tier for tier in self.tiers if tier.role == "cold")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "tier_param_version": self.tier_param_version,
            "tiers": [tier.to_dict() for tier in self.tiers],
            "objects": [obj.to_dict() for obj in self.objects],
            "rule": self.rule.to_dict(),
            "injection_sequence": list(self.injection_sequence),
        }
