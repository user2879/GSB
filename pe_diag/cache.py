"""Content-fingerprint cache keys and precise invalidation.

Every cached artifact is keyed by the exact tuple of factors that can change
it, so an edit only invalidates the affected sample/stage:

* scheme parameter fingerprint (the stage-relevant :class:`EffectiveConfig`
  subset);
* sample content fingerprint (id, tier, positions, features);
* precision tier;
* scale-tier identifier (also embedded in the config subset);
* the selected strategy decision fingerprint (active dimension / block method).
"""

from __future__ import annotations

from dataclasses import dataclass

from .common import content_hash, short_hash, stable_task_key
from .config import EffectiveConfig, Sample

STAGES = ("encode", "score", "error", "sensitivity")


def scheme_param_fingerprint(config: EffectiveConfig, stage: str) -> str:
    return short_hash(config.param_subset(stage))


def sample_fingerprint(sample: Sample) -> str:
    return short_hash(sample.content_dict())


def decision_fingerprint(decision: dict) -> str:
    return short_hash(decision)


def make_cache_key(stage: str, scheme: str, config: EffectiveConfig,
                   sample: Sample | None, precision: str,
                   decision: dict, sample_fp: str | None = None) -> str:
    parts = {
        "stage": stage,
        "scheme": scheme,
        "scheme_param_fp": scheme_param_fingerprint(config, stage),
        "tier_id": config.tier_id,
        "precision": precision,
        "decision_fp": decision_fingerprint(decision),
    }
    if sample is not None:
        parts["sample_fp"] = (
            sample_fp if sample_fp is not None else sample_fingerprint(sample))
        parts["sample_id"] = sample.sample_id
    return content_hash(parts)


def task_key(scheme: str, sample_id: str, stage: str) -> str:
    return stable_task_key(scheme, sample_id, stage)


@dataclass
class CacheAudit:
    reused: list
    recomputed: list
    invalidated: list

    def to_dict(self) -> dict:
        return {
            "reused": sorted(self.reused),
            "recomputed": sorted(self.recomputed),
            "invalidated": sorted(self.invalidated),
            "counts": {
                "reused": len(self.reused),
                "recomputed": len(self.recomputed),
                "invalidated": len(self.invalidated),
            },
        }
