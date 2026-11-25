"""Content-fingerprint cache keys and precise invalidation boundaries."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json


from quantcalib.fingerprint import fingerprint, fingerprint_json


@dataclass
class CacheKeys:
    layer_content: str
    group_content: str
    params: str
    quantization_level: str
    calibration_set: str

    def stage_key(self, stage: str, group_granularity: int | None = None) -> str:
        payload = {
            "stage": stage,
            "layer_content": self.layer_content,
            "group_content": self.group_content,
            "params": self.params,
            "quantization_level": self.quantization_level,
            "calibration_set": self.calibration_set,
        }
        if group_granularity is not None:
            payload["group_granularity"] = group_granularity
        return fingerprint_json(payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_layer_cache_keys(layer, group_index: int, cfg,
                           calib_fingerprint: str) -> CacheKeys:
    flat = layer.weights.reshape(-1)
    start = group_index * layer.group_granularity
    group_values = flat[start:start + layer.group_granularity]

    layer_fp = fingerprint(layer.weights)
    group_fp = fingerprint(group_values)
    # Granularity is excluded: it partitions the layer, while group_content
    # already identifies the exact bytes.
    param_fp = fingerprint_json({
        "scheme": cfg.scheme,
        "q_lo": cfg.q_lo_permille,
        "q_hi": cfg.q_hi_permille,
        "fractional_bits": cfg.fractional_bits,
        "rounding_mode": cfg.rounding_mode,
        "saturation_policy": cfg.saturation_policy,
        "tolerance": cfg.error_tolerance,
    })
    level_fp = fingerprint_json({
        "bits": cfg.layer_bits(layer.name),
        "levels": cfg.quantization_levels,
    })
    return CacheKeys(
        layer_content=layer_fp,
        group_content=group_fp,
        params=param_fp,
        quantization_level=level_fp,
        calibration_set=calib_fingerprint,
    )


class ReuseLedger:
    """Records hit/recompute decisions and upstream references per stage."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def record(self, layer: str, group_index: int, stage: str,
               key: str, hit: bool, upstream: list[str],
               calibration_refs: list[int]) -> None:
        self.records.append({
            "layer": layer,
            "group_index": group_index,
            "stage": stage,
            "cache_key": key,
            "hit": bool(hit),
            "upstream": list(upstream),
            "calibration_refs": list(calibration_refs),
        })

    def hits(self) -> list[dict[str, Any]]:
        return [r for r in self.records if r["hit"]]

    def recomputes(self) -> list[dict[str, Any]]:
        return [r for r in self.records if not r["hit"]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "records": self.records,
            "hit_count": len(self.hits()),
            "recompute_count": len(self.recomputes()),
        }


def invalidation_boundary(layer, group_index: int,
                          same_content_existed: bool,
                          new_keys: CacheKeys,
                          prior_same_content_keys: CacheKeys | None) -> list[str]:
    """Return stages invalidated by a content/config change.

    ``same_content_existed`` says whether any prior artifact stored these exact
    group bytes. When false the group is brand new (e.g. replaced weights or a
    new partition) and the full chain recomputes. When true, only the stages
    whose parameter/level fingerprints changed are invalidated.
    """
    if not same_content_existed or prior_same_content_keys is None:
        return ["calibration", "quantization", "error", "distillation", "receipt"]

    stages: set[str] = set()
    if prior_same_content_keys.params != new_keys.params:
        stages.update(["calibration", "quantization", "error"])
    if prior_same_content_keys.quantization_level != new_keys.quantization_level:
        stages.update(["quantization", "error"])
    if prior_same_content_keys.calibration_set != new_keys.calibration_set:
        stages.update(["distillation", "receipt"])
    if stages:
        stages.update(["distillation", "receipt"])
    return sorted(stages)


def load_prior_group_keys(stage_dir: str | Path, layer_name: str,
                          group_index: int, group_fingerprint: str):
    """Load all stored CacheKeys whose group bytes match the current group."""
    stage_dir = Path(stage_dir)
    prefix = f"{layer_name}__g{group_index:04d}__"
    matched: list[CacheKeys] = []
    for path in sorted(stage_dir.glob(prefix + "*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        keys = CacheKeys(**payload["cache_keys"])
        if keys.group_content == group_fingerprint:
            matched.append(keys)
    return matched
