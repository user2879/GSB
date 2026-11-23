"""Classification / scheduling configuration loading and validation."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from . import errors

CLASSES = ["farmland", "water", "building", "road"]
CLASS_CODE = {name: idx for idx, name in enumerate(CLASSES)}
FEATURES = ["veg", "water", "texture", "slope"]

REQUIRED_FIELDS = [
    "window_size", "stride", "context_overlap", "index_weights",
    "min_connected_pixels", "fusion_strategy_version",
    "declared_fusion_versions", "fixed_point_frac_bits", "quantum",
    "thread_pool_size", "global_pixel_budget", "per_tile_quota",
    "merge_batch_limit", "snapshot_commit_interval",
    "declared_band_set_versions",
]


@dataclass
class Config:
    window_size: int
    stride: int
    context_overlap: int
    index_weights: dict  # class -> {feature: int}
    min_connected_pixels: int
    fusion_strategy_version: str
    declared_fusion_versions: list
    fixed_point_frac_bits: int
    quantum: int
    thread_pool_size: int
    global_pixel_budget: int
    per_tile_quota: int
    merge_batch_limit: int
    snapshot_commit_interval: int
    declared_band_set_versions: list
    raw: dict = field(default_factory=dict)

    def weight_vector(self, class_name: str):
        weights = self.index_weights[class_name]
        return [int(weights[feat]) for feat in FEATURES]


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(config: Config) -> str:
    return hashlib.sha256(canonical_json(config.raw).encode("utf-8")).hexdigest()


def weights_fingerprint(config: Config) -> str:
    payload = canonical_json(config.index_weights)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def validate_config_dict(raw: dict, tile_size: int | None = None):
    """Return list of (code, detail) errors; empty means valid."""
    errs = []
    for name in REQUIRED_FIELDS:
        if name not in raw:
            errs.append((errors.E_CONFIG_INVALID, f"missing field {name}"))
    if errs:
        return errs
    if int(raw["window_size"]) <= 0 or int(raw["stride"]) <= 0:
        errs.append((errors.E_CONFIG_INVALID, "window_size and stride must be positive"))
    if int(raw["context_overlap"]) < 0:
        errs.append((errors.E_CONFIG_INVALID, "context_overlap must be >= 0"))
    if int(raw["min_connected_pixels"]) < 0:
        errs.append((errors.E_CONFIG_INVALID, "min_connected_pixels must be >= 0"))
    if int(raw["fixed_point_frac_bits"]) <= 0 or int(raw["quantum"]) <= 0:
        errs.append((errors.E_CONFIG_INVALID, "fixed point params must be positive"))
    if raw["fusion_strategy_version"] not in raw["declared_fusion_versions"]:
        errs.append((errors.E_CONFIG_INVALID, "fusion_strategy_version not declared"))
    for cls in CLASSES:
        if cls not in raw["index_weights"]:
            errs.append((errors.E_CONFIG_INVALID, f"index_weights missing class {cls}"))
            continue
        for feat in FEATURES:
            if feat not in raw["index_weights"][cls]:
                errs.append((errors.E_CONFIG_INVALID, f"index_weights[{cls}] missing {feat}"))
    if tile_size is not None and int(raw["window_size"]) > 0:
        if tile_size % int(raw["window_size"]) != 0:
            errs.append((errors.E_CONFIG_INVALID, "window_size does not divide tile_size"))
    return errs


def load_config(path, tile_size: int | None = None) -> Config:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    errs = validate_config_dict(raw, tile_size=tile_size)
    if errs:
        raise ValueError("; ".join(f"{code}: {detail}" for code, detail in errs))
    return Config(
        window_size=int(raw["window_size"]),
        stride=int(raw["stride"]),
        context_overlap=int(raw["context_overlap"]),
        index_weights=raw["index_weights"],
        min_connected_pixels=int(raw["min_connected_pixels"]),
        fusion_strategy_version=raw["fusion_strategy_version"],
        declared_fusion_versions=list(raw["declared_fusion_versions"]),
        fixed_point_frac_bits=int(raw["fixed_point_frac_bits"]),
        quantum=int(raw["quantum"]),
        thread_pool_size=int(raw["thread_pool_size"]),
        global_pixel_budget=int(raw["global_pixel_budget"]),
        per_tile_quota=int(raw["per_tile_quota"]),
        merge_batch_limit=int(raw["merge_batch_limit"]),
        snapshot_commit_interval=int(raw["snapshot_commit_interval"]),
        declared_band_set_versions=list(raw["declared_band_set_versions"]),
        raw=raw,
    )
