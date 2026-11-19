"""Segmentation configuration model with deterministic canonical hashing.

The configuration is pure JSON-serialisable data. Fingerprints are computed
from a canonical JSON encoding (sorted keys, fixed separators) so that
semantically identical configurations always produce identical hashes,
independent of dict insertion order.
"""

from __future__ import annotations

import copy
import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import BusinessError, ErrorCode

# Class encoding order is part of the deterministic tie-break contract.
CLASS_CODES = (1, 2, 3, 4)
CLASS_NAMES = {1: "farmland", 2: "water", 3: "building", 4: "road"}
BAND_ORDER = ("red", "green", "nir", "elevation")
FUSION_STRATEGIES = ("distance_weighted", "first_wins")


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_fingerprint(data: Any) -> str:
    payload = canonical_json(data).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class SegmentationConfig:
    window_size: int
    stride: int
    context_overlap: int
    feature_thresholds: dict[str, float]
    class_weights: dict[str, dict[str, float]]
    min_component_area: int
    fusion_strategy: str
    recompute_budget: int
    workers: int
    snapshot_commit_interval: int

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "SegmentationConfig":
        try:
            cfg = cls(
                window_size=int(raw["window_size"]),
                stride=int(raw["stride"]),
                context_overlap=int(raw["context_overlap"]),
                feature_thresholds=dict(raw["feature_thresholds"]),
                class_weights={str(k): dict(v) for k, v in raw["class_weights"].items()},
                min_component_area=int(raw["min_component_area"]),
                fusion_strategy=str(raw["fusion_strategy"]),
                recompute_budget=int(raw["recompute_budget"]),
                workers=int(raw["workers"]),
                snapshot_commit_interval=int(raw["snapshot_commit_interval"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise BusinessError(ErrorCode.E_CONFIG_INVALID, f"malformed config: {exc}") from exc
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.window_size <= 0 or self.stride <= 0 or self.context_overlap < 0:
            raise BusinessError(ErrorCode.E_CONFIG_INVALID,
                                "window_size/stride must be positive, context_overlap non-negative")
        if self.context_overlap >= self.window_size:
            raise BusinessError(ErrorCode.E_CONFIG_INVALID, "context_overlap must be smaller than window_size")
        if self.workers < 1 or self.snapshot_commit_interval < 1:
            raise BusinessError(ErrorCode.E_CONFIG_INVALID, "workers and commit interval must be >= 1")
        if self.recompute_budget < 0:
            raise BusinessError(ErrorCode.E_CONFIG_INVALID, "recompute_budget must be >= 0")
        if self.min_component_area < 1:
            raise BusinessError(ErrorCode.E_CONFIG_INVALID, "min_component_area must be >= 1")
        if self.fusion_strategy not in FUSION_STRATEGIES:
            raise BusinessError(ErrorCode.E_CONFIG_INVALID,
                                f"fusion_strategy must be one of {FUSION_STRATEGIES}")
        required_features = {"ndvi", "ndwi", "edge", "slope"}
        if not required_features.issubset(self.feature_thresholds):
            raise BusinessError(ErrorCode.E_CONFIG_INVALID,
                                f"feature_thresholds must contain {sorted(required_features)}")
        for code in CLASS_CODES:
            name = CLASS_NAMES[code]
            if name not in self.class_weights:
                raise BusinessError(ErrorCode.E_CONFIG_INVALID, f"missing class weights for {name}")
            if not required_features.issubset(self.class_weights[name]):
                raise BusinessError(ErrorCode.E_CONFIG_INVALID,
                                    f"weights for {name} must cover {sorted(required_features)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_size": self.window_size,
            "stride": self.stride,
            "context_overlap": self.context_overlap,
            "feature_thresholds": copy.deepcopy(self.feature_thresholds),
            "class_weights": copy.deepcopy(self.class_weights),
            "min_component_area": self.min_component_area,
            "fusion_strategy": self.fusion_strategy,
            "recompute_budget": self.recompute_budget,
            "workers": self.workers,
            "snapshot_commit_interval": self.snapshot_commit_interval,
        }

    def fingerprint(self) -> str:
        return stable_fingerprint(self.to_dict())


def load_config(path: str | Path) -> SegmentationConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return SegmentationConfig.from_dict(raw)
