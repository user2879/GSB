"""Detection / matching / injection configuration loading and validation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from .errors import DetboxError, E_WEIGHT_NONFINITE, E_CAND_FORMAT
from .models import ClassTable
from .util import read_json, is_finite_number


DEFAULT_CLASS_TABLE = ClassTable(
    {0: "circle", 1: "square", 2: "triangle"}
)


@dataclass
class DetectionParams:
    window_sizes: list[int]
    step: int
    color_mean_threshold: float
    edge_density_threshold: float
    score_weights: dict[str, float]
    pool_capacity: int

    @classmethod
    def load(cls, path: str | Path) -> "DetectionParams":
        data = read_json(path)
        required = [
            "window_sizes",
            "step",
            "color_mean_threshold",
            "edge_density_threshold",
            "score_weights",
            "pool_capacity",
        ]
        missing = [key for key in required if key not in data]
        if missing:
            raise DetboxError(f"detection params missing keys: {missing}", code=E_CAND_FORMAT)
        weights = data["score_weights"]
        if not isinstance(weights, dict) or not weights:
            raise DetboxError("score_weights must be a non-empty object", code=E_WEIGHT_NONFINITE)
        for name, value in weights.items():
            if not is_finite_number(value):
                raise DetboxError(
                    f"non-finite score weight: {name}={value!r}", code=E_WEIGHT_NONFINITE
                )
        numeric_fields = [
            ("step", data["step"]),
            ("color_mean_threshold", data["color_mean_threshold"]),
            ("edge_density_threshold", data["edge_density_threshold"]),
            ("pool_capacity", data["pool_capacity"]),
        ]
        for name, value in numeric_fields:
            if not is_finite_number(value):
                raise DetboxError(f"non-finite parameter: {name}", code=E_WEIGHT_NONFINITE)
        windows = data["window_sizes"]
        if not isinstance(windows, list) or not windows:
            raise DetboxError("window_sizes must be a non-empty list", code=E_CAND_FORMAT)
        for size in windows:
            if not is_finite_number(size) or int(size) <= 0:
                raise DetboxError("invalid window size", code=E_CAND_FORMAT)
        if int(data["step"]) <= 0 or int(data["pool_capacity"]) <= 0:
            raise DetboxError("step and pool_capacity must be positive", code=E_CAND_FORMAT)
        return cls(
            window_sizes=[int(v) for v in windows],
            step=int(data["step"]),
            color_mean_threshold=float(data["color_mean_threshold"]),
            edge_density_threshold=float(data["edge_density_threshold"]),
            score_weights={str(k): float(v) for k, v in weights.items()},
            pool_capacity=int(data["pool_capacity"]),
        )

    def to_dict(self) -> dict:
        return {
            "window_sizes": list(self.window_sizes),
            "step": self.step,
            "color_mean_threshold": self.color_mean_threshold,
            "edge_density_threshold": self.edge_density_threshold,
            "score_weights": dict(self.score_weights),
            "pool_capacity": self.pool_capacity,
        }


@dataclass
class MatchConfig:
    iou_threshold: float
    max_gap_frames: int
    eviction_rule: str

    @classmethod
    def load(cls, path: str | Path) -> "MatchConfig":
        data = read_json(path)
        required = ["iou_threshold", "max_gap_frames", "eviction_rule"]
        missing = [key for key in required if key not in data]
        if missing:
            raise DetboxError(f"match config missing keys: {missing}", code=E_CAND_FORMAT)
        if not is_finite_number(data["iou_threshold"]):
            raise DetboxError("non-finite iou_threshold", code=E_WEIGHT_NONFINITE)
        iou = float(data["iou_threshold"])
        if not 0.0 < iou <= 1.0:
            raise DetboxError("iou_threshold out of (0, 1]", code=E_CAND_FORMAT)
        gap = int(data["max_gap_frames"])
        if gap < 0:
            raise DetboxError("max_gap_frames must be >= 0", code=E_CAND_FORMAT)
        rule = str(data["eviction_rule"])
        if rule not in ("score_desc_coord_lex",):
            raise DetboxError(f"unsupported eviction rule: {rule}", code=E_CAND_FORMAT)
        return cls(iou_threshold=iou, max_gap_frames=gap, eviction_rule=rule)

    def to_dict(self) -> dict:
        return {
            "iou_threshold": self.iou_threshold,
            "max_gap_frames": self.max_gap_frames,
            "eviction_rule": self.eviction_rule,
        }


@dataclass
class Injection:
    injection_id: str
    sequence: str
    frame: int
    stage: str
    fault_type: str


def load_injections(path: str | Path | None) -> list[Injection]:
    if path is None or not Path(path).exists():
        return []
    data = read_json(path)
    items = data.get("injections", [])
    result: list[Injection] = []
    for item in items:
        result.append(
            Injection(
                injection_id=str(item["injection_id"]),
                sequence=str(item["sequence"]),
                frame=int(item["frame"]),
                stage=str(item["stage"]),
                fault_type=str(item["fault_type"]),
            )
        )
    return result

