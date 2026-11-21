"""核心数据模型。"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass(frozen=True)
class Tile:
    tile_id: str
    scene_id: str
    sensor_id: str
    band: int
    timestamp_ms: int
    counts: tuple[int, ...]
    rows: int
    cols: int
    saturated_flags: tuple[bool, ...]
    fp_override: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CalibRecord:
    tile_id: str
    scene_id: str
    sensor_id: str
    band: int
    fixed_values: list[int]
    saturation_count: int
    pixel_count: int
    key: str
    upstream_fingerprint: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RegressionModel:
    model_id: str
    source_sensor: str
    reference_sensor: str
    band: int
    feature_orders: list[int]
    weights: list[float]
    regularization: float
    near_singular: bool
    condition_number: float | None
    sample_count: int
    key: str
    referenced_sensors: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AlignmentRecord:
    tile_id: str
    scene_id: str
    source_sensor: str
    reference_sensor: str
    band: int
    aligned_values: list[int]
    residual_mean: float
    residual_max: float
    rmse: float
    model_id: str
    model_key: str
    key: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LedgerEntry:
    batch_id: int
    tile_id: str
    band: int
    pixels: int
    cumulative_pixels: int
    budget_grant: int
    status: str  # PROCESSED / PENDING / REUSED / FAILED

    def to_dict(self) -> dict:
        return asdict(self)
