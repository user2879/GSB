"""Typed records exchanged between pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CompatRecord:
    shard_id: str
    codebook_version: str
    grid_version: str
    semantic_mapping: dict[str, str]
    unmapped_codes: list[str]
    resampled: bool
    resample_residuals: dict[str, float]
    native_frame_shift_ms: int
    target_frame_shift_ms: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "codebook_version": self.codebook_version,
            "grid_version": self.grid_version,
            "semantic_mapping": dict(sorted(self.semantic_mapping.items())),
            "unmapped_codes": sorted(self.unmapped_codes),
            "resampled": self.resampled,
            "resample_residuals": dict(sorted(self.resample_residuals.items())),
            "native_frame_shift_ms": self.native_frame_shift_ms,
            "target_frame_shift_ms": self.target_frame_shift_ms,
        }


@dataclass
class LoadedShard:
    shard_id: str
    start_ms: int
    end_ms: int
    first_frame: int
    last_frame: int
    matrix: Any  # numpy array on the global grid, canonical class columns
    compat: CompatRecord
    unavailable_class_frames: dict[str, list[int]]
    duplicate_frame_kept: str | None


@dataclass
class BoundaryDecision:
    boundary_id: str
    sort_key_ms: int
    left_shard_id: str
    right_shard_id: str
    boundary_ms: int
    left_mean: dict[str, float]
    right_mean: dict[str, float]
    mean_diff: dict[str, float]
    criterion: str
    continuous: bool
    dominant_class: str | None
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary_id": self.boundary_id,
            "sort_key_ms": self.sort_key_ms,
            "left_shard_id": self.left_shard_id,
            "right_shard_id": self.right_shard_id,
            "boundary_ms": self.boundary_ms,
            "left_mean": _round_map(self.left_mean),
            "right_mean": _round_map(self.right_mean),
            "mean_diff": _round_map(self.mean_diff),
            "criterion": self.criterion,
            "continuous": self.continuous,
            "dominant_class": self.dominant_class,
            "evidence": self.evidence,
        }


@dataclass
class Trajectory:
    trajectory_id: str
    sort_key_frame: int
    event_class: str
    first_frame: int
    last_frame: int
    start_ms: int
    end_ms: int
    source_shard_ids: list[str]
    crossed_boundaries: list[str]
    boundary_evidence: list[dict[str, Any]]
    smoothing: dict[str, Any]
    owning_shard_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "sort_key_frame": self.sort_key_frame,
            "class": self.event_class,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "source_shard_ids": list(self.source_shard_ids),
            "crossed_boundaries": list(self.crossed_boundaries),
            "boundary_evidence": list(self.boundary_evidence),
            "smoothing": dict(self.smoothing),
            "owning_shard_id": self.owning_shard_id,
        }


def _round_map(values: dict[str, float], digits: int = 6) -> dict[str, float]:
    return {key: round(float(values[key]), digits) for key in sorted(values)}


def boundary_from_dict(row: dict[str, Any]) -> "BoundaryDecision":
    return BoundaryDecision(
        boundary_id=row["boundary_id"],
        sort_key_ms=row["sort_key_ms"],
        left_shard_id=row["left_shard_id"],
        right_shard_id=row["right_shard_id"],
        boundary_ms=row["boundary_ms"],
        left_mean=dict(row["left_mean"]),
        right_mean=dict(row["right_mean"]),
        mean_diff=dict(row["mean_diff"]),
        criterion=row["criterion"],
        continuous=row["continuous"],
        dominant_class=row["dominant_class"],
        evidence=dict(row["evidence"]),
    )


def trajectory_from_dict(row: dict[str, Any]) -> "Trajectory":
    return Trajectory(
        trajectory_id=row["trajectory_id"],
        sort_key_frame=row["sort_key_frame"],
        event_class=row["class"],
        first_frame=row["first_frame"],
        last_frame=row["last_frame"],
        start_ms=row["start_ms"],
        end_ms=row["end_ms"],
        source_shard_ids=list(row["source_shard_ids"]),
        crossed_boundaries=list(row["crossed_boundaries"]),
        boundary_evidence=list(row["boundary_evidence"]),
        smoothing=dict(row["smoothing"]),
        owning_shard_id=row["owning_shard_id"],
    )
