"""Immutable domain models used by the workbench pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .constants import DEFAULT_THRESHOLDS, FPS, SKELETON
from .errors import BusinessError, ErrorCode


def _readonly(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class Motion:
    seed: int
    frame_count: int
    fps: int
    joint_names: tuple[str, ...]
    quats: np.ndarray  # (frames, joints, 4), wxyz
    roots: np.ndarray  # (frames, 3)
    content_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "joint_names", tuple(self.joint_names))
        object.__setattr__(self, "quats", _readonly(self.quats))
        object.__setattr__(self, "roots", _readonly(self.roots))

    @property
    def joint_count(self) -> int:
        return len(self.joint_names)

    def check_frame(self, frame: int) -> None:
        if not isinstance(frame, int) or frame < 0 or frame >= self.frame_count:
            raise BusinessError(
                ErrorCode.E_FRAME_MISSING,
                "frame index is outside the record",
                {"frame": frame, "frame_count": self.frame_count},
            )

    def frame_hash(self, frame: int) -> str:
        import hashlib

        self.check_frame(frame)
        quat = np.rint(self.quats[frame] * 1_000_000_000.0).astype("<i8").tobytes()
        root = np.rint(self.roots[frame] * 1_000_000_000.0).astype("<i8").tobytes()
        return hashlib.sha256(quat + root).hexdigest()


@dataclass(frozen=True)
class DefectEntry:
    frame: int
    joint: str
    defect_type: str
    segment_id: int = 0

    def key(self) -> tuple[int, str, str]:
        return (self.frame, self.joint, self.defect_type)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "joint": self.joint,
            "type": self.defect_type,
            "segment_id": self.segment_id,
        }


@dataclass(frozen=True)
class Thresholds:
    joint_jump_deg: float = float(DEFAULT_THRESHOLDS["joint_jump_deg"])
    joint_jitter_deg: float = float(DEFAULT_THRESHOLDS["joint_jitter_deg"])
    invalid_norm_min: float = float(DEFAULT_THRESHOLDS["invalid_norm_min"])
    root_slip_m: float = float(DEFAULT_THRESHOLDS["root_slip_m"])

    def to_dict(self) -> dict[str, float]:
        return {
            "joint_jump_deg": self.joint_jump_deg,
            "joint_jitter_deg": self.joint_jitter_deg,
            "invalid_norm_min": self.invalid_norm_min,
            "root_slip_m": self.root_slip_m,
        }

    def hash(self) -> str:
        from .io import sha256_json

        return sha256_json(self.to_dict())


@dataclass(frozen=True)
class VersionStamp:
    sequence: int
    param_version: str
    record_hash: str
    threshold_hash: str
    keyframe_hash: str
    detection_hash: str
    repair_hash: str
    evaluation_hash: str
    content_hash: str

    @property
    def label(self) -> str:
        return f"v{self.sequence:04d}-{self.content_hash[:12]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "label": self.label,
            "param_version": self.param_version,
            "record_hash": self.record_hash,
            "threshold_hash": self.threshold_hash,
            "keyframe_hash": self.keyframe_hash,
            "detection_hash": self.detection_hash,
            "repair_hash": self.repair_hash,
            "evaluation_hash": self.evaluation_hash,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class Evidence:
    """Threshold-independent detector measurements.

    Arrays are read-only and can be reused for instantaneous threshold previews.
    """

    invalid: np.ndarray  # (N, J) bool
    midpoint_residual_deg: np.ndarray  # (N, J) float
    angular_second_diff_deg: np.ndarray  # (N, J) float
    step_prev_deg: np.ndarray
    step_next_deg: np.ndarray
    root_midpoint_residual_m: np.ndarray  # (N,)
    root_step_prev_m: np.ndarray
    evidence_hash: str

    def __post_init__(self) -> None:
        for name in (
            "invalid",
            "midpoint_residual_deg",
            "angular_second_diff_deg",
            "step_prev_deg",
            "step_next_deg",
            "root_midpoint_residual_m",
            "root_step_prev_m",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)))


@dataclass(frozen=True)
class DetectionResult:
    thresholds: Thresholds
    entries: tuple[DefectEntry, ...]
    joint_bad: np.ndarray  # (N,J) bool
    joint_type: np.ndarray  # (N,J) object/str or empty
    root_bad: np.ndarray  # (N,) bool
    root_type: np.ndarray
    comparison: dict[str, Any]
    detection_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "joint_bad", _readonly(self.joint_bad.astype(bool)))
        object.__setattr__(self, "root_bad", _readonly(self.root_bad.astype(bool)))

    @property
    def frames(self) -> set[int]:
        result: set[int] = set()
        for entry in self.entries:
            result.add(entry.frame)
        return result

    def entries_as_dicts(self) -> list[dict[str, Any]]:
        return [entry.to_dict() for entry in self.entries]


@dataclass(frozen=True)
class AnchorModel:
    # Per-joint anchors include automatic defect-boundary anchors and user keys
    # that are clean for that joint. Root has a parallel anchor stream.
    joint_auto: tuple[tuple[int, ...], ...]
    joint_user: tuple[tuple[int, ...], ...]
    root_auto: tuple[int, ...]
    root_user: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_auto": {
                name: list(anchors) for name, anchors in zip(SKELETON, self.joint_auto)
            },
            "joint_user": {
                name: list(anchors) for name, anchors in zip(SKELETON, self.joint_user)
            },
            "root_auto": list(self.root_auto),
            "root_user": list(self.root_user),
        }


@dataclass(frozen=True)
class EvaluationDetail:
    frame: int
    joint: str
    defect_type: str
    joint_angle_error_deg: float | None
    raw_joint_angle_error_deg: float | None
    root_error_m: float | None
    raw_root_error_m: float | None
    trusted: bool
    tolerance: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "joint": self.joint,
            "defect_type": self.defect_type,
            "joint_angle_error_deg": self.joint_angle_error_deg,
            "raw_joint_angle_error_deg": self.raw_joint_angle_error_deg,
            "root_error_m": self.root_error_m,
            "raw_root_error_m": self.raw_root_error_m,
            "trusted": self.trusted,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True)
class EvaluationResult:
    details: tuple[EvaluationDetail, ...]
    interval_metrics: dict[str, dict[str, Any]]
    acceleration_details: dict[str, list[dict[str, Any]]]
    summary: dict[str, Any]
    evaluation_hash: str

    @property
    def trusted(self) -> bool:
        return bool(self.summary.get("trusted", False))


@dataclass(frozen=True)
class RepairResult:
    quats: np.ndarray
    roots: np.ndarray
    frame_hashes: tuple[str, ...]
    changed_joint_frames: tuple[tuple[int, int], ...]
    changed_root_frames: tuple[int, ...]
    repair_hash: str
    policy: dict[str, str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "quats", _readonly(self.quats))
        object.__setattr__(self, "roots", _readonly(self.roots))


@dataclass(frozen=True)
class ClosureReport:
    trigger: str
    changed_keys: tuple[str, ...]
    dirty_joint_coords: tuple[tuple[int, int], ...]
    dirty_root_frames: tuple[int, ...]
    evaluation_intervals: tuple[str, ...]
    graph: dict[str, Any]
    actual_recomputed_joint_coords: tuple[tuple[int, int], ...] = ()
    actual_recomputed_root_frames: tuple[int, ...] = ()
    exact: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "mocap-closure/v1",
            "trigger": self.trigger,
            "changed_keys": list(self.changed_keys),
            "dirty_joint_coords": [
                {"frame": f, "joint": SKELETON[j]} for f, j in self.dirty_joint_coords
            ],
            "dirty_root_frames": list(self.dirty_root_frames),
            "evaluation_intervals": list(self.evaluation_intervals),
            "graph": self.graph,
            "actual_recomputed_joint_coords": [
                {"frame": f, "joint": SKELETON[j]}
                for f, j in self.actual_recomputed_joint_coords
            ],
            "actual_recomputed_root_frames": list(self.actual_recomputed_root_frames),
            "exact": self.exact,
        }


@dataclass(frozen=True)
class ConsistencyReport:
    checked_frames: int
    checked_joint_coords: int
    tolerance: dict[str, float]
    max_quaternion_component_abs: float
    max_joint_angular_deg: float
    max_root_component_abs_m: float
    mismatch_count: int
    first_mismatch: dict[str, Any] | None
    incremental_hash: str
    full_hash: str
    consistent: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "mocap-consistency/v1",
            "checked_frames": self.checked_frames,
            "checked_joint_coords": self.checked_joint_coords,
            "tolerance": self.tolerance,
            "max_quaternion_component_abs": self.max_quaternion_component_abs,
            "max_joint_angular_deg": self.max_joint_angular_deg,
            "max_root_component_abs_m": self.max_root_component_abs_m,
            "mismatch_count": self.mismatch_count,
            "first_mismatch": self.first_mismatch,
            "incremental_hash": self.incremental_hash,
            "full_hash": self.full_hash,
            "consistent": self.consistent,
            "comparison_policy": (
                "Every frame and every joint is compared; no RMS-only shortcut. A "
                "coordinate fails on quaternion component >1e-9 or normalized angular >1e-5 deg; "
                "root fails on component >1e-9 m."
            ),
        }


@dataclass(frozen=True)
class Snapshot:
    record: Motion
    truth: Motion
    thresholds: Thresholds
    keyframes: frozenset[int]
    evidence: Evidence
    detection: DetectionResult
    anchors: AnchorModel
    repair: RepairResult
    evaluation: EvaluationResult
    version: VersionStamp
    suppressed_auto: frozenset[int] = frozenset()
    closure: ClosureReport | None = None
    consistency: ConsistencyReport | None = None
    committed: bool = False

    @property
    def frame_count(self) -> int:
        return self.record.frame_count

    @property
    def content_hash(self) -> str:
        return self.version.content_hash

    def require_base(self, expected_content_hash: str) -> None:
        if self.content_hash != expected_content_hash:
            raise BusinessError(
                ErrorCode.E_SNAPSHOT_STALE,
                "snapshot base version no longer matches the current view",
                {
                    "expected": expected_content_hash,
                    "actual": self.content_hash,
                    "version": self.version.label,
                },
            )

    def to_meta_dict(self) -> dict[str, Any]:
        return {
            "version": self.version.to_dict(),
            "thresholds": self.thresholds.to_dict(),
            "keyframes": sorted(self.keyframes),
            "suppressed_auto": sorted(self.suppressed_auto),
            "trusted": self.evaluation.trusted,
            "publication_status": (
                "TRUSTED" if self.evaluation.trusted else "WITHHELD_UNTRUSTED"
            ),
            "closure": self.closure.to_dict() if self.closure else None,
            "consistency": self.consistency.to_dict() if self.consistency else None,
        }
