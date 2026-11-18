"""Immutable domain values for samples, intervals and profiling policy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .errors import ParameterError

CURRENT_METRIC_VERSION = "self-leaf-cum-presence-v1"
SNAPSHOT_SCHEMA_VERSION = "slowprof.snapshot.v1"
INTERVAL_BOUNDS = "inclusive"

_FRAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*\.[A-Za-z_][A-Za-z0-9_-]*$")


def is_int(value: Any) -> bool:
    return type(value) is int


def frame_is_valid(frame: Any) -> bool:
    return isinstance(frame, str) and _FRAME_RE.fullmatch(frame) is not None


@dataclass(frozen=True, slots=True)
class Interval:
    """Inclusive millisecond interval ``[start_ms, end_ms]``."""

    start_ms: int
    end_ms: int
    interval_version: str = "obs-v1"

    def __post_init__(self) -> None:
        if not is_int(self.start_ms):
            raise ParameterError("start_ms must be an integer", field="start_ms")
        if not is_int(self.end_ms):
            raise ParameterError("end_ms must be an integer", field="end_ms")
        if self.start_ms > self.end_ms:
            raise ParameterError("start_ms must be <= end_ms", field="start_ms")
        if not isinstance(self.interval_version, str) or not self.interval_version:
            raise ParameterError("interval_version must be a non-empty string", field="interval_version")

    def includes(self, timestamp_ms: int) -> bool:
        return self.start_ms <= timestamp_ms <= self.end_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "interval_version": self.interval_version,
            "bounds": INTERVAL_BOUNDS,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Interval":
        if not isinstance(value, dict):
            raise ParameterError("interval must be an object", field="interval")
        for key in ("start_ms", "end_ms", "interval_version"):
            if key not in value:
                raise ParameterError(f"interval is missing {key}", field=f"interval.{key}")
        return cls(
            start_ms=value["start_ms"],
            end_ms=value["end_ms"],
            interval_version=value["interval_version"],
        )


@dataclass(frozen=True, slots=True)
class ProfilePolicy:
    """Profiling thresholds and fixed-point parameters.

    The full-fold path is selected only when *both* the unique stack count and
    the selected total weight remain strictly below their thresholds.
    """

    max_depth: int = 64
    full_unique_stack_threshold: int = 64
    full_total_weight_threshold: int = 4096
    scale: int = 1_000_000
    top_k: int = 5
    metric_version: str = CURRENT_METRIC_VERSION

    def __post_init__(self) -> None:
        for name in (
            "max_depth",
            "full_unique_stack_threshold",
            "full_total_weight_threshold",
            "scale",
            "top_k",
        ):
            value = getattr(self, name)
            if not is_int(value):
                raise ParameterError(f"{name} must be an integer", field=name)
        if self.max_depth < 1:
            raise ParameterError("max_depth must be >= 1", field="max_depth")
        if self.full_unique_stack_threshold < 1:
            raise ParameterError(
                "full_unique_stack_threshold must be >= 1",
                field="full_unique_stack_threshold",
            )
        if self.full_total_weight_threshold < 1:
            raise ParameterError(
                "full_total_weight_threshold must be >= 1",
                field="full_total_weight_threshold",
            )
        if self.scale < 2:
            raise ParameterError("scale must be >= 2", field="scale")
        if self.top_k < 1:
            raise ParameterError("top_k must be >= 1", field="top_k")
        if self.metric_version != CURRENT_METRIC_VERSION:
            raise ParameterError(
                "unsupported metric_version",
                field="metric_version",
                details={
                    "expected": CURRENT_METRIC_VERSION,
                    "actual": self.metric_version,
                },
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_depth": self.max_depth,
            "full_unique_stack_threshold": self.full_unique_stack_threshold,
            "full_total_weight_threshold": self.full_total_weight_threshold,
            "scale": self.scale,
            "top_k": self.top_k,
            "metric_version": self.metric_version,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProfilePolicy":
        if not isinstance(value, dict):
            raise ParameterError("policy must be an object", field="policy")
        kwargs: dict[str, Any] = {}
        for name in (
            "max_depth",
            "full_unique_stack_threshold",
            "full_total_weight_threshold",
            "scale",
            "top_k",
            "metric_version",
        ):
            if name in value:
                kwargs[name] = value[name]
        return cls(**kwargs)


@dataclass(frozen=True, slots=True)
class Sample:
    request_id: str
    timestamp_ms: int
    duration_ms: int
    service_id: str
    frames: tuple[str, ...]
    weight: int
    line_no: int = 0

    @property
    def weighted_duration_ms(self) -> int:
        return self.duration_ms * self.weight


@dataclass(frozen=True, slots=True)
class FrameStat:
    frame: str
    self_time_ms: int
    cumulative_time_ms: int
    participant_weight: int
    occurrence_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "self_time_ms": self.self_time_ms,
            "cumulative_time_ms": self.cumulative_time_ms,
            "participant_weight": self.participant_weight,
            "occurrence_count": self.occurrence_count,
        }


@dataclass(frozen=True, slots=True)
class ContextStat:
    frame: str
    inclusive_time_ms: int
    participant_weight: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "inclusive_time_ms": self.inclusive_time_ms,
            "participant_weight": self.participant_weight,
        }


@dataclass(frozen=True, slots=True)
class PathSelection:
    path: str
    forced: bool
    unique_stack_count: int
    total_weight: int
    unique_threshold: int
    weight_threshold: int
    rule: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "forced": self.forced,
            "unique_stack_count": self.unique_stack_count,
            "selected_total_weight": self.total_weight,
            "full_unique_stack_threshold": self.unique_threshold,
            "full_total_weight_threshold": self.weight_threshold,
            "rule": self.rule,
        }


def choose_path(
    policy: ProfilePolicy,
    unique_stack_count: int,
    total_weight: int,
    *,
    force_path: str | None = None,
) -> PathSelection:
    rule = (
        "full iff unique_stack_count < full_unique_stack_threshold "
        "and selected_total_weight < full_total_weight_threshold"
    )
    if force_path is not None:
        if force_path not in {"full", "streaming"}:
            raise ParameterError("force_path must be 'full' or 'streaming'", field="force_path")
        selected = force_path
        forced = True
    else:
        selected = (
            "full"
            if unique_stack_count < policy.full_unique_stack_threshold
            and total_weight < policy.full_total_weight_threshold
            else "streaming"
        )
        forced = False
    return PathSelection(
        path=selected,
        forced=forced,
        unique_stack_count=unique_stack_count,
        total_weight=total_weight,
        unique_threshold=policy.full_unique_stack_threshold,
        weight_threshold=policy.full_total_weight_threshold,
        rule=rule,
    )


def metric_definition() -> dict[str, Any]:
    return {
        "metric_version": CURRENT_METRIC_VERSION,
        "self_time": (
            "weight * duration_ms for a frame only when it is the leaf/top frame; "
            "a frame participates at most once per stack observation"
        ),
        "cumulative_time": (
            "weight * duration_ms once for every stack observation containing the frame; "
            "recursive duplicate frame names in one stack still count once"
        ),
        "participant_weight": (
            "sum of weights of stack observations containing the frame at least once"
        ),
        "fold_key": "the ordered frame path only; weights merge for identical paths",
        "ordering": "input-row order independent; output frames sorted lexicographically",
    }

