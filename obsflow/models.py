"""Core data models for obsflow.

These dataclasses are the single source of truth shared by decoding,
aggregation, scheduling, leasing and persistence.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

# Supported fixed-point scale factors (integer powers of ten). Any record or
# session scale outside this set is rejected with E_FORMAT.
SUPPORTED_SCALES: tuple[int, ...] = (1, 10, 100, 1_000, 10_000)

# Supported on-disk historical record format versions.
SUPPORTED_FORMATS: tuple[int, ...] = (1, 2, 3)

# Saturation bounds for fixed-point integer arithmetic.
INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1


@dataclass(frozen=True, slots=True)
class NormalizedRecord:
    """Internal unified view of one observation record."""

    source_id: str
    seq: int
    ts_ms: int
    metric_key: str
    # Stored integer measurement together with its record-level scale.
    raw_value: int
    record_scale: int
    # Historical format version this record originated from (1, 2 or 3).
    source_format: int
    # Fields that were filled rather than read from the raw record.
    filled_fields: tuple[str, ...]
    # field -> human readable reason describing the fill basis.
    fill_basis: dict[str, str]
    # 1-based line number of the record inside its stream file.
    line_no: int

    def with_line_no(self, line_no: int) -> "NormalizedRecord":
        return replace(self, line_no=line_no)


@dataclass(frozen=True, slots=True)
class SourceDeclaration:
    """One observation source declaration loaded from the declarations JSON."""

    source_id: str
    # On-disk record format version used by this source's stream file.
    format_version: int
    # Metric key used for records that do not carry their own key (v1/v2).
    default_metric_key: str
    # Record-level scale used when the record itself omits it (v1/v2).
    default_scale: int
    # Optional raw metric key -> canonical metric key mapping.
    metric_mapping: dict[str, str] = field(default_factory=dict)
    # Path of the stream file, relative to the declaration file directory.
    file_path: str = ""


@dataclass(frozen=True, slots=True)
class WindowParams:
    """Session window / fixed-point parameters."""

    window_size: int
    # Slide step in records of the per-metric ordered sequence (default 1).
    slide_step: int
    # Session fixed-point scale Q used for quantisation.
    scale: int

    def validate(self) -> None:
        if not isinstance(self.window_size, int) or self.window_size < 1:
            raise ValueError("window_size must be a positive integer")
        if not isinstance(self.slide_step, int) or self.slide_step < 1:
            raise ValueError("slide_step must be a positive integer")
        if self.slide_step > self.window_size:
            raise ValueError("slide_step must not exceed window_size")
        if self.scale not in SUPPORTED_SCALES:
            raise ValueError(f"unsupported session scale: {self.scale}")


@dataclass(frozen=True, slots=True)
class Lease:
    """Sliding-window lease."""

    lease_id: str
    metric_key: str
    window_index: int
    start_seq: int
    end_seq: int
    granted_ms: int
    expiry_ms: int
    # Optimistic version; every successful renewal increments it.
    version: int
    # "active" | "expired" | "revoked"
    state: str = "active"

    def is_expired(self, now_ms: int) -> bool:
        return now_ms >= self.expiry_ms


@dataclass(frozen=True, slots=True)
class WindowResult:
    """One fixed-length sliding window aggregation conclusion."""

    metric_key: str
    window_index: int
    start_source_id: str
    end_source_id: str
    start_seq: int
    end_seq: int
    count: int
    # Fixed-point integer sum / mean at the session scale.
    sum_q: int
    mean_q: int
    # Strict upper bounds expressed in quantised integer units: the absolute
    # error is strictly smaller than the stored bound.
    value_error_bound_q: int
    sum_error_bound_q: int
    scale: int
    truncation: str
    saturation: str
    # Monotonic per-window version; bumped only when the window is recomputed
    # with changed inputs.
    window_version: int
    status: str  # "ready" | "unknown"
    input_digest: str
    result_digest: str
    lease_id: Optional[str] = None


@dataclass(frozen=True, slots=True)
class PropagationEvent:
    """One invalidation propagation step (emitted in stable order)."""

    order: int
    reason: str
    source_id: Optional[str]
    metric_key: Optional[str]
    changed_field: str
    old_value: Optional[str]
    new_value: Optional[str]
    affected_windows: tuple[tuple[str, int], ...]
    basis: str


@dataclass(frozen=True, slots=True)
class TaskTiming:
    """Scheduler timing record for one window aggregation task."""

    metric_key: str
    window_index: int
    source_id: str
    enqueued_ms: int
    wait_start_ms: int
    wait_end_ms: int
    run_start_ms: int
    run_end_ms: int
