"""Shared immutable-ish data models for the fingerprint pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class FingerprintConfig:
    frame_length: int
    hop_length: int
    base_sample_rate: int
    bands: tuple[tuple[float, float], ...]
    freq_neighbor_bins: int
    time_neighbor_frames: int
    min_peak_db: float
    fanout: int
    max_pair_delta_frames: int
    peaks_per_band: int
    window: str = "hann"


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    bucket_capacity: int
    voting_window_frames: int
    min_matches: int
    max_candidates: int
    overlap_policy: str
    boundary_gap_frames: int
    boundary_padding_frames: int


@dataclass(frozen=True, slots=True)
class ToleranceConfig:
    score_scale: int
    score_tolerance: float
    time_offset_tolerance_samples: float


@dataclass(frozen=True, slots=True)
class ConcurrencyConfig:
    max_workers: int
    spectrum_buffer_bytes: int


@dataclass(frozen=True, slots=True)
class TrackSpec:
    track_id: str
    path: Path
    role: str
    declared_sample_rate: int | None = None
    fingerprint_override: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class InjectionSpec:
    target_id: str
    kind: str
    expected_code: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_path: Path
    base_dir: Path
    references: tuple[TrackSpec, ...]
    queries: tuple[TrackSpec, ...]
    fingerprint: FingerprintConfig
    retrieval: RetrievalConfig
    tolerance: ToleranceConfig
    concurrency: ConcurrencyConfig
    allowed_sample_rates: tuple[int, ...]
    state_dir: Path
    report_path: Path
    results_path: Path
    injections: tuple[InjectionSpec, ...]
    raw: dict[str, Any]


@dataclass(frozen=True, slots=True)
class FrameMeta:
    index: int
    source_start_sample: int
    source_end_sample: int
    theoretical_start_seconds: Fraction
    theoretical_end_seconds: Fraction
    theoretical_start_sample: Fraction
    theoretical_end_sample: Fraction
    start_mapping_error_samples: Fraction
    end_mapping_error_samples: Fraction

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "source_start_sample": self.source_start_sample,
            "source_end_sample": self.source_end_sample,
            "theoretical_start_seconds": round(float(self.theoretical_start_seconds), 12),
            "theoretical_end_seconds": round(float(self.theoretical_end_seconds), 12),
            "theoretical_start_sample": round(float(self.theoretical_start_sample), 9),
            "theoretical_end_sample": round(float(self.theoretical_end_sample), 9),
            "start_mapping_error_samples": round(float(self.start_mapping_error_samples), 9),
            "end_mapping_error_samples": round(float(self.end_mapping_error_samples), 9),
        }


@dataclass(frozen=True, slots=True)
class Peak:
    frame_index: int
    band_index: int
    freq_bin: int
    freq_hz: float
    magnitude: float
    relative_db: float
    rank: int = 0


@dataclass(frozen=True, slots=True)
class FingerprintItem:
    key: tuple[int, int, int, int, int, int, int]
    anchor_frame: int
    target_frame: int
    anchor_band: int
    target_band: int
    anchor_freq_hz: float
    target_freq_hz: float
    quality: float


@dataclass(slots=True)
class TrackAudio:
    track_id: str
    path: Path
    role: str
    sample_rate: int
    declared_sample_rate: int | None
    samples: np.ndarray
    duration_samples: int


@dataclass(slots=True)
class TrackFingerprint:
    track_id: str
    path: Path
    role: str
    sample_rate: int
    duration_samples: int
    frame_samples: int
    hop_samples_rounded: int
    frames: tuple[FrameMeta, ...]
    peaks: tuple[Peak, ...]
    items: tuple[FingerprintItem, ...]
    dropped_peaks_by_band_cap: int = 0

    @property
    def frame_count(self) -> int:
        return len(self.frames)


@dataclass(frozen=True, slots=True)
class ReferenceEntry:
    reference_id: str
    reference_order: int
    item: FingerprintItem


@dataclass(slots=True)
class FingerprintMatch:
    key: tuple[int, int, int, int, int, int, int]
    reference_id: str
    query_id: str
    reference_frame: int
    query_frame: int
    reference_target_frame: int
    query_target_frame: int
    float_weight: float
    fixed_units: int
    offset_frames: int
    reference_time_seconds: float
    query_time_seconds: float
    offset_seconds: float


@dataclass(slots=True)
class Fragment:
    reference_id: str
    query_id: str
    match: tuple[tuple[int, int, int, int, int, int, int], int, int, int, int]
    reference_start_sample: int
    reference_end_sample: int
    query_start_sample: int
    query_end_sample: int
    reference_start_seconds: float
    reference_end_seconds: float
    query_start_seconds: float
    query_end_seconds: float
    score_float: float
    score_fixed_units: int
    score_fixed: float
    first_exceed: dict[str, Any] | None
    score_trace: list[dict[str, Any]]
    integer_frame_offset: int
    offset_seconds_float: float
    offset_reference_samples_float: float
    offset_analysis_samples_float: float
    integer_offset_analysis_samples: float
    offset_conversion_error_samples: float
    alignment: dict[str, Any]
    drift: dict[str, Any]
    anchor_count: int


@dataclass(slots=True)
class TrackError:
    track_id: str
    role: str
    path: str
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
