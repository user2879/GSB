"""Configuration loading and deterministic hashing."""

import hashlib
import json
from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class EventParams:
    change_pixel_ratio_threshold: float
    color_match_tolerance: int
    min_event_duration_ms: int
    min_gap_ms: int
    event_categories: tuple
    change_pixel_value_threshold: int = 16
    track_max_gap_frames: int = 2
    block_size_frames: int = 8
    hist_bins: int = 4


@dataclass(frozen=True)
class ScheduleConfig:
    max_concurrent_sequences: int
    global_frame_budget: int
    per_sequence_frame_quota: int
    frame_block_residency_budget: int
    snapshot_commit_interval_frames: int


def _load_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_event_params(path) -> EventParams:
    raw = _load_json(path)
    return EventParams(
        change_pixel_ratio_threshold=float(raw["change_pixel_ratio_threshold"]),
        color_match_tolerance=int(raw["color_match_tolerance"]),
        min_event_duration_ms=int(raw["min_event_duration_ms"]),
        min_gap_ms=int(raw["min_gap_ms"]),
        event_categories=tuple(raw["event_categories"]),
        change_pixel_value_threshold=int(raw.get("change_pixel_value_threshold", 16)),
        track_max_gap_frames=int(raw.get("track_max_gap_frames", 2)),
        block_size_frames=int(raw.get("block_size_frames", 8)),
        hist_bins=int(raw.get("hist_bins", 4)),
    )


def load_schedule_config(path) -> ScheduleConfig:
    raw = _load_json(path)
    return ScheduleConfig(
        max_concurrent_sequences=int(raw["max_concurrent_sequences"]),
        global_frame_budget=int(raw["global_frame_budget"]),
        per_sequence_frame_quota=int(raw["per_sequence_frame_quota"]),
        frame_block_residency_budget=int(raw["frame_block_residency_budget"]),
        snapshot_commit_interval_frames=int(raw["snapshot_commit_interval_frames"]),
    )


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def config_hash(event_params: EventParams, schedule: ScheduleConfig) -> str:
    payload = {
        "event_params": asdict(event_params),
        "schedule": asdict(schedule),
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
