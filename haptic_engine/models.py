"""Immutable domain objects describing devices, events and global settings."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Device:
    device_id: str
    sample_rate_hz: int
    frame_duration_ms: int
    min_intensity: float
    max_intensity: float
    levels: int
    max_commands_per_frame: int

    @property
    def frame_samples(self) -> int:
        # Both operands are positive integers; the rounded result is always >= 1.
        return max(1, round(self.sample_rate_hz * self.frame_duration_ms / 1000))


@dataclass(frozen=True)
class CurvePoint:
    time_ms: int
    intensity: float


@dataclass(frozen=True)
class Event:
    event_id: str
    event_type: str
    start_ms: int
    duration_ms: int
    target_device: str
    curve: tuple[CurvePoint, ...]

    @property
    def end_ms(self) -> int:
        return self.start_ms + self.duration_ms


@dataclass(frozen=True)
class SyncAnchor:
    anchor_id: str
    time_ms: int
    device_ids: tuple[str, ...]
    reference_device: str
    allowed_samples: int


@dataclass(frozen=True)
class GlobalConfig:
    concurrency: int
    inflight_limit: int


@dataclass(frozen=True)
class Manifest:
    devices: tuple[Device, ...]
    events: tuple[Event, ...]
    sync_anchors: tuple[SyncAnchor, ...]
    global_config: GlobalConfig
    state_dir: str
    output_file: str
    extra: dict = field(default_factory=dict, compare=False)

    def device_map(self) -> dict[str, Device]:
        return {device.device_id: device for device in self.devices}
