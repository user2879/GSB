"""Configuration objects with deterministic defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Any

from .errors import require


@dataclass(frozen=True)
class MediaConfig:
    seed: int = 20240517
    duration_seconds: float = 30.0
    fps: int = 30
    width: int = 1280
    height: int = 720
    slice_seconds: float = 2.0
    sample_rate: int = 48000
    channels: int = 2

    def validate(self) -> None:
        require(self.fps > 0, "E_MEDIA_FORMAT", "fps must be positive")
        require(self.width >= 64 and self.height >= 36, "E_MEDIA_FORMAT", "video geometry too small")
        require(self.width % 2 == 0 and self.height % 2 == 0, "E_MEDIA_FORMAT", "H.264 requires even dimensions")
        require(self.duration_seconds > 0, "E_MEDIA_FORMAT", "duration must be positive")
        require(self.slice_seconds > 0, "E_MEDIA_FORMAT", "slice duration must be positive")
        require(self.sample_rate == 48000, "E_MEDIA_FORMAT", "this workbench requires 48 kHz audio")
        require(self.channels == 2, "E_MEDIA_FORMAT", "this workbench requires stereo audio")
        require(
            abs(self.slice_seconds * self.fps - round(self.slice_seconds * self.fps)) < 1e-9,
            "E_MEDIA_FORMAT",
            "slice duration must contain an integer number of frames",
        )
        require(
            abs(self.duration_seconds * self.fps - round(self.duration_seconds * self.fps)) < 1e-9,
            "E_MEDIA_FORMAT",
            "duration must contain an integer number of frames",
        )

    @property
    def total_frames(self) -> int:
        return int(round(self.duration_seconds * self.fps))

    @property
    def frames_per_slice(self) -> int:
        return int(round(self.slice_seconds * self.fps))

    @property
    def slice_count(self) -> int:
        frames = self.total_frames
        per_slice = self.frames_per_slice
        return (frames + per_slice - 1) // per_slice

    @property
    def total_audio_samples(self) -> int:
        return int(round(self.duration_seconds * self.sample_rate))

    def frame_count(self, slice_index: int) -> int:
        return min(self.frames_per_slice, self.total_frames - slice_index * self.frames_per_slice)

    def audio_samples(self, slice_index: int) -> int:
        start = int(round(slice_index * self.slice_seconds * self.sample_rate))
        end = min(
            self.total_audio_samples,
            int(round((slice_index + 1) * self.slice_seconds * self.sample_rate)),
        )
        return end - start

    @property
    def video_time_base(self) -> Fraction:
        return Fraction(1, self.fps)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProfileConfig:
    name: str
    width: int
    height: int
    video_bitrate_kbps: int
    audio_bitrate_kbps: int
    priority: int
    memory_mb: int
    cpu_ms: int
    virtual_transcode_ms: int
    base_delay_ms: int
    jitter_ms: int
    delay_spike_ms: int = 0
    delay_spike_sequences: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["delay_spike_sequences"] = list(self.delay_spike_sequences)
        return value


DEFAULT_PROFILES: tuple[ProfileConfig, ...] = (
    ProfileConfig(
        name="360p",
        width=640,
        height=360,
        video_bitrate_kbps=700,
        audio_bitrate_kbps=96,
        priority=0,
        memory_mb=90,
        cpu_ms=60,
        virtual_transcode_ms=55,
        base_delay_ms=110,
        jitter_ms=90,
    ),
    ProfileConfig(
        name="720p",
        width=1280,
        height=720,
        video_bitrate_kbps=1800,
        audio_bitrate_kbps=128,
        priority=1,
        memory_mb=180,
        cpu_ms=110,
        virtual_transcode_ms=95,
        base_delay_ms=340,
        jitter_ms=220,
        delay_spike_ms=9300,
        delay_spike_sequences=(4, 7),
    ),
)


@dataclass(frozen=True)
class EncoderConfig:
    h264_preset: str = "ultrafast"
    source_crf: int = 18
    threads: int = 1
    audio_frame_samples: int = 1024

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BudgetConfig:
    max_workers: int = 4
    memory_budget_mb: int = 320
    cpu_budget_ms: int = 300

    def validate(self) -> None:
        require(self.max_workers >= 1, "E_BUDGET_DENIED", "max_workers must be at least 1")
        require(self.memory_budget_mb > 0, "E_BUDGET_DENIED", "memory budget must be positive")
        require(self.cpu_budget_ms > 0, "E_BUDGET_DENIED", "CPU budget must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetryConfig:
    max_retries: int = 2
    base_delay_ms: int = 15
    multiplier: float = 2.0
    max_delay_ms: int = 500

    def validate(self) -> None:
        require(self.max_retries >= 0, "E_RETRY_EXHAUSTED", "max_retries must be non-negative")
        require(self.base_delay_ms >= 0, "E_RETRY_EXHAUSTED", "backoff must be non-negative")
        require(self.multiplier >= 1.0, "E_RETRY_EXHAUSTED", "backoff multiplier must be >= 1")

    def delay_for_retry(self, retry_number: int) -> int:
        if retry_number <= 0:
            return 0
        value = self.base_delay_ms * (self.multiplier ** (retry_number - 1))
        return int(round(min(value, self.max_delay_ms)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FaultConfig:
    task_key: str | None = None
    unit_id: str | None = None
    failures: int = 0

    def validate(self) -> None:
        require(self.failures >= 0, "E_RETRY_EXHAUSTED", "fault failure count must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ClientBufferConfig:
    capacity_ms: int = 8000
    threshold_ms: int = 2600
    step_ms: int = 100
    resume_buffer_ms: int = 300
    startup_buffer_ms: int = 0

    def validate(self) -> None:
        require(6000 <= self.capacity_ms <= 20000, "E_BUFFER_UNDERRUN", "buffer capacity must be 6-20 seconds")
        require(self.threshold_ms >= 0, "E_BUFFER_UNDERRUN", "buffer threshold must be non-negative")
        require(self.step_ms > 0, "E_BUFFER_UNDERRUN", "buffer step must be positive")
        require(self.resume_buffer_ms >= 0, "E_BUFFER_UNDERRUN", "resume buffer must be non-negative")
        require(self.startup_buffer_ms >= 0, "E_BUFFER_UNDERRUN", "startup buffer must be non-negative")
        require(
            self.threshold_ms <= self.capacity_ms,
            "E_BUFFER_UNDERRUN",
            "buffer threshold cannot exceed buffer capacity",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecisionConfig:
    bandwidth_kbps: int = 2600
    bandwidth_safety_ratio: float = 0.85

    def validate(self) -> None:
        require(self.bandwidth_kbps > 0, "E_BUFFER_UNDERRUN", "bandwidth estimate must be positive")
        require(0 < self.bandwidth_safety_ratio <= 1, "E_BUFFER_UNDERRUN", "safety ratio must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkbenchConfig:
    media: MediaConfig = field(default_factory=MediaConfig)
    profiles: tuple[ProfileConfig, ...] = DEFAULT_PROFILES
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    fault: FaultConfig = field(default_factory=FaultConfig)
    client_buffer: ClientBufferConfig = field(default_factory=ClientBufferConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)

    def validate(self) -> None:
        self.media.validate()
        self.budget.validate()
        self.retry.validate()
        self.fault.validate()
        self.client_buffer.validate()
        self.decision.validate()
        names = [profile.name for profile in self.profiles]
        require(len(names) == len(set(names)), "E_MEDIA_FORMAT", "profile names must be unique")
        require("360p" in names and "720p" in names, "E_MEDIA_FORMAT", "360p and 720p profiles are required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "media": self.media.to_dict(),
            "profiles": [profile.to_dict() for profile in self.profiles],
            "encoder": self.encoder.to_dict(),
            "budget": self.budget.to_dict(),
            "retry": self.retry.to_dict(),
            "fault": self.fault.to_dict(),
            "client_buffer": self.client_buffer.to_dict(),
            "decision": self.decision.to_dict(),
        }
