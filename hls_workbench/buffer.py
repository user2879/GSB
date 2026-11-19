"""Deterministic client download/playback buffer model."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import write_canonical
from .config import ClientBufferConfig, MediaConfig, ProfileConfig
from .errors import require


def build_delay_table(
    profiles: tuple[ProfileConfig, ...],
    media: MediaConfig,
    seed: int,
) -> dict[str, Any]:
    """Build the local pseudo-network table used instead of real downloads."""

    values: dict[str, list[int]] = {}
    for profile in sorted(profiles, key=lambda item: item.priority):
        delays: list[int] = []
        for sequence in range(media.slice_count):
            token = f"{seed}:{profile.name}:{sequence}:pseudo-download".encode("utf-8")
            number = int.from_bytes(hashlib.sha256(token).digest()[:8], "big")
            delay = profile.base_delay_ms + (number % max(1, profile.jitter_ms))
            if sequence in profile.delay_spike_sequences:
                delay += profile.delay_spike_ms
            delays.append(int(delay))
        values[profile.name] = delays
    return {
        "schema_version": 1,
        "seed": seed,
        "source": "local-deterministic-table",
        "delays_ms": values,
    }


def write_delay_table(path: Path, table: dict[str, Any]) -> str:
    return write_canonical(path, table)


@dataclass(frozen=True)
class BufferSimulation:
    profile: str
    threshold_ms: int
    capacity_ms: int
    minimum_buffer_ms: int
    peak_buffer_ms: int
    playback_start_ms: int | None
    playback_end_ms: int | None
    underrun_events: tuple[dict[str, Any], ...]
    switch_recommendations: tuple[dict[str, Any], ...]
    points: tuple[dict[str, Any], ...]
    downloads: tuple[dict[str, Any], ...]

    @property
    def underrun_count(self) -> int:
        return len(self.underrun_events)

    @property
    def safe(self) -> bool:
        return self.underrun_count == 0 and self.minimum_buffer_ms >= self.threshold_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "threshold_ms": self.threshold_ms,
            "capacity_ms": self.capacity_ms,
            "minimum_buffer_ms": self.minimum_buffer_ms,
            "peak_buffer_ms": self.peak_buffer_ms,
            "playback_start_ms": self.playback_start_ms,
            "playback_end_ms": self.playback_end_ms,
            "underrun_count": self.underrun_count,
            "underrun_events": list(self.underrun_events),
            "switch_recommendations": list(self.switch_recommendations),
            "safe": self.safe,
            "points": list(self.points),
            "downloads": list(self.downloads),
        }


def simulate_buffer(
    *,
    profile: ProfileConfig,
    lower_profile: str | None,
    ready_times_ms: dict[int, int],
    delays_ms: list[int],
    segment_durations_ms: list[int],
    media_duration_ms: int,
    config: ClientBufferConfig,
) -> BufferSimulation:
    """Simulate one sequential downloader and one playback clock in a shared ms timebase."""

    config.validate()
    segment_count = len(segment_durations_ms)
    require(len(delays_ms) == segment_count, "E_BUFFER_UNDERRUN", "delay table length mismatch")
    require(
        sorted(ready_times_ms) == list(range(segment_count)),
        "E_BUFFER_UNDERRUN",
        "ready-time map is missing a segment",
        expected=segment_count,
        actual=sorted(ready_times_ms),
    )

    step = config.step_ms
    horizon_ms = (
        media_duration_ms
        + sum(delays_ms)
        + max(ready_times_ms.values(), default=0)
        + 4 * step
    )
    next_sequence = 0
    active_download: dict[str, Any] | None = None
    buffer_ms = 0
    consumed_ms = 0
    downloaded_ms = 0
    playback_start_ms: int | None = None
    playback_end_ms: int | None = None
    stalled = False
    stall_start_ms: int | None = None
    underruns: list[dict[str, Any]] = []
    switch_recommendations: list[dict[str, Any]] = []
    below_threshold = False
    points: list[dict[str, Any]] = []
    downloads: list[dict[str, Any]] = []
    minimum_buffer_ms: int | None = None
    peak_buffer_ms = 0
    first_segment_completed = False
    risk_window_ms = max(config.threshold_ms, max(segment_durations_ms, default=0))

    for clock_ms in range(0, horizon_ms + step, step):
        completed = False
        if active_download is not None and active_download["end_ms"] <= clock_ms:
            duration = active_download["duration_ms"]
            buffer_ms += duration
            downloaded_ms += duration
            completed = True
            downloads.append(
                {
                    "sequence": active_download["sequence"],
                    "ready_ms": active_download["ready_ms"],
                    "start_ms": active_download["start_ms"],
                    "end_ms": active_download["end_ms"],
                    "delay_ms": active_download["delay_ms"],
                    "duration_ms": duration,
                }
            )
            first_segment_completed = True
            active_download = None

        if (
            active_download is None
            and next_sequence < segment_count
            and clock_ms >= ready_times_ms[next_sequence]
            and buffer_ms < config.capacity_ms
        ):
            sequence = next_sequence
            start_ms = max(clock_ms, ready_times_ms[sequence])
            delay_ms = delays_ms[sequence]
            active_download = {
                "sequence": sequence,
                "ready_ms": ready_times_ms[sequence],
                "start_ms": start_ms,
                "end_ms": start_ms + delay_ms,
                "delay_ms": delay_ms,
                "duration_ms": segment_durations_ms[sequence],
            }
            next_sequence += 1

        if playback_start_ms is None and first_segment_completed:
            startup_target = config.startup_buffer_ms or config.threshold_ms
            startup_target = min(max(1, startup_target), media_duration_ms)
            if buffer_ms >= startup_target or (next_sequence >= segment_count and active_download is None):
                playback_start_ms = clock_ms

        if playback_start_ms is not None and clock_ms >= playback_start_ms:
            if not stalled:
                if buffer_ms >= step:
                    buffer_ms -= step
                    consumed_ms += step
                elif buffer_ms > 0:
                    consumed_ms += buffer_ms
                    buffer_ms = 0
                    stalled = True
                    stall_start_ms = clock_ms
                else:
                    stalled = True
                    stall_start_ms = clock_ms
            elif buffer_ms >= max(config.resume_buffer_ms, step):
                if stall_start_ms is not None:
                    underruns.append(
                        {
                            "code": "E_BUFFER_UNDERRUN",
                            "start_ms": stall_start_ms,
                            "end_ms": clock_ms,
                            "duration_ms": max(0, clock_ms - stall_start_ms),
                            "profile": profile.name,
                            "reason": "buffer exhausted before the next segment completed",
                        }
                    )
                stall_start_ms = None
                stalled = False
                buffer_ms -= step
                consumed_ms += step

        risk_window_active = (
            playback_start_ms is not None
            and media_duration_ms - consumed_ms > risk_window_ms
        )
        if risk_window_active:
            peak_buffer_ms = max(peak_buffer_ms, buffer_ms)
            minimum_buffer_ms = buffer_ms if minimum_buffer_ms is None else min(minimum_buffer_ms, buffer_ms)
            if buffer_ms < config.threshold_ms:
                if not below_threshold:
                    below_threshold = True
                    switch_recommendations.append(
                        {
                            "time_ms": clock_ms,
                            "code": "E_BUFFER_UNDERRUN",
                            "profile": profile.name,
                            "suggested_profile": lower_profile,
                            "buffer_ms": buffer_ms,
                            "threshold_ms": config.threshold_ms,
                            "trigger": "buffer_below_threshold",
                            "evidence": {
                                "downloaded_ms": downloaded_ms,
                                "consumed_ms": consumed_ms,
                                "stalled": stalled,
                            },
                        }
                    )
            else:
                below_threshold = False

        points.append(
            {
                "time_ms": clock_ms,
                "buffer_ms": buffer_ms,
                "downloaded_ms": downloaded_ms,
                "consumed_ms": consumed_ms,
                "stalled": stalled,
                "download_active": active_download is not None,
            }
        )

        all_downloads_complete = next_sequence >= segment_count and active_download is None
        all_media_consumed = consumed_ms >= media_duration_ms and not stalled
        if all_downloads_complete and all_media_consumed and playback_start_ms is not None:
            playback_end_ms = clock_ms
            break

    if stall_start_ms is not None:
        underruns.append(
            {
                "code": "E_BUFFER_UNDERRUN",
                "start_ms": stall_start_ms,
                "end_ms": horizon_ms,
                "duration_ms": max(0, horizon_ms - stall_start_ms),
                "profile": profile.name,
                "reason": "buffer remained empty through the end of the simulated window",
            }
        )

    return BufferSimulation(
        profile=profile.name,
        threshold_ms=config.threshold_ms,
        capacity_ms=config.capacity_ms,
        minimum_buffer_ms=minimum_buffer_ms if minimum_buffer_ms is not None else 0,
        peak_buffer_ms=peak_buffer_ms,
        playback_start_ms=playback_start_ms,
        playback_end_ms=playback_end_ms,
        underrun_events=tuple(underruns),
        switch_recommendations=tuple(switch_recommendations),
        points=tuple(points),
        downloads=tuple(downloads),
    )
