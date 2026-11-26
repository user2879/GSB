from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .config import AnalysisConfig, EndpointConfig
from .util import json_hash


@dataclass(frozen=True)
class FrameMetric:
    frame_index: int
    start_sample: int
    end_sample: int
    energy: float
    spectral_flatness: float
    score: float
    deterministic_value: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "energy": self.energy,
            "spectral_flatness": self.spectral_flatness,
            "score": self.score,
            "deterministic_value": self.deterministic_value,
        }


@dataclass(frozen=True)
class SpeechCandidate:
    segment_index: int
    start_frame: int
    end_frame: int
    start_sample: int
    end_sample: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "segment_index": self.segment_index,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
        }


@dataclass(frozen=True)
class WindowGeometry:
    window_id: str
    window_index: int
    sample_start: int
    sample_end: int
    frame_indices: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "window_index": self.window_index,
            "sample_start": self.sample_start,
            "sample_end": self.sample_end,
            "frame_indices": list(self.frame_indices),
        }


@dataclass(frozen=True)
class EndpointResult:
    frame_metrics: tuple[FrameMetric, ...]
    candidates: tuple[SpeechCandidate, ...]
    windows: tuple[WindowGeometry, ...]
    content_hash: str
    frame_metrics_digest: str

    def active_frame_mask(self) -> tuple[bool, ...]:
        mask = [False] * len(self.frame_metrics)
        for candidate in self.candidates:
            for index in range(candidate.start_frame, candidate.end_frame):
                if 0 <= index < len(mask):
                    mask[index] = True
        return tuple(mask)

    def as_cache_payload(self) -> dict[str, Any]:
        return {
            "frame_metrics": [metric.as_dict() for metric in self.frame_metrics],
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "windows": [window.as_dict() for window in self.windows],
            "content_hash": self.content_hash,
            "frame_metrics_digest": self.frame_metrics_digest,
        }


def frame_metrics(pcm: np.ndarray, frame_length: int, frame_shift: int) -> tuple[FrameMetric, ...]:
    samples = np.asarray(pcm, dtype=np.float32) / 32768.0
    if samples.size < frame_length:
        return ()
    starts = np.arange(0, samples.size - frame_length + 1, frame_shift, dtype=np.int64)
    window = np.hanning(frame_length).astype(np.float32)
    spectrum_window = window
    power_floor = np.float32(1e-12)
    result: list[FrameMetric] = []
    for frame_index, start in enumerate(starts.tolist()):
        frame = samples[start : start + frame_length]
        energy = float(np.sqrt(np.mean(frame * frame, dtype=np.float64)))
        windowed = frame * spectrum_window
        spectrum = np.fft.rfft(windowed)
        power = (spectrum.real * spectrum.real + spectrum.imag * spectrum.imag).astype(np.float32)
        if power.size:
            shifted = power + power_floor
            geometric = float(np.exp(np.mean(np.log(shifted), dtype=np.float64)))
            arithmetic = float(np.mean(power, dtype=np.float64))
            flatness = geometric / (arithmetic + float(power_floor))
        else:
            flatness = 0.0
        score = energy * (1.0 - 0.55 * min(1.0, max(0.0, flatness)))
        result.append(
            FrameMetric(
                frame_index=frame_index,
                start_sample=int(start),
                end_sample=int(start + frame_length),
                energy=round(energy, 12),
                spectral_flatness=round(flatness, 12),
                score=round(score, 12),
                deterministic_value=int(round(energy * 1_000_000_000.0)),
            )
        )
    return tuple(result)


def _raw_hysteresis_segments(
    metrics: tuple[FrameMetric, ...],
    endpoint: EndpointConfig,
) -> list[tuple[int, int]]:
    min_silence_frames = max(1, int(np.ceil(endpoint.min_silence_ms * 16.0 / endpoint.frame_shift)))
    min_speech_frames = max(1, int(np.ceil(endpoint.min_speech_ms * 16.0 / endpoint.frame_shift)))
    segments: list[tuple[int, int]] = []
    active = False
    start_frame = 0
    last_active = -1
    for metric in metrics:
        above_low = metric.score >= endpoint.low_energy
        if not active and metric.score >= endpoint.high_energy:
            active = True
            start_frame = metric.frame_index
            last_active = metric.frame_index
        elif active and above_low:
            last_active = metric.frame_index
        if active and not above_low:
            if metric.frame_index - last_active >= min_silence_frames:
                if last_active - start_frame + 1 >= min_speech_frames:
                    segments.append((start_frame, last_active + 1))
                active = False
                last_active = -1
    if active and last_active >= start_frame:
        if last_active - start_frame + 1 >= min_speech_frames:
            segments.append((start_frame, last_active + 1))

    # Explicit post-processing pass: remove short candidates and merge gaps
    # shorter than the declared minimum silence.  Sorting makes the operation
    # independent of how the streaming frame run was discovered.
    filtered: list[tuple[int, int]] = []
    min_silence = max(1, min_silence_frames)
    for start, end in sorted(segments):
        if end - start < min_speech_frames:
            continue
        if filtered and start - filtered[-1][1] <= min_silence:
            filtered[-1] = (filtered[-1][0], max(filtered[-1][1], end))
        else:
            filtered.append((start, end))
    return filtered


def detect_speech(
    pcm: np.ndarray,
    *,
    endpoint: EndpointConfig,
    content_hash: str,
) -> EndpointResult:
    metrics = frame_metrics(pcm, endpoint.frame_length, endpoint.frame_shift)
    segments = _raw_hysteresis_segments(metrics, endpoint)
    candidates: list[SpeechCandidate] = []
    for index, (start_frame, end_frame) in enumerate(segments):
        first = metrics[start_frame]
        last = metrics[end_frame - 1]
        candidates.append(
            SpeechCandidate(
                segment_index=index,
                start_frame=start_frame,
                end_frame=end_frame,
                start_sample=first.start_sample,
                end_sample=last.end_sample,
            )
        )
    metrics_digest = json_hash([metric.as_dict() for metric in metrics])
    return EndpointResult(
        frame_metrics=metrics,
        candidates=tuple(candidates),
        windows=(),
        content_hash=content_hash,
        frame_metrics_digest=metrics_digest,
    )


def build_windows(
    *,
    timeline_end: int,
    analysis: AnalysisConfig,
) -> tuple[WindowGeometry, ...]:
    if timeline_end <= 0:
        return ()
    length = analysis.window_length
    shift = analysis.window_shift
    starts: list[int] = []
    if timeline_end <= length:
        starts = [0]
    else:
        starts = list(range(0, timeline_end - length + 1, shift))
    windows: list[WindowGeometry] = []
    for index, start in enumerate(starts):
        end = min(timeline_end, start + length)
        frame_length = analysis.frame_length
        frame_shift = analysis.frame_shift
        frame_indices: list[int] = []
        frame_index = 0
        while True:
            frame_start = frame_index * frame_shift
            center_twice = 2 * frame_start + frame_length
            if center_twice >= 2 * start and center_twice < 2 * end:
                frame_indices.append(frame_index)
            if frame_start >= end:
                break
            frame_index += 1
        windows.append(
            WindowGeometry(
                window_id=f"W{index:06d}",
                window_index=index,
                sample_start=start,
                sample_end=end,
                frame_indices=tuple(frame_indices),
            )
        )
    return tuple(windows)


def attach_windows(result: EndpointResult, windows: Iterable[WindowGeometry]) -> EndpointResult:
    return EndpointResult(
        frame_metrics=result.frame_metrics,
        candidates=result.candidates,
        windows=tuple(windows),
        content_hash=result.content_hash,
        frame_metrics_digest=result.frame_metrics_digest,
    )
