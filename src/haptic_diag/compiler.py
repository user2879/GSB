"""Deterministic effect-graph to binary frame compilation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .constants import CHANNELS, PROTOCOL_V2, SUPPORTED_PROTOCOLS
from .frames import (
    BatchDigest,
    ChannelRecord,
    FrameRecord,
    batch_key_for_range,
    make_batch_digest,
)
from .model import EffectGraph, evaluate_channel
from .quantization import DEFAULT_QUANT_PROFILE, QuantProfile


@dataclass(frozen=True)
class FrameQuantizationReport:
    seq: int
    timestamp_us: int
    channel_errors: tuple[dict[str, float | str | None], ...]
    max_amplitude_error: float
    max_frequency_error: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "timestamp_us": self.timestamp_us,
            "channel_errors": list(self.channel_errors),
            "max_amplitude_error": self.max_amplitude_error,
            "max_frequency_error": self.max_frequency_error,
        }


@dataclass(frozen=True)
class CompileReport:
    profile: dict[str, float | int]
    frame_count: int
    protocol_version: int
    max_amplitude_error: float
    max_frequency_error: float | None
    amplitude_error_bound: float
    frequency_error_bound: float
    frames: tuple[FrameQuantizationReport, ...]

    @property
    def error_bound_satisfied(self) -> bool:
        amp_ok = self.max_amplitude_error <= self.amplitude_error_bound + 1e-15
        freq_ok = (
            self.max_frequency_error is None
            or self.max_frequency_error <= self.frequency_error_bound + 1e-15
        )
        return amp_ok and freq_ok

    def to_dict(self, include_frames: bool = True) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "protocol_version": self.protocol_version,
            "frame_count": self.frame_count,
            "max_amplitude_error": self.max_amplitude_error,
            "max_frequency_error": self.max_frequency_error,
            "amplitude_error_bound": self.amplitude_error_bound,
            "frequency_error_bound": self.frequency_error_bound,
            "error_bound_satisfied": self.error_bound_satisfied,
            "frames": [item.to_dict() for item in self.frames] if include_frames else [],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CompileReport":
        frames = tuple(
            FrameQuantizationReport(
                seq=int(item["seq"]),
                timestamp_us=int(item["timestamp_us"]),
                channel_errors=tuple(item["channel_errors"]),
                max_amplitude_error=float(item["max_amplitude_error"]),
                max_frequency_error=(
                    None
                    if item.get("max_frequency_error") is None
                    else float(item["max_frequency_error"])
                ),
            )
            for item in value.get("frames", [])
        )
        return cls(
            profile=dict(value["profile"]),
            frame_count=int(value["frame_count"]),
            protocol_version=int(value["protocol_version"]),
            max_amplitude_error=float(value["max_amplitude_error"]),
            max_frequency_error=(
                None
                if value.get("max_frequency_error") is None
                else float(value["max_frequency_error"])
            ),
            amplitude_error_bound=float(value["amplitude_error_bound"]),
            frequency_error_bound=float(value["frequency_error_bound"]),
            frames=frames,
        )


@dataclass(frozen=True)
class CompileResult:
    graph: EffectGraph
    protocol_version: int
    frames: tuple[FrameRecord, ...]
    digests: tuple[BatchDigest, ...]
    report: CompileReport

    @property
    def frames_by_seq(self) -> dict[int, FrameRecord]:
        return {frame.seq: frame for frame in self.frames}


def _compile_frame(
    graph: EffectGraph,
    seq: int,
    protocol_version: int,
    profile: QuantProfile,
) -> tuple[FrameRecord, FrameQuantizationReport]:
    timestamp_us = seq * graph.frame_period_us
    channel_records: list[ChannelRecord] = []
    errors: list[dict[str, float | str | None]] = []
    for channel in CHANNELS:
        value = evaluate_channel(graph, channel, timestamp_us)
        if value is None:
            continue
        amplitude, frequency_hz = value
        if amplitude <= 0.0:
            continue
        amplitude_code = profile.encode_amplitude(amplitude)
        amplitude_error = abs(profile.decode_amplitude(amplitude_code) - amplitude)
        if protocol_version == PROTOCOL_V2:
            frequency_code = profile.encode_frequency(frequency_hz)
            frequency_error: float | None = abs(
                profile.decode_frequency(frequency_code) - frequency_hz
            )
        else:
            frequency_code = None
            frequency_error = None
        channel_records.append(
            ChannelRecord(
                channel=channel,
                amplitude=amplitude,
                frequency_hz=frequency_hz,
                amplitude_code=amplitude_code,
                frequency_code=frequency_code,
                amplitude_error=amplitude_error,
                frequency_error=frequency_error,
            )
        )
        errors.append(
            {
                "channel": channel,
                "amplitude": amplitude,
                "amplitude_code": amplitude_code,
                "amplitude_error": amplitude_error,
                "frequency_hz": frequency_hz,
                "frequency_code": frequency_code,
                "frequency_error": frequency_error,
            }
        )
    frame = FrameRecord(
        seq=seq,
        timestamp_us=timestamp_us,
        protocol_version=protocol_version,
        channels=tuple(channel_records),
    )
    frame_report = FrameQuantizationReport(
        seq=seq,
        timestamp_us=timestamp_us,
        channel_errors=tuple(errors),
        max_amplitude_error=max((item.amplitude_error for item in channel_records), default=0.0),
        max_frequency_error=(
            None
            if protocol_version != PROTOCOL_V2
            else max(
                (
                    item.frequency_error
                    for item in channel_records
                    if item.frequency_error is not None
                ),
                default=0.0,
            )
        ),
    )
    return frame, frame_report


def compile_frame_record(
    graph: EffectGraph,
    seq: int,
    protocol_version: int,
    profile: QuantProfile = DEFAULT_QUANT_PROFILE,
) -> FrameRecord:
    if protocol_version not in SUPPORTED_PROTOCOLS:
        raise ValueError(f"unsupported protocol version: {protocol_version}")
    if not 0 <= seq < graph.frame_count:
        raise ValueError(f"frame sequence {seq} outside graph timeline")
    frame, _ = _compile_frame(graph, seq, protocol_version, profile)
    return frame


def compile_graph(
    graph: EffectGraph,
    protocol_version: int,
    profile: QuantProfile = DEFAULT_QUANT_PROFILE,
) -> CompileResult:
    if protocol_version not in SUPPORTED_PROTOCOLS:
        raise ValueError(f"unsupported protocol version: {protocol_version}")
    graph.validate_integrity()
    frames: list[FrameRecord] = []
    reports: list[FrameQuantizationReport] = []
    for seq in range(graph.frame_count):
        frame, frame_report = _compile_frame(graph, seq, protocol_version, profile)
        frames.append(frame)
        reports.append(frame_report)
    digests: list[BatchDigest] = []
    for start in range(0, len(frames), graph.batch_size):
        batch = frames[start : start + graph.batch_size]
        batch_key = batch_key_for_range(batch[0].seq, batch[-1].seq)
        digests.append(make_batch_digest(batch, batch_key))
    max_amp = max((item.max_amplitude_error for item in reports), default=0.0)
    max_freq = (
        None
        if protocol_version != PROTOCOL_V2
        else max((item.max_frequency_error or 0.0 for item in reports), default=0.0)
    )
    report = CompileReport(
        profile=profile.to_dict(),
        frame_count=len(frames),
        protocol_version=protocol_version,
        max_amplitude_error=max_amp,
        max_frequency_error=max_freq,
        amplitude_error_bound=profile.amplitude_max_error,
        frequency_error_bound=profile.frequency_max_error,
        frames=tuple(reports),
    )
    if not report.error_bound_satisfied:
        raise AssertionError("quantization error exceeded the declared bound")
    return CompileResult(
        graph=graph,
        protocol_version=protocol_version,
        frames=tuple(frames),
        digests=tuple(digests),
        report=report,
    )


def compute_all_batch_digests(
    frames: Iterable[FrameRecord],
    batch_size: int,
) -> tuple[BatchDigest, ...]:
    ordered = sorted(frames, key=lambda frame: frame.seq)
    if not ordered:
        return ()
    digests: list[BatchDigest] = []
    for start in range(0, len(ordered), batch_size):
        batch = ordered[start : start + batch_size]
        batch_key = batch_key_for_range(batch[0].seq, batch[-1].seq)
        digests.append(make_batch_digest(batch, batch_key))
    return tuple(digests)

