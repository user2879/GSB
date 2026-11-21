from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .audio import AudioFile, AudioScan
from .config import AnalysisConfig, digest_json
from .errors import E_AUDIO_FORMAT, BusinessError
from .model import AudioDocument, Segment, make_segment_id


@dataclass(frozen=True)
class FrameEnergy:
    frame_index: int
    start_sample: int
    rms: float
    db: float


def frame_energies(samples: np.ndarray, frame_length: int, hop: int) -> list[FrameEnergy]:
    if samples.size < frame_length:
        return []
    frame_count = 1 + (samples.size - frame_length) // hop
    result: list[FrameEnergy] = []
    for frame_index in range(frame_count):
        start = frame_index * hop
        frame = samples[start : start + frame_length].astype(np.float64) / 32768.0
        rms = float(np.sqrt(np.mean(frame * frame)))
        db = 20.0 * math.log10(max(rms, 1e-12))
        result.append(FrameEnergy(frame_index, start, rms, db))
    return result


def hysteresis_ranges(energies: list[FrameEnergy], enter_db: float, exit_db: float) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    active = False
    start = 0
    for energy in energies:
        if not active:
            if energy.db >= enter_db:
                active = True
                start = energy.frame_index
        elif energy.db < exit_db:
            ranges.append((start, energy.frame_index))
            active = False
    if active:
        ranges.append((start, len(energies)))
    return ranges


def cleanup_ranges(
    ranges: list[tuple[int, int]],
    min_segment_frames: int,
    min_gap_frames: int,
) -> list[tuple[int, int]]:
    current = [(start, end) for start, end in ranges if end > start]
    changed = True
    while changed:
        changed = False
        merged: list[tuple[int, int]] = []
        for item in current:
            if merged and item[0] - merged[-1][1] < min_gap_frames:
                merged[-1] = (merged[-1][0], max(merged[-1][1], item[1]))
                changed = True
            else:
                merged.append(item)
        kept = [(start, end) for start, end in merged if end - start >= min_segment_frames]
        if len(kept) != len(merged):
            changed = True
        current = kept
    return current


def annotate_segments(
    segments: list[Segment],
    annotations: tuple[dict[str, Any], ...],
) -> None:
    for annotation in annotations:
        file_name = annotation.get("file")
        speaker = annotation.get("speaker")
        style_hint = annotation.get("style_hint")
        contains_ms = annotation.get("contains_ms")
        for segment in segments:
            if file_name is not None and segment.file != str(file_name):
                continue
            if contains_ms is not None:
                point = float(contains_ms)
                if not (segment.start_sample / 16.0 <= point < segment.end_sample / 16.0):
                    continue
            if speaker is not None:
                segment.speaker = str(speaker)
            if style_hint is not None:
                segment.style_hint = str(style_hint)


def segment_audio(
    audio: AudioFile,
    config: AnalysisConfig,
    annotations: tuple[dict[str, Any], ...] = (),
) -> tuple[list[Segment], dict[str, Any]]:
    frame_length = config.frame_length_samples
    hop = config.frame_shift_samples
    energies = frame_energies(audio.samples, frame_length, hop)
    if not energies:
        raise BusinessError(
            E_AUDIO_FORMAT,
            "audio is shorter than one analysis frame",
            file=audio.name,
            frame_length_samples=frame_length,
        )
    raw_ranges = hysteresis_ranges(energies, config.energy_enter_db, config.energy_exit_db)
    min_segment_frames = max(1, int(math.ceil(config.min_segment_ms / config.frame_shift_ms)))
    min_gap_frames = max(0, int(math.ceil(config.min_gap_ms / config.frame_shift_ms)))
    ranges = cleanup_ranges(raw_ranges, min_segment_frames, min_gap_frames)
    segments: list[Segment] = []
    for start_frame, end_frame in ranges:
        start_sample = start_frame * hop
        end_sample = min(audio.sample_count, (end_frame - 1) * hop + frame_length)
        if end_sample <= start_sample:
            continue
        segments.append(
            Segment(
                segment_id=make_segment_id(audio.name, start_sample, end_sample),
                file=audio.name,
                start_sample=start_sample,
                end_sample=end_sample,
                start_frame=start_frame,
                end_frame=end_frame,
                origin="segmentation",
            )
        )
    annotate_segments(segments, annotations)
    summary = {
        "file": audio.name,
        "frame_count": len(energies),
        "raw_active_ranges": [list(item) for item in raw_ranges],
        "final_active_ranges": [list(item) for item in ranges],
        "min_segment_frames": min_segment_frames,
        "min_gap_frames": min_gap_frames,
        "segmentation_fingerprint": digest_json(
            {
                "audio": audio.content_sha256(),
                "energy": [config.energy_enter_db, config.energy_exit_db],
                "frame": [frame_length, hop],
                "minimums": [min_segment_frames, min_gap_frames],
            }
        ),
    }
    return segments, summary


def build_document(
    scan: AudioScan,
    config: AnalysisConfig,
) -> tuple[AudioDocument, list[dict[str, Any]], list[dict[str, Any]]]:
    segments: list[Segment] = []
    summaries: list[dict[str, Any]] = []
    failures = list(scan.failures)
    for file_name in sorted(scan.files):
        audio = scan.files[file_name]
        try:
            file_segments, summary = segment_audio(audio, config, config.initial_annotations)
        except BusinessError as exc:
            failures.append({"code": exc.code, "message": exc.message, "details": {"file": file_name, **exc.details}})
            continue
        segments.extend(file_segments)
        summaries.append(summary)
    document = AudioDocument(files=dict(scan.files), segments=segments, input_annotations=config.initial_annotations)
    document.refresh_frame_indices(config)
    return document, summaries, failures
