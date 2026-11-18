from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .fingerprint import digest_json
from .pcm import I16_MAX, I16_MIN, int16_vector_sha256, int64_matrix_sha256

MIX_ALGORITHM_VERSION = "fixed-q23-mix-v1"


@dataclass(frozen=True)
class GainArtifact:
    track: str
    gain_q: int
    mute: bool
    weights: np.ndarray
    gain_parameter: float


@dataclass(frozen=True)
class FragmentPlan:
    clip_id: str
    fragment_index: int
    sample_start: int
    sample_end: int
    output_start: int
    output_end: int
    tracks: tuple[str, ...]
    track_trims: tuple[tuple[str, int, int], ...]


def quantize_gain(gain: float, quantization: int) -> int:
    return int(np.floor(float(gain) * quantization + 0.5))


def weighted_contributions(samples_q23: np.ndarray, gain_q: int, mute: bool) -> np.ndarray:
    values = np.asarray(samples_q23, dtype=np.int64)
    if mute or gain_q == 0:
        return np.zeros(values.shape, dtype=np.int64)
    return values * int(gain_q)


def weighted_to_i16(weighted: np.ndarray, quantization: int) -> np.ndarray:
    numerator = np.asarray(weighted, dtype=np.int64)
    denominator = int(quantization) * 256
    magnitude = (np.abs(numerator) + denominator // 2) // denominator
    rounded = np.where(numerator < 0, -magnitude, magnitude)
    return np.clip(rounded, I16_MIN, I16_MAX).astype(np.int16)


def weighted_to_q23(weighted: np.ndarray, quantization: int) -> np.ndarray:
    numerator = np.asarray(weighted, dtype=np.int64)
    denominator = int(quantization)
    magnitude = (np.abs(numerator) + denominator // 2) // denominator
    rounded = np.where(numerator < 0, -magnitude, magnitude)
    return np.clip(rounded, -8_388_608, 8_388_607).astype(np.int64)


def plan_fragments(
    *,
    clip_id: str,
    frames: int,
    render_frame_length: int,
    clip_start: int,
    clip_end: int,
    track_trims: dict[str, tuple[int, int]],
    muted: set[str],
) -> list[FragmentPlan]:
    plans: list[FragmentPlan] = []
    for block_start in range(0, frames, render_frame_length):
        block_end = min(frames, block_start + render_frame_length)
        output_start = max(block_start, clip_start)
        output_end = min(block_end, clip_end)
        if output_start >= output_end:
            continue
        references = []
        for track in sorted(track_trims):
            if track in muted:
                continue
            track_start, track_end = track_trims[track]
            if max(output_start, track_start) < min(output_end, track_end):
                references.append(track)
        plans.append(
            FragmentPlan(
                clip_id=clip_id,
                fragment_index=block_start // render_frame_length,
                sample_start=block_start,
                sample_end=block_end,
                output_start=output_start,
                output_end=output_end,
                tracks=tuple(references),
                track_trims=tuple(
                    (track, *track_trims[track]) for track in sorted(track_trims)
                ),
            )
        )
    return plans


def render_fragment(
    plan: FragmentPlan,
    gains: dict[str, GainArtifact],
    *,
    quantization: int,
    all_tracks: Iterable[str],
) -> dict[str, Any]:
    frames = plan.output_end - plan.output_start
    names = list(plan.tracks)
    weighted = np.zeros((frames, len(names)), dtype=np.int64)
    trims = {
        track: (start, end) for track, start, end in plan.track_trims
    }
    for column, track in enumerate(names):
        track_start, track_end = trims[track]
        lo = max(plan.output_start, track_start)
        hi = min(plan.output_end, track_end)
        if lo >= hi:
            continue
        artifact = gains[track]
        if artifact.mute or artifact.gain_q == 0:
            continue
        weighted[lo - plan.output_start : hi - plan.output_start, column] = (
            artifact.weights[lo:hi]
        )
    mixed_weighted = np.zeros(frames, dtype=np.int64)
    for column in range(len(names)):
        mixed_weighted += weighted[:, column]
    pcm = weighted_to_i16(mixed_weighted, quantization)
    mixed_q23 = weighted_to_q23(mixed_weighted, quantization)
    component_meta = {
        "clip_id": plan.clip_id,
        "fragment_index": plan.fragment_index,
        "sample_start": plan.sample_start,
        "sample_end": plan.sample_end,
        "output_start": plan.output_start,
        "output_end": plan.output_end,
        "tracks": names,
        "quantization": quantization,
    }
    component_sha = int64_matrix_sha256(weighted) + ":" + digest_json(component_meta)
    peak = int(np.max(np.abs(pcm.astype(np.int32)))) if pcm.size else 0
    sums = {
        track: int(np.sum(weighted[:, column], dtype=np.int64))
        for column, track in enumerate(names)
    }
    first_frame = {
        "frame": plan.output_start,
        "tracks": {
            track: int(weighted[0, column])
            for column, track in enumerate(names)
        },
        "mixed_q23": int(mixed_q23[0]) if frames else 0,
    }
    last_frame = {
        "frame": plan.output_end - 1,
        "tracks": {
            track: int(weighted[-1, column])
            for column, track in enumerate(names)
        },
        "mixed_q23": int(mixed_q23[-1]) if frames else 0,
    }
    return {
        "pcm": pcm,
        "weighted": weighted,
        "mixed_q23": mixed_q23,
        "meta": {
            **component_meta,
            "frames": frames,
            "peak": peak,
            "pcm_sha256": int16_vector_sha256(pcm),
            "component_sha256": component_sha,
            "component_sums": sums,
            "first_frame": first_frame,
            "last_frame": last_frame,
        },
    }


def full_component_matrix(
    plan_fragment_results: list[dict[str, Any]],
    gains: dict[str, GainArtifact],
    *,
    clip_start: int,
    clip_end: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    names = sorted(gains)
    length = clip_end - clip_start
    matrix = np.zeros((length, len(names)), dtype=np.int64)
    for fragment in plan_fragment_results:
        meta = fragment["meta"]
        local_start = meta["output_start"] - clip_start
        local_end = meta["output_end"] - clip_start
        frag_names = meta["tracks"]
        weighted = fragment["weighted"]
        for column, track in enumerate(frag_names):
            matrix[local_start:local_end, names.index(track)] = weighted[:, column]
    mixed = np.zeros(length, dtype=np.int64)
    for column in range(len(names)):
        mixed += matrix[:, column]
    return matrix, mixed, names
