"""Deterministic spectral-peak fingerprints and cross-format diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.ndimage import maximum_filter

from .config import FingerprintConfig
from .errors import FormatDriftError


@dataclass(frozen=True, order=True)
class Peak:
    frame: int
    band: int
    frequency: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "band": self.band,
            "frequency": round(self.frequency, 6),
            "score": round(self.score, 6),
        }


@dataclass(frozen=True, order=True)
class Fingerprint:
    hash: str
    anchor_frame: int
    target_frame: int
    band1: int
    band2: int
    delta_frames: int

    @property
    def time_signature(self) -> tuple[int, int]:
        return self.anchor_frame, self.target_frame

    def to_dict(self) -> dict[str, Any]:
        return {
            "hash": self.hash,
            "anchor_frame": self.anchor_frame,
            "target_frame": self.target_frame,
            "band1": self.band1,
            "band2": self.band2,
            "delta_frames": self.delta_frames,
        }


@dataclass
class FingerprintSequence:
    config: FingerprintConfig
    duration: float
    peaks: list[Peak]
    fingerprints: list[Fingerprint]

    @property
    def version(self) -> str:
        return self.config.version

    @property
    def config_digest(self) -> str:
        return self.config.digest()

    @property
    def encoding_space(self) -> str:
        return self.config.encoding_space

    def canonical_rows(self) -> list[tuple[str, int, int, int, int, int]]:
        return [
            (
                item.hash,
                item.anchor_frame,
                item.target_frame,
                item.band1,
                item.band2,
                item.delta_frames,
            )
            for item in self.fingerprints
        ]

    def content_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.config.digest().encode("ascii"))
        for row in self.canonical_rows():
            digest.update(("|".join(str(value) for value in row) + "\n").encode("ascii"))
        return digest.hexdigest()

    def fingerprint_counts_by_band(self) -> list[int]:
        counts = [0] * self.config.band_count
        for item in self.fingerprints:
            counts[item.band1] += 1
            counts[item.band2] += 1
        return counts

    def to_dict(self, include_peaks: bool = False) -> dict[str, Any]:
        value: dict[str, Any] = {
            "version": self.version,
            "config_digest": self.config_digest,
            "encoding_space": self.encoding_space,
            "duration": round(self.duration, 9),
            "peak_count": len(self.peaks),
            "fingerprint_count": len(self.fingerprints),
            "fingerprint_content_hash": self.content_hash(),
        }
        if include_peaks:
            value["peaks"] = [peak.to_dict() for peak in self.peaks]
            value["fingerprints"] = [item.to_dict() for item in self.fingerprints]
        return value


def _normalize_audio(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.float64).reshape(-1)
    if len(values) == 0:
        return values
    values = values - float(values.mean())
    peak = float(np.max(np.abs(values)))
    if peak > 1e-12:
        values = values / peak
    return values


def stft_magnitude(samples: np.ndarray, config: FingerprintConfig) -> tuple[np.ndarray, np.ndarray]:
    """Small, deterministic STFT with an explicit frame grid (no centering)."""

    values = _normalize_audio(samples)
    if len(values) < config.n_fft:
        values = np.pad(values, (0, config.n_fft - len(values)))
    frame_count = 1 + int(math.ceil((len(values) - config.n_fft) / config.hop_length))
    required = config.n_fft + (frame_count - 1) * config.hop_length
    if required > len(values):
        values = np.pad(values, (0, required - len(values)))
    frames = np.lib.stride_tricks.sliding_window_view(values, config.n_fft)[:: config.hop_length]
    window = np.hanning(config.n_fft + 1)[:-1]
    spectrum = np.abs(np.fft.rfft(frames * window[None, :], axis=1))
    frequencies = np.fft.rfftfreq(config.n_fft, d=1.0 / config.sample_rate)
    return spectrum, frequencies


def _band_matrix(
    spectrum: np.ndarray,
    frequencies: np.ndarray,
    config: FingerprintConfig,
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.asarray(config.band_edges(), dtype=np.float64)
    centers = np.sqrt(edges[:-1] * edges[1:])
    matrix = np.empty((spectrum.shape[0], config.band_count), dtype=np.float64)
    for band in range(config.band_count):
        lower = edges[band]
        upper = edges[band + 1]
        if band == 0:
            indices = np.flatnonzero((frequencies >= lower) & (frequencies < upper))
        else:
            indices = np.flatnonzero((frequencies >= lower) & (frequencies <= upper))
        if len(indices) == 0:
            indices = np.asarray([int(np.argmin(np.abs(frequencies - centers[band])))])
        matrix[:, band] = np.max(spectrum[:, indices], axis=1)
    return matrix, centers


def pick_peaks(samples: np.ndarray, config: FingerprintConfig) -> list[Peak]:
    spectrum, frequencies = stft_magnitude(samples, config)
    band_magnitude, centers = _band_matrix(spectrum, frequencies, config)
    db = 20.0 * np.log10(band_magnitude + 1e-10)
    frame_floor = np.max(db, axis=1, keepdims=True) - 8.0
    local_max = maximum_filter(
        band_magnitude,
        size=(2 * config.neighborhood_frames + 1, 2 * config.neighborhood_bands + 1),
        mode="nearest",
    )
    candidate_mask = (band_magnitude >= local_max) & (db >= frame_floor)
    # Boundaries are not true local maxima and are often codec-unstable.
    candidate_mask[: config.neighborhood_frames, :] = False
    candidate_mask[-config.neighborhood_frames :, :] = False
    candidate_mask[:, : config.neighborhood_bands] = False
    candidate_mask[:, -config.neighborhood_bands :] = False
    frame_indices, band_indices = np.nonzero(candidate_mask)
    if len(frame_indices) == 0:
        return []

    scores = band_magnitude[frame_indices, band_indices]
    candidates = [
        Peak(
            frame=int(frame),
            band=int(band),
            frequency=float(centers[band]),
            score=float(score),
        )
        for frame, band, score in zip(frame_indices, band_indices, scores)
    ]
    duration = len(samples) / float(config.sample_rate)
    target_count = max(1, int(round(duration * config.peak_density)))
    if len(candidates) > target_count:
        candidates.sort(key=lambda item: (-item.score, item.frame, item.band))
        candidates = candidates[:target_count]
    candidates.sort(key=lambda item: (item.frame, item.band, -item.score))
    return candidates


def pair_peaks(
    peaks: Iterable[Peak],
    config: FingerprintConfig,
) -> list[Fingerprint]:
    ordered = sorted(peaks, key=lambda item: (item.frame, item.band, -item.score))
    result: list[Fingerprint] = []
    for anchor_index, anchor in enumerate(ordered):
        candidates: list[Peak] = []
        for target in ordered[anchor_index + 1 :]:
            delta = target.frame - anchor.frame
            if delta < config.target_min_frames:
                continue
            if delta > config.target_max_frames:
                break
            candidates.append(target)
        if len(candidates) > config.max_pairs_per_peak:
            candidates.sort(key=lambda item: (-item.score, item.frame, item.band))
            candidates = candidates[: config.max_pairs_per_peak]
        for target in candidates:
            delta_frames = target.frame - anchor.frame
            fingerprint_hash = config.hash_for(anchor.band, target.band, delta_frames)
            result.append(
                Fingerprint(
                    hash=fingerprint_hash,
                    anchor_frame=anchor.frame,
                    target_frame=target.frame,
                    band1=anchor.band,
                    band2=target.band,
                    delta_frames=delta_frames,
                )
            )
    result.sort(
        key=lambda item: (
            item.anchor_frame,
            item.target_frame,
            item.band1,
            item.band2,
            item.hash,
        )
    )
    return result


def extract_fingerprints(samples: np.ndarray, config: FingerprintConfig) -> FingerprintSequence:
    values = np.asarray(samples, dtype=np.float64)
    peaks = pick_peaks(values, config)
    fingerprints = pair_peaks(peaks, config)
    return FingerprintSequence(
        config=config,
        duration=len(values) / float(config.sample_rate),
        peaks=peaks,
        fingerprints=fingerprints,
    )


@dataclass
class SequenceMatch:
    matched: int
    total_a: int
    total_b: int
    hit_rate: float
    dice: float
    median_anchor_delta: float
    unmatched_band_delta: float
    reason: str
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched": self.matched,
            "total_a": self.total_a,
            "total_b": self.total_b,
            "hit_rate": round(self.hit_rate, 6),
            "dice": round(self.dice, 6),
            "median_anchor_delta": round(self.median_anchor_delta, 6),
            "unmatched_band_delta": round(self.unmatched_band_delta, 6),
            "reason": self.reason,
            "evidence": self.evidence,
        }


def match_fingerprint_sequences(
    left: FingerprintSequence,
    right: FingerprintSequence,
    tolerance_frames: int | None = None,
) -> SequenceMatch:
    """Greedy one-to-one hash match close on the common time axis."""

    tolerance = tolerance_frames or left.config.match_anchor_tolerance_frames
    by_hash: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for right_index, item in enumerate(right.fingerprints):
        by_hash[item.hash].append((item.anchor_frame, right_index))
    for positions in by_hash.values():
        positions.sort(key=lambda value: (value[0], value[1]))
    used: set[int] = set()
    deltas: list[int] = []
    matched_left: list[Fingerprint] = []
    unmatched_left: list[Fingerprint] = []
    for item in left.fingerprints:
        positions = by_hash.get(item.hash)
        best_position: int | None = None
        best_index: int | None = None
        best_distance = tolerance + 1
        if positions:
            for position, right_index in positions:
                if right_index in used:
                    continue
                distance = abs(position - item.anchor_frame)
                if distance <= tolerance and distance < best_distance:
                    best_position = position
                    best_distance = distance
                    best_index = right_index
                if position > item.anchor_frame + tolerance:
                    break
        if best_position is None or best_index is None:
            unmatched_left.append(item)
        else:
            used.add(best_index)
            deltas.append(best_position - item.anchor_frame)
            matched_left.append(item)

    matched_count = len(matched_left)
    denominator = min(len(left.fingerprints), len(right.fingerprints))
    hit_rate = matched_count / denominator if denominator else 0.0
    dice_denominator = len(left.fingerprints) + len(right.fingerprints)
    dice = (2.0 * matched_count / dice_denominator) if dice_denominator else 0.0
    left_unmatched_bands = [item.band1 for item in unmatched_left]
    right_unmatched_bands = [
        item.band1 for right_index, item in enumerate(right.fingerprints) if right_index not in used
    ]
    left_median = float(np.median(left_unmatched_bands)) if left_unmatched_bands else 0.0
    right_median = float(np.median(right_unmatched_bands)) if right_unmatched_bands else 0.0
    band_delta = abs(left_median - right_median)
    if matched_count == 0:
        reason = "no_hash_overlap"
    elif hit_rate >= left.config.format_hit_threshold:
        reason = "consistent"
    elif band_delta >= 2.0:
        reason = "frequency_band_shift"
    elif abs(float(np.median(deltas)) if deltas else 0.0) > tolerance:
        reason = "time_grid_drift"
    else:
        reason = "codec_magnitude_instability"
    return SequenceMatch(
        matched=matched_count,
        total_a=len(left.fingerprints),
        total_b=len(right.fingerprints),
        hit_rate=hit_rate,
        dice=dice,
        median_anchor_delta=float(np.median(deltas)) if deltas else 0.0,
        unmatched_band_delta=band_delta,
        reason=reason,
        evidence={
            "left_version": left.version,
            "right_version": right.version,
            "left_encoding_space": left.encoding_space,
            "right_encoding_space": right.encoding_space,
            "left_unmatched_median_band": round(left_median, 3),
            "right_unmatched_median_band": round(right_median, 3),
            "left_duration": round(left.duration, 6),
            "right_duration": round(right.duration, 6),
            "suspected_causes": _suspected_causes(reason, left, right),
        },
    )


def _suspected_causes(
    reason: str,
    left: FingerprintSequence,
    right: FingerprintSequence,
) -> list[str]:
    causes: list[str] = []
    if reason == "frequency_band_shift":
        causes.append("resampling_or_codec_frequency_shift")
    if left.config.sample_rate != right.config.sample_rate:
        causes.append("analysis_sample_rate_mismatch")
    if reason == "time_grid_drift":
        causes.append("decoder_time_base_offset")
    if reason in {"codec_magnitude_instability", "no_hash_overlap"}:
        causes.append("lossy_codec_peak_perturbation")
    return causes


def verify_pair_consistency(
    matches: list[SequenceMatch],
    config: FingerprintConfig,
    pair_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    labels = pair_labels or {}
    failing: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        value = match.to_dict()
        value["pair"] = labels.get(str(index), f"pair-{index}")
        value["threshold"] = config.format_hit_threshold
        value["passed"] = match.hit_rate >= config.format_hit_threshold
        pairs.append(value)
        if not value["passed"]:
            failing.append(value)
    report = {
        "threshold": config.format_hit_threshold,
        "passed": not failing,
        "pairs": pairs,
    }
    if failing:
        raise FormatDriftError(
            "Cross-format fingerprint hit rate is below the declared threshold",
            {
                "threshold": config.format_hit_threshold,
                "failing_pairs": failing,
                "hint": "Inspect frequency-band and time-grid evidence; do not mask extraction drift with retrieval recall.",
            },
        )
    return report


def ranking_first_difference(
    left: list[str],
    right: list[str],
) -> dict[str, Any] | None:
    length = max(len(left), len(right))
    for index in range(length):
        left_value = left[index] if index < len(left) else None
        right_value = right[index] if index < len(right) else None
        if left_value != right_value:
            return {
                "position": index + 1,
                "left": left_value,
                "right": right_value,
            }
    return None


def counter_json(counter: Counter[Any], limit: int = 12) -> list[dict[str, Any]]:
    values = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return [{"value": key, "votes": count} for key, count in values[:limit]]


def sequence_to_json(sequence: FingerprintSequence, include_peaks: bool = False) -> str:
    return json.dumps(
        sequence.to_dict(include_peaks=include_peaks),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
