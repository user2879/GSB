"""Frame-synchronous FFT peak fingerprints.

Frame length and hop are expressed at ``base_sample_rate``. For files at other
sample rates every boundary is derived from the exact rational ratio
``hop_length * rate / base_sample_rate`` and recorded with its integer sample
position and theoretical (fractional) position.
"""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .models import (
    FingerprintConfig,
    FingerprintItem,
    FrameMeta,
    Peak,
    TrackAudio,
    TrackFingerprint,
)
from .runtime import CancelToken, FrameBudget
from .wavio import read_wav


def _round_half_up(value: Fraction) -> int:
    if value < 0:
        return -_round_half_up(-value)
    return (value.numerator * 2 + value.denominator) // (2 * value.denominator)


def build_frame_map(total_samples: int, source_rate: int, config: FingerprintConfig) -> tuple[FrameMeta, ...]:
    frame_exact = Fraction(config.frame_length * source_rate, config.base_sample_rate)
    hop_exact = Fraction(config.hop_length * source_rate, config.base_sample_rate)
    frame_samples = _round_half_up(frame_exact)
    frames: list[FrameMeta] = []
    index = 0
    while True:
        theoretical_start = index * hop_exact
        start = _round_half_up(theoretical_start)
        end = start + frame_samples
        if end > total_samples:
            break
        theoretical_end = theoretical_start + frame_exact
        frames.append(
            FrameMeta(
                index=index,
                source_start_sample=start,
                source_end_sample=end,
                theoretical_start_seconds=theoretical_start / source_rate,
                theoretical_end_seconds=theoretical_end / source_rate,
                theoretical_start_sample=theoretical_start,
                theoretical_end_sample=theoretical_end,
                start_mapping_error_samples=Fraction(start) - theoretical_start,
                end_mapping_error_samples=Fraction(end) - theoretical_end,
            )
        )
        index += 1
    return tuple(frames)


def _window(name: str, length: int) -> np.ndarray:
    if name == "hann":
        return np.hanning(length).astype(np.float64)
    if name == "hamming":
        return np.hamming(length).astype(np.float64)
    return np.ones(length, dtype=np.float64)


def _band_bins(config: FingerprintConfig, source_rate: int, fft_bins: int) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for low, high in config.bands:
        low_bin = max(1, int(np.ceil(low * fft_bins * 2 / source_rate)))
        high_bin = min(fft_bins, int(np.floor(high * fft_bins * 2 / source_rate)) + 1)
        if high_bin <= low_bin:
            low_bin = min(max(1, low_bin), fft_bins - 1)
            high_bin = min(fft_bins, low_bin + 1)
        result.append((low_bin, high_bin))
    return result


def _detect_chunk_peaks(
    magnitude: np.ndarray,
    frame_indices: np.ndarray,
    config: FingerprintConfig,
    source_rate: int,
    band_ranges: list[tuple[int, int]],
) -> list[Peak]:
    eps = 1e-20
    db = 20.0 * np.log10(magnitude + eps)
    global_max = np.max(db, axis=1, keepdims=True)
    relative = db - global_max
    peaks: list[Peak] = []
    fft_bins = magnitude.shape[1]
    for band_index, (low_bin, high_bin) in enumerate(band_ranges):
        values = relative[:, low_bin:high_bin]
        if values.size == 0:
            continue
        eligible = values >= config.min_peak_db
        radius = config.freq_neighbor_bins
        padded = np.pad(values, ((0, 0), (radius, radius)), constant_values=-np.inf)
        local_max = np.ones_like(values, dtype=bool)
        width = high_bin - low_bin
        for offset in range(0, radius * 2 + 1):
            if offset == radius:
                continue
            neighbor = padded[:, offset : offset + width]
            local_max &= values >= neighbor
        local_max &= eligible
        for row in np.nonzero(np.any(local_max, axis=1))[0]:
            candidates = np.nonzero(local_max[row])[0]
            if candidates.size == 0:
                continue
            global_bins = candidates + low_bin
            order = np.lexsort((global_bins, -values[row, candidates]))[: config.peaks_per_band]
            for rank, selected in enumerate(order):
                local_bin = int(candidates[selected])
                global_bin = int(global_bins[selected])
                freq_hz = global_bin * source_rate / (2.0 * (fft_bins - 1))
                peaks.append(
                    Peak(
                        frame_index=int(frame_indices[row]),
                        band_index=band_index,
                        freq_bin=global_bin,
                        freq_hz=float(freq_hz),
                        magnitude=float(magnitude[row, global_bin]),
                        relative_db=float(values[row, local_bin]),
                        rank=rank,
                    )
                )
    return peaks


def _time_local_peaks(peaks: Iterable[Peak], radius: int) -> tuple[list[Peak], int]:
    by_band_frame: dict[tuple[int, int], list[Peak]] = defaultdict(list)
    all_peaks = list(peaks)
    for peak in all_peaks:
        by_band_frame[(peak.band_index, peak.frame_index)].append(peak)
    accepted: list[Peak] = []
    dropped = 0
    for peak in all_peaks:
        is_peak = True
        for frame in range(peak.frame_index - radius, peak.frame_index + radius + 1):
            if frame == peak.frame_index:
                continue
            for neighbor in by_band_frame.get((peak.band_index, frame), ()):
                if neighbor.relative_db > peak.relative_db + 1e-9:
                    is_peak = False
                    break
            if not is_peak:
                break
        if is_peak:
            accepted.append(peak)
        else:
            dropped += 1
    accepted.sort(key=lambda p: (p.frame_index, p.band_index, p.freq_bin))
    return accepted, dropped


def _extract_spectrum_peaks(
    audio: TrackAudio,
    frames: tuple[FrameMeta, ...],
    config: FingerprintConfig,
    *,
    budget: FrameBudget | None,
    priority: int,
    token: CancelToken,
) -> tuple[list[Peak], int]:
    if not frames:
        return [], 0
    frame_samples = frames[0].source_end_sample - frames[0].source_start_sample
    fft_bins = frame_samples // 2 + 1
    band_ranges = _band_bins(config, audio.sample_rate, fft_bins)
    window = _window(config.window, frame_samples)
    starts = np.asarray([frame.source_start_sample for frame in frames], dtype=np.int64)
    frame_indices = np.asarray([frame.index for frame in frames], dtype=np.int64)
    if budget is not None:
        chunk_frames = max(1, min(budget.capacity_frames, 96))
    else:
        chunk_frames = 96
    candidates: list[Peak] = []
    for begin in range(0, starts.size, chunk_frames):
        token.raise_if_cancelled()
        end = min(starts.size, begin + chunk_frames)
        count = end - begin
        if budget is not None:
            budget.acquire(count, priority, token)
        try:
            block = np.stack([audio.samples[start : start + frame_samples] for start in starts[begin:end]])
            spectrum = np.fft.rfft(block * window[np.newaxis, :], axis=1)
            magnitude = np.abs(spectrum)
            candidates.extend(
                _detect_chunk_peaks(magnitude, frame_indices[begin:end], config, audio.sample_rate, band_ranges)
            )
        finally:
            if budget is not None:
                budget.release(count)
    accepted, dropped = _time_local_peaks(candidates, config.time_neighbor_frames)
    return accepted, dropped


def _make_items(peaks: list[Peak], config: FingerprintConfig) -> tuple[FingerprintItem, ...]:
    quality_by_band = []
    for peak in peaks:
        strength = 0.0
        if config.min_peak_db < 0:
            strength = min(1.0, max(0.0, 1.0 - peak.relative_db / config.min_peak_db))
        quality_by_band.append(strength)
    items: list[FingerprintItem] = []
    def pitch_bucket(frequency_hz: float) -> int:
        # Six whole-tone-scale buckets per octave tolerate the demo's +/-3%
        # time/frequency warping while sharply reducing unrelated collisions.
        return int(math.floor(6.0 * math.log2(max(frequency_hz, 1.0) / 55.0) + 0.5))

    def interval_bucket(anchor_hz: float, target_hz: float) -> int:
        ratio = max(target_hz, 1.0) / max(anchor_hz, 1.0)
        return int(math.floor(24.0 * math.log2(ratio) + 0.5))

    for index, anchor in enumerate(peaks):
        emitted = 0
        for target_index in range(index + 1, len(peaks)):
            target = peaks[target_index]
            delta = target.frame_index - anchor.frame_index
            if delta > config.max_pair_delta_frames:
                break
            if delta <= 0:
                continue
            quality = 1.0 + 0.5 * min(quality_by_band[index], quality_by_band[target_index])
            items.append(
                FingerprintItem(
                    key=(
                        anchor.band_index,
                        target.band_index,
                        delta,
                        pitch_bucket(anchor.freq_hz),
                        interval_bucket(anchor.freq_hz, target.freq_hz),
                        anchor.rank,
                        target.rank,
                    ),
                    anchor_frame=anchor.frame_index,
                    target_frame=target.frame_index,
                    anchor_band=anchor.band_index,
                    target_band=target.band_index,
                    anchor_freq_hz=anchor.freq_hz,
                    target_freq_hz=target.freq_hz,
                    quality=quality,
                )
            )
            emitted += 1
            if emitted >= config.fanout:
                break
    items.sort(
        key=lambda item: (
            item.anchor_frame,
            item.target_frame,
            item.anchor_band,
            item.target_band,
            item.anchor_freq_hz,
            item.target_freq_hz,
        )
    )
    return tuple(items)


def fingerprint_audio(
    audio: TrackAudio,
    config: FingerprintConfig,
    *,
    budget: FrameBudget | None = None,
    priority: int = 0,
    token: CancelToken | None = None,
) -> TrackFingerprint:
    cancellation = token or CancelToken()
    frames = build_frame_map(audio.duration_samples, audio.sample_rate, config)
    peaks, dropped = _extract_spectrum_peaks(
        audio, frames, config, budget=budget, priority=priority, token=cancellation
    )
    items = _make_items(peaks, config)
    frame_samples = 0
    hop_rounded = 0
    if frames:
        frame_samples = frames[0].source_end_sample - frames[0].source_start_sample
        if len(frames) > 1:
            hop_rounded = frames[1].source_start_sample - frames[0].source_start_sample
        else:
            hop_rounded = _round_half_up(Fraction(config.hop_length * audio.sample_rate, config.base_sample_rate))
    return TrackFingerprint(
        track_id=audio.track_id,
        path=audio.path,
        role=audio.role,
        sample_rate=audio.sample_rate,
        duration_samples=audio.duration_samples,
        frame_samples=frame_samples,
        hop_samples_rounded=hop_rounded,
        frames=frames,
        peaks=tuple(peaks),
        items=items,
        dropped_peaks_by_band_cap=dropped,
    )


def extract_fingerprint(
    path: str | Path,
    config: FingerprintConfig,
    *,
    allowed_sample_rates: tuple[int, ...] = (16000, 44100, 48000),
    declared_sample_rate: int | None = None,
    track_id: str | None = None,
    role: str = "track",
    budget: FrameBudget | None = None,
    priority: int = 0,
    token: CancelToken | None = None,
) -> TrackFingerprint:
    audio = read_wav(
        path,
        track_id=track_id or Path(path).stem,
        role=role,
        allowed_sample_rates=allowed_sample_rates,
        declared_sample_rate=declared_sample_rate,
    )
    return fingerprint_audio(audio, config, budget=budget, priority=priority, token=token)
