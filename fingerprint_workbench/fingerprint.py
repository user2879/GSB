from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter

from .canonical import digest_object, sha256_bytes
from .config import SAMPLE_RATE, FingerprintConfig
from .errors import WorkbenchError


FINGERPRINT_SCHEMA = 1


@dataclass(frozen=True)
class FingerprintResult:
    document: dict[str, Any]
    content_hash: str


def _local_peaks(
    magnitude: np.ndarray,
    config: FingerprintConfig,
    frequency_bins: np.ndarray,
) -> np.ndarray:
    if magnitude.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    max_amplitude = float(np.max(magnitude))
    if max_amplitude <= 0.0:
        return np.empty((0, 3), dtype=np.float64)
    floor = max_amplitude * (10.0 ** (config.amplitude_floor_db / 20.0))
    time_radius = config.neighborhood_time_frames
    frequency_radius = config.neighborhood_freq_bins
    candidates: list[tuple[float, int, int, int]] = []
    bin_edges = np.searchsorted(frequency_bins, np.asarray(config.band_edges_hz), side="left")
    bin_edges = np.clip(bin_edges, 0, magnitude.shape[1])
    for band_index, (left, right) in enumerate(zip(bin_edges, bin_edges[1:])):
        if right <= left:
            continue
        band = magnitude[:, left:right]
        filtered = maximum_filter(
            band,
            size=(2 * time_radius + 1, 2 * frequency_radius + 1),
            mode="constant",
            cval=-np.inf,
        )
        mask = (band == filtered) & (band >= floor)
        frame_indices, local_bins = np.nonzero(mask)
        for frame, local_bin in zip(frame_indices.tolist(), local_bins.tolist()):
            candidates.append(
                (float(band[frame, local_bin]), int(frame), int(left + local_bin), band_index)
            )
    if not candidates:
        return np.empty((0, 3), dtype=np.float64)

    frames_per_block = max(1, int(round(SAMPLE_RATE / config.hop_samples)))
    block_seconds = frames_per_block * config.hop_samples / SAMPLE_RATE
    per_block = max(1, int(round(config.peak_density * block_seconds)))
    by_block: dict[int, list[tuple[float, int, int, int]]] = {}
    for candidate in candidates:
        block = candidate[1] // frames_per_block
        by_block.setdefault(block, []).append(candidate)
    selected: list[tuple[float, int, int]] = []
    for block in sorted(by_block):
        ordered = sorted(
            by_block[block],
            key=lambda item: (-item[0], item[3], item[2], item[1]),
        )
        for amplitude, frame, frequency_bin, _ in ordered[:per_block]:
            selected.append((amplitude, frame, frequency_bin))
    selected.sort(key=lambda item: (item[1], item[2]))
    return np.asarray(selected, dtype=np.float64)


def _pair_hash(kind: str, payload: str) -> str:
    return sha256_bytes(f"{kind}|{payload}".encode("ascii"))[:16]


def _time_bucket(delta_frames: int) -> int:
    # Half-log buckets are invariant to small tempo changes (e.g. 0.85x..1.15x).
    return int(math.floor(math.log2(max(1, delta_frames)) * 0.5))


def extract_fingerprint(
    samples: np.ndarray,
    config: FingerprintConfig,
    *,
    audio_sha256: str,
) -> FingerprintResult:
    if samples.dtype.kind not in {"i", "u"}:
        raise WorkbenchError("E_AUDIO_FORMAT", "fingerprinting expects integer PCM samples")
    if len(samples) < config.window_samples:
        raise WorkbenchError(
            "E_AUDIO_FORMAT",
            "audio is shorter than one fingerprint window",
            {"samples": int(len(samples)), "window_samples": config.window_samples},
        )
    waveform = np.asarray(samples, dtype=np.float64) / 32768.0
    frequencies, _, complex_spectrum = signal.stft(
        waveform,
        fs=SAMPLE_RATE,
        window="hann",
        nperseg=config.window_samples,
        noverlap=config.window_samples - config.hop_samples,
        nfft=config.window_samples,
        boundary=None,
        padded=False,
        detrend=False,
    )
    # SciPy returns (frequency, time); peak picking and time-difference hashes use (time, frequency).
    magnitude = np.abs(complex_spectrum).T
    peaks = _local_peaks(magnitude, config, frequencies)
    if len(peaks) < 2:
        raise WorkbenchError(
            "E_AUDIO_FORMAT",
            "audio produced too few spectral peaks",
            {"peak_count": int(len(peaks))},
        )

    records: list[dict[str, Any]] = []
    peak_times = peaks[:, 1].astype(np.int64)
    peak_bins = peaks[:, 2].astype(np.int64)
    peak_amplitudes = peaks[:, 0]
    peak_frequencies = frequencies[peak_bins]
    for anchor_index in range(len(peaks)):
        anchor_time = int(peak_times[anchor_index])
        anchor_bin = int(peak_bins[anchor_index])
        anchor_freq = float(peak_frequencies[anchor_index])
        later = np.nonzero(
            (peak_times > anchor_time)
            & (peak_times <= anchor_time + config.target_zone_frames)
        )[0]
        if len(later) == 0:
            continue
        ordered_targets = sorted(
            later.tolist(),
            key=lambda index: (
                -float(peak_amplitudes[index]),
                abs(int(peak_bins[index]) - anchor_bin),
                int(peak_times[index]),
                int(peak_bins[index]),
            ),
        )[: config.fanout]
        selected_targets: list[tuple[int, int, int, float, int]] = []
        for target_index in ordered_targets:
            target_time = int(peak_times[target_index])
            target_bin = int(peak_bins[target_index])
            target_freq = float(peak_frequencies[target_index])
            delta_frames = target_time - anchor_time
            selected_targets.append(
                (target_index, target_time, target_bin, target_freq, delta_frames)
            )
            exact_hash = _pair_hash("E", f"{anchor_bin}|{target_bin}|{delta_frames}")
            records.append(
                {
                    "hash": exact_hash,
                    "kind": "exact",
                    "anchor_frame": anchor_time,
                    "target_frame": target_time,
                    "anchor_bin": anchor_bin,
                    "target_bin": target_bin,
                }
            )
            semitones = 12.0 * math.log2(max(target_freq, 1e-9) / max(anchor_freq, 1e-9))
            if abs(semitones) > 36.0:
                continue
            quarter_tone = int(round(semitones * 2.0))
            delta_bucket = _time_bucket(delta_frames)
            for interval_tolerance in (-1, 0, 1):
                robust_hash = _pair_hash(
                    "R", f"{quarter_tone + interval_tolerance}|{delta_bucket}"
                )
                records.append(
                    {
                        "hash": robust_hash,
                        "kind": "robust",
                        "anchor_frame": anchor_time,
                        "target_frame": target_time,
                        "anchor_bin": anchor_bin,
                        "target_bin": target_bin,
                    }
                )
        for first, second in zip(selected_targets, selected_targets[1:]):
            _, first_time, first_bin, first_freq, first_delta = first
            _, second_time, second_bin, second_freq, second_delta = second
            first_semitones = 12.0 * math.log2(
                max(first_freq, 1e-9) / max(anchor_freq, 1e-9)
            )
            second_semitones = 12.0 * math.log2(
                max(second_freq, 1e-9) / max(anchor_freq, 1e-9)
            )
            between_semitones = 12.0 * math.log2(
                max(second_freq, 1e-9) / max(first_freq, 1e-9)
            )
            if max(abs(first_semitones), abs(second_semitones), abs(between_semitones)) > 48.0:
                continue
            first_q = int(round(first_semitones * 2.0))
            second_q = int(round(second_semitones * 2.0))
            between_q = int(round(between_semitones * 2.0))
            first_bucket = _time_bucket(first_delta)
            second_bucket = _time_bucket(second_delta)
            for tolerance in (-1, 0, 1):
                context_hash = _pair_hash(
                    "C",
                    f"{first_q + tolerance}|{first_bucket}|"
                    f"{second_q + tolerance}|{second_bucket}|{between_q}",
                )
                records.append(
                    {
                        "hash": context_hash,
                        "kind": "context",
                        "anchor_frame": anchor_time,
                        "target_frame": second_time,
                        "anchor_bin": anchor_bin,
                        "target_bin": second_bin,
                    }
                )
        for first, second, third in zip(
            selected_targets, selected_targets[1:], selected_targets[2:]
        ):
            _, first_time, first_bin, first_freq, first_delta = first
            _, second_time, second_bin, second_freq, second_delta = second
            _, third_time, third_bin, third_freq, third_delta = third
            interval_values = []
            for other_freq in (first_freq, second_freq, third_freq):
                semitones = 12.0 * math.log2(
                    max(other_freq, 1e-9) / max(anchor_freq, 1e-9)
                )
                if abs(semitones) > 48.0:
                    break
                interval_values.append(int(round(semitones * 2.0)))
            if len(interval_values) != 3:
                continue
            bucket_values = [
                _time_bucket(first_delta),
                _time_bucket(second_delta),
                _time_bucket(third_delta),
            ]
            for tolerance in (-1, 0, 1):
                sequence_payload = "|".join(
                    str(value + tolerance) for value in interval_values
                ) + "|" + "|".join(str(value) for value in bucket_values)
                sequence_hash = _pair_hash("S", sequence_payload)
                records.append(
                    {
                        "hash": sequence_hash,
                        "kind": "sequence",
                        "anchor_frame": anchor_time,
                        "target_frame": third_time,
                        "anchor_bin": anchor_bin,
                        "target_bin": third_bin,
                    }
                )

    records.sort(
        key=lambda item: (
            item["hash"],
            item["kind"],
            item["anchor_frame"],
            item["target_frame"],
            item["anchor_bin"],
            item["target_bin"],
        )
    )
    payload = {
        "schema": FINGERPRINT_SCHEMA,
        "audio_sha256": audio_sha256,
        "sample_rate": SAMPLE_RATE,
        "duration_samples": int(len(samples)),
        "config": config.to_dict(),
        "config_digest": config.digest,
        "frame_count": int(magnitude.shape[0]),
        "peak_count": int(len(peaks)),
        "fingerprint_count": len(records),
        "fingerprints": records,
    }
    content_hash = digest_object(payload)
    document = dict(payload)
    document["content_hash"] = content_hash
    return FingerprintResult(document=document, content_hash=content_hash)


def validate_fingerprint_document(document: dict[str, Any]) -> dict[str, Any]:
    if document.get("schema") != FINGERPRINT_SCHEMA:
        raise WorkbenchError(
            "E_INDEX_VERSION",
            "unsupported fingerprint schema",
            {"schema": document.get("schema"), "expected": FINGERPRINT_SCHEMA},
        )
    payload = {key: value for key, value in document.items() if key != "content_hash"}
    content_hash = digest_object(payload)
    if content_hash != document.get("content_hash"):
        raise WorkbenchError(
            "E_PARTITION_HASH",
            "fingerprint content hash mismatch",
            {"stored": document.get("content_hash"), "actual": content_hash},
        )
    config = FingerprintConfig.from_dict(document["config"])
    if config.digest != document["config_digest"]:
        raise WorkbenchError(
            "E_CONFIG_RANGE",
            "fingerprint config digest mismatch",
            {"stored": document["config_digest"], "actual": config.digest},
        )
    return document
