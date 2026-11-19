"""RIR event extraction, spherical sampling map, and SH projection inputs."""

from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import find_peaks

from .canonical import canonical_hash, json_safe, sha256_file
from .config import Config
from .sphere import fibonacci_sphere, nearest_indices, spherical_weights


def read_pcm_wav(path: Path) -> tuple[np.ndarray, dict[str, int]]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.getnframes()
        payload = handle.readframes(frames)
    if width != 2:
        raise ValueError(f"expected 16-bit PCM WAV: {path}")
    data = np.frombuffer(payload, dtype="<i2").astype(np.float64) / 32768.0
    if channels <= 0 or data.size != frames * channels:
        raise ValueError(f"malformed PCM payload: {path}")
    return data.reshape(frames, channels), {
        "channels": channels,
        "sample_rate": rate,
        "frames": frames,
        "bit_depth": width * 8,
    }


def extract_foa_events(
    samples: np.ndarray,
    sample_rate: int,
    relative_peak_threshold: float = 1.0e-6,
) -> list[dict[str, Any]]:
    """Extract separated FOA arrivals from a four-channel RIR.

    Channel 0 is W and channels 1..3 are X/Y/Z.  Each detected energy peak
    produces one observation with an FOA-derived direction and a local energy.
    No truth metadata is consumed here.
    """

    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 4:
        raise ValueError("FOA RIR must have shape (frames, 4)")
    energy = np.sum(samples * samples, axis=1)
    maximum = float(np.max(energy)) if energy.size else 0.0
    if maximum <= 0.0:
        return []
    peaks, _ = find_peaks(
        energy,
        height=maximum * float(relative_peak_threshold),
        distance=2,
    )
    events: list[dict[str, Any]] = []
    for peak in peaks:
        lower = max(0, int(peak) - 1)
        upper = min(len(samples), int(peak) + 2)
        window = samples[lower:upper]
        # Active-intensity direction: W multiplied by X/Y/Z preserves the
        # pressure sign, avoiding a 180-degree flip for odd-order reflections.
        vector = np.sum(window[:, 0:1] * window[:, 1:4], axis=0)
        norm = float(np.linalg.norm(vector))
        if norm <= 1.0e-12:
            continue
        direction = vector / norm
        events.append(
            {
                "arrival_samples": int(peak),
                "arrival_seconds": int(peak) / float(sample_rate),
                "direction": direction.tolist(),
                "energy": float(np.sum(energy[lower:upper])),
            }
        )
    return events


def _event_bins(events: list[dict[str, Any]], sample_directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not events:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    directions = np.asarray([event["direction"] for event in events], dtype=np.float64)
    energies = np.asarray([event["energy"] for event in events], dtype=np.float64)
    bins = np.argmax(directions @ sample_directions.T, axis=1).astype(np.int64)
    return bins, energies


def project_tier(
    config: Config,
    tier: dict[str, Any],
    manifest: dict[str, Any],
    tier_dir: Path,
) -> dict[str, Any]:
    sphere_points = int(tier["sphere_points"])
    sample_directions = fibonacci_sphere(sphere_points)
    sample_weights = spherical_weights(sphere_points)
    counts = np.zeros(sphere_points, dtype=np.int64)
    energy_sum = np.zeros(sphere_points, dtype=np.float64)
    earliest = np.full(sphere_points, np.inf, dtype=np.float64)
    latest = np.full(sphere_points, -np.inf, dtype=np.float64)
    event_total = 0
    invalid_events = 0
    source_event_counts: list[int] = []

    for source in manifest["sources"]:
        wav_path = Path(tier_dir) / source["wav"]["path"]
        expected_hash = source["wav"]["sha256"]
        if sha256_file(wav_path) != expected_hash:
            raise ValueError(f"RIR hash mismatch: {wav_path}")
        samples, audio_info = read_pcm_wav(wav_path)
        if audio_info["channels"] != 4 or audio_info["sample_rate"] != int(config.audio()["sample_rate"]):
            raise ValueError(f"unexpected RIR format: {wav_path}")
        events = extract_foa_events(samples, audio_info["sample_rate"])
        if not events:
            invalid_events += 1
        source_event_counts.append(len(events))
        event_total += len(events)
        bins, energies = _event_bins(events, sample_directions)
        for event, bin_index, energy in zip(events, bins, energies):
            index = int(bin_index)
            counts[index] += 1
            energy_sum[index] += float(energy)
            arrival = float(event["arrival_seconds"])
            earliest[index] = min(float(earliest[index]), arrival)
            latest[index] = max(float(latest[index]), arrival)

    min_samples = int(config.projection()["min_effective_samples"])
    low_confidence = counts < min_samples
    source_count = int(tier["source_positions"])
    mean_energy = energy_sum / float(source_count)
    source_event_counts_array = np.asarray(source_event_counts, dtype=np.int64)
    if int(np.sum(counts)) != event_total:
        raise RuntimeError("sampling bins lost or duplicated an event")

    projection: dict[str, Any] = {
        "schema": "spatial-diagnostic-projection/1",
        "tier": tier["name"],
        "source_positions": source_count,
        "sphere_points": sphere_points,
        "sample_directions": sample_directions.tolist(),
        "sample_weights": sample_weights.tolist(),
        "weight_rule": "equal-area Fibonacci quadrature: w_i = 4*pi/N",
        "normalization": "orthonormal real spherical harmonics, ACN ordering",
        "confidence_rule": f"effective_count >= {min_samples}",
        "min_effective_samples": min_samples,
        "effective_sample_count": counts.tolist(),
        "observed_energy": mean_energy.tolist(),
        "low_confidence": low_confidence.astype(np.int8).tolist(),
        "earliest_arrival_seconds": [
            None if not np.isfinite(value) else float(value) for value in earliest
        ],
        "latest_arrival_seconds": [
            None if not np.isfinite(value) else float(value) for value in latest
        ],
        "event_total": int(event_total),
        "binned_event_total": int(np.sum(counts)),
        "invalid_source_total": int(invalid_events),
        "source_event_counts": source_event_counts_array.tolist(),
        "low_confidence_direction_count": int(np.sum(low_confidence)),
        "fit_eligible_direction_count": int(np.sum(~low_confidence)),
        "fit_eligible_mass_fraction": float(
            np.sum(energy_sum[~low_confidence]) / np.sum(energy_sum) if np.sum(energy_sum) > 0 else 0.0
        ),
        "conservation": {
            "events_equal_bin_counts": bool(np.sum(counts) == event_total),
            "every_event_has_one_bin": True,
            "low_confidence_excluded_from_fit": True,
        },
    }
    projection["content_hash"] = canonical_hash(projection)
    return json_safe(projection)
