"""Per-segment deterministic spectral mapping and parametric synthesis."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .config import WorkbenchConfig
from .dsp import _dct_matrix, _hann_window, frame_ranges
from .errors import StageError
from .util import canonical_json, q_div, q_from_float, q_mul, sha256_bytes, sha256_value


def _deq(value: int, cfg: WorkbenchConfig) -> float:
    return int(value) / cfg.scale


def _warp_target_mean(target_mean: list[int], warp: float, cfg: WorkbenchConfig) -> list[float]:
    bands = len(target_mean)
    if bands == 0:
        return []
    center = (bands - 1) / 2.0
    warped: list[float] = []
    for band in range(bands):
        source_position = center + (band - center) / warp
        source_position = max(0.0, min(float(bands - 1), source_position))
        lower = int(math.floor(source_position))
        upper = min(bands - 1, lower + 1)
        fraction = source_position - lower
        value = _deq(target_mean[lower], cfg) * (1.0 - fraction) + _deq(target_mean[upper], cfg) * fraction
        warped.append(value)
    return warped


def _smoothed_band(value: int, previous: int, following: int, smoothing_q: int, cfg: WorkbenchConfig) -> int:
    if smoothing_q <= 0:
        return int(value)
    outer = min(smoothing_q, cfg.scale // 4)
    middle = cfg.scale - 2 * outer
    numerator = outer * int(previous) + middle * int(value) + outer * int(following)
    return q_div(numerator, cfg.scale)


def _cepstral_distance(source_mean: list[int], target_mean: list[int], cfg: WorkbenchConfig) -> float:
    dimensions = min(len(source_mean), len(target_mean))
    if dimensions <= 1:
        return 0.0
    total = 0
    for index in range(1, dimensions):
        difference = int(target_mean[index]) - int(source_mean[index])
        total += difference * difference
    return math.sqrt(total) / cfg.scale


def mapping_key(
    *,
    analysis_key: str,
    local_stats_fingerprint: str,
    target_stats_fingerprint: str,
    effective: dict[str, Any],
    config_fingerprint: str,
) -> str:
    return sha256_value(
        {
            "stage": "segment-mapping",
            "analysis_key": analysis_key,
            "local_stats": local_stats_fingerprint,
            "target_stats": target_stats_fingerprint,
            "params": effective,
            "config": config_fingerprint,
        }
    )


def convert_frames(
    analysis: dict[str, Any],
    local_stats: dict[str, Any],
    target_stats: dict[str, Any],
    effective: dict[str, Any],
    cfg: WorkbenchConfig,
) -> tuple[list[dict[str, Any]], int]:
    source_mean = [int(value) for value in local_stats.get("logmel", {}).get("mean_q", [])]
    target_mean = [int(value) for value in target_stats.get("logmel", {}).get("mean_q", [])]
    if not source_mean:
        source_mean = [0 for _ in range(cfg.mel_bands)]
    if not target_mean:
        target_mean = list(source_mean)
    source_cep = [int(value) for value in local_stats.get("cepstrum", {}).get("mean_q", [])]
    target_cep = [int(value) for value in target_stats.get("cepstrum", {}).get("mean_q", [])]
    strength = _deq(int(effective["timbre_strength_q"]), cfg)
    smoothing_q = int(effective["smoothing_q"])
    pitch_offset = _deq(int(effective["pitch_offset_semitones_q"]), cfg)
    distance = _cepstral_distance(source_cep, target_cep, cfg)
    bend = strength * (distance / (1.0 + distance)) if distance > 0 else 0.0
    bend_q = q_from_float(bend, cfg.scale)
    # The explicit formant control is curved by the strength-weighted long-term
    # source/target cepstral distance. This term is part of the mapping identity
    # through the effective params and config fingerprints.
    warp = max(0.5, _deq(int(effective["formant_warp_q"]), cfg) * (1.0 + 0.35 * _deq(bend_q, cfg)))
    warped_target = _warp_target_mean(target_mean, warp, cfg)
    dct = _dct_matrix(cfg.cepstrum_order, cfg.mel_bands)
    converted_frames: list[dict[str, Any]] = []
    for frame in analysis.get("frames", []):
        source_logmel = [int(value) for value in frame["mel_energy_q"]]
        converted = [0 for _ in range(cfg.mel_bands)]
        for band in range(cfg.mel_bands):
            delta = warped_target[band] - _deq(source_mean[band], cfg)
            base = _deq(source_logmel[band], cfg)
            converted[band] = q_from_float(base + strength * delta, cfg.scale)
        smoothed = [
            _smoothed_band(
                converted[band],
                converted[max(0, band - 1)],
                converted[min(cfg.mel_bands - 1, band + 1)],
                smoothing_q,
                cfg,
            )
            for band in range(cfg.mel_bands)
        ]
        logmel_float = np.asarray([_deq(value, cfg) for value in smoothed], dtype=np.float64)
        cepstrum = dct @ logmel_float
        source_f0 = int(frame.get("source_f0_hz_q", 0))
        if source_f0 > 0:
            output_f0_q = q_from_float(_deq(source_f0, cfg) * (2.0 ** (pitch_offset / 12.0)), cfg.scale)
        else:
            output_f0_q = 0
        converted_frames.append(
            {
                "frame_index": int(frame["frame_index"]),
                "start_sample": int(frame.get("start_sample", 0)),
                "valid_samples": int(frame.get("valid_samples", cfg.frame_length_samples)),
                "padded": bool(frame.get("padded", False)),
                "source_f0_unit": int(frame.get("f0_unit", -1)),
                "source_f0_hz_q": source_f0,
                "output_f0_hz_q": int(output_f0_q),
                "energy_q": int(frame.get("energy_q", 0)),
                "envelope_energy_q": [int(value) for value in smoothed],
                "cepstrum_q": [q_from_float(float(value), cfg.scale) for value in cepstrum],
            }
        )
    return converted_frames, bend_q


def _mel_position(frequency_hz: float, cfg: WorkbenchConfig) -> float:
    def hz_to_mel(hz: float) -> float:
        return 2595.0 * math.log10(1.0 + hz / 700.0)

    mel_low = hz_to_mel(0.0)
    mel_high = hz_to_mel(cfg.mel_max_hz)
    position = (hz_to_mel(max(0.0, frequency_hz)) - mel_low) / (mel_high - mel_low) * cfg.mel_bands - 0.5
    return max(0.0, min(cfg.mel_bands - 1.0, position))


def _envelope_level(levels: list[int], frequency_hz: float, cfg: WorkbenchConfig) -> float:
    position = _mel_position(frequency_hz, cfg)
    lower = int(math.floor(position))
    upper = min(cfg.mel_bands - 1, lower + 1)
    fraction = position - lower
    return _deq(levels[lower], cfg) * (1.0 - fraction) + _deq(levels[upper], cfg) * fraction


def _deterministic_noise(seed_text: str, count: int) -> np.ndarray:
    seed = int(sha256_bytes(seed_text.encode("utf-8"))[:16], 16)
    state = seed & ((1 << 64) - 1)
    values = np.empty(count, dtype=np.float64)
    for index in range(count):
        state = (6364136223846793005 * state + 1442695040888963407) & ((1 << 64) - 1)
        values[index] = ((state >> 11) / float(1 << 53)) * 2.0 - 1.0
    return values


def synthesize_pcm(
    frames: list[dict[str, Any]],
    sample_count: int,
    segment_id: str,
    cfg: WorkbenchConfig,
) -> bytes:
    if sample_count <= 0:
        return b""
    length = cfg.frame_length_samples
    shift = cfg.frame_shift_samples
    output = np.zeros(sample_count, dtype=np.float64)
    window_sum = np.zeros(sample_count, dtype=np.float64)
    window = _hann_window(length)
    nyquist = cfg.sample_rate / 2.0
    max_harmonic_hz = min(5000.0, nyquist * 0.98)
    for frame in frames:
        frame_index = int(frame["frame_index"])
        start = frame_index * shift
        if start >= sample_count:
            continue
        local_len = min(length, sample_count - start)
        f0_hz = _deq(int(frame["output_f0_hz_q"]), cfg)
        rms = _deq(int(frame.get("energy_q", 0)), cfg)
        levels = [int(value) for value in frame["envelope_energy_q"]]
        mean_level = _deq(sum(levels), cfg) / max(1, len(levels))
        waveform = np.zeros(length, dtype=np.float64)
        if f0_hz >= cfg.f0_min_hz / 2.0:
            max_harmonic = max(1, int(math.floor(min(max_harmonic_hz, nyquist) / f0_hz)))
            max_harmonic = min(max_harmonic, 80)
            for harmonic in range(1, max_harmonic + 1):
                frequency = harmonic * f0_hz
                relative_db = _envelope_level(levels, frequency, cfg) - mean_level
                relative_db = max(-45.0, min(24.0, relative_db))
                amplitude = (10.0 ** (relative_db / 20.0)) / math.sqrt(harmonic)
                phase = 0.13 * harmonic * harmonic + 0.017 * frame_index * harmonic
                for sample_index in range(length):
                    t = (start + sample_index) / cfg.sample_rate
                    waveform[sample_index] += amplitude * math.sin(2.0 * math.pi * frequency * t + phase)
        else:
            waveform[:] = _deterministic_noise(f"{segment_id}:{frame_index}", length)
        current_rms = float(np.sqrt(np.mean(np.square(waveform), dtype=np.float64)))
        target_rms = rms if rms > 1e-9 else 0.0
        if current_rms > 1e-12 and target_rms > 0:
            waveform *= target_rms / current_rms
        else:
            waveform.fill(0.0)
        if local_len > 0:
            output[start : start + local_len] += waveform[:local_len] * window[:local_len]
            window_sum[start : start + local_len] += window[:local_len]
    valid = window_sum > 1e-9
    output[valid] /= window_sum[valid]
    output[~valid] = 0.0
    gain = 10.0 ** (cfg.output_gain_db / 20.0)
    output *= gain
    pcm = np.clip(np.rint(output * 32767.0), -32768, 32767).astype("<i2", copy=False)
    return pcm.tobytes()


def build_mapping(
    *,
    segment_id: str,
    segment_index: int,
    analysis: dict[str, Any],
    local_stats: dict[str, Any],
    target_stats: dict[str, Any],
    effective: dict[str, Any],
    cfg: WorkbenchConfig,
) -> dict[str, Any]:
    key = mapping_key(
        analysis_key=str(analysis["analysis_key"]),
        local_stats_fingerprint=str(local_stats["fingerprint"]),
        target_stats_fingerprint=str(target_stats.get("fingerprint", "empty")),
        effective=effective,
        config_fingerprint=cfg.mapping_fingerprint(),
    )
    try:
        frames, bend_q = convert_frames(analysis, local_stats, target_stats, effective, cfg)
    except Exception as exc:
        raise StageError("mapping", "E_SEGMENT_FAILURE", type(exc).__name__, f"spectral mapping failed: {exc}") from exc
    try:
        pcm = synthesize_pcm(frames, int(analysis["sample_count"]), segment_id, cfg)
    except Exception as exc:
        raise StageError("synthesis", "E_SEGMENT_FAILURE", type(exc).__name__, f"segment synthesis failed: {exc}") from exc
    pcm_sha = sha256_bytes(pcm)
    segment_fingerprint = sha256_value(
        {
            "mapping_key": key,
            "pcm_sha256": pcm_sha,
            "sample_count": len(pcm) // 2,
            "frame_summaries": frames,
        }
    )
    return {
        "segment_id": segment_id,
        "segment_index": int(segment_index),
        "mapping_key": key,
        "analysis_key": str(analysis["analysis_key"]),
        "local_stats_key": str(local_stats.get("cache_key", "")),
        "local_stats_fingerprint": str(local_stats.get("fingerprint", "")),
        "target_stats_key": str(target_stats.get("cache_key", "")),
        "target_stats_fingerprint": str(target_stats.get("fingerprint", "empty")),
        "params": dict(sorted(effective.items())),
        "bend_q": int(bend_q),
        "frames": frames,
        "pcm_sha256": pcm_sha,
        "sample_count": len(pcm) // 2,
        "segment_fingerprint": segment_fingerprint,
        "mapping_status": "mapped",
        "pcm": pcm,
    }


def public_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in mapping.items() if key not in {"pcm", "piece_file", "mapping_status"}}
    result["frame_summaries"] = result.pop("frames", [])
    return result


def mapping_identity(mapping: dict[str, Any]) -> str:
    return sha256_value(
        {
            "mapping_key": mapping.get("mapping_key"),
            "segment_fingerprint": mapping.get("segment_fingerprint"),
        }
    )


def effective_public(effective: dict[str, Any]) -> str:
    return canonical_json(effective)
