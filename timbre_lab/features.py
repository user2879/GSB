from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .audio import AudioFile
from .config import AnalysisConfig, digest_json
from .model import Segment


EPS = 1e-12


@dataclass
class FrameFeatures:
    frame_index: int
    absolute_frame_index: int
    start_sample: int
    rms: float
    rms_db: float
    f0_hz: float | None
    f1_hz: float | None
    f2_hz: float | None
    voiced: bool


def _db(rms: float) -> float:
    return 20.0 * math.log10(max(rms, 1e-12))


def _estimate_f0(frame: np.ndarray, config: AnalysisConfig) -> tuple[float | None, float]:
    x = frame.astype(np.float64)
    x = x - float(np.mean(x))
    if float(np.sqrt(np.mean(x * x))) < 1e-7:
        return None, 0.0
    x = x * np.hanning(x.size)
    n = x.size
    nfft = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(x, nfft)
    corr = np.fft.irfft(spectrum * np.conj(spectrum), nfft)[:n]
    if corr[0] <= EPS:
        return None, 0.0
    corr = corr / corr[0]
    min_lag = max(1, int(math.floor(config.sample_rate / config.f0_max_hz)))
    max_lag = min(n - 2, int(math.ceil(config.sample_rate / config.f0_min_hz)))
    if max_lag <= min_lag + 1:
        return None, 0.0
    search = corr[min_lag : max_lag + 1]
    best_relative = int(np.argmax(search))
    best = float(search[best_relative])
    if best < config.voiced_autocorr:
        return None, best
    local_peaks: list[int] = []
    for index in range(1, len(search) - 1):
        if search[index] >= search[index - 1] and search[index] >= search[index + 1]:
            if search[index] >= best * 0.9:
                local_peaks.append(index)
    chosen = local_peaks[0] if local_peaks else best_relative
    lag = min_lag + chosen
    refined = float(lag)
    if 0 < lag < n - 1:
        left, center, right = corr[lag - 1], corr[lag], corr[lag + 1]
        denominator = left - 2.0 * center + right
        if abs(denominator) > EPS:
            refined = lag + 0.5 * (left - right) / denominator
    if refined <= 0:
        return None, best
    f0 = config.sample_rate / refined
    if not (config.f0_min_hz <= f0 <= config.f0_max_hz):
        return None, best
    return float(f0), best


def _lpc_formants(frame: np.ndarray, config: AnalysisConfig) -> tuple[float | None, float | None]:
    x = frame.astype(np.float64)
    x = (x - float(np.mean(x))) * np.hamming(x.size)
    order = max(4, min(config.lpc_order, x.size - 2))
    nfft = 1 << (2 * x.size - 1).bit_length()
    spectrum = np.fft.rfft(x, nfft)
    corr = np.fft.irfft(spectrum * np.conj(spectrum), nfft)[: order + 1]
    if corr[0] <= EPS:
        return None, None
    corr = corr / corr[0]
    corr[0] += 1e-8
    matrix = np.empty((order, order), dtype=np.float64)
    for row in range(order):
        for column in range(order):
            matrix[row, column] = corr[abs(row - column)]
    try:
        coefficients = np.linalg.solve(matrix, corr[1 : order + 1])
        roots = np.roots(np.concatenate(([1.0], -coefficients)))
    except (np.linalg.LinAlgError, ValueError):
        return _spectral_formants(x, config)
    candidates: list[tuple[float, float, float]] = []
    for root in roots:
        if root.imag <= 0:
            continue
        frequency = abs(float(np.angle(root))) / (2.0 * math.pi) * config.sample_rate
        radius = abs(root)
        if radius <= EPS or frequency < 80.0 or frequency >= config.sample_rate / 2.0:
            continue
        bandwidth = -math.log(min(radius, 1.0)) / math.pi * config.sample_rate
        if bandwidth > 700.0:
            continue
        candidates.append((frequency, bandwidth, radius))
    f1_candidates = [item for item in candidates if config.f1_range_hz[0] <= item[0] <= config.f1_range_hz[1]]
    if not f1_candidates:
        return _spectral_formants(x, config)
    f1 = min(f1_candidates, key=lambda item: (item[1], item[0]))[0]
    f2_candidates = [
        item
        for item in candidates
        if config.f2_range_hz[0] <= item[0] <= config.f2_range_hz[1] and item[0] > f1 + 120.0
    ]
    if not f2_candidates:
        return float(f1), _spectral_peak(x, config, config.f2_range_hz)
    f2 = min(f2_candidates, key=lambda item: (item[1], item[0]))[0]
    return float(f1), float(f2)


def _spectral_peak(frame: np.ndarray, config: AnalysisConfig, frequency_range: tuple[float, float]) -> float | None:
    spectrum = np.abs(np.fft.rfft(frame * np.hanning(frame.size)))
    frequencies = np.fft.rfftfreq(frame.size, d=1.0 / config.sample_rate)
    mask = (frequencies >= frequency_range[0]) & (frequencies <= frequency_range[1])
    if not np.any(mask):
        return None
    indices = np.flatnonzero(mask)
    local = spectrum[indices]
    if local.size >= 9:
        kernel = np.ones(9) / 9.0
        local = np.convolve(local, kernel, mode="same")
    return float(frequencies[indices[int(np.argmax(local))]])


def _spectral_formants(frame: np.ndarray, config: AnalysisConfig) -> tuple[float | None, float | None]:
    return (
        _spectral_peak(frame, config, config.f1_range_hz),
        _spectral_peak(frame, config, config.f2_range_hz),
    )


def _speech_rate(frame_db: np.ndarray, hop_seconds: float) -> tuple[float, int]:
    if frame_db.size == 0:
        return 0.0, 0
    envelope = frame_db - float(np.mean(frame_db))
    if frame_db.size >= 5:
        envelope = np.convolve(envelope, np.ones(5) / 5.0, mode="same")
    threshold = float(np.std(envelope)) * 0.15
    max_peaks = max(1, int(round(0.12 / hop_seconds)))
    peaks: list[int] = []
    last_peak = -max_peaks
    for index in range(1, envelope.size - 1):
        if envelope[index] < threshold:
            continue
        if envelope[index] >= envelope[index - 1] and envelope[index] > envelope[index + 1]:
            if index - last_peak >= max_peaks:
                peaks.append(index)
                last_peak = index
    duration = max(frame_db.size * hop_seconds, hop_seconds)
    return float(len(peaks) / duration), len(peaks)


def _quantile(values: list[float], q: float, method: str) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=np.float64), q, method=method))


def _content_fingerprint(samples: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(len(samples).to_bytes(8, "little", signed=False))
    digest.update(samples.astype("<i2", copy=False).tobytes())
    return digest.hexdigest()


def segment_content_fingerprint(audio: AudioFile, segment: Segment) -> str:
    start, end = segment.start_sample, segment.end_sample
    start = max(0, min(start, audio.sample_count))
    end = max(start, min(end, audio.sample_count))
    return _content_fingerprint(audio.samples[start:end])


def feature_cache_key(content_fingerprint: str, config: AnalysisConfig, segment: Segment | None = None) -> str:
    binding = None
    if segment is not None:
        binding = [segment.file, segment.start_sample, segment.end_sample, segment.start_frame]
    return digest_json(
        {
            "content": content_fingerprint,
            "feature_config": config.feature_fingerprint(),
            "frame": [config.frame_length_samples, config.frame_shift_samples],
            "segment_binding": binding,
        }
    )


def extract_segment_features(audio: AudioFile, segment: Segment, config: AnalysisConfig) -> dict[str, Any]:
    start = max(0, min(segment.start_sample, audio.sample_count))
    end = max(start, min(segment.end_sample, audio.sample_count))
    samples = audio.samples[start:end]
    content_fingerprint = _content_fingerprint(samples)
    cache_key = feature_cache_key(content_fingerprint, config, segment)
    frame_length = config.frame_length_samples
    hop = config.frame_shift_samples
    if samples.size < frame_length:
        raise ValueError(f"segment {segment.segment_id} is shorter than one frame")
    frame_count = 1 + (samples.size - frame_length) // hop
    frames: list[FrameFeatures] = []
    for ordinal in range(frame_count):
        frame_start = ordinal * hop
        frame = samples[frame_start : frame_start + frame_length]
        normalized = frame.astype(np.float64) / 32768.0
        rms = float(np.sqrt(np.mean(normalized * normalized)))
        rms_db = _db(rms)
        if rms_db < config.energy_exit_db:
            frames.append(
                FrameFeatures(
                    frame_index=ordinal,
                    absolute_frame_index=(start + frame_start)//hop,
                    start_sample=start + frame_start,
                    rms=rms,
                    rms_db=rms_db,
                    f0_hz=None,
                    f1_hz=None,
                    f2_hz=None,
                    voiced=False,
                )
            )
            continue
        f0, autocorr_score = _estimate_f0(frame, config)
        voiced = f0 is not None
        if voiced:
            f1, f2 = _lpc_formants(frame, config)
        else:
            f1, f2 = None, None
        frames.append(
            FrameFeatures(
                frame_index=ordinal,
                absolute_frame_index=(start + frame_start)//hop,
                start_sample=start + frame_start,
                rms=rms,
                rms_db=rms_db,
                f0_hz=f0,
                f1_hz=f1,
                f2_hz=f2,
                voiced=voiced,
            )
        )
    valid_frames = [frame.rms_db >= config.energy_exit_db for frame in frames]
    effective_count = int(sum(valid_frames))
    f0_values = [float(frame.f0_hz) for frame in frames if frame.f0_hz is not None]
    f1_values = [float(frame.f1_hz) for frame in frames if frame.f1_hz is not None]
    f2_values = [float(frame.f2_hz) for frame in frames if frame.f2_hz is not None]
    db_values = [float(frame.rms_db) for frame in frames if frame.rms_db >= config.energy_exit_db]
    speech_rate, syllable_count = _speech_rate(np.asarray(db_values, dtype=np.float64), hop / config.sample_rate)
    f0_mean = float(np.mean(f0_values)) if f0_values else 0.0
    f0_std = float(np.std(f0_values)) if f0_values else 0.0
    f1_mean = float(np.mean(f1_values)) if f1_values else 0.0
    f2_mean = float(np.mean(f2_values)) if f2_values else 0.0
    voiced_ratio = len(f0_values) / effective_count if effective_count else 0.0
    rms_db = float(np.mean(db_values)) if db_values else -120.0
    if f0_std >= config.high_variation_f0_std_hz:
        style = config.high_variation_label
    elif speech_rate < config.style_rate_bins_sps[0]:
        style = config.style_labels[0]
    elif speech_rate > config.style_rate_bins_sps[1]:
        style = config.style_labels[2]
    else:
        style = config.style_labels[1]
    statistics = {
        "f0_mean_hz": f0_mean,
        "f0_std_hz": f0_std,
        "f0_median_hz": _quantile(f0_values, 0.5, config.quantile_interpolation),
        "f0_p10_hz": _quantile(f0_values, 0.1, config.quantile_interpolation),
        "f0_p90_hz": _quantile(f0_values, 0.9, config.quantile_interpolation),
        "f1_mean_hz": f1_mean,
        "f2_mean_hz": f2_mean,
        "speech_rate_sps": speech_rate,
        "voiced_ratio": voiced_ratio,
        "rms_db": rms_db,
    }
    return {
        "segment_id": segment.segment_id,
        "file": segment.file,
        "content_fingerprint": content_fingerprint,
        "cache_key": cache_key,
        "frame_count": frame_count,
        "effective_frame_count": effective_count,
        "voiced_frame_count": len(f0_values),
        "syllable_count": syllable_count,
        "quantile_interpolation": config.quantile_interpolation,
        "statistics": statistics,
        "pronunciation_style": style,
        "style_hint": segment.style_hint,
        "frame_index_order": [frame.frame_index for frame in frames],
        "frame_summary": [
            {
                "frame_index": frame.frame_index,
                "absolute_frame_index": frame.absolute_frame_index,
                "rms_db": frame.rms_db,
                "f0_hz": frame.f0_hz,
                "f1_hz": frame.f1_hz,
                "f2_hz": frame.f2_hz,
                "voiced": frame.voiced,
            }
            for frame in frames
        ],
    }


def canonical_features_for_compare(features: dict[str, Any]) -> dict[str, Any]:
    ignored = {"cache_key", "frame_summary", "frame_index_order", "cache_status", "upstream"}
    return {key: value for key, value in features.items() if key not in ignored}
