"""Frame-level acoustic features and word-level emission scoring.

All arithmetic is float64. Rounding to the configured significant digits is
applied only when values are written into artifacts; the same inputs always
produce byte-identical outputs (no RNG, fixed filterbank geometry).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from .audio import SAMPLE_RATE, FramePlan, plan_frames
from .config import AcousticConfig, PhonemeMap, round_sig


def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(n_bands: int, n_fft: int, sample_rate: int) -> np.ndarray:
    """Deterministic triangular mel filterbank, shape (n_bands, n_fft//2+1)."""
    low_mel = _hz_to_mel(0.0)
    high_mel = _hz_to_mel(sample_rate / 2.0)
    points_mel = np.linspace(low_mel, high_mel, n_bands + 2, dtype=np.float64)
    points_hz = np.array([_mel_to_hz(value) for value in points_mel], dtype=np.float64)
    bins = np.floor((n_fft + 1) * points_hz / sample_rate).astype(np.int64)
    n_spec = n_fft // 2 + 1
    bank = np.zeros((n_bands, n_spec), dtype=np.float64)
    for band in range(n_bands):
        left, center, right = int(bins[band]), int(bins[band + 1]), int(bins[band + 2])
        if center == left:
            center = left + 1
        if right <= center:
            right = center + 1
        for spec_bin in range(left, min(center, n_spec)):
            if center > left:
                bank[band, spec_bin] = (spec_bin - left) / (center - left)
        for spec_bin in range(center, min(right, n_spec)):
            if right > center:
                bank[band, spec_bin] = (right - spec_bin) / (right - center)
    return bank


def extract_frame_waveforms(samples: np.ndarray, plan: FramePlan) -> np.ndarray:
    """Return frames as (n_frames, frame_length), zero padding the last frame."""
    n_frames = plan.n_frames
    frames = np.zeros((n_frames, plan.frame_length), dtype=np.float64)
    for index in range(n_frames):
        start = plan.frame_start(index)
        end = min(start + plan.frame_length, samples.shape[0])
        frames[index, :end - start] = samples[start:end]
    return frames


def log_mel_energies(samples: np.ndarray, plan: FramePlan, n_bands: int) -> np.ndarray:
    frames = extract_frame_waveforms(samples, plan)
    n_fft = 1
    while n_fft < plan.frame_length:
        n_fft *= 2
    window = np.hanning(plan.frame_length).astype(np.float64) + 1e-12
    spectrum = np.fft.rfft(frames * window, n=n_fft, axis=1)
    power = (np.abs(spectrum) ** 2).astype(np.float64)
    bank = mel_filterbank(n_bands, n_fft, SAMPLE_RATE)
    mel_energy = power @ bank.T
    log_mel = np.log(mel_energy + 1e-9).astype(np.float64)
    return log_mel


@dataclass(frozen=True)
class FrameFeatures:
    phonemes: Tuple[str, ...]
    emissions: np.ndarray  # (n_frames, n_words) float64, rounded artifact values
    log_mel: np.ndarray    # (n_frames, n_bands) float64, rounded artifact values
    plan: FramePlan
    words: Tuple[str, ...]


def _classify(log_mel: np.ndarray, centroids: Dict[str, List[float]]) -> Tuple[str, ...]:
    phonemes = sorted(centroids)
    matrix = np.array([centroids[phoneme] for phoneme in phonemes], dtype=np.float64)
    # squared L2 distance, ties broken by sorted phoneme order (argmax of -dist)
    distances = ((log_mel[:, None, :] - matrix[None, :, :]) ** 2).sum(axis=2)
    indices = np.argmax(-distances, axis=1)
    return tuple(phonemes[int(index)] for index in indices)


def compute_features(samples: np.ndarray, acoustic: AcousticConfig,
                     phoneme_map: PhonemeMap, significant_digits: int) -> FrameFeatures:
    plan = plan_frames(samples.shape[0], acoustic.frame_length_ms, acoustic.frame_shift_ms)
    log_mel = log_mel_energies(samples, plan, acoustic.mel_bands)
    phonemes = _classify(log_mel, acoustic.centroids)
    words = tuple(phoneme_map.words)
    smooth = acoustic.log_smooth
    emissions = np.full((plan.n_frames, len(words)), smooth, dtype=np.float64)
    word_index = {word: idx for idx, word in enumerate(words)}
    for frame_index, phoneme in enumerate(phonemes):
        rows = acoustic.emission_table.get(phoneme, {})
        for word, score in rows.items():
            emissions[frame_index, word_index[word]] = float(score)
        for idx in range(len(words)):
            word = words[idx]
            if word not in rows:
                emissions[frame_index, idx] = smooth
    rounded_emissions = round_sig(emissions, significant_digits)
    rounded_log_mel = round_sig(log_mel, significant_digits)
    return FrameFeatures(phonemes=phonemes, emissions=rounded_emissions,
                         log_mel=rounded_log_mel, plan=plan, words=words)


def features_artifact(features: FrameFeatures, source: str,
                      frame_shift_ms: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for frame_index in range(features.plan.n_frames):
        start = features.plan.frame_start(frame_index)
        is_last = frame_index == features.plan.n_frames - 1 and features.plan.last_frame_padded
        rows.append({
            "source": source,
            "frame_index": frame_index,
            "start_sample": int(start),
            "start_ms": int(start * 1000 / SAMPLE_RATE),
            "last_frame_zero_padded": bool(is_last),
            "phoneme": features.phonemes[frame_index],
            "log_mel": [float(value) for value in features.log_mel[frame_index]],
            "emissions": {
                word: float(features.emissions[frame_index, idx])
                for idx, word in enumerate(features.words)
            },
        })
    return rows
