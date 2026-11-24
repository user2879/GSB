"""Overlapping window planning, frame features and deterministic embeddings."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import DiarConfig, SAMPLE_RATE

CEPSTRUM_ORDER = 12
F0_MIN_HZ = 50.0
F0_MAX_HZ = 400.0
VOICING_THRESHOLD = 0.3


@dataclass(frozen=True)
class WindowInfo:
    index: int
    start_sample: int
    end_sample: int
    frame_indices: Tuple[int, ...]
    overlaps_with: Tuple[int, ...]


@dataclass
class FrameFeature:
    index: int
    start_sample: int
    end_sample: int
    energy: float
    f0: float
    cepstrum: Tuple[float, ...]


@dataclass
class WindowEmbedding:
    index: int
    info: WindowInfo
    content_key: str
    accumulation_order: Tuple[int, ...]
    raw_feature: Tuple[float, ...]
    embedding: np.ndarray
    frame_features: List[FrameFeature] = field(default_factory=list)


def plan_frames(num_samples: int, cfg: DiarConfig) -> List[Tuple[int, int]]:
    frame_len = int(round(cfg.frame_ms * SAMPLE_RATE / 1000.0))
    frame_hop = int(round(cfg.frame_hop_ms * SAMPLE_RATE / 1000.0))
    frames = []
    start = 0
    while start + frame_len <= num_samples:
        frames.append((start, start + frame_len))
        start += frame_hop
    return frames


def plan_windows(num_samples: int, cfg: DiarConfig) -> List[WindowInfo]:
    win_len = int(round(cfg.window_ms * SAMPLE_RATE / 1000.0))
    hop = int(round(cfg.hop_ms * SAMPLE_RATE / 1000.0))
    starts = list(range(0, max(0, num_samples - win_len) + 1, hop))
    windows: List[WindowInfo] = []
    frame_bounds = plan_frames(num_samples, cfg)
    for index, start in enumerate(starts):
        end = start + win_len
        covered = tuple(f for f, (fs, fe) in enumerate(frame_bounds)
                        if fs >= start and fe <= end)
        windows.append(WindowInfo(index=index, start_sample=start,
                                  end_sample=end, frame_indices=covered,
                                  overlaps_with=()))
    relations: Dict[int, List[int]] = {w.index: [] for w in windows}
    for i, left in enumerate(windows):
        left_frames = set(left.frame_indices)
        for right in windows[i + 1:]:
            if left_frames.intersection(right.frame_indices):
                relations[left.index].append(right.index)
                relations[right.index].append(left.index)
    return [WindowInfo(index=w.index, start_sample=w.start_sample,
                       end_sample=w.end_sample, frame_indices=w.frame_indices,
                       overlaps_with=tuple(relations[w.index]))
            for w in windows]


def _frame_energy(segment: np.ndarray) -> float:
    return float(np.sqrt(np.mean(segment.astype(np.float64) ** 2)))


def _estimate_f0(segment: np.ndarray) -> float:
    windowed = segment * np.hanning(segment.size)
    corr = np.correlate(windowed, windowed, mode="full")[segment.size - 1:]
    if corr[0] <= 0:
        return 0.0
    norm = corr / corr[0]
    min_lag = max(1, int(SAMPLE_RATE / F0_MAX_HZ))
    max_lag = min(segment.size - 1, int(SAMPLE_RATE / F0_MIN_HZ))
    if max_lag <= min_lag:
        return 0.0
    peak_lag = min_lag + int(np.argmax(norm[min_lag:max_lag + 1]))
    if norm[peak_lag] < VOICING_THRESHOLD:
        return 0.0
    return float(SAMPLE_RATE / peak_lag)


def _cepstrum(segment: np.ndarray) -> np.ndarray:
    windowed = segment * np.hanning(segment.size)
    spectrum = np.fft.rfft(windowed)
    log_mag = np.log(np.abs(spectrum) + 1e-8)
    cep = np.fft.irfft(log_mag)
    return cep[:CEPSTRUM_ORDER].astype(np.float64)


def compute_frame_features(samples: Sequence[float],
                           cfg: DiarConfig) -> List[FrameFeature]:
    audio = np.asarray(samples, dtype=np.float64)
    bounds = plan_frames(audio.size, cfg)
    features: List[FrameFeature] = []
    for index, (start, end) in enumerate(bounds):
        segment = audio[start:end]
        cep = _cepstrum(segment)
        features.append(FrameFeature(
            index=index, start_sample=start, end_sample=end,
            energy=_frame_energy(segment), f0=_estimate_f0(segment),
            cepstrum=tuple(float(v) for v in cep),
        ))
    return features


def projection_matrix(raw_dim: int, cfg: DiarConfig) -> np.ndarray:
    """Deterministic fixed projection derived from the config fingerprint.

    Instead of a fully random orthogonal map (which dilutes discriminative
    features), the first columns are identity-aligned so that pitch and
    timbre statistics occupy stable, reproducible embedding coordinates; the
    remaining coordinates use seeded pseudo-random mixing for extra capacity.
    """
    seed_blob = cfg.fingerprint().encode("utf-8")
    seed = int.from_bytes(seed_blob[:8], "big")
    rng = np.random.default_rng(seed ^ (raw_dim << 17) ^ cfg.embed_dim)
    matrix = np.zeros((raw_dim, cfg.embed_dim), dtype=np.float64)
    identity_span = min(raw_dim, cfg.embed_dim)
    for i in range(identity_span):
        matrix[i, i] = 1.0
    if cfg.embed_dim > identity_span:
        mix = rng.standard_normal(
            (raw_dim, cfg.embed_dim - identity_span)).astype(np.float64)
        matrix[:, identity_span:] = 0.35 * mix
    return matrix


def window_content_key(samples: Sequence[float], info: WindowInfo) -> str:
    block = np.asarray(samples[info.start_sample:info.end_sample],
                       dtype=np.float64).tobytes()
    return hashlib.sha256(b"window-content:" + block).hexdigest()


def _window_raw(features: List[FrameFeature], order: Sequence[int]) -> np.ndarray:
    ordered = [features[i] for i in order]
    energies = np.array([f.energy for f in ordered], dtype=np.float64)
    f0_values = np.array([f.f0 for f in ordered if f.f0 > 0], dtype=np.float64)
    if f0_values.size == 0:
        f0_values = np.zeros(1, dtype=np.float64)
    cep_stack = np.vstack([np.asarray(f.cepstrum, dtype=np.float64)
                           for f in ordered])
    # Speaker-discriminative weighting: pitch and low-quefrency cepstral
    # coefficients (voice timbre/formants) dominate; energy is down-weighted
    # because loudness is not a stable speaker trait.
    raw = np.concatenate([
        [0.25 * float(np.mean(energies)),
         0.25 * float(np.std(energies)),
         3.0 * float(np.mean(f0_values)) / F0_MAX_HZ,
         2.0 * float(np.std(f0_values)) / F0_MAX_HZ,
         3.0 * float(np.max(f0_values)) / F0_MAX_HZ],
        2.0 * np.mean(cep_stack, axis=0),
        np.std(cep_stack, axis=0),
    ]).astype(np.float64)
    return raw


def raw_feature_dim() -> int:
    return 5 + 2 * CEPSTRUM_ORDER


def fit_scaler(all_raw: Sequence[np.ndarray]) -> np.ndarray:
    """Per-audio deterministic standardization scales (1/std, floor 1e-6)."""
    stack = np.vstack(list(all_raw))
    std = np.std(stack, axis=0)
    std[std < 1e-6] = 1.0
    return std.astype(np.float64)


def embed_window(raw: np.ndarray, scaler: np.ndarray,
                 projection: np.ndarray, info: WindowInfo,
                 content_key: str) -> WindowEmbedding:
    scaled = raw / scaler
    projected = scaled @ projection
    norm = float(np.linalg.norm(projected))
    embedding = projected / norm if norm > 0 else projected
    return WindowEmbedding(
        index=info.index, info=info, content_key=content_key,
        accumulation_order=tuple(sorted(info.frame_indices)),
        raw_feature=tuple(float(v) for v in raw),
        embedding=embedding.astype(np.float64),
    )
