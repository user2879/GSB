"""Frame splitting and deterministic per-frame feature extraction.

All arithmetic runs in ``numpy.float64``.  Every emitted number is rounded
to 6 decimal places so that repeated runs produce byte-identical records.
"""

import numpy as np

from .audioio import SAMPLE_RATE
from .errors import FeatureError
from .fingerprint import round6

_EPS = 1e-12


def frame_offsets(n_samples, frame_len, frame_shift):
    """Return zero-based sample offsets of every frame.

    The final frame is kept even when fewer than ``frame_len`` samples
    remain; callers zero-pad it and flag ``padded=True``.
    """
    if n_samples <= 0:
        return []
    offsets = list(range(0, n_samples, frame_shift))
    if offsets[-1] + frame_len > n_samples and offsets[-1] == 0:
        return offsets
    return offsets


def extract_frame(samples, start, frame_len, feature_dim):
    """Extract one zero-padded frame and its feature vector."""
    n = len(samples)
    available = max(0, min(frame_len, n - start))
    padded = start + frame_len > n
    frame = np.zeros(frame_len, dtype=np.float64)
    frame[:available] = samples[start : start + available]

    windowed = frame * np.hanning(frame_len).astype(np.float64)
    energy = float(np.mean(frame * frame))

    spectrum = np.fft.rfft(windowed).astype(np.complex128)
    power = (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float64)
    positive = power[1:] if power.size > 1 else power
    geometric = float(np.exp(np.mean(np.log(positive + _EPS))))
    arithmetic = float(np.mean(positive)) + _EPS
    flatness = geometric / arithmetic

    voiceprint = _voiceprint(power, feature_dim)
    return {
        "energy": round6(energy),
        "spectral_flatness": round6(flatness),
        "voiceprint": [round6(v) for v in voiceprint],
        "padded": bool(padded),
    }


def _voiceprint(power, feature_dim):
    """Fixed-dim voiceprint: log band energies + DCT + centroid."""
    n_bins = power.size
    bands = max(1, feature_dim - 1)
    edges = np.linspace(0, n_bins, bands + 1).astype(np.int64)
    log_energies = np.empty(bands, dtype=np.float64)
    for idx in range(bands):
        lo, hi = int(edges[idx]), max(int(edges[idx + 1]), int(edges[idx]) + 1)
        log_energies[idx] = np.log(float(np.mean(power[lo:hi])) + _EPS)
    # Orthogonal DCT-II over log band energies (MFCC style).
    dct = np.empty(bands, dtype=np.float64)
    for k in range(bands):
        dct[k] = np.sum(
            log_energies
            * np.cos(
                np.pi * (np.arange(bands) + 0.5) * k / bands
            )
        )
    centroid = float(
        np.sum(np.arange(n_bins) * power) / (np.sum(power) + _EPS)
    )
    return np.concatenate([dct, np.array([centroid])])[:feature_dim]


def extract_features(samples, config, worker_pool=None):
    """Extract frame records for one audio, optionally in a thread pool."""
    frame_len = int(round(config.data["frame_length_ms"] * SAMPLE_RATE / 1000))
    frame_shift = int(
        round(config.data["frame_shift_ms"] * SAMPLE_RATE / 1000)
    )
    feature_dim = int(config.data["feature_dim"])
    offsets = frame_offsets(len(samples), frame_len, frame_shift)

    def job(index_offset):
        index, offset = index_offset
        try:
            record = extract_frame(samples, offset, frame_len, feature_dim)
        except (FloatingPointError, ValueError) as exc:
            raise FeatureError(
                f"feature extraction failed at frame {index}: {exc}",
                stage="feature",
                interval_index=index,
            ) from exc
        record["frame_index"] = index
        record["offset_sample"] = offset
        record["start_ms"] = round6(offset * 1000.0 / SAMPLE_RATE)
        record["end_ms"] = round6(
            (offset + frame_len) * 1000.0 / SAMPLE_RATE
        )
        return record

    items = list(enumerate(offsets))
    if worker_pool is not None:
        records = worker_pool.map_ordered(
            job, items, sort_key=lambda item: item[0]
        )
    else:
        records = [job(item) for item in items]
    return {
        "frame_len": frame_len,
        "frame_shift": frame_shift,
        "n_samples": int(len(samples)),
        "frames": records,
    }
