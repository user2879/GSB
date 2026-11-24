"""Analysis-window slicing and per-window short-time features."""

from dataclasses import dataclass

import numpy as np

from .hashing import array_fingerprint, combine_keys, stable_hash

BASE_FEATURES = ("rms", "kurtosis", "peak")


@dataclass
class WindowData:
    index: int
    start: int
    end: int
    features: dict
    feature_names: tuple
    content_hash: str
    param_hash: str
    fingerprint: str
    segments: tuple
    sample_indices: np.ndarray

    def vector(self):
        return np.array([self.features[name] for name in self.feature_names],
                        dtype=np.float64)


def window_param_hash(cfg):
    return stable_hash([
        "window-params",
        int(cfg["window"]["length"]),
        int(cfg["window"]["hop"]),
        int(cfg["features"]["sample_rate"]),
        [list(map(float, band)) for band in cfg["features"]["bands"]],
    ])


def feature_names(cfg):
    names = list(BASE_FEATURES)
    names.extend(f"band_energy_{i}" for i in range(len(cfg["features"]["bands"])))
    return tuple(names)


def window_starts(n_samples, length, hop):
    if n_samples < length:
        return []
    last = n_samples - length
    count = last // hop + 1
    return [k * hop for k in range(count)]


def _window_features(samples, cfg):
    # Fixed accumulation order: sample-index ascending via ordered reductions.
    n = samples.shape[0]
    mean = np.add.reduce(samples) / n
    deviations = samples - mean
    variance = np.add.reduce(deviations * deviations) / n
    rms = float(np.sqrt(np.add.reduce(samples * samples) / n))
    if variance > 0.0:
        kurtosis = float(np.add.reduce(deviations ** 4) / n / (variance ** 2) - 3.0)
    else:
        kurtosis = 0.0
    peak = float(np.max(np.abs(samples)))

    spectrum = np.fft.rfft(samples)
    magnitude_sq = (spectrum.real ** 2 + spectrum.imag ** 2)
    bin_count = magnitude_sq.shape[0]
    freqs = np.arange(bin_count, dtype=np.float64) / n
    # normalize frequency by Nyquist (0.5) so configured edges are in [0, 0.5]
    norm_freqs = freqs * 2.0
    bands = []
    for band_index, edge in enumerate(cfg["features"]["bands"]):
        low, high = float(edge[0]), float(edge[1])
        if band_index == len(cfg["features"]["bands"]) - 1:
            selected = (norm_freqs >= low) & (norm_freqs <= high)
        else:
            selected = (norm_freqs >= low) & (norm_freqs < high)
        if np.any(selected):
            # bin accumulation order is ascending (rfft natural order)
            energy = float(np.add.reduce(magnitude_sq[selected]) /
                           float(np.count_nonzero(selected)) / (n * n))
        else:
            energy = 0.0
        bands.append(energy)

    features = {"rms": rms, "kurtosis": kurtosis, "peak": peak}
    for i, energy in enumerate(bands):
        features[f"band_energy_{i}"] = energy
    return features


def build_windows(wave, cfg):
    length = int(cfg["window"]["length"])
    hop = int(cfg["window"]["hop"])
    names = feature_names(cfg)
    param_hash = window_param_hash(cfg)
    starts = window_starts(wave.values.shape[0], length, hop)
    windows = []
    for index, start in enumerate(starts):
        end = start + length
        samples = wave.values[start:end]
        sample_indices = np.arange(start, end, dtype=np.int64)
        features = _window_features(samples, cfg)
        content_hash = array_fingerprint(samples)
        fingerprint = combine_keys(content_hash, param_hash,
                                   stable_hash(index))
        overlapping = tuple(sorted({
            seg.seg_id for seg in wave.segments
            if not (seg.end <= start or seg.start >= end)
        }))
        windows.append(WindowData(
            index=index, start=start, end=end, features=features,
            feature_names=names, content_hash=content_hash,
            param_hash=param_hash, fingerprint=fingerprint,
            segments=overlapping, sample_indices=sample_indices))
    return windows
