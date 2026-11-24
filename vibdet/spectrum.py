"""Cross-window feature-spectrum construction.

For every feature channel the ``order``-th adjacent-window difference is
computed over the per-window feature series, then sliding statistics (mean,
dispersion and extreme) over ``stat_window`` differences form one spectrum
position per analysis window.  Left-edge values are filled by repeating the
earliest *real* value and every padded position is explicitly flagged.

Each spectrum position records the exact set of real window indices it
depends on, including windows reached through overlapping segment boundaries.
"""

from dataclasses import dataclass

import numpy as np

from .errors import E_SPECTRUM_TOO_SHORT, PipelineError
from .hashing import array_fingerprint, combine_keys, stable_hash
from .windows import feature_names


STAT_NAMES = ("mean", "dispersion", "extreme")


@dataclass
class SpectrumPosition:
    position: int
    vector: np.ndarray
    channel_names: tuple
    dependencies: tuple
    segments: tuple
    diff_padded: bool
    stat_padded: bool
    padding_count: int
    diff_values: dict
    cache_key: str

    def to_dict(self):
        return {
            "position": self.position,
            "vector": [float(x) for x in self.vector],
            "channel_names": list(self.channel_names),
            "dependencies": list(self.dependencies),
            "segments": list(self.segments),
            "diff_padded": self.diff_padded,
            "stat_padded": self.stat_padded,
            "padding_count": int(self.padding_count),
            "diff_values": {k: float(v) for k, v in self.diff_values.items()},
            "cache_key": self.cache_key,
        }


@dataclass
class SpectrumResult:
    device_id: str
    positions: list
    channel_names: tuple
    matrix: np.ndarray
    param_hash: str

    def to_manifest(self):
        return {
            "device_id": self.device_id,
            "position_count": len(self.positions),
            "channel_count": len(self.channel_names),
            "channel_names": list(self.channel_names),
            "param_hash": self.param_hash,
            "matrix_fingerprint": array_fingerprint(self.matrix),
        }


def spectrum_param_hash(cfg):
    return stable_hash([
        "spectrum-params",
        int(cfg["spectrum"]["order"]),
        int(cfg["spectrum"]["stat_window"]),
    ])


def _ordered_diff(series, order):
    """Order-th adjacent difference with constant left-edge padding.

    ``series`` is an ascending-order array; reductions always walk ascending
    indices.  Returns (diff_values, padded_flag_per_position).
    """
    current = series.astype(np.float64).copy()
    n = current.shape[0]
    diff_pad_flags = np.zeros(n, dtype=bool)
    for step in range(1, order + 1):
        length = current.shape[0]
        nxt = np.empty(length, dtype=np.float64)
        fill = current[0]
        for i in range(length):
            left_index = i - 1
            if left_index < 0:
                nxt[i] = current[i] - fill
                if step == order:
                    diff_pad_flags[i] = True
            else:
                nxt[i] = current[i] - current[left_index]
        current = nxt
    return current, diff_pad_flags


def build_spectrum(device_id, windows, cfg, shutdown_event=None):
    order = int(cfg["spectrum"]["order"])
    stat_window = int(cfg["spectrum"]["stat_window"])
    n_windows = len(windows)
    if n_windows < 2 or n_windows < order:
        raise PipelineError(
            E_SPECTRUM_TOO_SHORT,
            f"device {device_id} yields {n_windows} windows, need at least "
            f"{max(2, order)} for order-{order} feature spectrum",
            device=device_id)

    names = feature_names(cfg)
    channel_names = tuple(
        f"{feature}:{stat}" for stat in STAT_NAMES for feature in names)
    matrix = np.empty((n_windows, len(channel_names)), dtype=np.float64)
    param_hash = spectrum_param_hash(cfg)

    diffs, diff_flags = {}, {}
    for name in names:
        series = np.array([win.features[name] for win in windows],
                          dtype=np.float64)
        diffs[name], diff_flags[name] = _ordered_diff(series, order)

    for channel_offset, name in enumerate(names):
        diff = diffs[name]
        for p in range(n_windows):
            if shutdown_event is not None and shutdown_event.is_set():
                from .scheduler import ShutdownRequested
                raise ShutdownRequested("shutdown during spectrum stage")
            indices = list(range(p - stat_window + 1, p + 1))
            real_indices = [i for i in indices if i >= 0]
            padding_count = stat_window - len(real_indices)
            stat_values = np.array([diff[i] for i in real_indices],
                                   dtype=np.float64)
            if padding_count:
                stat_values = np.concatenate([
                    np.full(padding_count, float(diff[0]), dtype=np.float64),
                    stat_values])
            mean = float(np.add.reduce(stat_values) / stat_window)
            centered = stat_values - mean
            variance = float(np.add.reduce(centered * centered) / stat_window)
            dispersion = float(np.sqrt(variance))
            extreme = float(np.max(np.abs(stat_values)))
            base = channel_offset * len(STAT_NAMES)
            matrix[p, base] = mean
            matrix[p, base + 1] = dispersion
            matrix[p, base + 2] = extreme

    positions = []
    for p in range(n_windows):
        stat_indices = [i for i in range(p - stat_window + 1, p + 1)
                        if i >= 0]
        dependency_set = set()
        for j in stat_indices:
            for back in range(order + 1):
                window_index = j - back
                if window_index >= 0:
                    dependency_set.add(window_index)
        dependencies = tuple(sorted(dependency_set))
        segments = tuple(sorted({
            seg for win_index in dependencies
            for seg in windows[win_index].segments
        }))
        diff_padded = bool(diff_flags[names[0]][p])
        stat_padded = bool(p < stat_window - 1)
        padding_count = max(0, stat_window - p - 1) + (1 if diff_padded else 0)
        diff_values = {name: float(diffs[name][p]) for name in names}
        ordered_window_keys = [windows[i].fingerprint for i in dependencies]
        cache_key = combine_keys(
            stable_hash(["spectrum", device_id, p]),
            stable_hash(ordered_window_keys),
            param_hash)
        positions.append(SpectrumPosition(
            position=p, vector=matrix[p].copy(),
            channel_names=channel_names, dependencies=dependencies,
            segments=segments, diff_padded=diff_padded,
            stat_padded=stat_padded, padding_count=padding_count,
            diff_values=diff_values, cache_key=cache_key))

    return SpectrumResult(device_id, positions, channel_names, matrix,
                          param_hash)
