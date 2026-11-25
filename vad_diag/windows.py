"""Overlapping analysis windows and deterministic fixed-point features.

Windows are cut on the *global sample timeline* (not per segment), so a window
may span several segments.  Each window records its global sample interval and
the exact set of segments it references; feature extraction only ever reads
samples inside that interval, never data from outside the window.

Per-frame real-valued features are quantized to fixed-point integers.  The
window feature vector is the saturating sum of frame vectors, accumulated in a
fixed order: frame index ascending, then feature dimension ascending.  Because
the inputs and order are fixed, identical (segment set, parameters, quant)
always produce byte-identical integer vectors.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from . import dsp
from .config import Config
from .fixedpoint import fp_div, fp_sum, to_fixed, to_float
from .timeline import Timeline


@dataclass(frozen=True)
class Window:
    window_id: str
    index: int
    start: int
    end: int
    frame_indices: Tuple[int, ...]
    segment_ids: Tuple[str, ...]
    intersections: Tuple[Tuple[int, int], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "frame_indices": list(self.frame_indices),
            "segment_ids": sorted(self.segment_ids),
            "intersections": [list(p) for p in self.intersections],
        }


@dataclass(frozen=True)
class WindowFeature:
    window_id: str
    int_features: Tuple[int, ...]
    frac_bits: int
    quant_mode: str
    frame_count: int
    cache_key: str = ""

    def dequantized(self) -> List[float]:
        return [to_float(v, self.frac_bits) for v in self.int_features]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_id": self.window_id,
            "int_features": list(self.int_features),
            "dequant_features": self.dequantized(),
            "frac_bits": self.frac_bits,
            "quant_mode": self.quant_mode,
            "frame_count": self.frame_count,
            "cache_key": self.cache_key,
        }


def cut_windows(timeline: Timeline, config: Config,
                frame_intervals: Sequence[Tuple[int, int]]) -> List[Window]:
    """Cut windows and bind each to the frames fully inside its interval."""
    total = timeline.total_samples
    win_intervals = dsp.window_intervals(
        total, config.window_length, config.window_shift)
    windows: List[Window] = []
    for index, (start, end) in enumerate(win_intervals):
        seg_ids, intersections = timeline.segments_for_interval(start, end)
        frame_indices = tuple(
            fi for fi, (fs, fe) in enumerate(frame_intervals)
            if fs >= start and fe <= end)
        windows.append(Window(
            window_id=f"W{index:04d}",
            index=index, start=start, end=end,
            frame_indices=frame_indices,
            segment_ids=tuple(sorted(seg_ids)),
            intersections=tuple(intersections)))
    return windows


def extract_window_feature(window: Window, timeline: Timeline,
                           config: Config,
                           frame_intervals: Sequence[Tuple[int, int]],
                           cache_key: str = "") -> WindowFeature:
    """Quantize per-frame features and accumulate in the fixed order."""
    flat = timeline.samples()
    accum = [0] * dsp.FEATURE_DIM
    used_frames = 0
    # Frame index ascending is guaranteed by window.frame_indices ordering.
    for fi in window.frame_indices:
        fs, fe = frame_intervals[fi]
        # Hard guarantee: never read outside the window's recorded interval.
        assert fs >= window.start and fe <= window.end
        frame_samples = flat[fs:fe]
        raw = dsp.raw_frame_feature(frame_samples, config.sample_rate)
        quantized = [to_fixed(v, config.frac_bits, config.quant_mode)
                     for v in raw]
        # Feature dimension ascending saturating accumulation.
        for dim in range(dsp.FEATURE_DIM):
            accum[dim] = fp_sum([accum[dim], quantized[dim]])
        used_frames += 1
    # Deterministic fixed-point mean so a window feature stays in the same Qn
    # range regardless of how many frames it contains.
    if used_frames > 0:
        count_q = to_fixed(float(used_frames), config.frac_bits,
                           config.quant_mode)
        accum = [fp_div(value, count_q, config.frac_bits,
                        config.quant_mode) for value in accum]
    return WindowFeature(
        window_id=window.window_id,
        int_features=tuple(accum),
        frac_bits=config.frac_bits,
        quant_mode=config.quant_mode,
        frame_count=used_frames,
        cache_key=cache_key)


def extract_all(windows: Sequence[Window], timeline: Timeline, config: Config,
                frame_intervals: Sequence[Tuple[int, int]],
                key_fn=None, max_workers: int = 1,
                feature_hook=None) -> Dict[str, WindowFeature]:
    """Run window feature extraction with a bounded thread pool.

    ``key_fn(window)`` optionally supplies the precomputed cache key;
    ``feature_hook(window_id, feature)`` allows the injection layer to stage a
    dimension mismatch.  Results are returned keyed by ``window_id`` and the
    dictionary order matches the input window order regardless of threading.
    """
    workers = max(1, min(max_workers, config.max_workers))

    def work(window: Window) -> WindowFeature:
        key = key_fn(window) if key_fn else ""
        feature = extract_window_feature(
            window, timeline, config, frame_intervals, cache_key=key)
        if feature_hook is not None:
            feature = feature_hook(window.window_id, feature)
        return feature

    results: Dict[str, WindowFeature] = {}
    if workers == 1 or len(windows) <= 1:
        for window in windows:
            feat = work(window)
            results[feat.window_id] = feat
        return results

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for feat in pool.map(work, windows):
            results[feat.window_id] = feat
    # Re-impose the deterministic input order.
    return {w.window_id: results[w.window_id] for w in windows}
