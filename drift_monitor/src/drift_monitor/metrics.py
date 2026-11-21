"""Per-column, per-window drift metrics with deterministic accumulation.

Accumulation order is always window index ascending then sample index
ascending.  Categorical buckets are broken and emitted in lexicographic value
order.  Missing entries are skipped by every metric but counted in the stats
block.  Approximate and exact quantile paths deliberately differ so the
strategy selector can observably change the computation path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .datagen import GeneratedColumn

APPROX_GRID = 32
HIST_BINS = 10


@dataclass(frozen=True)
class WindowStats:
    column: str
    kind: str
    window_index: int
    period: str
    n_observed: int
    n_missing: int
    mean: float
    std: float
    quantiles_approx: tuple[float, ...]
    quantiles_exact: tuple[float, ...]
    histogram: tuple[float, ...]
    bin_edges: tuple[float, ...]
    frequencies: tuple[tuple[str, int], ...]
    observed_values_fingerprint: str


def _present(values: np.ndarray) -> list:
    out = []
    for value in values:  # sample index ascending
        if value is not None:
            out.append(value)
    return out


def _quantile_grid(sorted_values: list[float], probs: list[float]) -> tuple[float, ...]:
    if not sorted_values:
        return tuple(0.0 for _ in probs)
    n = len(sorted_values)
    if n == 1:
        return tuple(float(sorted_values[0]) for _ in probs)
    out = []
    for p in probs:
        pos = p * (n - 1)
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            out.append(float(sorted_values[lo]))
        else:
            frac = pos - lo
            out.append(float(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac))
    return tuple(out)


def _approx_quantiles(values: list[float]) -> tuple[float, ...]:
    """Fixed-grid centroid approximation (coarser than full interpolation)."""
    if not values:
        return tuple(0.0 for _ in (0.25, 0.5, 0.75))
    lo, hi = min(values), max(values)
    if lo == hi:
        return (float(lo), float(lo), float(lo))
    width = (hi - lo) / APPROX_GRID
    centroids = [0.0] * APPROX_GRID
    counts = [0] * APPROX_GRID
    for value in values:  # sample order ascending, deterministic accumulation
        idx = min(APPROX_GRID - 1, int((value - lo) / width))
        counts[idx] += 1
        centroids[idx] += (value - centroids[idx]) / counts[idx]
    total = len(values)
    cumulative = 0
    targets = [(0.25, 0), (0.5, 1), (0.75, 2)]
    results = [0.0, 0.0, 0.0]
    for idx in range(APPROX_GRID):
        next_cum = cumulative + counts[idx]
        for p, slot in targets:
            if results[slot] == 0.0 and cumulative <= p * total <= next_cum and counts[idx] > 0:
                results[slot] = centroids[idx]
        cumulative = next_cum
    for slot, p in ((0, 0.25), (1, 0.5), (2, 0.75)):
        if results[slot] == 0.0:
            results[slot] = _quantile_grid(sorted(values), [p])[0]
    return tuple(results)


def _histogram(values: list[float]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not values:
        edges = tuple(0.0 for _ in range(HIST_BINS + 1))
        return tuple(0.0 for _ in range(HIST_BINS)), edges
    lo, hi = min(values), max(values)
    if lo == hi:
        edges = tuple(lo for _ in range(HIST_BINS + 1))
        return tuple(0.0 for _ in range(HIST_BINS)), edges
    width = (hi - lo) / HIST_BINS
    counts = [0] * HIST_BINS
    for value in values:
        idx = min(HIST_BINS - 1, int((value - lo) / width))
        counts[idx] += 1
    total = len(values)
    density = tuple(count / (total * width) for count in counts)
    edges = tuple(lo + idx * width for idx in range(HIST_BINS + 1))
    return density, edges


def histogram_on_grid(values: list[float],
                      edges: tuple[float, ...]) -> tuple[float, ...]:
    """Density of ``values`` projected onto an externally supplied bin grid."""
    if not values or len(edges) != HIST_BINS + 1 or edges[-1] <= edges[0]:
        return tuple(0.0 for _ in range(HIST_BINS))
    width = (edges[-1] - edges[0]) / HIST_BINS
    counts = [0] * HIST_BINS
    for value in values:
        if value < edges[0]:
            idx = 0
        elif value >= edges[-1]:
            idx = HIST_BINS - 1
        else:
            idx = min(HIST_BINS - 1, int((value - edges[0]) / width))
        counts[idx] += 1
    return tuple(c / (len(values) * width) for c in counts)


def _frequencies(values: list) -> tuple[tuple[str, int], ...]:
    counts: dict[str, int] = {}
    for value in values:  # sample index ascending
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return tuple(sorted(counts.items(), key=lambda kv: kv[0]))  # lexicographic buckets


def build_stats(column: GeneratedColumn, window_index: int, period: str,
                start: int, end: int) -> WindowStats:
    window_values = column.values[start:end]
    present = _present(window_values)
    n_missing = len(window_values) - len(present)
    observed_fp = "|".join(str(v) for v in present[:64])
    if column.kind == "numeric":
        nums = [float(v) for v in present]
        n = len(nums)
        mean = sum(nums) / n if n else 0.0
        variance = sum((v - mean) ** 2 for v in nums) / n if n else 0.0
        std = math.sqrt(variance)
        ordered = sorted(nums)
        probs = [0.25, 0.5, 0.75]
        exact = _quantile_grid(ordered, probs)
        approx = _approx_quantiles(nums)
        density, edges = _histogram(nums)
        return WindowStats(
            column=column.name, kind=column.kind, window_index=window_index,
            period=period, n_observed=n, n_missing=n_missing, mean=mean, std=std,
            quantiles_approx=approx, quantiles_exact=exact, histogram=density,
            bin_edges=edges, frequencies=(), observed_values_fingerprint=observed_fp,
        )
    freqs = _frequencies(present)
    return WindowStats(
        column=column.name, kind=column.kind, window_index=window_index,
        period=period, n_observed=len(present), n_missing=n_missing,
        mean=0.0, std=0.0, quantiles_approx=(), quantiles_exact=(),
        histogram=(), bin_edges=(), frequencies=freqs,
        observed_values_fingerprint=observed_fp,
    )


def _mean_shift(ref: WindowStats, cur: WindowStats) -> float:
    pooled = math.sqrt((ref.std ** 2 + cur.std ** 2) / 2.0)
    if pooled == 0.0:
        return abs(cur.mean - ref.mean)
    return abs(cur.mean - ref.mean) / pooled


def _quantile_distance(ref: WindowStats, cur: WindowStats, approx: bool) -> float:
    rq = ref.quantiles_approx if approx else ref.quantiles_exact
    cq = cur.quantiles_approx if approx else cur.quantiles_exact
    if not rq:
        return 0.0
    scale = max(1e-12, sum(abs(v) for v in rq) / len(rq))
    return sum(abs(a - b) for a, b in zip(rq, cq)) / (len(rq) * scale)


def _hist_density_distance(ref: WindowStats, cur: WindowStats) -> float:
    if not ref.histogram or not cur.histogram or not ref.bin_edges:
        return 0.0
    edges = ref.bin_edges
    width = (edges[-1] - edges[0]) / HIST_BINS
    # Both histograms must be built on the shared reference grid by the engine
    # (see histogram_on_grid); WindowStats.histogram is then grid-aligned.
    return 0.5 * sum(abs(a - b) for a, b in zip(ref.histogram, cur.histogram)) * width


def _freq_tv_distance(ref: WindowStats, cur: WindowStats) -> float:
    keys = sorted({k for k, _ in ref.frequencies} | {k for k, _ in cur.frequencies})
    ref_total = max(1, ref.n_observed)
    cur_total = max(1, cur.n_observed)
    ref_map = dict(ref.frequencies)
    cur_map = dict(cur.frequencies)
    return 0.5 * sum(
        abs(ref_map.get(k, 0) / ref_total - cur_map.get(k, 0) / cur_total) for k in keys
    )


def _chi_square(ref: WindowStats, cur: WindowStats) -> float:
    keys = sorted({k for k, _ in ref.frequencies} | {k for k, _ in cur.frequencies})
    ref_map = dict(ref.frequencies)
    cur_map = dict(cur.frequencies)
    n_ref = max(1, ref.n_observed)
    n_cur = max(1, cur.n_observed)
    total = 0.0
    for key in keys:  # lexicographic bucket order
        expected = n_cur * (ref_map.get(key, 0) / n_ref)
        observed = cur_map.get(key, 0)
        if expected > 0:
            total += (observed - expected) ** 2 / expected
    # Normalize by degrees of freedom scale so the score stays O(1) and shares
    # the same threshold space with the other metrics.
    return total / max(1, len(keys))


def metric_score(metric: str, ref: WindowStats, cur: WindowStats,
                 approx_quantile: bool) -> float:
    if metric == "mean_shift":
        return _mean_shift(ref, cur)
    if metric == "quantile":
        return _quantile_distance(ref, cur, approx_quantile)
    if metric == "hist_density":
        return _hist_density_distance(ref, cur)
    if metric == "freq_diff":
        return _freq_tv_distance(ref, cur)
    if metric == "chi_square":
        return _chi_square(ref, cur)
    raise ValueError(f"unknown metric {metric}")
