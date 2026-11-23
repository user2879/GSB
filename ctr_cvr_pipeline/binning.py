from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Iterable

from .canonical import fingerprint
from .config import FeatureConfig, FeatureSpec
from .samples import effective_dense_value
from .state import Sample


@dataclass(frozen=True)
class EdgeTable:
    feature: str
    mode: str
    bins: int
    source: str
    edges: tuple[float, ...]
    interior: tuple[float, ...]
    degenerate: bool
    fallback_rule: str
    bin_counts: tuple[int, ...]

    def fingerprint(self) -> str:
        return fingerprint(
            {
                "feature": self.feature,
                "mode": self.mode,
                "bins": self.bins,
                "source": self.source,
                "edges": list(self.edges),
                "degenerate": self.degenerate,
                "fallback_rule": self.fallback_rule,
            }
        )

    def assign(self, value: float) -> int:
        if self.degenerate:
            return 0
        return min(self.bins - 1, max(0, bisect_right(self.interior, value)))

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "mode": self.mode,
            "bins": self.bins,
            "source": self.source,
            "edges": list(self.edges),
            "degenerate": self.degenerate,
            "fallback_rule": self.fallback_rule,
            "bin_counts": list(self.bin_counts),
            "fingerprint": self.fingerprint(),
        }


def _equal_width_edges(values: list[float], bins: int) -> tuple[tuple[float, ...], bool]:
    low = min(values)
    high = max(values)
    if high == low:
        return tuple([low] * (bins + 1)), True
    width = (high - low) / bins
    edges = [low + index * width for index in range(bins + 1)]
    edges[0] = low
    edges[-1] = high
    return tuple(edges), False


def _equal_frequency_edges(values: list[float], bins: int) -> tuple[tuple[float, ...], bool, str]:
    ordered = sorted(values)
    low = ordered[0]
    high = ordered[-1]
    if high == low:
        return tuple([low] * (bins + 1)), True, "DEGENERATE_RANGE_SINGLE_BIN"
    edges = [low]
    for index in range(1, bins):
        position = min(len(ordered) - 1, (index * len(ordered)) // bins)
        edges.append(ordered[position])
    edges.append(high)
    return tuple(float(value) for value in edges), False, "TIES_ASSIGN_TO_UPPER_BIN_LAST_BIN_REMAINDER"


def build_edge_table(
    feature: FeatureSpec,
    samples: Iterable[Sample],
    explicit_edges: tuple[float, ...] | None = None,
) -> EdgeTable:
    assert feature.kind == "dense" and feature.bins is not None and feature.binning is not None
    values = [effective_dense_value(sample, feature.name) for sample in samples]
    if not values:
        values = [0.0]
    if explicit_edges is not None:
        edges = tuple(float(value) for value in explicit_edges)
        degenerate = len(set(edges)) == 1
        source = "explicit"
        fallback = "EXPLICIT_STRICTLY_INCREASING_EDGES"
    elif feature.binning == "equal_width":
        edges, degenerate = _equal_width_edges(values, feature.bins)
        source = "auto_equal_width"
        fallback = "DEGENERATE_RANGE_SINGLE_BIN" if degenerate else "LINEAR_HALF_OPEN_BINS"
    else:
        edges, degenerate, fallback = _equal_frequency_edges(values, feature.bins)
        source = "auto_equal_frequency"
    interior = tuple(edges[1:-1])
    counts = [0] * feature.bins
    if degenerate:
        counts[0] = len(values)
    else:
        for value in values:
            counts[bisect_right(interior, value)] += 1
    return EdgeTable(
        feature=feature.name,
        mode=feature.binning,
        bins=feature.bins,
        source=source,
        edges=edges,
        interior=interior,
        degenerate=degenerate,
        fallback_rule=fallback,
        bin_counts=tuple(counts),
    )


def build_edge_tables(
    config: FeatureConfig,
    samples: list[Sample],
    explicit_edges: dict[str, tuple[float, ...]],
) -> dict[str, EdgeTable]:
    result: dict[str, EdgeTable] = {}
    for feature in config.dense_features:
        result[feature.name] = build_edge_table(
            feature,
            samples,
            explicit_edges.get(feature.name),
        )
    return result
