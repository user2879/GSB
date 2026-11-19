from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Iterable

from .fixed import FixedGrid, fraction_text, stable_descending_rank
from .spec import ContentEntry, Dependency, Spec

HIT_SOURCES = frozenset({"cache_hit", "expired_revalidated", "stale_hit"})
VALID_SOURCES = frozenset({"cache_hit", "expired_revalidated", "stale_hit", "miss_origin"})


@dataclass(frozen=True)
class BatchRequestRecord:
    seq: int
    request_id: str
    node_id: str
    path: str
    query: str
    at_ms: int
    hit_source: str
    returned_etag: str
    authoritative_etag: str
    version_consistent: bool
    remaining_ttl_ms: int
    entry_version: int
    stale: bool
    error_code: str | None
    cache_key: str = ""
    written_at_ms: int = 0
    written_mono_ms: int = 0
    monotonic_base_ms: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "request_id": self.request_id,
            "node_id": self.node_id,
            "path": self.path,
            "query": self.query,
            "at_ms": self.at_ms,
            "hit_source": self.hit_source,
            "returned_etag": self.returned_etag,
            "authoritative_etag": self.authoritative_etag,
            "version_consistent": self.version_consistent,
            "remaining_ttl_ms": self.remaining_ttl_ms,
            "entry_version": self.entry_version,
            "stale": self.stale,
            "error_code": self.error_code,
            "cache_key": self.cache_key,
            "written_at_ms": self.written_at_ms,
            "written_mono_ms": self.written_mono_ms,
            "monotonic_base_ms": self.monotonic_base_ms,
        }

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> "BatchRequestRecord":
        return cls(
            seq=int(obj["seq"]),
            request_id=str(obj["request_id"]),
            node_id=str(obj["node_id"]),
            path=str(obj["path"]),
            query=str(obj.get("query", "")),
            at_ms=int(obj["at_ms"]),
            hit_source=str(obj["hit_source"]),
            returned_etag=str(obj.get("returned_etag", "")),
            authoritative_etag=str(obj.get("authoritative_etag", "")),
            version_consistent=bool(obj["version_consistent"]),
            remaining_ttl_ms=int(obj.get("remaining_ttl_ms", 0)),
            entry_version=int(obj.get("entry_version", 0)),
            stale=bool(obj.get("stale", False)),
            error_code=obj.get("error_code"),
            cache_key=str(obj.get("cache_key", "")),
            written_at_ms=int(obj.get("written_at_ms", 0)),
            written_mono_ms=int(obj.get("written_mono_ms", 0)),
            monotonic_base_ms=int(obj.get("monotonic_base_ms", 0)),
        )


def _segment(
    source: str,
    contribution_units: Fraction,
    total_bound_units: Fraction,
    grid: FixedGrid,
    detail: str,
) -> dict[str, Any]:
    if total_bound_units <= 0:
        share = Fraction(0)
    else:
        share = contribution_units / total_bound_units
    return {
        "source": source,
        "contribution_units": fraction_text(contribution_units),
        "contribution_units_num": contribution_units.numerator,
        "contribution_units_den": contribution_units.denominator,
        "share_units": grid.q_ratio_fraction(share),
        "detail": detail,
    }


def _dominant_segment(segments: Iterable[tuple[str, Fraction]], fallback: str = "none") -> str:
    best_source = fallback
    best_value: Fraction | None = None
    for source, value in segments:
        if best_value is None or value > best_value or (value == best_value and source < best_source):
            best_source = source
            best_value = value
    return best_source


def _node_metric(
    metric_name: str,
    node_id: str,
    value_units: int,
    sample_count: int,
    grid: FixedGrid,
    detail: str,
) -> dict[str, Any]:
    if sample_count <= 0:
        return {
            "metric": metric_name,
            "value_units": 0,
            "sample_count": 0,
            "error_bound_units": 0,
            "error_bound_exclusive": True,
            "dominant_error_source": "none",
            "segments": [],
            "detail": "no samples",
        }
    source = f"node:{node_id}:floor_quantization"
    total_bound = Fraction(1)
    segments = [_segment(source, Fraction(1), total_bound, grid, detail)]
    return {
        "metric": metric_name,
        "value_units": value_units,
        "sample_count": sample_count,
        "error_bound_units": grid.single_value_error_bound_units,
        "error_bound_exclusive": True,
        "dominant_error_source": source,
        "segments": segments,
        "detail": detail,
    }


def _aggregate_metric(
    metric_name: str,
    node_ids: list[str],
    node_counts: dict[str, int],
    node_values: dict[str, int],
    total_count: int,
    grid: FixedGrid,
) -> dict[str, Any]:
    if total_count <= 0:
        return {
            "metric": metric_name,
            "value_units": 0,
            "sample_count": 0,
            "error_bound_units": 0,
            "error_bound_exclusive": True,
            "dominant_error_source": "none",
            "segments": [],
            "detail": "no samples",
        }
    numerator = 0
    node_segments: list[tuple[str, Fraction]] = []
    segment_payloads: list[dict[str, Any]] = []
    cross_node_bound = Fraction(1)
    for node_id in sorted(node_ids):
        count = node_counts.get(node_id, 0)
        if count <= 0:
            continue
        numerator += node_values.get(node_id, 0) * count
        contribution = Fraction(count, total_count)
        node_segments.append((f"node:{node_id}:weighted_quantization", contribution))
    value_units = numerator // total_count
    total_bound = cross_node_bound + Fraction(1)
    for source, contribution in node_segments:
        segment_payloads.append(
            _segment(
                source,
                contribution,
                total_bound,
                grid,
                "weight = node_request_count / batch_request_count",
            )
        )
    final_source = "aggregate:final_truncation"
    segment_payloads.append(
        _segment(
            final_source,
            Fraction(1),
            total_bound,
            grid,
            "one final integer floor division across nodes",
        )
    )
    comparable = node_segments + [(final_source, Fraction(1))]
    return {
        "metric": metric_name,
        "value_units": value_units,
        "sample_count": total_count,
        "error_bound_units": int(total_bound),
        "error_bound_exclusive": True,
        "dominant_error_source": _dominant_segment(comparable),
        "segments": segment_payloads,
        "detail": "sum(node_quantized_value * node_count) // total_count; bound = 1 cross-node + 1 final unit",
    }


def _cross_node_comparisons(
    node_ids: list[str], node_values: dict[str, int], grid: FixedGrid
) -> dict[str, Any]:
    rankings = stable_descending_rank(
        [(node_id, node_values[node_id]) for node_id in sorted(node_ids)], grid
    )
    pairwise: list[dict[str, Any]] = []
    sorted_ids = sorted(node_ids)
    for left_index, left in enumerate(sorted_ids):
        for right in sorted_ids[left_index + 1 :]:
            comparison = grid.compare_with_bound(node_values[left], node_values[right])
            if comparison["within_error_bound"]:
                pairwise.append(
                    {
                        "left": left,
                        "right": right,
                        "left_units": node_values[left],
                        "right_units": node_values[right],
                        **comparison,
                        "stable_tie_order": sorted([left, right]),
                        "note": "ranking must use stable id tie-break; difference is covered by the bound",
                    }
                )
    return {"rankings": rankings, "pairwise_within_bound": pairwise}


def _node_block(records: list[BatchRequestRecord], node_id: str, grid: FixedGrid) -> dict[str, Any]:
    samples = [record for record in records if record.node_id == node_id]
    request_count = len(samples)
    source_counts = {source: 0 for source in sorted(VALID_SOURCES)}
    for record in samples:
        if record.hit_source not in source_counts:
            source_counts[record.hit_source] = 0
        source_counts[record.hit_source] += 1
    hits = sum(count for source, count in source_counts.items() if source in HIT_SOURCES)
    fresh_hits = source_counts.get("cache_hit", 0)
    revalidated = source_counts.get("expired_revalidated", 0)
    stale_hits = source_counts.get("stale_hit", 0)
    misses = source_counts.get("miss_origin", 0)
    consistent = sum(1 for record in samples if record.version_consistent)
    ttl_total_ms = sum(record.remaining_ttl_ms for record in samples)
    hit_rate = _node_metric(
        "hit_rate",
        node_id,
        grid.q_ratio(hits, request_count) if request_count else 0,
        request_count,
        grid,
        "floor(hit_count * S / request_count); hit includes fresh, revalidated and stale cache hits",
    )
    fresh_rate = _node_metric(
        "fresh_hit_rate",
        node_id,
        grid.q_ratio(fresh_hits, request_count) if request_count else 0,
        request_count,
        grid,
        "floor(fresh_hit_count * S / request_count)",
    )
    version_rate = _node_metric(
        "version_consistency_rate",
        node_id,
        grid.q_ratio(consistent, request_count) if request_count else 0,
        request_count,
        grid,
        "floor(version_consistent_count * S / request_count)",
    )
    ttl_avg = _node_metric(
        "remaining_ttl_seconds",
        node_id,
        grid.q_average_ms(ttl_total_ms, request_count) if request_count else 0,
        request_count,
        grid,
        "floor(sum(remaining_ttl_ms) * S / (1000 * request_count))",
    )
    return {
        "node_id": node_id,
        "request_count": request_count,
        "hit_count": hits,
        "fresh_hit_count": fresh_hits,
        "revalidated_count": revalidated,
        "stale_hit_count": stale_hits,
        "miss_count": misses,
        "version_consistent_count": consistent,
        "remaining_ttl_ms_sum": ttl_total_ms,
        "source_counts": source_counts,
        "metrics": {
            "hit_rate": hit_rate,
            "fresh_hit_rate": fresh_rate,
            "version_consistency_rate": version_rate,
            "remaining_ttl_seconds": ttl_avg,
        },
    }


def _aggregate_block(
    records: list[BatchRequestRecord], node_ids: list[str], node_blocks: dict[str, dict[str, Any]], grid: FixedGrid
) -> dict[str, Any]:
    counts = {node_id: node_blocks[node_id]["request_count"] for node_id in node_ids}
    total = len(records)
    metrics: dict[str, Any] = {}
    for metric_name in ("hit_rate", "fresh_hit_rate", "version_consistency_rate", "remaining_ttl_seconds"):
        values = {
            node_id: int(node_blocks[node_id]["metrics"][metric_name]["value_units"])
            for node_id in node_ids
        }
        metrics[metric_name] = _aggregate_metric(
            metric_name, node_ids, counts, values, total, grid
        )
    return {
        "request_count": total,
        "hit_count": sum(int(node_blocks[node_id]["hit_count"]) for node_id in node_ids),
        "version_consistent_count": sum(
            int(node_blocks[node_id]["version_consistent_count"]) for node_id in node_ids
        ),
        "remaining_ttl_ms_sum": sum(
            int(node_blocks[node_id]["remaining_ttl_ms_sum"]) for node_id in node_ids
        ),
        "metrics": metrics,
    }


def _topological_order(
    content: dict[str, ContentEntry],
    topology: dict[str, tuple[Dependency, ...]] | None = None,
) -> list[str]:
    visited: set[str] = set()
    visiting: set[str] = set()
    order: list[str] = []

    def dependencies_for(path: str) -> tuple[Dependency, ...]:
        if topology is not None and path in topology:
            return topology[path]
        return content[path].dependencies

    def visit(path: str) -> None:
        if path in visited:
            return
        if path in visiting:
            raise ValueError("cyclic content graph")
        visiting.add(path)
        for dep in sorted(dependencies_for(path), key=lambda item: item.path):
            visit(dep.path)
        visiting.remove(path)
        visited.add(path)
        order.append(path)

    for path in sorted(content):
        visit(path)
    return order


def _derived_weights(
    spec: Spec,
    grid: FixedGrid,
    topology: dict[str, tuple[Dependency, ...]] | None = None,
) -> dict[str, Any]:
    scale = grid.scale
    exact_weights: dict[str, dict[str, Fraction]] = {}
    fixed_weights: dict[str, dict[str, int]] = {}
    error_bounds: dict[str, dict[str, Fraction]] = {}
    path_edges: dict[str, list[dict[str, Any]]] = {}

    def dependencies_for(path: str) -> tuple[Dependency, ...]:
        if topology is not None and path in topology:
            return topology[path]
        return spec.content[path].dependencies

    for path in _topological_order(spec.content, topology):
        dependencies = dependencies_for(path)
        if not dependencies:
            exact_weights[path] = {path: Fraction(1)}
            fixed_weights[path] = {path: scale}
            error_bounds[path] = {path: Fraction(0)}
            continue
        coefficient_sum = sum(dep.coefficient for dep in dependencies)
        exact: dict[str, Fraction] = {}
        fixed: dict[str, int] = {}
        bounds: dict[str, Fraction] = {}
        edge_payloads: list[dict[str, Any]] = []
        for dep in sorted(dependencies, key=lambda item: item.path):
            share = Fraction(dep.coefficient, coefficient_sum)
            coefficient_units = (dep.coefficient * scale) // coefficient_sum
            child_exact = exact_weights[dep.path]
            child_fixed = fixed_weights[dep.path]
            child_bounds = error_bounds[dep.path]
            edge_payloads.append(
                {
                    "child_path": dep.path,
                    "coefficient": dep.coefficient,
                    "normalized_coefficient": fraction_text(share),
                    "coefficient_units": coefficient_units,
                }
            )
            for leaf in sorted(child_exact):
                leaf_share = share * child_exact[leaf]
                exact[leaf] = exact.get(leaf, Fraction(0)) + leaf_share
                child_fixed_units = child_fixed.get(leaf, 0)
                fixed[leaf] = fixed.get(leaf, 0) + (coefficient_units * child_fixed_units) // scale
                child_bound = child_bounds.get(leaf, Fraction(0))
                segment_bound = (
                    share * child_bound
                    + (child_exact[leaf] + child_bound) / scale
                    + Fraction(1, scale)
                )
                bounds[leaf] = bounds.get(leaf, Fraction(0)) + segment_bound
        exact_weights[path] = exact
        fixed_weights[path] = fixed
        error_bounds[path] = bounds
        path_edges[path] = edge_payloads

    paths_payload: dict[str, Any] = {}
    leaf_rank_input: dict[str, list[tuple[str, int]]] = {}
    for path in sorted(spec.content):
        dependencies = dependencies_for(path)
        if not dependencies:
            continue
        coefficient_sum = sum(dep.coefficient for dep in dependencies)
        leaf_payloads: dict[str, Any] = {}
        for leaf in sorted(exact_weights[path]):
            exact = exact_weights[path][leaf]
            fixed_units = fixed_weights[path].get(leaf, 0)
            total_bound = error_bounds[path].get(leaf, Fraction(0))
            actual_error = exact - Fraction(fixed_units, scale)
            segments: list[dict[str, Any]] = []
            segment_fractions: list[tuple[str, Fraction]] = []
            for dep in sorted(dependencies, key=lambda item: item.path):
                child_exact = exact_weights[dep.path].get(leaf, Fraction(0))
                child_bound = error_bounds[dep.path].get(leaf, Fraction(0))
                share = Fraction(dep.coefficient, coefficient_sum)
                segment_real = (
                    share * child_bound
                    + (child_exact + child_bound) / scale
                    + Fraction(1, scale)
                )
                segment_fraction_units = segment_real * scale
                source = f"derive:{path}->{dep.path}:leaf={leaf}"
                segment_fractions.append((source, segment_real))
                segments.append(
                    _segment(
                        source,
                        segment_fraction_units,
                        total_bound * scale,
                        grid,
                        "w_i*b_child + (w_leaf_i+b_child)/S + 1/S",
                    )
                )
            leaf_payloads[leaf] = {
                "exact_weight": fraction_text(exact),
                "fixed_weight_units": fixed_units,
                "fixed_weight_approx": fraction_text(Fraction(fixed_units, scale)),
                "realized_error_units": grid.error_units_from_fraction(actual_error),
                "error_bound_units": grid.error_units_from_fraction(total_bound),
                "error_bound_units_fraction": fraction_text(total_bound * scale),
                "dominant_error_segment": _dominant_segment(segment_fractions),
                "segments": segments,
            }
            leaf_rank_input.setdefault(leaf, []).append((path, fixed_units))
        sum_fixed = sum(fixed_weights[path].values())
        max_bound_units = max(
            (grid.error_units_from_fraction(value) for value in error_bounds[path].values()),
            default=0,
        )
        paths_payload[path] = {
            "path": path,
            "edges": path_edges.get(path, []),
            "normalized_fixed_sum_units": sum_fixed,
            "normalization_shortfall_units": max(0, scale - sum_fixed),
            "max_error_bound_units": max_bound_units,
            "leaves": leaf_payloads,
        }

    leaf_rankings: dict[str, Any] = {}
    for leaf in sorted(leaf_rank_input):
        values = leaf_rank_input[leaf]
        rankings = stable_descending_rank(values, grid, tie_key="path")
        pairwise: list[dict[str, Any]] = []
        for left_index, left in enumerate(values):
            for right in values[left_index + 1 :]:
                comparison = grid.compare_with_bound(left[1], right[1])
                if comparison["within_error_bound"]:
                    pairwise.append(
                        {
                            "left": left[0],
                            "right": right[0],
                            "left_units": left[1],
                            "right_units": right[1],
                            **comparison,
                            "stable_tie_order": sorted([left[0], right[0]]),
                        }
                    )
        leaf_rankings[leaf] = {
            "rankings": rankings,
            "pairwise_within_bound": pairwise,
        }

    return {
        "leaf": "consistent unit weights propagated from integer dependency coefficients",
        "propagation": "topological integer multiply-add; coefficient share and every product use floor division",
        "error_formula": "B(p,l) = sum_i(w_i*B(child_i,l) + (w_i,l+B(child_i,l))/S + 1/S)",
        "paths": paths_payload,
        "leaf_rankings": leaf_rankings,
    }


def compute_batch_metrics(
    spec: Spec,
    grid: FixedGrid,
    batch_id: int,
    records: list[BatchRequestRecord],
    content_topology: dict[str, tuple[Dependency, ...]] | None = None,
) -> dict[str, Any]:
    ordered_records = sorted(records, key=lambda item: item.seq)
    node_ids = sorted(spec.nodes)
    node_blocks = {
        node_id: _node_block(ordered_records, node_id, grid) for node_id in node_ids
    }
    aggregate = _aggregate_block(ordered_records, node_ids, node_blocks, grid)
    comparisons: dict[str, Any] = {}
    for metric_name in ("hit_rate", "fresh_hit_rate", "version_consistency_rate", "remaining_ttl_seconds"):
        values = {
            node_id: int(node_blocks[node_id]["metrics"][metric_name]["value_units"])
            for node_id in node_ids
        }
        comparisons[metric_name] = _cross_node_comparisons(node_ids, values, grid)
    derived = _derived_weights(spec, grid, content_topology)
    source_counts = {source: 0 for source in sorted(VALID_SOURCES)}
    for record in ordered_records:
        source_counts[record.hit_source] = source_counts.get(record.hit_source, 0) + 1
    return {
        "batch_id": batch_id,
        "grid": grid.as_dict(),
        "request_count": len(ordered_records),
        "source_counts": source_counts,
        "nodes": node_blocks,
        "aggregate": aggregate,
        "cross_node_comparisons": comparisons,
        "derived_weights": derived,
        "ordering": {
            "node_order": node_ids,
            "tie_break": "descending fixed value, then lexicographic node/path id",
            "processing_order_independent": True,
        },
    }


def error_summary_from_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"nodes": {}, "aggregate": {}}
    for node_id in sorted(metrics["nodes"]):
        node_metrics = metrics["nodes"][node_id]["metrics"]
        summary["nodes"][node_id] = {
            name: {
                "value_units": node_metrics[name]["value_units"],
                "error_bound_units": node_metrics[name]["error_bound_units"],
                "dominant_error_source": node_metrics[name]["dominant_error_source"],
            }
            for name in sorted(node_metrics)
        }
    for name in sorted(metrics["aggregate"]["metrics"]):
        metric = metrics["aggregate"]["metrics"][name]
        summary["aggregate"][name] = {
            "value_units": metric["value_units"],
            "error_bound_units": metric["error_bound_units"],
            "dominant_error_source": metric["dominant_error_source"],
        }
    return summary
