"""Dual-path layer inference, canonical summaries and error attribution."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np

from .graph import DependencyGraph
from .quant import (
    error_bound_components,
    error_statistics,
    fixed_forward_rows,
    float_forward_rows,
    stable_array_digest,
)
from .samples import SampleFactory
from .schema import Layer, Plan, layer_memory_bytes
from .util import digest_value, sha256_bytes


QUANTIZATION_POLICY = {
    "rounding": "round_half_away_from_zero",
    "operand_saturation": "signed_int24",
    "accumulator_saturation": "signed_int64",
    "reduction_order": "ascending_input_index",
    "dequantization": "accumulator / scale**2",
    "cast": "float32",
    "lease_reordering_numeric_effect": 0,
}


@dataclass
class LayerBatchResult:
    layer_id: str
    batch_id: str
    sample_ids: tuple[str, ...]
    layer_version: int
    plan_version: int
    float_output: np.ndarray
    quant_output: np.ndarray
    errors: dict[str, Any]
    error_bound: dict[str, float]
    saturation_count: int
    occupancy_bytes: int
    occupancy_blocks: int
    lease_id: str | None = None
    block_start: int | None = None
    block_count: int | None = None
    input_digest: str = ""
    float_digest: str = ""
    quant_digest: str = ""
    per_sample: dict[str, dict[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.float_digest:
            self.float_digest = stable_array_digest(self.float_output)
        if not self.quant_digest:
            self.quant_digest = stable_array_digest(self.quant_output)
        if not self.per_sample:
            for index, sample_id in enumerate(self.sample_ids):
                self.per_sample[sample_id] = {
                    "float": stable_array_digest(self.float_output[index]),
                    "quant": stable_array_digest(self.quant_output[index]),
                }

    def ledger_payload(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "batch_id": self.batch_id,
            "sample_ids": list(self.sample_ids),
            "layer_version": self.layer_version,
            "plan_version": self.plan_version,
            "float_digest": self.float_digest,
            "quant_digest": self.quant_digest,
            "per_sample": {key: self.per_sample[key] for key in sorted(self.per_sample)},
            "errors": self.errors,
            "error_bound": self.error_bound,
            "saturation_count": self.saturation_count,
            "occupancy_bytes": self.occupancy_bytes,
            "occupancy_blocks": self.occupancy_blocks,
            "lease_id": self.lease_id,
            "block_start": self.block_start,
            "block_count": self.block_count,
            "input_digest": self.input_digest,
        }

    def to_dict(self) -> dict[str, Any]:
        result = self.ledger_payload()
        result["float_output"] = np.asarray(self.float_output, dtype="<f4", order="C").tolist()
        result["quant_output"] = np.asarray(self.quant_output, dtype="<f4", order="C").tolist()
        return result

    def clone_with_lease(self, lease: Any) -> "LayerBatchResult":
        return LayerBatchResult(
            layer_id=self.layer_id,
            batch_id=self.batch_id,
            sample_ids=self.sample_ids,
            layer_version=self.layer_version,
            plan_version=self.plan_version,
            float_output=self.float_output,
            quant_output=self.quant_output,
            errors=self.errors,
            error_bound=self.error_bound,
            saturation_count=self.saturation_count,
            occupancy_bytes=self.occupancy_bytes,
            occupancy_blocks=self.occupancy_blocks,
            lease_id=lease.lease_id,
            block_start=lease.block_start,
            block_count=lease.block_count,
            input_digest=self.input_digest,
            float_digest=self.float_digest,
            quant_digest=self.quant_digest,
            per_sample=self.per_sample,
        )


def _compose_input(
    layer: Layer,
    sample_ids: tuple[str, ...],
    factory: SampleFactory,
    upstream: Mapping[str, LayerBatchResult],
) -> np.ndarray:
    if not layer.depends_on:
        return factory.root_matrix(layer.id, sample_ids)
    matrices: list[np.ndarray] = []
    for dependency in layer.depends_on:
        if dependency not in upstream:
            raise KeyError(f"missing upstream result for {dependency}")
        matrices.append(np.asarray(upstream[dependency].float_output, dtype=np.float32, order="C"))
    result = np.concatenate(matrices, axis=1)
    if result.shape != (len(sample_ids), layer.input_dim):
        raise ValueError(
            f"composed input for {layer.id} has shape {result.shape}, expected "
            f"{(len(sample_ids), layer.input_dim)}"
        )
    return np.asarray(result, dtype=np.float32, order="C")


def compute_layer_result(
    plan: Plan,
    factory: SampleFactory,
    layer: Layer,
    sample_ids: tuple[str, ...],
    batch_id: str,
    layer_version: int,
    upstream: Mapping[str, LayerBatchResult],
) -> LayerBatchResult:
    inputs = _compose_input(layer, sample_ids, factory, upstream)
    weights, bias = factory.weights(layer.id)
    float_output = float_forward_rows(inputs, weights, bias)
    fixed = fixed_forward_rows(inputs, weights, bias, layer.scale)
    errors = error_statistics(float_output, fixed.values)
    bound = error_bound_components(inputs, weights, bias, layer.scale)
    occupancy_bytes = layer_memory_bytes(plan, layer.id, len(sample_ids))
    occupancy_blocks = occupancy_bytes // plan.block_size_bytes
    input_digest = sha256_bytes(
        np.asarray(inputs, dtype="<f4", order="C").tobytes(order="C")
    )
    per_sample: dict[str, dict[str, Any]] = {}
    for index, sample_id in enumerate(sample_ids):
        sample_error = error_statistics(
            float_output[index : index + 1], fixed.values[index : index + 1]
        )
        per_sample[sample_id] = {
            "float": stable_array_digest(float_output[index]),
            "quant": stable_array_digest(fixed.values[index]),
            "sum_abs_error": sample_error["sum_abs_error"],
            "max_abs_error": sample_error["max_abs_error"],
            "max_rel_error": sample_error["max_rel_error"],
            "element_count": sample_error["element_count"],
        }
    return LayerBatchResult(
        layer_id=layer.id,
        batch_id=batch_id,
        sample_ids=tuple(sample_ids),
        layer_version=layer_version,
        plan_version=plan.version,
        float_output=float_output,
        quant_output=fixed.values,
        errors=errors,
        error_bound=bound,
        saturation_count=fixed.saturation_count,
        occupancy_bytes=occupancy_bytes,
        occupancy_blocks=occupancy_blocks,
        input_digest=input_digest,
        per_sample=per_sample,
    )


def compute_batch(
    plan: Plan,
    graph: DependencyGraph,
    factory: SampleFactory,
    sample_ids: Iterable[str],
    batch_id: str,
    layer_versions: Mapping[str, int],
) -> dict[str, LayerBatchResult]:
    ids = tuple(sample_ids)
    results: dict[str, LayerBatchResult] = {}
    for layer_id in graph.order:
        layer = plan.layer_by_id[layer_id]
        results[layer_id] = compute_layer_result(
            plan,
            factory,
            layer,
            ids,
            batch_id,
            layer_versions[layer_id],
            results,
        )
    return results


def aggregate_layer_digest(results: Iterable[LayerBatchResult]) -> dict[str, Any]:
    by_sample: dict[str, dict[str, str]] = {}
    for result in results:
        for sample_id, hashes in result.per_sample.items():
            if sample_id in by_sample:
                previous = by_sample[sample_id]
                if previous["float"] != hashes["float"] or previous["quant"] != hashes["quant"]:
                    raise ValueError(f"conflicting duplicate result for {sample_id}")
            by_sample[sample_id] = {
                "float": hashes["float"],
                "quant": hashes["quant"],
            }
    ordered = [[sample_id, by_sample[sample_id]["float"], by_sample[sample_id]["quant"]] for sample_id in sorted(by_sample)]
    return {
        "sample_count": len(ordered),
        "float_digest": sha256_bytes(
            "".join(row[1] for row in ordered).encode("ascii")
        ),
        "quant_digest": sha256_bytes(
            "".join(row[2] for row in ordered).encode("ascii")
        ),
        "combined_digest": digest_value(ordered),
        "samples": ordered,
    }


def aggregate_layer_outputs(results: Iterable[LayerBatchResult]) -> dict[str, np.ndarray]:
    rows: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for result in results:
        for index, sample_id in enumerate(result.sample_ids):
            if sample_id in rows:
                raise ValueError(f"duplicate sample in results: {sample_id}")
            rows[sample_id] = (result.float_output[index], result.quant_output[index])
    float_output = np.stack([rows[key][0] for key in sorted(rows)], axis=0)
    quant_output = np.stack([rows[key][1] for key in sorted(rows)], axis=0)
    return {"float": float_output, "quant": quant_output}


def aggregate_layer_errors(results: Iterable[LayerBatchResult]) -> dict[str, Any]:
    items = list(results)
    if not items:
        return {
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "sum_abs_error": 0.0,
            "max_rel_error": 0.0,
            "element_count": 0,
            "saturation_count": 0,
        }
    samples: dict[str, Mapping[str, Any]] = {}
    for item in items:
        for sample_id, stats in item.per_sample.items():
            previous = samples.get(sample_id)
            if previous is not None and previous != stats:
                raise ValueError(f"conflicting per-sample error stats for {sample_id}")
            samples[sample_id] = stats
    ordered = [samples[sample_id] for sample_id in sorted(samples)]
    total_elements = sum(int(stats["element_count"]) for stats in ordered)
    sum_abs = math.fsum(float(stats["sum_abs_error"]) for stats in ordered)
    mean_abs = sum_abs / total_elements if total_elements else 0.0
    return {
        "max_abs_error": max(float(stats["max_abs_error"]) for stats in ordered),
        "mean_abs_error": mean_abs,
        "sum_abs_error": sum_abs,
        "max_rel_error": max(float(stats["max_rel_error"]) for stats in ordered),
        "element_count": total_elements,
        "saturation_count": sum(item.saturation_count for item in items),
    }


def _aggregate_bound(results: Iterable[LayerBatchResult]) -> dict[str, float]:
    components = ("quantization_truncation", "accumulation_order", "lease_reordering")
    maxima = {
        name: max(float(result.error_bound[name]) for result in results)
        for name in components
    }
    maxima["combined"] = sum(maxima.values())
    return maxima


def attribute_errors(
    plan: Plan,
    graph: DependencyGraph,
    results_by_layer: Mapping[str, Iterable[LayerBatchResult]],
) -> dict[str, Any]:
    """Produce scheduling-order-independent per-layer attribution."""

    layers: list[dict[str, Any]] = []
    total_abs = 0.0
    for layer_id in graph.order:
        items = list(results_by_layer.get(layer_id, ()))
        stats = aggregate_layer_errors(items)
        bound = _aggregate_bound(items) if items else {
            "quantization_truncation": 0.0,
            "accumulation_order": 0.0,
            "lease_reordering": 0.0,
            "combined": 0.0,
        }
        total_abs += float(stats["sum_abs_error"])
        layers.append(
            {
                "layer_id": layer_id,
                "max_abs_error": stats["max_abs_error"],
                "mean_abs_error": stats["mean_abs_error"],
                "sum_abs_error": stats["sum_abs_error"],
                "max_rel_error": stats["max_rel_error"],
                "element_count": stats["element_count"],
                "saturation_count": stats["saturation_count"],
                "error_bound": bound,
                "within_bound": stats["max_abs_error"] <= bound["combined"] + 1e-30,
                "cumulative_share": 0.0,
            }
        )
    for layer in layers:
        layer["cumulative_share"] = (
            layer["sum_abs_error"] / total_abs if total_abs > 0 else 0.0
        )
    accumulated = 0.0
    for layer in layers:
        accumulated += layer["cumulative_share"]
        layer["cumulative_share"] = accumulated

    ranked = sorted(
        layers,
        key=lambda item: (-item["max_abs_error"], item["layer_id"]),
    )
    top = ranked[0] if ranked else None
    tie_candidates: list[dict[str, Any]] = []
    error_affected = False
    if top and len(ranked) > 1:
        runner_up = ranked[1]
        overlap = min(top["error_bound"]["combined"], runner_up["error_bound"]["combined"])
        if abs(top["max_abs_error"] - runner_up["max_abs_error"]) <= overlap:
            error_affected = True
            threshold = top["max_abs_error"] - max(overlap, 0.0)
            tie_candidates = sorted(
                [item for item in layers if item["max_abs_error"] >= threshold],
                key=lambda item: item["layer_id"],
            )
            top = tie_candidates[0]

    component_totals = {
        "quantization_truncation": sum(
            layer["error_bound"]["quantization_truncation"] for layer in layers
        ),
        "accumulation_order": sum(
            layer["error_bound"]["accumulation_order"] for layer in layers
        ),
        "lease_reordering": sum(
            layer["error_bound"]["lease_reordering"] for layer in layers
        ),
    }
    bound_total = sum(component_totals.values())
    shares = {
        name: (value / bound_total if bound_total > 0 else 0.0)
        for name, value in component_totals.items()
    }
    ranking = sorted(shares, key=lambda name: (-shares[name], name))
    first, second = ranking[0], ranking[1]
    if shares[first] >= 0.60 and shares[first] - shares[second] >= 0.10:
        dominant = first
        rule = "share>=0.60 and lead>=0.10"
    else:
        dominant = "mixed"
        rule = "otherwise mixed"
    return {
        "policy": dict(QUANTIZATION_POLICY),
        "layers": layers,
        "max_deviation_layer": top["layer_id"] if top else None,
        "max_deviation_tie_candidates": [item["layer_id"] for item in tie_candidates],
        "max_deviation_error_affected": error_affected,
        "tie_rule": "stable layer_id ascending when empirical gap <= combined bound",
        "component_totals": component_totals,
        "component_shares": shares,
        "total_error_bound": bound_total,
        "dominant_component": dominant,
        "dominant_rule": rule,
        "ordering_independent": True,
    }


def compare_aggregates(
    left: Mapping[str, Mapping[str, Any]],
    right: Mapping[str, Mapping[str, Any]],
    *,
    graph: DependencyGraph,
) -> dict[str, Any] | None:
    """Locate the first aggregate/field mismatch in stable topological order."""

    fields = (
        "float_digest",
        "quant_digest",
        "combined_digest",
        "occupancy_bytes",
        "occupancy_blocks",
        "max_abs_error",
        "mean_abs_error",
        "error_bound",
        "lease_range",
        "layer_version",
    )
    for layer_id in graph.order:
        if layer_id not in left:
            return {"layer_id": layer_id, "field": "<missing-left>"}
        if layer_id not in right:
            return {"layer_id": layer_id, "field": "<missing-right>"}
        for field_name in fields:
            if left[layer_id].get(field_name) != right[layer_id].get(field_name):
                return {
                    "layer_id": layer_id,
                    "field": field_name,
                    "left": left[layer_id].get(field_name),
                    "right": right[layer_id].get(field_name),
                }
    return None
