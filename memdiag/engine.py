"""Incremental inference, block-pool allocation and diagnostics engine."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from .deps import DependencyGraph, build_dependency_graph, propagate_invalidation
from .errors import IncrementalError, OOMError
from .models import (
    BatchRecord,
    DiagnosticRun,
    InvalidationStep,
    LayerBatchCommit,
    LayerResult,
    LayerSpec,
    Plan,
    SampleSet,
    batch_sample_groups,
    digest_sample_subset,
)
from .quant import (
    accumulation_order_delta,
    adapt_input,
    deterministic_layer_parameters,
    fixed_point_layer_forward,
    float32_layer_forward,
)
from .util import array_digest, canonical_digest


ProgressCallback = Callable[
    [str, LayerResult, tuple[LayerBatchCommit, ...], dict[str, Any]], None
]


@dataclass(frozen=True)
class Allocation:
    blocks: tuple[int, ...]
    mode: str
    runs: tuple[tuple[int, int], ...]


class MemoryBlockPool:
    """A deterministic block pool with contiguous-first allocation."""

    def __init__(
        self,
        capacity_blocks: int,
        block_size_bytes: int,
        allocations: Mapping[str, Iterable[int]] | None = None,
        modes: Mapping[str, str] | None = None,
    ) -> None:
        if capacity_blocks < 0:
            raise ValueError("capacity_blocks must be non-negative")
        if block_size_bytes <= 0:
            raise ValueError("block_size_bytes must be positive")
        self.capacity_blocks = int(capacity_blocks)
        self.block_size_bytes = int(block_size_bytes)
        self._allocations: dict[str, Allocation] = {}
        for layer_id, raw_blocks in (allocations or {}).items():
            blocks = tuple(sorted(int(block) for block in raw_blocks))
            self._validate_blocks(blocks, layer_id)
            for other_id, other in self._allocations.items():
                overlap = set(blocks) & set(other.blocks)
                if overlap:
                    raise ValueError(
                        f"block overlap between {layer_id} and {other_id}: {sorted(overlap)}"
                    )
            mode = str((modes or {}).get(layer_id, "restored"))
            self._allocations[str(layer_id)] = Allocation(
                blocks=blocks,
                mode=mode,
                runs=_runs_from_blocks(blocks),
            )

    def _validate_blocks(self, blocks: tuple[int, ...], layer_id: str) -> None:
        if len(blocks) != len(set(blocks)):
            raise ValueError(f"duplicate blocks for {layer_id}")
        if any(block < 0 or block >= self.capacity_blocks for block in blocks):
            raise ValueError(f"out-of-range block for {layer_id}")

    def clone(self) -> "MemoryBlockPool":
        return MemoryBlockPool(
            self.capacity_blocks,
            self.block_size_bytes,
            {key: value.blocks for key, value in self._allocations.items()},
            {key: value.mode for key, value in self._allocations.items()},
        )

    @property
    def allocations(self) -> dict[str, tuple[int, ...]]:
        return {key: value.blocks for key, value in sorted(self._allocations.items())}

    @property
    def modes(self) -> dict[str, str]:
        return {key: value.mode for key, value in sorted(self._allocations.items())}

    @property
    def runs(self) -> dict[str, tuple[tuple[int, int], ...]]:
        return {key: value.runs for key, value in sorted(self._allocations.items())}

    @property
    def occupied_blocks(self) -> int:
        return sum(len(value.blocks) for value in self._allocations.values())

    @property
    def free_blocks(self) -> int:
        return self.capacity_blocks - self.occupied_blocks

    def occupied_set(self) -> set[int]:
        return {block for value in self._allocations.values() for block in value.blocks}

    def free_set(self) -> set[int]:
        return set(range(self.capacity_blocks)) - self.occupied_set()

    def largest_contiguous_free(self) -> int:
        free = self.free_set()
        largest = 0
        current = 0
        for block in range(self.capacity_blocks):
            if block in free:
                current += 1
                largest = max(largest, current)
            else:
                current = 0
        return largest

    @property
    def fragmentation_blocks(self) -> int:
        return self.free_blocks - self.largest_contiguous_free()

    def free(self, layer_id: str) -> tuple[int, ...]:
        allocation = self._allocations.pop(str(layer_id), None)
        return () if allocation is None else allocation.blocks

    def allocate(self, layer_id: str, count: int, *, strategy: str = "auto") -> Allocation:
        if count <= 0:
            raise ValueError("count must be positive")
        if strategy not in {"auto", "contiguous", "single_block"}:
            raise ValueError("strategy must be auto, contiguous or single_block")
        layer_key = str(layer_id)
        if layer_key in self._allocations:
            raise ValueError(f"layer {layer_key} is already allocated")
        free = sorted(self.free_set())
        if len(free) < count:
            raise OOMError(
                "insufficient free blocks for layer allocation",
                layer_id=layer_key,
                needed_blocks=count,
                free_blocks=len(free),
                capacity_blocks=self.capacity_blocks,
                occupied_blocks=self.occupied_blocks,
            )

        contiguous = _free_runs(free)
        chosen: tuple[int, ...] | None = None
        mode = ""
        if strategy in {"auto", "contiguous"}:
            for start, length in contiguous:
                if length >= count:
                    chosen = tuple(range(start, start + count))
                    mode = "contiguous"
                    break
            if chosen is None and strategy == "contiguous":
                raise OOMError(
                    "no contiguous block run is large enough",
                    layer_id=layer_key,
                    needed_blocks=count,
                    free_blocks=len(free),
                    largest_contiguous_free_blocks=self.largest_contiguous_free(),
                    capacity_blocks=self.capacity_blocks,
                    occupied_blocks=self.occupied_blocks,
                )
        if chosen is None:
            chosen = tuple(free[:count])
            mode = "single_block"
        allocation = Allocation(
            blocks=chosen,
            mode=mode,
            runs=_runs_from_blocks(chosen),
        )
        self._allocations[layer_key] = allocation
        return allocation

    def allocation(self, layer_id: str) -> Allocation:
        return self._allocations[layer_id]


def _free_runs(free_blocks: Iterable[int]) -> tuple[tuple[int, int], ...]:
    ordered = sorted(int(block) for block in free_blocks)
    if not ordered:
        return ()
    runs: list[tuple[int, int]] = []
    start = ordered[0]
    previous = ordered[0]
    for block in ordered[1:]:
        if block == previous + 1:
            previous = block
            continue
        runs.append((start, previous - start + 1))
        start = block
        previous = block
    runs.append((start, previous - start + 1))
    return tuple(runs)


def _runs_from_blocks(blocks: Iterable[int]) -> tuple[tuple[int, int], ...]:
    ordered = sorted(int(block) for block in blocks)
    if not ordered:
        return ()
    runs: list[tuple[int, int]] = []
    start = ordered[0]
    previous = ordered[0]
    for block in ordered[1:]:
        if block == previous + 1:
            previous = block
            continue
        runs.append((start, previous - start + 1))
        start = block
        previous = block
    runs.append((start, previous - start + 1))
    return tuple(runs)


def layer_request_bytes(layer: LayerSpec, batch_size: int) -> int:
    weight_bytes = layer.param_count * layer.dtype_bytes
    activation_bytes = max(1, batch_size) * (layer.input_dim + layer.output_dim) * layer.dtype_bytes
    scratch_bytes = max(1, batch_size) * max(layer.input_dim, layer.output_dim) * (
        8 if layer.precision == "fixed" else 4
    )
    return int(weight_bytes + activation_bytes + scratch_bytes + 256)


def layer_block_count(layer: LayerSpec, plan: Plan) -> tuple[int, int]:
    request = layer_request_bytes(layer, plan.max_batch_size)
    return request, int(math.ceil(request / plan.block_size_bytes))


def _empty_numeric_cache() -> dict[str, dict[str, np.ndarray]]:
    return {}


def _parents_input(
    layer: LayerSpec,
    graph: DependencyGraph,
    sample_id: str,
    samples: SampleSet,
    cache: Mapping[str, Mapping[str, np.ndarray]],
) -> np.ndarray:
    parents = graph.parents(layer.layer_id)
    if not parents:
        source = np.asarray(samples.sample_map[sample_id].values, dtype=np.float32)
    else:
        pieces = [np.asarray(cache[parent][sample_id], dtype=np.float32) for parent in parents]
        source = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)
    return adapt_input(source, layer.input_dim)


def _compute_layer_values(
    plan: Plan,
    layer: LayerSpec,
    graph: DependencyGraph,
    samples: SampleSet,
    actual_cache: Mapping[str, Mapping[str, np.ndarray]],
    float_cache: Mapping[str, Mapping[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float, np.ndarray | None, np.ndarray | None]:
    weights, bias = deterministic_layer_parameters(
        plan.seed, layer.layer_id, layer.input_dim, layer.output_dim
    )
    actual_outputs: dict[str, np.ndarray] = {}
    float_outputs: dict[str, np.ndarray] = {}
    accumulation_delta = 0.0
    sample_actual: np.ndarray | None = None
    sample_float: np.ndarray | None = None
    for sample_id in samples.sorted_ids:
        actual_input = _parents_input(layer, graph, sample_id, samples, actual_cache)
        float_input = _parents_input(layer, graph, sample_id, samples, float_cache)
        reference = float32_layer_forward(float_input, weights, bias)
        if layer.precision == "fixed":
            actual, _ = fixed_point_layer_forward(
                actual_input, weights, bias, layer.scale, accumulation_order="standard"
            )
            accumulation_delta = max(
                accumulation_delta,
                accumulation_order_delta(actual_input, weights, bias, layer.scale),
            )
        else:
            actual = float32_layer_forward(actual_input, weights, bias)
        actual_outputs[sample_id] = np.asarray(actual, dtype=np.float32)
        float_outputs[sample_id] = np.asarray(reference, dtype=np.float32)
        if sample_actual is None:
            sample_actual = actual_outputs[sample_id].reshape(1, -1)
            sample_float = float_outputs[sample_id].reshape(1, -1)
        else:
            sample_actual = np.concatenate([sample_actual, actual_outputs[sample_id].reshape(1, -1)])
            sample_float = np.concatenate([sample_float, float_outputs[sample_id].reshape(1, -1)])
    return actual_outputs, float_outputs, accumulation_delta, sample_actual, sample_float


def _digest_layer_outputs(sample_digests: Mapping[str, str]) -> str:
    return canonical_digest([[key, sample_digests[key]] for key in sorted(sample_digests)])


def _base_component_bound(layer: LayerSpec) -> float:
    if layer.precision != "fixed":
        return 0.0
    local = (layer.input_dim + layer.output_dim + 1) * 0.5 * (2.0 ** (-layer.scale))
    return float(local)


def _error_for_layer(
    layer: LayerSpec,
    actual: np.ndarray | None,
    reference: np.ndarray | None,
    accumulation_delta: float,
    allocation: Allocation,
    parent_results: Mapping[str, LayerResult],
) -> tuple[float, float, dict[str, float], str]:
    if actual is None or reference is None or actual.size == 0:
        actual_error = 0.0
    else:
        actual_error = float(np.max(np.abs(actual.astype(np.float64) - reference.astype(np.float64))))
    local_quantization = _base_component_bound(layer)
    inherited = max(
        (
            parent_results[parent].error_components.get("quantization_truncation", 0.0)
            for parent in sorted(parent_results)
        ),
        default=0.0,
    )
    quantization = max(local_quantization + inherited, 0.0)
    accumulation = max(0.0, float(accumulation_delta))
    allocation_gaps = max(0, len(allocation.runs) - 1)
    memory_reordering = (
        allocation_gaps * (2.0 ** (-layer.scale)) if layer.precision == "fixed" else 0.0
    )
    quantization = max(quantization, actual_error - accumulation - memory_reordering)
    components = {
        "quantization_truncation": float(quantization),
        "accumulation_order": float(accumulation),
        "memory_reordering": float(memory_reordering),
    }
    dominant = sorted(components, key=lambda key: (-components[key], key))[0]
    return float(sum(components.values())), actual_error, components, dominant


def rank_attribution(
    results: Mapping[str, LayerResult], topo_ids: Iterable[str]
) -> dict[str, Any]:
    ordered_ids = tuple(topo_ids)
    total_bound = sum(max(0.0, results[layer_id].error_bound) for layer_id in ordered_ids)
    cumulative = 0.0
    cumulative_ratios: dict[str, float] = {}
    for layer_id in ordered_ids:
        cumulative += max(0.0, results[layer_id].error_bound)
        cumulative_ratios[layer_id] = cumulative / total_bound if total_bound > 0 else 0.0

    ranked = sorted(
        ordered_ids,
        key=lambda layer_id: (-results[layer_id].max_abs_error, layer_id),
    )
    rankings: list[dict[str, Any]] = []
    for rank, layer_id in enumerate(ranked, start=1):
        result = results[layer_id]
        rankings.append(
            {
                "rank": rank,
                "layer_id": layer_id,
                "max_abs_error": result.max_abs_error,
                "error_bound": result.error_bound,
                "error_interval": [result.max_abs_error, result.error_bound],
                "cumulative_error_ratio": cumulative_ratios[layer_id],
                "dominant_segment": result.dominant_segment,
            }
        )

    overlaps: list[dict[str, Any]] = []
    affected: set[str] = set()
    for index in range(len(rankings) - 1):
        left = results[rankings[index]["layer_id"]]
        right = results[rankings[index + 1]["layer_id"]]
        if left.max_abs_error <= right.error_bound and right.max_abs_error <= left.error_bound:
            pair = [rankings[index]["layer_id"], rankings[index + 1]["layer_id"]]
            affected.update(pair)
            overlaps.append(
                {
                    "layers": sorted(pair),
                    "left_interval": [left.max_abs_error, left.error_bound],
                    "right_interval": [right.max_abs_error, right.error_bound],
                }
            )
    for entry in rankings:
        entry["tie_affected"] = entry["layer_id"] in affected

    return {
        "ranking_rule": "max_abs_error_desc_then_layer_id_asc",
        "tie_rule": "overlapping_error_intervals_marked_tie_affected",
        "total_error_bound": total_bound,
        "top_layer": ranked[0] if ranked else None,
        "top_decision_tie_affected": bool(ranked and ranked[0] in affected),
        "tie_affected_layers": sorted(affected),
        "overlaps": overlaps,
        "rankings": rankings,
        "cumulative_error_ratio": {
            layer_id: cumulative_ratios[layer_id] for layer_id in ordered_ids
        },
    }


def _build_batch_groups(samples: SampleSet, max_batch_size: int) -> tuple[tuple[str, tuple[str, ...]], ...]:
    groups = batch_sample_groups(samples, max_batch_size)
    return tuple((f"b{index:04d}", group) for index, group in enumerate(groups))


def _layer_batch_commits(
    layer: LayerSpec,
    sample_digests: Mapping[str, str],
    sample_float_digests: Mapping[str, str],
    allocation: Allocation,
    groups: tuple[tuple[str, tuple[str, ...]], ...],
    pool: MemoryBlockPool,
) -> tuple[LayerBatchCommit, ...]:
    commits: list[LayerBatchCommit] = []
    for batch_id, sample_ids in groups:
        commits.append(
            LayerBatchCommit(
                batch_id=batch_id,
                sample_ids=tuple(sample_ids),
                block_ids=allocation.blocks,
                request_bytes=layer_request_bytes(layer, len(sample_ids)),
                peak_blocks=pool.occupied_blocks,
                fragmentation_blocks=pool.fragmentation_blocks,
                output_digest=digest_sample_subset(sample_digests, sample_ids),
                float_digest=digest_sample_subset(sample_float_digests, sample_ids),
            )
        )
    return tuple(commits)


def _batch_records(
    plan: Plan,
    samples: SampleSet,
    results: Mapping[str, LayerResult],
    pool: MemoryBlockPool,
    groups: tuple[tuple[str, tuple[str, ...]], ...],
) -> tuple[BatchRecord, ...]:
    all_blocks = tuple(sorted(pool.occupied_set()))
    records: list[BatchRecord] = []
    layer_map = plan.layer_map
    ordered_layers = tuple(sorted(results))
    for batch_id, sample_ids in groups:
        request_bytes = sum(
            layer_request_bytes(layer_map[layer_id], len(sample_ids))
            for layer_id in ordered_layers
        )
        digests = {
            layer_id: digest_sample_subset(
                results[layer_id].sample_output_digests, sample_ids
            )
            for layer_id in ordered_layers
        }
        float_digests = {
            layer_id: digest_sample_subset(
                results[layer_id].sample_float_digests, sample_ids
            )
            for layer_id in ordered_layers
        }
        records.append(
            BatchRecord(
                batch_id=batch_id,
                sample_ids=tuple(sample_ids),
                layer_ids=ordered_layers,
                block_ids=all_blocks,
                request_bytes=request_bytes,
                peak_blocks=pool.occupied_blocks,
                fragmentation_blocks=pool.fragmentation_blocks,
                layer_output_digests=digests,
                layer_float_digests=float_digests,
            )
        )
    return tuple(records)


def _compute_content_digest(run: DiagnosticRun) -> str:
    payload = run.result_dict(include_samples=True)
    payload.pop("content_digest", None)
    payload.pop("snapshot_format_version", None)
    return canonical_digest(payload)


def _compute_run(
    *,
    plan: Plan,
    samples: SampleSet,
    base_run: DiagnosticRun | None,
    changed_sources: Mapping[str, str],
    numeric_invalid: set[str],
    memory_invalid: set[str],
    invalidation_override: tuple[tuple[str, ...], tuple[InvalidationStep, ...]] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> DiagnosticRun:
    graph = build_dependency_graph(plan)
    if invalidation_override is None:
        invalidated, invalidation_trace = propagate_invalidation(
            plan, None if base_run is None else base_run.plan, changed_sources
        )
    else:
        invalidated, invalidation_trace = invalidation_override

    if base_run is None:
        pool = MemoryBlockPool(plan.capacity_blocks, plan.block_size_bytes)
    else:
        pool = MemoryBlockPool(
            plan.capacity_blocks,
            plan.block_size_bytes,
            base_run.allocations,
            base_run.allocation_modes,
        )
        for layer_id in sorted(set(pool.allocations) - set(plan.layer_ids)):
            pool.free(layer_id)
    for layer_id in sorted(memory_invalid):
        if layer_id in pool.allocations:
            pool.free(layer_id)
    for layer_id in graph.topo:
        if layer_id not in memory_invalid:
            continue
        layer = plan.layer_map[layer_id]
        _, blocks_needed = layer_block_count(layer, plan)
        pool.allocate(layer_id, blocks_needed, strategy="auto")

    actual_cache: dict[str, dict[str, np.ndarray]] = {}
    float_cache: dict[str, dict[str, np.ndarray]] = {}
    if base_run is not None:
        for layer_id in plan.layer_ids:
            if layer_id in base_run.actual_values:
                actual_cache[layer_id] = base_run.actual_values[layer_id]
            if layer_id in base_run.float_values:
                float_cache[layer_id] = base_run.float_values[layer_id]

    groups = _build_batch_groups(samples, plan.max_batch_size)
    results: dict[str, LayerResult] = {}
    for layer_id in graph.topo:
        layer = plan.layer_map[layer_id]
        allocation = pool.allocation(layer_id)
        previous_result = None if base_run is None else base_run.layers.get(layer_id)

        if base_run is not None and layer_id not in numeric_invalid and previous_result is not None:
            if layer_id not in actual_cache or layer_id not in float_cache:
                raise RuntimeError(
                    f"runtime numeric cache is missing for unaffected layer {layer_id}; hydrate the run first"
                )
            actual_cache[layer_id] = base_run.actual_values[layer_id]
            float_cache[layer_id] = base_run.float_values[layer_id]
            result = copy.deepcopy(previous_result)
            result.spec = layer.to_dict()
            result.block_ids = allocation.blocks
            result.allocation_mode = allocation.mode
            result.allocation_runs = allocation.runs
            result.request_bytes = layer_request_bytes(layer, plan.max_batch_size)
        else:
            parent_results = {
                parent: results[parent]
                for parent in graph.parents(layer_id)
                if parent in results
            }
            actual_values, float_values, accumulation_delta, sample_actual, sample_float = _compute_layer_values(
                plan, layer, graph, samples, actual_cache, float_cache
            )
            actual_cache[layer_id] = actual_values
            float_cache[layer_id] = float_values
            sample_digests = {
                sample_id: array_digest(actual_values[sample_id])
                for sample_id in samples.sorted_ids
            }
            float_digests = {
                sample_id: array_digest(float_values[sample_id])
                for sample_id in samples.sorted_ids
            }
            error_bound, max_abs_error, components, dominant = _error_for_layer(
                layer,
                sample_actual,
                sample_float,
                accumulation_delta,
                allocation,
                parent_results,
            )
            version = 1 if previous_result is None else previous_result.version + (
                1 if layer_id in memory_invalid else 0
            )
            result = LayerResult(
                layer_id=layer_id,
                version=version,
                spec=layer.to_dict(),
                output_digest=_digest_layer_outputs(sample_digests),
                float_digest=_digest_layer_outputs(float_digests),
                sample_output_digests=sample_digests,
                sample_float_digests=float_digests,
                block_ids=allocation.blocks,
                allocation_mode=allocation.mode,
                allocation_runs=allocation.runs,
                request_bytes=layer_request_bytes(layer, plan.max_batch_size),
                error_bound=error_bound,
                max_abs_error=max_abs_error,
                error_components=components,
                dominant_segment=dominant,
            )
        results[layer_id] = result
        if progress_callback is not None:
            batch_commits = _layer_batch_commits(
                layer,
                result.sample_output_digests,
                result.sample_float_digests,
                allocation,
                groups,
                pool,
            )
            progress_callback(
                layer_id,
                result,
                batch_commits,
                {
                    "invalidated": list(invalidated),
                    "memory_invalid": sorted(memory_invalid),
                    "occupied_blocks": pool.occupied_blocks,
                    "free_blocks": pool.free_blocks,
                },
            )

    batches = _batch_records(plan, samples, results, pool, groups)
    attribution = rank_attribution(results, graph.topo)
    run = DiagnosticRun(
        plan=plan,
        samples=samples,
        layers=results,
        pool_capacity_blocks=pool.capacity_blocks,
        allocations=pool.allocations,
        allocation_modes=pool.modes,
        allocation_runs=pool.runs,
        batches=batches,
        invalidation=invalidation_trace,
        attribution=attribution,
        actual_values=actual_cache,
        float_values=float_cache,
    )
    run.content_digest = _compute_content_digest(run)
    return run


def run_initial(
    plan: Plan,
    samples: SampleSet,
    *,
    progress_callback: ProgressCallback | None = None,
) -> DiagnosticRun:
    changed = {layer_id: "initial_compute" for layer_id in plan.layer_ids}
    invalid = set(plan.layer_ids)
    return _compute_run(
        plan=plan,
        samples=samples,
        base_run=None,
        changed_sources=changed,
        numeric_invalid=invalid,
        memory_invalid=invalid,
        progress_callback=progress_callback,
    )


def apply_incremental(
    plan: Plan,
    samples: SampleSet,
    base_run: DiagnosticRun,
    changed_sources: Mapping[str, str],
    *,
    progress_callback: ProgressCallback | None = None,
) -> tuple[DiagnosticRun, tuple[str, ...]]:
    if samples.digest != base_run.samples.digest:
        raise ValueError("incremental inference requires the same deterministic sample set")
    base_run = hydrate_run(base_run)
    invalidated, trace = propagate_invalidation(plan, base_run.plan, changed_sources)
    invalid_set = set(invalidated)
    run = _compute_run(
        plan=plan,
        samples=samples,
        base_run=base_run,
        changed_sources=changed_sources,
        numeric_invalid=invalid_set,
        memory_invalid=invalid_set,
        invalidation_override=(invalidated, trace),
        progress_callback=progress_callback,
    )
    return run, invalidated


def recompute_full_numeric(
    plan: Plan,
    samples: SampleSet,
    base_run: DiagnosticRun,
    invalidated: Iterable[str],
    invalidation_trace: Iterable[InvalidationStep] | None = None,
    *,
    progress_callback: ProgressCallback | None = None,
) -> DiagnosticRun:
    """Recompute every numeric conclusion while anchoring unchanged block layouts.

    Block numbers are history-dependent by design: incremental mode must retain
    unaffected allocations.  The full oracle therefore recomputes all layer
    numerics/errors from scratch but uses the same invalidation trace and
    allocation anchors for unaffected layers.
    """

    base_run = hydrate_run(base_run)
    invalidated_tuple = tuple(invalidated)
    graph = build_dependency_graph(plan)
    trace = tuple(invalidation_trace) if invalidation_trace is not None else tuple(
        InvalidationStep(
            layer_id=layer_id,
            trigger_source=layer_id,
            via=None,
            depth=0,
            reason="full_recompute",
            path=(layer_id,),
        )
        for layer_id in graph.topo
    )
    return _compute_run(
        plan=plan,
        samples=samples,
        base_run=base_run,
        changed_sources={layer_id: "full_recompute" for layer_id in graph.topo},
        numeric_invalid=set(graph.topo),
        memory_invalid=set(invalidated_tuple),
        invalidation_override=(invalidated_tuple, trace),
        progress_callback=progress_callback,
    )


def compare_runs(incremental: DiagnosticRun, full: DiagnosticRun) -> dict[str, Any]:
    if incremental.plan.digest != full.plan.digest:
        raise IncrementalError(
            "incremental and full recomputation used different plans",
            layer_id="<plan>",
            field="plan_digest",
            incremental=incremental.plan.digest,
            full=full.plan.digest,
        )
    ordered = sorted(incremental.layers, key=lambda item: incremental.topo_ids.index(item))
    fields = [
        "output_digest",
        "float_digest",
        "sample_output_digests",
        "sample_float_digests",
        "block_ids",
        "allocation_mode",
        "allocation_runs",
        "request_bytes",
        "error_bound",
        "max_abs_error",
        "error_components",
        "dominant_segment",
    ]
    for layer_id in ordered:
        if layer_id not in full.layers:
            raise IncrementalError(
                "layer missing from full recomputation",
                layer_id=layer_id,
                field="layer",
                incremental="present",
                full="missing",
            )
        left = incremental.layers[layer_id]
        right = full.layers[layer_id]
        for field_name in fields:
            left_value = getattr(left, field_name)
            right_value = getattr(right, field_name)
            if left_value != right_value:
                raise IncrementalError(
                    "incremental result differs from full recomputation",
                    layer_id=layer_id,
                    field=field_name,
                    incremental=left_value,
                    full=right_value,
                )
    if incremental.batches != full.batches:
        left_batches = {batch.batch_id: batch for batch in incremental.batches}
        right_batches = {batch.batch_id: batch for batch in full.batches}
        for batch_id in sorted(set(left_batches) | set(right_batches)):
            if left_batches.get(batch_id) != right_batches.get(batch_id):
                layer_id = "<batch>"
                if batch_id in left_batches and left_batches[batch_id].layer_ids:
                    layer_id = left_batches[batch_id].layer_ids[0]
                elif batch_id in right_batches and right_batches[batch_id].layer_ids:
                    layer_id = right_batches[batch_id].layer_ids[0]
                raise IncrementalError(
                    "incremental batch conclusion differs from full recomputation",
                    layer_id=layer_id,
                    field=f"batch[{batch_id}]",
                    incremental=None if batch_id not in left_batches else left_batches[batch_id].to_dict(),
                    full=None if batch_id not in right_batches else right_batches[batch_id].to_dict(),
                )
    if incremental.memory_summary != full.memory_summary:
        raise IncrementalError(
            "incremental memory summary differs from full recomputation",
            layer_id="<memory>",
            field="memory_summary",
            incremental=incremental.memory_summary,
            full=full.memory_summary,
        )
    if incremental.invalidation != full.invalidation:
        raise IncrementalError(
            "incremental invalidation trace differs from full recomputation",
            layer_id="<invalidation>",
            field="invalidation_trace",
            incremental=[step.to_dict() for step in incremental.invalidation],
            full=[step.to_dict() for step in full.invalidation],
        )
    if incremental.attribution != full.attribution:
        raise IncrementalError(
            "incremental attribution differs from full recomputation",
            layer_id="<attribution>",
            field="attribution",
            incremental=incremental.attribution,
            full=full.attribution,
        )
    return {
        "ok": True,
        "checked_layers": list(ordered),
        "fields": fields,
        "incremental_content_digest": incremental.content_digest,
        "full_content_digest": full.content_digest,
    }


def hydrate_run(run: DiagnosticRun) -> DiagnosticRun:
    """Attach deterministic NumPy caches to a persisted/restored run.

    Layer results, versions and physical block numbers stay exactly as recorded.
    Only transient intermediates are reconstructed from the plan and samples.
    """

    if all(
        layer_id in run.actual_values and layer_id in run.float_values
        for layer_id in run.plan.layer_ids
    ):
        return run
    cold = run_initial(run.plan, run.samples)
    hydrated = copy.deepcopy(run)
    hydrated.actual_values = cold.actual_values
    hydrated.float_values = cold.float_values
    return hydrated


def hydrate_runtime(plan: Plan, samples: SampleSet) -> DiagnosticRun:
    """Rebuild a runtime cache from a declarative plan and sample set."""

    return run_initial(plan, samples)
