"""Immutable plan, sample, memory and diagnostic result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .util import array_digest, canonical_digest


SUPPORTED_PRECISIONS = ("float32", "fixed")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class LayerSpec:
    layer_id: str
    input_dim: int
    output_dim: int
    param_count: int
    precision: str
    scale: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.layer_id,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "param_count": self.param_count,
            "precision": self.precision,
            "scale": self.scale,
        }

    @property
    def dtype_bytes(self) -> int:
        return 4 if self.precision == "float32" else 2

    @property
    def spec_digest(self) -> str:
        return canonical_digest(self.to_dict())


@dataclass(frozen=True)
class Plan:
    format_version: int
    plan_version: int
    seed: int
    block_size_kb: int
    memory_budget_mib: int
    max_batch_size: int
    layers: tuple[LayerSpec, ...]
    consumers: Mapping[str, tuple[str, ...]]
    memory_derived_from: Mapping[str, tuple[str, ...]]

    @property
    def layer_ids(self) -> tuple[str, ...]:
        return tuple(layer.layer_id for layer in self.layers)

    @property
    def layer_map(self) -> dict[str, LayerSpec]:
        return {layer.layer_id: layer for layer in self.layers}

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "plan_version": self.plan_version,
            "seed": self.seed,
            "block_size_kb": self.block_size_kb,
            "memory_budget_mib": self.memory_budget_mib,
            "max_batch_size": self.max_batch_size,
            "layers": [layer.to_dict() for layer in self.layers],
            "consumers": {
                key: list(self.consumers[key]) for key in sorted(self.consumers)
            },
            "memory_derived_from": {
                key: list(self.memory_derived_from[key])
                for key in sorted(self.memory_derived_from)
            },
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_dict())

    @property
    def block_size_bytes(self) -> int:
        return self.block_size_kb * 1024

    @property
    def budget_bytes(self) -> int:
        return self.memory_budget_mib * 1024 * 1024

    @property
    def capacity_blocks(self) -> int:
        return self.budget_bytes // self.block_size_bytes


@dataclass(frozen=True)
class Sample:
    sample_id: str
    values: tuple[float, ...]


@dataclass(frozen=True)
class SampleSet:
    format_version: int
    seed: int
    dim: int
    samples: tuple[Sample, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "seed": self.seed,
            "dim": self.dim,
            "count": len(self.samples),
            "samples": [
                {"id": sample.sample_id, "values": list(sample.values)}
                for sample in self.samples
            ],
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.to_dict())

    @property
    def sample_map(self) -> dict[str, Sample]:
        return {sample.sample_id: sample for sample in self.samples}

    @property
    def ordered_ids(self) -> tuple[str, ...]:
        return tuple(sample.sample_id for sample in self.samples)

    @property
    def sorted_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.sample_map))


@dataclass
class LayerResult:
    layer_id: str
    version: int
    spec: dict[str, Any]
    output_digest: str
    float_digest: str
    sample_output_digests: dict[str, str]
    sample_float_digests: dict[str, str]
    block_ids: tuple[int, ...]
    allocation_mode: str
    allocation_runs: tuple[tuple[int, int], ...]
    request_bytes: int
    error_bound: float
    max_abs_error: float
    error_components: dict[str, float]
    dominant_segment: str

    @property
    def precision(self) -> str:
        return str(self.spec["precision"])

    @property
    def scale(self) -> int:
        return int(self.spec["scale"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "version": self.version,
            "spec": dict(self.spec),
            "output_digest": self.output_digest,
            "float_digest": self.float_digest,
            "sample_output_digests": {
                key: self.sample_output_digests[key]
                for key in sorted(self.sample_output_digests)
            },
            "sample_float_digests": {
                key: self.sample_float_digests[key]
                for key in sorted(self.sample_float_digests)
            },
            "block_ids": list(self.block_ids),
            "allocation_mode": self.allocation_mode,
            "allocation_runs": [list(run) for run in self.allocation_runs],
            "request_bytes": self.request_bytes,
            "error_bound": self.error_bound,
            "max_abs_error": self.max_abs_error,
            "error_components": dict(self.error_components),
            "dominant_segment": self.dominant_segment,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LayerResult":
        return cls(
            layer_id=str(payload["layer_id"]),
            version=int(payload["version"]),
            spec=dict(payload["spec"]),
            output_digest=str(payload["output_digest"]),
            float_digest=str(payload["float_digest"]),
            sample_output_digests={
                str(key): str(value)
                for key, value in payload["sample_output_digests"].items()
            },
            sample_float_digests={
                str(key): str(value)
                for key, value in payload.get("sample_float_digests", {}).items()
            },
            block_ids=tuple(int(value) for value in payload["block_ids"]),
            allocation_mode=str(payload["allocation_mode"]),
            allocation_runs=tuple(
                (int(run[0]), int(run[1])) for run in payload["allocation_runs"]
            ),
            request_bytes=int(payload["request_bytes"]),
            error_bound=float(payload["error_bound"]),
            max_abs_error=float(payload["max_abs_error"]),
            error_components={
                str(key): float(value)
                for key, value in payload["error_components"].items()
            },
            dominant_segment=str(payload["dominant_segment"]),
        )


@dataclass(frozen=True)
class InvalidationStep:
    layer_id: str
    trigger_source: str
    via: str | None
    depth: int
    reason: str
    path: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "trigger_source": self.trigger_source,
            "via": self.via,
            "depth": self.depth,
            "reason": self.reason,
            "path": list(self.path),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "InvalidationStep":
        return cls(
            layer_id=str(payload["layer_id"]),
            trigger_source=str(payload["trigger_source"]),
            via=None if payload.get("via") is None else str(payload["via"]),
            depth=int(payload["depth"]),
            reason=str(payload["reason"]),
            path=tuple(str(item) for item in payload["path"]),
        )


@dataclass(frozen=True)
class LayerBatchCommit:
    batch_id: str
    sample_ids: tuple[str, ...]
    block_ids: tuple[int, ...]
    request_bytes: int
    peak_blocks: int
    fragmentation_blocks: int
    output_digest: str
    float_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "sample_ids": list(self.sample_ids),
            "block_ids": list(self.block_ids),
            "request_bytes": self.request_bytes,
            "peak_blocks": self.peak_blocks,
            "fragmentation_blocks": self.fragmentation_blocks,
            "output_digest": self.output_digest,
            "float_digest": self.float_digest,
        }


@dataclass(frozen=True)
class BatchRecord:
    batch_id: str
    sample_ids: tuple[str, ...]
    layer_ids: tuple[str, ...]
    block_ids: tuple[int, ...]
    request_bytes: int
    peak_blocks: int
    fragmentation_blocks: int
    layer_output_digests: Mapping[str, str]
    layer_float_digests: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "sample_ids": list(self.sample_ids),
            "layer_ids": list(self.layer_ids),
            "block_ids": list(self.block_ids),
            "request_bytes": self.request_bytes,
            "peak_blocks": self.peak_blocks,
            "fragmentation_blocks": self.fragmentation_blocks,
            "layer_output_digests": {
                key: self.layer_output_digests[key]
                for key in sorted(self.layer_output_digests)
            },
            "layer_float_digests": {
                key: self.layer_float_digests[key]
                for key in sorted(self.layer_float_digests)
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BatchRecord":
        return cls(
            batch_id=str(payload["batch_id"]),
            sample_ids=tuple(str(item) for item in payload["sample_ids"]),
            layer_ids=tuple(str(item) for item in payload["layer_ids"]),
            block_ids=tuple(int(item) for item in payload["block_ids"]),
            request_bytes=int(payload["request_bytes"]),
            peak_blocks=int(payload["peak_blocks"]),
            fragmentation_blocks=int(payload["fragmentation_blocks"]),
            layer_output_digests={
                str(key): str(value)
                for key, value in payload["layer_output_digests"].items()
            },
            layer_float_digests={
                str(key): str(value)
                for key, value in payload.get("layer_float_digests", {}).items()
            },
        )


@dataclass
class DiagnosticRun:
    plan: Plan
    samples: SampleSet
    layers: dict[str, LayerResult]
    pool_capacity_blocks: int
    allocations: dict[str, tuple[int, ...]]
    allocation_modes: dict[str, str]
    allocation_runs: dict[str, tuple[tuple[int, int], ...]]
    batches: tuple[BatchRecord, ...]
    invalidation: tuple[InvalidationStep, ...]
    attribution: dict[str, Any]
    content_digest: str = ""
    actual_values: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    float_values: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

    @property
    def topo_ids(self) -> tuple[str, ...]:
        return tuple(self.layers)

    def result_dict(self, *, include_samples: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format_version": 3,
            "plan_version": self.plan.plan_version,
            "plan_digest": self.plan.digest,
            "samples_digest": self.samples.digest,
            "content_digest": self.content_digest,
            "plan": self.plan.to_dict(),
            "layers": [self.layers[layer_id].to_dict() for layer_id in self.layers],
            "pool": {
                "capacity_blocks": self.pool_capacity_blocks,
                "allocations": {
                    layer_id: list(self.allocations[layer_id])
                    for layer_id in sorted(self.allocations)
                },
                "allocation_modes": {
                    layer_id: self.allocation_modes[layer_id]
                    for layer_id in sorted(self.allocation_modes)
                },
                "allocation_runs": {
                    layer_id: [list(run) for run in self.allocation_runs[layer_id]]
                    for layer_id in sorted(self.allocation_runs)
                },
            },
            "batches": [batch.to_dict() for batch in self.batches],
            "invalidation": [step.to_dict() for step in self.invalidation],
            "attribution": self.attribution,
        }
        if include_samples:
            payload["samples"] = self.samples.to_dict()
        return payload

    @property
    def memory_summary(self) -> dict[str, int | float]:
        occupied = sum(len(block_ids) for block_ids in self.allocations.values())
        free = max(0, self.pool_capacity_blocks - occupied)
        free_set = set(range(self.pool_capacity_blocks)) - {
            block for blocks in self.allocations.values() for block in blocks
        }
        largest = 0
        current = 0
        for block in range(self.pool_capacity_blocks):
            if block in free_set:
                current += 1
                largest = max(largest, current)
            else:
                current = 0
        fragmentation = free - largest
        peak = max((batch.peak_blocks for batch in self.batches), default=occupied)
        return {
            "capacity_blocks": self.pool_capacity_blocks,
            "occupied_blocks": occupied,
            "free_blocks": free,
            "largest_contiguous_free_blocks": largest,
            "fragmentation_blocks": fragmentation,
            "peak_blocks": peak,
            "block_size_kb": self.plan.block_size_kb,
            "budget_mib": self.plan.memory_budget_mib,
        }

    def to_snapshot_dict(self) -> dict[str, Any]:
        payload = self.result_dict(include_samples=True)
        payload["snapshot_format_version"] = 1
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, plan: Plan, samples: SampleSet) -> "DiagnosticRun":
        layers = {
            str(item["layer_id"]): LayerResult.from_dict(item)
            for item in payload["layers"]
        }
        ordered = {
            layer_id: layers[layer_id]
            for layer_id in sorted(layers, key=lambda item: _topo_index(payload, item))
        }
        pool = payload.get("pool", {})
        allocations = {
            str(key): tuple(int(value) for value in values)
            for key, values in pool.get("allocations", {}).items()
        }
        modes = {str(key): str(value) for key, value in pool.get("allocation_modes", {}).items()}
        runs = {
            str(key): tuple((int(pair[0]), int(pair[1])) for pair in values)
            for key, values in pool.get("allocation_runs", {}).items()
        }
        return cls(
            plan=plan,
            samples=samples,
            layers=ordered,
            pool_capacity_blocks=int(pool.get("capacity_blocks", plan.capacity_blocks)),
            allocations=allocations,
            allocation_modes=modes,
            allocation_runs=runs,
            batches=tuple(BatchRecord.from_dict(item) for item in payload.get("batches", [])),
            invalidation=tuple(
                InvalidationStep.from_dict(item)
                for item in payload.get("invalidation", [])
            ),
            attribution=dict(payload.get("attribution", {})),
            content_digest=str(payload.get("content_digest", "")),
        )


def _topo_index(payload: Mapping[str, Any], layer_id: str) -> int:
    for index, item in enumerate(payload.get("layers", [])):
        if str(item.get("layer_id")) == layer_id:
            return index
    return 10**9


def batch_sample_groups(samples: SampleSet, max_batch_size: int) -> tuple[tuple[str, ...], ...]:
    ids = list(samples.ordered_ids)
    return tuple(
        tuple(ids[start : start + max_batch_size])
        for start in range(0, len(ids), max_batch_size)
    )


def digest_sample_subset(
    sample_digests: Mapping[str, str], sample_ids: Iterable[str]
) -> str:
    return canonical_digest(
        [[sample_id, sample_digests[sample_id]] for sample_id in sorted(sample_ids)]
    )
