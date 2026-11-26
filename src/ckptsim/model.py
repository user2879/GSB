from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .canonical import canonical_bytes, sha256_value
from .errors import spec_error
from .spec import LayerSpec, ParameterVersion, ShardSpec


MAX_SAFE_DIMENSION = 4096
MAX_SAFE_SCALE = 10_000_000


def _seed_int(material: Any) -> int:
    return int.from_bytes(hashlib.sha256(canonical_bytes(material)).digest()[:16], "big")


def _rng_int(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def _input_dim(layer: LayerSpec, parents: list["LayerTensor"]) -> int:
    return sum(parent.dimension for parent in parents) if parents else layer.dimension


@dataclass
class LayerTensor:
    layer_id: str
    dimension: int
    scale_factor: int
    derived_from: tuple[str, ...]
    input_dimension: int
    weights: np.ndarray
    digest: str


@dataclass
class ParameterSet:
    shard_id: str
    parameter_version: int
    generation: int
    digest: str
    layers: dict[str, LayerTensor]
    order: tuple[str, ...]

    @property
    def root_layers(self) -> tuple[str, ...]:
        return tuple(
            layer_id
            for layer_id in self.order
            if not self.layers[layer_id].derived_from
        )

    @property
    def sink_layers(self) -> tuple[str, ...]:
        parents = {
            parent
            for layer in self.layers.values()
            for parent in layer.derived_from
        }
        return tuple(layer_id for layer_id in self.order if layer_id not in parents)


@dataclass
class LossReport:
    total: int
    per_sample: tuple[int, ...]
    sample_ids: tuple[int, ...]
    samples_digest: str
    payload: dict[str, Any]
    digest: str


@dataclass
class BatchComputation:
    shard_id: str
    batch_id: str
    parameter_version: int
    generation: int
    parameter_digest: str
    loss: LossReport
    duration_ms: int

    def to_response(self) -> dict[str, Any]:
        return {
            "ok": True,
            "shard_id": self.shard_id,
            "batch_id": self.batch_id,
            "parameter_version": self.parameter_version,
            "generation": self.generation,
            "parameter_digest": self.parameter_digest,
            "samples_digest": self.loss.samples_digest,
            "sample_ids": list(self.loss.sample_ids),
            "loss": self.loss.payload,
            "loss_digest": self.loss.digest,
            "duration_ms": self.duration_ms,
        }


def _check_numeric_bounds(shard: ShardSpec, version: ParameterVersion) -> None:
    for layer in version.layers:
        if layer.dimension > MAX_SAFE_DIMENSION:
            raise spec_error(
                f"layer dimension {layer.dimension} exceeds safe bound {MAX_SAFE_DIMENSION}",
                item=layer.layer_id,
                line=layer.line,
            )
        if layer.scale_factor > MAX_SAFE_SCALE:
            raise spec_error(
                f"layer scale_factor {layer.scale_factor} exceeds safe bound {MAX_SAFE_SCALE}",
                item=layer.layer_id,
                line=layer.line,
            )
        width = max(layer.dimension, sum(
            version.by_id()[parent].dimension for parent in layer.derived_from
        ) or layer.dimension)
        if width * layer.scale_factor * layer.scale_factor > 4_000_000_000_000_000_000:
            raise spec_error(
                f"layer {layer.layer_id!r} fixed-point bounds may overflow int64",
                item=shard.shard_id,
                line=layer.line,
            )


def generate_parameter_set(
    shard: ShardSpec, version: ParameterVersion, *, cache: dict[int, ParameterSet] | None = None
) -> ParameterSet:
    if cache is not None and version.parameter_version in cache:
        return cache[version.parameter_version]
    _check_numeric_bounds(shard, version)
    by_id = version.by_id()
    built: dict[str, LayerTensor] = {}
    order: list[str] = []
    for layer in version.topological_layers():
        parents = [built[parent] for parent in sorted(layer.derived_from)]
        input_dimension = _input_dim(layer, parents)
        parent_material = [
            [parent.layer_id, parent.dimension, parent.scale_factor, parent.digest]
            for parent in parents
        ]
        seed = _seed_int(
            {
                "kind": "layer-parameters",
                "shard_id": shard.shard_id,
                "shard_seed": shard.seed,
                "parameter_version": version.parameter_version,
                "layer_id": layer.layer_id,
                "dimension": layer.dimension,
                "scale_factor": layer.scale_factor,
                "derived_from": sorted(layer.derived_from),
                "parents": parent_material,
            }
        )
        rng = _rng_int(seed)
        half = max(1, layer.scale_factor // 2)
        weights = rng.integers(
            low=-half,
            high=half + 1,
            size=(input_dimension, layer.dimension),
            dtype=np.int64,
        )
        weights = np.ascontiguousarray(weights, dtype=np.int64)
        digest = hashlib.sha256()
        digest.update(
            canonical_bytes(
                {
                    "kind": "layer-tensor",
                    "shard_id": shard.shard_id,
                    "parameter_version": version.parameter_version,
                    "generation": version.generation,
                    "layer_id": layer.layer_id,
                    "dimension": layer.dimension,
                    "input_dimension": input_dimension,
                    "scale_factor": layer.scale_factor,
                    "derived_from": sorted(layer.derived_from),
                    "parent_digests": [
                        [parent.layer_id, parent.digest] for parent in parents
                    ],
                    "dtype": "<i8",
                    "shape": list(weights.shape),
                }
            )
        )
        digest.update(weights.astype("<i8", copy=False).tobytes(order="C"))
        built[layer.layer_id] = LayerTensor(
            layer_id=layer.layer_id,
            dimension=layer.dimension,
            scale_factor=layer.scale_factor,
            derived_from=tuple(sorted(layer.derived_from)),
            input_dimension=input_dimension,
            weights=weights,
            digest=digest.hexdigest(),
        )
        order.append(layer.layer_id)
    parameter_digest = sha256_value(
        {
            "kind": "parameter-set",
            "shard_id": shard.shard_id,
            "shard_seed": shard.seed,
            "parameter_version": version.parameter_version,
            "generation": version.generation,
            "order": order,
            "layers": [
                {
                    "layer_id": built[layer_id].layer_id,
                    "dimension": built[layer_id].dimension,
                    "scale_factor": built[layer_id].scale_factor,
                    "derived_from": list(built[layer_id].derived_from),
                    "digest": built[layer_id].digest,
                }
                for layer_id in order
            ],
        }
    )
    result = ParameterSet(
        shard_id=shard.shard_id,
        parameter_version=version.parameter_version,
        generation=version.generation,
        digest=parameter_digest,
        layers=built,
        order=tuple(order),
    )
    if cache is not None:
        cache[version.parameter_version] = result
    return result


def _sample_seed(
    shard: ShardSpec,
    batch_id: str,
    sample_id: int,
    layer_id: str,
    purpose: str,
) -> int:
    return _seed_int(
        {
            "kind": purpose,
            "shard_id": shard.shard_id,
            "shard_seed": shard.seed,
            "batch_id": batch_id,
            "sample_id": sample_id,
            "layer_id": layer_id,
        }
    )


def _generate_matrix(
    shard: ShardSpec,
    batch_id: str,
    sample_ids: Iterable[int],
    layer_id: str,
    dimension: int,
    scale_factor: int,
    purpose: str,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    half = max(1, scale_factor // 2)
    for sample_id in sorted(sample_ids):
        rng = _rng_int(_sample_seed(shard, batch_id, sample_id, layer_id, purpose))
        rows.append(
            rng.integers(
                low=-half,
                high=half + 1,
                size=(dimension,),
                dtype=np.int64,
            )
        )
    if not rows:
        return np.empty((0, dimension), dtype=np.int64)
    return np.ascontiguousarray(np.stack(rows, axis=0), dtype=np.int64)


def build_batch_samples(
    shard: ShardSpec,
    parameters: ParameterSet,
    batch_id: str,
    sample_ids: Iterable[int],
) -> dict[str, list[list[int]]]:
    ordered_ids = sorted(sample_ids)
    samples: dict[str, list[list[int]]] = {}
    for layer_id in parameters.root_layers:
        layer = parameters.layers[layer_id]
        matrix = _generate_matrix(
            shard,
            batch_id,
            ordered_ids,
            layer_id,
            layer.dimension,
            layer.scale_factor,
            "batch-sample",
        )
        samples[layer_id] = matrix.tolist()
    return samples


def samples_digest(samples: dict[str, list[list[int]]]) -> str:
    return sha256_value(
        {
            layer_id: [list(row) for row in samples[layer_id]]
            for layer_id in sorted(samples)
        }
    )


def _validated_samples(
    shard: ShardSpec,
    parameters: ParameterSet,
    batch_id: str,
    sample_ids: list[int],
    samples: dict[str, Any] | None,
) -> tuple[dict[str, np.ndarray], str]:
    ordered_ids = sorted(sample_ids)
    normalized: dict[str, list[list[int]]] = {}
    matrices: dict[str, np.ndarray] = {}
    roots = set(parameters.root_layers)
    if samples is None:
        normalized = build_batch_samples(shard, parameters, batch_id, ordered_ids)
    else:
        if not isinstance(samples, dict):
            raise ValueError("samples must be an object keyed by root layer id")
        for layer_id, raw_matrix in samples.items():
            if layer_id not in roots:
                raise ValueError(f"sample matrix {layer_id!r} is not a root layer")
            if not isinstance(raw_matrix, list):
                raise ValueError(f"sample matrix {layer_id!r} must be an array")
            matrix = np.asarray(raw_matrix, dtype=np.int64)
            if matrix.ndim != 2 or matrix.shape[0] != len(ordered_ids):
                raise ValueError(
                    f"sample matrix {layer_id!r} must have shape ({len(ordered_ids)}, dim)"
                )
            if matrix.shape[1] != parameters.layers[layer_id].dimension:
                raise ValueError(
                    f"sample matrix {layer_id!r} dimension does not match declaration"
                )
            normalized[layer_id] = matrix.tolist()
    missing = sorted(roots - set(normalized))
    if missing:
        generated = build_batch_samples(shard, parameters, batch_id, ordered_ids)
        for layer_id in missing:
            normalized[layer_id] = generated[layer_id]
    for layer_id, rows in normalized.items():
        matrices[layer_id] = np.ascontiguousarray(np.asarray(rows, dtype=np.int64))
    return matrices, samples_digest(normalized)


def compute_loss(
    shard: ShardSpec,
    parameters: ParameterSet,
    batch_id: str,
    sample_ids: Iterable[int],
    samples: dict[str, Any] | None = None,
) -> LossReport:
    ordered_ids = tuple(sorted(sample_ids))
    if not ordered_ids or len(set(ordered_ids)) != len(ordered_ids):
        raise ValueError("sample_ids must be a non-empty unique set")
    root_samples, sample_digest = _validated_samples(
        shard, parameters, batch_id, list(ordered_ids), samples
    )
    outputs: dict[str, np.ndarray] = {}
    for layer_id in parameters.order:
        layer = parameters.layers[layer_id]
        if layer.derived_from:
            pieces: list[np.ndarray] = []
            for parent_id in sorted(layer.derived_from):
                parent = parameters.layers[parent_id]
                converted = (
                    outputs[parent_id].astype(np.int64) * layer.scale_factor
                ) // parent.scale_factor
                pieces.append(converted)
            activation = np.concatenate(pieces, axis=1)
        else:
            activation = root_samples[layer_id]
        if activation.shape != (len(ordered_ids), layer.input_dimension):
            raise ValueError(f"activation shape mismatch for layer {layer_id}")
        output = (activation @ layer.weights) // layer.scale_factor
        outputs[layer_id] = np.ascontiguousarray(output, dtype=np.int64)
    per_sample = np.zeros(len(ordered_ids), dtype=object)
    for layer_id in parameters.sink_layers:
        layer = parameters.layers[layer_id]
        target = _generate_matrix(
            shard,
            batch_id,
            ordered_ids,
            layer_id,
            layer.dimension,
            layer.scale_factor,
            "loss-target",
        )
        diff = outputs[layer_id] - target
        squared = np.multiply(diff, diff, dtype=object)
        row_losses = np.sum(squared, axis=1, dtype=object) // layer.scale_factor
        per_sample = per_sample + row_losses
    per_sample_ints = tuple(int(value) for value in per_sample.tolist())
    total = int(sum(per_sample_ints))
    payload = {
        "shard_id": shard.shard_id,
        "batch_id": batch_id,
        "sample_ids": list(ordered_ids),
        "parameter_version": parameters.parameter_version,
        "parameter_digest": parameters.digest,
        "samples_digest": sample_digest,
        "loss_units": total,
        "per_sample_loss": list(per_sample_ints),
    }
    digest = sha256_value(payload)
    return LossReport(
        total=total,
        per_sample=per_sample_ints,
        sample_ids=ordered_ids,
        samples_digest=sample_digest,
        payload=payload,
        digest=digest,
    )


def compute_batch(
    shard: ShardSpec,
    version: ParameterVersion,
    batch_id: str,
    sample_ids: Iterable[int],
    *,
    samples: dict[str, Any] | None = None,
    cache: dict[int, ParameterSet] | None = None,
    duration_ms: int = 0,
) -> BatchComputation:
    parameters = generate_parameter_set(shard, version, cache=cache)
    loss = compute_loss(shard, parameters, batch_id, sample_ids, samples=samples)
    return BatchComputation(
        shard_id=shard.shard_id,
        batch_id=batch_id,
        parameter_version=version.parameter_version,
        generation=version.generation,
        parameter_digest=parameters.digest,
        loss=loss,
        duration_ms=duration_ms,
    )
