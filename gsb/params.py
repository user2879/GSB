"""Deterministic integer fixed-point model execution."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Sequence

import numpy as np

from .canonical import canonical_bytes, canonical_json, sha256_hex
from .model import BatchSpec, LayerSpec, ShardSpec

WEIGHT_LOW = -512
WEIGHT_HIGH = 512
BIAS_LOW = -128
BIAS_HIGH = 128
SAMPLE_LOW = -1000
SAMPLE_HIGH = 1001


def _seed_entropy(*parts: Any) -> list[int]:
    material = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    return [int.from_bytes(digest[i : i + 4], "little") for i in range(0, 32, 4)]


def topological_layers(shard: ShardSpec) -> tuple[LayerSpec, ...]:
    by_id = {layer.layer_id: layer for layer in shard.layers}
    visited: set[str] = set()
    result: list[LayerSpec] = []

    def visit(layer: LayerSpec) -> None:
        if layer.layer_id in visited:
            return
        for dep in layer.depends_on:
            visit(by_id[dep])
        visited.add(layer.layer_id)
        result.append(layer)

    for layer in shard.layers:
        visit(layer)
    return tuple(result)


def sink_layers(shard: ShardSpec) -> tuple[LayerSpec, ...]:
    used = {dep for layer in shard.layers for dep in layer.depends_on}
    return tuple(layer for layer in shard.layers if layer.layer_id not in used)


@dataclass(frozen=True)
class LayerParameters:
    layer: LayerSpec
    weights: np.ndarray
    bias: np.ndarray
    digest: str

    def summary(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer.layer_id,
            "input_dim": self.layer.input_dim,
            "output_dim": self.layer.output_dim,
            "scale": self.layer.scale,
            "depends_on": list(self.layer.depends_on),
            "weight_range": [WEIGHT_LOW, WEIGHT_HIGH - 1],
            "bias_range": [BIAS_LOW, BIAS_HIGH - 1],
            "digest": self.digest,
        }


@dataclass(frozen=True)
class ParameterSet:
    shard_id: str
    parameter_version: int
    layers: dict[str, LayerParameters]
    order: tuple[LayerSpec, ...]
    digest: str

    def summary(self) -> dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "parameter_version": self.parameter_version,
            "digest": self.digest,
            "layers": [self.layers[layer.layer_id].summary() for layer in self.order],
        }


def _array_digest(array: np.ndarray) -> str:
    little = np.ascontiguousarray(array, dtype="<i8")
    payload = canonical_json(
        {"dtype": "int64-le", "shape": list(little.shape)}
    ).encode("utf-8") + b"\0" + little.tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def generate_layer_parameters(
    shard: ShardSpec, layer: LayerSpec, parameter_version: int, position: int
) -> LayerParameters:
    entropy = _seed_entropy(
        shard.seed,
        shard.shard_id,
        parameter_version,
        position,
        layer.layer_id,
        layer.input_dim,
        layer.output_dim,
        layer.scale,
        ",".join(layer.depends_on),
    )
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    weights = rng.integers(
        WEIGHT_LOW,
        WEIGHT_HIGH,
        size=(layer.input_dim, layer.output_dim),
        dtype=np.int64,
    )
    bias = rng.integers(
        BIAS_LOW, BIAS_HIGH, size=(layer.output_dim,), dtype=np.int64
    )
    metadata = {
        "shard_id": shard.shard_id,
        "parameter_version": parameter_version,
        "layer": layer.to_dict(),
        "weights": _array_digest(weights),
        "bias": _array_digest(bias),
    }
    digest = sha256_hex(metadata)
    weights.flags.writeable = False
    bias.flags.writeable = False
    return LayerParameters(layer=layer, weights=weights, bias=bias, digest=digest)


@lru_cache(maxsize=128)
def _parameter_set_cached(
    shard_id: str,
    seed: int,
    parameter_version: int,
    sample_dim: int,
    layers_tuple: tuple[tuple[str, int, int, int, tuple[str, ...]], ...],
) -> ParameterSet:
    layers = tuple(
        LayerSpec(layer_id, input_dim, output_dim, scale, deps)
        for layer_id, input_dim, output_dim, scale, deps in layers_tuple
    )
    shard = ShardSpec(
        shard_id=shard_id,
        host="127.0.0.1",
        port=1,
        lease_limit=0,
        concurrency_limit=0,
        seed=seed,
        parameter_version=parameter_version,
        sample_dim=sample_dim,
        lease_ttl_ms=1,
        layers=layers,
    )
    order = topological_layers(shard)
    generated: dict[str, LayerParameters] = {}
    for position, layer in enumerate(shard.layers):
        generated[layer.layer_id] = generate_layer_parameters(
            shard, layer, parameter_version, position
        )
    digest_payload = {
        "shard_id": shard_id,
        "seed": seed,
        "parameter_version": parameter_version,
        "sample_dim": sample_dim,
        "layers": [generated[layer.layer_id].summary() for layer in order],
    }
    return ParameterSet(
        shard_id=shard_id,
        parameter_version=parameter_version,
        layers=generated,
        order=order,
        digest=sha256_hex(digest_payload),
    )


def get_parameter_set(shard: ShardSpec, parameter_version: int) -> ParameterSet:
    layers_tuple = tuple(
        (layer.layer_id, layer.input_dim, layer.output_dim, layer.scale, layer.depends_on)
        for layer in shard.layers
    )
    return _parameter_set_cached(
        shard.shard_id,
        shard.seed,
        parameter_version,
        shard.sample_dim,
        layers_tuple,
    )


def generate_sample(shard: ShardSpec, sample_id: int) -> tuple[np.ndarray, np.ndarray]:
    entropy = _seed_entropy(shard.seed, shard.shard_id, "sample", sample_id)
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    inputs = rng.integers(
        SAMPLE_LOW, SAMPLE_HIGH, size=(shard.sample_dim,), dtype=np.int64
    )
    target_dim = sum(layer.output_dim for layer in sink_layers(shard))
    targets = rng.integers(
        SAMPLE_LOW, SAMPLE_HIGH, size=(target_dim,), dtype=np.int64
    )
    return inputs, targets


def _layer_input(
    layer: LayerSpec, inputs: np.ndarray, outputs: dict[str, np.ndarray]
) -> np.ndarray:
    if not layer.depends_on:
        return inputs
    parts = [outputs[dep] for dep in layer.depends_on]
    return np.concatenate(parts).astype(object, copy=False)


def forward_sample(
    shard: ShardSpec,
    parameters: ParameterSet,
    sample_id: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    inputs, targets = generate_sample(shard, sample_id)
    outputs: dict[str, np.ndarray] = {}
    for layer in parameters.order:
        layer_input = _layer_input(layer, inputs, outputs)
        layer_parameters = parameters.layers[layer.layer_id]
        raw = np.matmul(
            layer_input.astype(object, copy=False),
            layer_parameters.weights.astype(object, copy=False),
        )
        raw = raw + layer_parameters.bias.astype(object, copy=False)
        outputs[layer.layer_id] = np.floor_divide(raw, layer.scale)
    prediction = np.concatenate(
        [outputs[layer.layer_id] for layer in sink_layers(shard)]
    ).astype(object, copy=False)
    return prediction, targets.astype(object, copy=False), outputs


def _sample_digest(
    shard: ShardSpec,
    sample_id: int,
    parameter_version: int,
    parameter_digest: str,
    prediction: np.ndarray,
    targets: np.ndarray,
) -> str:
    return sha256_hex(
        {
            "shard_id": shard.shard_id,
            "sample_id": sample_id,
            "parameter_version": parameter_version,
            "parameter_digest": parameter_digest,
            "prediction": [int(value) for value in prediction],
            "target": [int(value) for value in targets],
        }
    )


def compute_batch_summary(
    shard: ShardSpec,
    batch_id: str,
    sample_ids: Sequence[int],
    parameter_version: int,
) -> dict[str, Any]:
    """Compute an order-independent loss summary for one batch."""

    parameters = get_parameter_set(shard, parameter_version)
    ordered_ids = sorted(int(sample_id) for sample_id in sample_ids)
    sample_digests: dict[str, str] = {}
    total_loss = np.array(0, dtype=object)
    for sample_id in ordered_ids:
        prediction, targets, _ = forward_sample(shard, parameters, sample_id)
        difference = prediction - targets
        sample_loss = np.sum(np.square(difference, dtype=object), dtype=object)
        total_loss = total_loss + sample_loss
        sample_digests[str(sample_id)] = _sample_digest(
            shard,
            sample_id,
            parameter_version,
            parameters.digest,
            prediction,
            targets,
        )
    loss_value = int(total_loss)
    payload = {
        "shard_id": shard.shard_id,
        "batch_id": batch_id,
        "sample_ids": ordered_ids,
        "parameter_version": parameter_version,
        "parameter_digest": parameters.digest,
        "loss": loss_value,
        "sample_digests": sample_digests,
    }
    return {
        **payload,
        "loss_digest": sha256_hex(payload),
    }


@dataclass(frozen=True)
class ErrorBound:
    per_layer: dict[str, int]
    sink_errors: dict[str, int]
    magnitude_bounds: dict[str, int]
    loss_bound: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "per_layer": self.per_layer,
            "sink_errors": self.sink_errors,
            "magnitude_bounds": self.magnitude_bounds,
            "loss_bound": self.loss_bound,
        }


def quantization_error_bound(
    shard: ShardSpec, sample_ids: Sequence[int], parameter_version: int
) -> ErrorBound:
    """Conservative integer bound propagated through the declared layer DAG."""

    parameters = get_parameter_set(shard, parameter_version)
    layer_errors: dict[str, int] = {}
    magnitude: dict[str, int] = {}
    for layer in parameters.order:
        params = parameters.layers[layer.layer_id]
        abs_weights = np.abs(params.weights.astype(object))
        output_abs_weights = np.sum(abs_weights, axis=0, dtype=object)
        max_abs_weight = int(np.max(output_abs_weights))
        max_abs_bias = int(np.max(np.abs(params.bias.astype(object))))
        weighted_error = 0
        magnitude_raw = max_abs_bias
        for dep in layer.depends_on:
            dep_params = parameters.layers[dep]
            dep_abs_weights = np.abs(dep_params.weights.astype(object))
            dep_max = int(np.max(np.sum(dep_abs_weights, axis=0, dtype=object)))
            weighted_error += layer_errors[dep] * dep_max
            magnitude_raw += magnitude[dep] * dep_max
        if not layer.depends_on:
            magnitude_raw += SAMPLE_HIGH * max_abs_weight
        if not layer.depends_on:
            magnitude_raw += SAMPLE_HIGH * max_abs_weight
        layer_errors[layer.layer_id] = weighted_error // layer.scale + 1
        magnitude[layer.layer_id] = magnitude_raw // layer.scale + 1

    sinks = sink_layers(shard)
    cardinality = len(set(int(sample_id) for sample_id in sample_ids))
    loss_bound = 0
    for layer in sinks:
        error = layer_errors[layer.layer_id]
        bound = magnitude[layer.layer_id]
        loss_bound += cardinality * layer.output_dim * (2 * bound * error + error * error)
    return ErrorBound(
        per_layer=layer_errors,
        sink_errors={layer.layer_id: layer_errors[layer.layer_id] for layer in sinks},
        magnitude_bounds={layer.layer_id: magnitude[layer.layer_id] for layer in sinks},
        loss_bound=int(loss_bound),
    )


def batch_error_bound(
    shard: ShardSpec, batch: BatchSpec, parameter_version: int
) -> int:
    return quantization_error_bound(shard, batch.sample_ids, parameter_version).loss_bound


def parameter_summary_bytes(parameters: ParameterSet) -> bytes:
    return canonical_bytes(parameters.summary())
