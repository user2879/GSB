"""Requirement 3: deterministic integer fixed-point forward and loss.

Everything here is a pure function of (shard seed, layer dimensions, parameter
version, sample ids). Results are independent of dispatch order, dictionary
order and concurrency because:

* RNG streams are derived from explicit integer seeds via NumPy
  ``SeedSequence`` / ``Generator`` with explicit integer dtypes;
* parameters and samples are stored in sorted containers before digesting;
* all arithmetic is integer fixed-point (Q14) in int64 with explicit overflow
  guards, never floating point.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .digest import canonical, digest
from .spec import LayerSpec, ShardSpec

FRAC_BITS = 14
SCALE_Q = 1 << FRAC_BITS
INT64_MAX = np.iinfo(np.int64).max
INT64_MIN = np.iinfo(np.int64).min


def _rng(*parts: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(list(parts)))


def _check_int64(values: np.ndarray, where: str) -> np.ndarray:
    if np.any(values > INT64_MAX) or np.any(values < INT64_MIN):
        raise OverflowError(f"int64 overflow while computing {where}")
    return values.astype(np.int64)


def generate_layer_params(shard: ShardSpec, param_version: int) -> dict[str, np.ndarray]:
    """Deterministic Q14 weight matrices keyed by sorted layer_id."""
    params: dict[str, np.ndarray] = {}
    for layer in shard.layers:
        rng = _rng(shard.seed, param_version, layer.dim_in, layer.dim_out, layer.scale)
        raw = rng.integers(-layer.scale, layer.scale + 1,
                           size=(layer.dim_out, layer.dim_in), dtype=np.int64,
                           endpoint=True)
        params[layer.layer_id] = raw
    return params


def generate_samples(shard: ShardSpec, sample_ids: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Deterministic int16-range Q14 input vectors keyed by sorted sample_id."""
    samples: dict[str, np.ndarray] = {}
    for sample_id in sorted(sample_ids):
        seed_part = int(sample_id.split("-")[-1]) if sample_id.rsplit("-", 1)[-1].isdigit() else 0
        rng = _rng(shard.seed, seed_part, shard.input_dim)
        samples[sample_id] = rng.integers(
            -8, 9, size=(shard.input_dim,), dtype=np.int64, endpoint=True
        )
    return samples


def _target(shard: ShardSpec, sample_id: str, layer: LayerSpec) -> np.ndarray:
    seed_part = int(sample_id.split("-")[-1]) if sample_id.rsplit("-", 1)[-1].isdigit() else 0
    rng = _rng(shard.seed, 777, seed_part, layer.dim_out)
    return rng.integers(-4, 5, size=(layer.dim_out,), dtype=np.int64, endpoint=True)


def _topo_order(layers: tuple[LayerSpec, ...]) -> list[str]:
    order: list[str] = []
    remaining = {layer.layer_id: layer for layer in layers}
    emitted: set[str] = set()
    while remaining:
        for layer_id in sorted(remaining):
            parent = remaining[layer_id].parent
            if parent is None or parent in emitted:
                order.append(layer_id)
                emitted.add(layer_id)
                del remaining[layer_id]
                break
        else:  # pragma: no cover - cycle is rejected at spec load time
            raise RuntimeError("unexpected layer cycle during topo sort")
    return order


@dataclass(frozen=True)
class ComputeResult:
    shard_id: str
    batch_id: str
    param_version: int
    sample_ids: tuple[str, ...]
    param_summary: str
    loss_summary: str
    total_loss: int
    per_sample_loss: dict[str, int]
    scope_layers: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "shard_id": self.shard_id,
            "batch_id": self.batch_id,
            "param_version": self.param_version,
            "sample_ids": list(self.sample_ids),
            "scope_layers": list(self.scope_layers),
            "param_summary": self.param_summary,
            "loss_summary": self.loss_summary,
            "total_loss": self.total_loss,
            "per_sample_loss": dict(sorted(self.per_sample_loss.items())),
        }


def _scope_closure(layers: tuple[LayerSpec, ...], scope: tuple[str, ...]) -> tuple[str, ...]:
    """Layers that must run for the scope: scope roots, their ancestors (input
    dependencies) and every layer derived (transitively) from them."""
    by_id = {layer.layer_id: layer for layer in layers}
    by_parent: dict[str, list[str]] = {}
    for layer in layers:
        if layer.parent is not None:
            by_parent.setdefault(layer.parent, []).append(layer.layer_id)
    chosen: set[str] = set(scope)
    # ancestor inputs required to materialize each scoped layer's activations
    for root in scope:
        current = by_id[root].parent
        while current is not None and current not in chosen:
            chosen.add(current)
            current = by_id[current].parent
    # transitive derived layers
    stack = list(scope)
    while stack:
        current = stack.pop()
        for child in by_parent.get(current, []):
            if child not in chosen:
                chosen.add(child)
                stack.append(child)
    return tuple(sorted(chosen))


def parameter_summary(shard: ShardSpec, param_version: int,
                      scope_layers: tuple[str, ...] = ()) -> tuple[str, tuple[str, ...]]:
    params = generate_layer_params(shard, param_version)
    chosen = _scope_closure(shard.layers, scope_layers) if scope_layers \
        else tuple(sorted(params))
    canonical_params = {
        layer_id: params[layer_id].tolist() for layer_id in chosen
    }
    record = {
        "shard_id": shard.shard_id,
        "param_version": param_version,
        "frac_bits": FRAC_BITS,
        "layers": canonical_params,
    }
    return digest(record), chosen


def compute_batch(shard: ShardSpec, batch_id: str, param_version: int,
                  sample_ids: tuple[str, ...],
                  scope_layers: tuple[str, ...] = ()) -> ComputeResult:
    params = generate_layer_params(shard, param_version)
    samples = generate_samples(shard, sorted(sample_ids))
    chosen_layers = _scope_closure(shard.layers, scope_layers) if scope_layers \
        else tuple(_topo_order(shard.layers))
    order = [layer_id for layer_id in _topo_order(shard.layers)
             if layer_id in chosen_layers]
    by_id = {layer.layer_id: layer for layer in shard.layers}

    per_sample_loss: dict[str, int] = {}
    per_sample_sq: dict[str, int] = {}
    for sample_id in sorted(samples):
        vector = samples[sample_id]
        activations: dict[str, np.ndarray] = {}
        for layer_id in order:
            layer = by_id[layer_id]
            source = vector if layer.parent is None else activations[layer.parent]
            if source.shape[0] != layer.dim_in:
                raise OverflowError(
                    f"layer {layer_id} dim_in {layer.dim_in} != source width "
                    f"{source.shape[0]}")
            accum = params[layer_id] @ source
            q = _check_int64(accum // SCALE_Q, f"layer {layer_id} matmul")
            activated = np.maximum(q, np.int64(0))
            activations[layer_id] = activated
        terminal = order[-1]
        target = _target(shard, sample_id, by_id[terminal])
        diff = _check_int64(activations[terminal] - target, f"batch {batch_id} diff")
        sq = int(np.sum(diff * diff, dtype=np.int64))
        per_sample_loss[sample_id] = sq
        per_sample_sq[sample_id] = sq

    total_loss = sum(per_sample_loss.values())
    param_sum, chosen = parameter_summary(shard, param_version, scope_layers)
    loss_record = {
        "shard_id": shard.shard_id,
        "batch_id": batch_id,
        "param_version": param_version,
        "sample_ids": sorted(sample_ids),
        "scope_layers": list(chosen),
        "total_loss": total_loss,
        "per_sample_loss": {k: per_sample_loss[k] for k in sorted(per_sample_loss)},
        "frac_bits": FRAC_BITS,
    }
    return ComputeResult(
        shard_id=shard.shard_id,
        batch_id=batch_id,
        param_version=param_version,
        sample_ids=tuple(sorted(sample_ids)),
        param_summary=param_sum,
        loss_summary=digest(loss_record),
        total_loss=total_loss,
        per_sample_loss=per_sample_sq,
        scope_layers=tuple(chosen),
    )


def canonical_loss_record(result: ComputeResult) -> str:
    """Expose the canonical bytes so callers can verify byte-level stability."""
    record = {
        "shard_id": result.shard_id,
        "batch_id": result.batch_id,
        "param_version": result.param_version,
        "sample_ids": list(result.sample_ids),
        "scope_layers": list(result.scope_layers),
        "total_loss": result.total_loss,
        "per_sample_loss": dict(sorted(result.per_sample_loss.items())),
        "frac_bits": FRAC_BITS,
    }
    return canonical(record)
