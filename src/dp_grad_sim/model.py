from __future__ import annotations

import hashlib
import math
from typing import Iterable

import numpy as np

from .config import SimConfig
from .data import Shard, ShardSet


def _rng_from_parts(*parts: object) -> np.random.Generator:
    material = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    words = [int.from_bytes(digest[offset : offset + 8], "little") for offset in (0, 8, 16, 24)]
    return np.random.default_rng(np.random.SeedSequence(words))


def initialize_parameters(config: SimConfig) -> dict[str, np.ndarray]:
    parameters: dict[str, np.ndarray] = {}
    for tensor_index, tensor in enumerate(config.model.expanded_tensors()):
        rng = _rng_from_parts("parameter", config.seed, tensor["name"], tensor_index)
        shape = tuple(tensor["shape"])
        scale = 0.02 / max(1.0, math.sqrt(config.model.hidden_dim))
        parameters[tensor["name"]] = (rng.standard_normal(shape) * scale).astype(np.float32)
    return parameters


def _shard_context(shards: Iterable[Shard]) -> str:
    return "|".join(
        f"{shard.shard_id}:{shard.samples}:{shard.feature_digest}:{shard.label_digest}"
        for shard in shards
    )


def derive_worker_gradient(
    config: SimConfig,
    shards: tuple[Shard, ...],
    worker_id: int,
    step: int,
) -> dict[str, np.ndarray]:
    total_samples = sum(shard.samples for shard in shards)
    context = _shard_context(shards)
    gradients: dict[str, np.ndarray] = {}
    for tensor_index, tensor in enumerate(config.model.expanded_tensors()):
        name = tensor["name"]
        shape = tuple(tensor["shape"])
        accumulator = np.zeros(shape, dtype=np.float64)
        for shard in shards:
            rng = _rng_from_parts(
                "gradient",
                config.seed,
                step,
                worker_id,
                tensor_index,
                name,
                context,
                shard.shard_id,
                shard.feature_digest,
                shard.label_digest,
            )
            weight = shard.samples / total_samples
            accumulator += rng.standard_normal(shape) * weight
        magnitude = 0.10 / max(1.0, math.sqrt(config.model.hidden_dim))
        gradients[name] = (accumulator * magnitude).astype(np.float32)
    return gradients


def derive_window_gradients(
    config: SimConfig,
    shard_set: ShardSet,
    window_start: int,
    window_end: int,
) -> dict[int, dict[str, np.ndarray]]:
    result: dict[int, dict[str, np.ndarray]] = {}
    tensor_names = config.tensor_names()
    tensor_shapes = {
        tensor["name"]: tuple(tensor["shape"]) for tensor in config.model.expanded_tensors()
    }
    for worker_id in range(config.workers):
        accumulated = {
            name: np.zeros(tensor_shapes[name], dtype=np.float64)
            for name in tensor_names
        }
        for step in range(window_start, window_end):
            step_gradient = derive_worker_gradient(
                config, shard_set.by_worker[worker_id], worker_id, step
            )
            for name in tensor_names:
                accumulated[name] += step_gradient[name].astype(np.float64)
        result[worker_id] = {
            name: accumulated[name].astype(np.float32) for name in tensor_names
        }
    return result


def sanitize_nonfinite(
    arrays_by_worker: dict[int, dict[str, np.ndarray]],
) -> tuple[dict[int, dict[str, np.ndarray]], list[dict[str, object]]]:
    """Return finite copies plus E_PARAM_NONFINITE records keyed by worker/tensor."""

    cleaned: dict[int, dict[str, np.ndarray]] = {}
    failures: list[dict[str, object]] = []
    for worker_id in sorted(arrays_by_worker):
        cleaned[worker_id] = {}
        for tensor_name in sorted(arrays_by_worker[worker_id]):
            values = arrays_by_worker[worker_id][tensor_name]
            if not np.all(np.isfinite(values)):
                nonfinite_count = int(np.size(values) - np.count_nonzero(np.isfinite(values)))
                failures.append(
                    {
                        "code": "E_PARAM_NONFINITE",
                        "worker_id": worker_id,
                        "tensor": tensor_name,
                        "nonfinite_count": nonfinite_count,
                        "exception_type": "NonFiniteParameterError",
                        "message": f"non-finite values in worker {worker_id} tensor {tensor_name}",
                    }
                )
                cleaned[worker_id][tensor_name] = np.nan_to_num(
                    values, nan=0.0, posinf=0.0, neginf=0.0
                ).astype(np.float32)
            else:
                cleaned[worker_id][tensor_name] = values.astype(np.float32, copy=True)
    return cleaned, failures


def exact_aggregate(
    arrays_by_worker: dict[int, np.ndarray], mode: str
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    ordered_workers = sorted(arrays_by_worker)
    if not ordered_workers:
        raise ValueError("cannot aggregate an empty worker set")
    truth = np.zeros_like(arrays_by_worker[ordered_workers[0]], dtype=np.float64)
    for worker_id in ordered_workers:
        truth += arrays_by_worker[worker_id].astype(np.float64)
    if mode == "mean":
        truth /= len(ordered_workers)
    return truth.astype(np.float32), truth
