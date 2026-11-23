"""Deterministic local model tensors and sample matrices."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .schema import Plan
from .util import digest_value, sha256_bytes


def _seed64(*parts: object) -> int:
    material = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little", signed=False)


def sample_ids(count: int) -> tuple[str, ...]:
    width = max(4, len(str(max(0, count - 1))))
    return tuple(f"s{index:0{width}d}" for index in range(count))


@dataclass(frozen=True)
class BatchSpec:
    batch_id: str
    sample_ids: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.sample_ids)

    def to_dict(self) -> dict[str, object]:
        return {"batch_id": self.batch_id, "sample_ids": list(self.sample_ids)}


def make_batches(ids: Sequence[str], batch_size_max: int) -> tuple[BatchSpec, ...]:
    if batch_size_max <= 0:
        raise ValueError("batch_size_max must be positive")
    ordered = tuple(ids)
    return tuple(
        BatchSpec(f"b{index:04d}", ordered[start : start + batch_size_max])
        for index, start in enumerate(range(0, len(ordered), batch_size_max))
    )


class SampleFactory:
    """Pure generators keyed only by seed, layer metadata and sample identity."""

    def __init__(self, plan: Plan, count: int) -> None:
        self.plan = plan
        self.count = count
        self.ids = sample_ids(count)
        self._input_cache: dict[tuple[str, str], np.ndarray] = {}
        self._activation_input_cache: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}
        self._weight_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._lock = threading.RLock()

    def weights(self, layer_id: str) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            cached = self._weight_cache.get(layer_id)
            if cached is not None:
                return cached
            layer = self.plan.layer_by_id[layer_id]
            seed = _seed64(
                self.plan.seed,
                "weights",
                layer.id,
                layer.input_dim,
                layer.output_dim,
                layer.param_count,
            )
            rng = np.random.Generator(np.random.PCG64(seed))
            weights = rng.uniform(-1.0, 1.0, size=(layer.output_dim, layer.input_dim))
            bias = rng.uniform(-1.0, 1.0, size=(layer.output_dim,))
            result = (
                np.asarray(weights, dtype=np.float32, order="C"),
                np.asarray(bias, dtype=np.float32, order="C"),
            )
            self._weight_cache[layer_id] = result
            return result

    def root_input(self, layer_id: str, sample_id: str) -> np.ndarray:
        key = (layer_id, sample_id)
        with self._lock:
            cached = self._input_cache.get(key)
            if cached is not None:
                return cached
            layer = self.plan.layer_by_id[layer_id]
            seed = _seed64(
                self.plan.seed,
                "input",
                layer.id,
                layer.input_dim,
                sample_id,
            )
            rng = np.random.Generator(np.random.PCG64(seed))
            values = rng.uniform(-1.0, 1.0, size=(layer.input_dim,))
            result = np.asarray(values, dtype=np.float32, order="C")
            self._input_cache[key] = result
            return result

    def root_matrix(self, layer_id: str, ids: Iterable[str]) -> np.ndarray:
        ordered = tuple(ids)
        if not ordered:
            return np.empty((0, self.plan.layer_by_id[layer_id].input_dim), dtype=np.float32)
        rows = [self.root_input(layer_id, sample_id) for sample_id in ordered]
        return np.stack(rows, axis=0).astype(np.float32, copy=False, order="C")

    def input_digest(self, layer_id: str, ids: Iterable[str]) -> str:
        ordered = tuple(sorted(ids))
        payload = (
            layer_id,
            [(sample_id, self.root_input(layer_id, sample_id).tobytes(order="C")) for sample_id in ordered],
        )
        return sha256_bytes(repr(payload).encode("utf-8"))

    def weights_digest(self, layer_id: str) -> str:
        weights, bias = self.weights(layer_id)
        return sha256_bytes(weights.tobytes(order="C") + bias.tobytes(order="C"))

    def dataset_digest(self) -> str:
        roots = [layer.id for layer in self.plan.layers if not layer.depends_on]
        payload = {
            "seed": self.plan.seed,
            "count": self.count,
            "roots": {
                layer_id: self.input_digest(layer_id, self.ids) for layer_id in roots
            },
            "weights": {
                layer.id: self.weights_digest(layer.id) for layer in self.plan.layers
            },
        }
        return digest_value(payload)

    def summary(self) -> dict[str, object]:
        roots = [layer.id for layer in self.plan.layers if not layer.depends_on]
        return {
            "seed": self.plan.seed,
            "sample_count": self.count,
            "sample_ids": list(self.ids),
            "input_matrices": {
                layer_id: {
                    "shape": [self.count, self.plan.layer_by_id[layer_id].input_dim],
                    "digest": self.input_digest(layer_id, self.ids),
                }
                for layer_id in roots
            },
            "weight_matrices": {
                layer.id: {
                    "shape": [layer.output_dim, layer.input_dim],
                    "digest": self.weights_digest(layer.id),
                }
                for layer in self.plan.layers
            },
            "dataset_digest": self.dataset_digest(),
        }
