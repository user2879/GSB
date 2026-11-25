"""Deterministic NumPy parameter generation, forward passes, and digests."""

from __future__ import annotations

import hashlib
import struct
from functools import lru_cache

import numpy as np

from .jsonio import canonical_bytes, sha256_bytes
from .spec import WorkerSpec


PARAMETER_ALGORITHM = "numpy-pcg64-mlp-v1"
OUTPUT_ALGORITHM = "numpy-float64-scalar-v1"


@lru_cache(maxsize=128)
def _parameters_cached(
    seed: int, input_dim: int, hidden_dim: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(seed))
    weight_1 = rng.standard_normal((input_dim, hidden_dim), dtype=np.float64) * 0.25
    bias_1 = rng.standard_normal(hidden_dim, dtype=np.float64) * 0.05
    weight_2 = rng.standard_normal((hidden_dim, 1), dtype=np.float64) * 0.25
    bias_2 = rng.standard_normal(1, dtype=np.float64) * 0.05
    for array in (weight_1, bias_1, weight_2, bias_2):
        array.setflags(write=False)
    return weight_1, bias_1, weight_2, bias_2


def parameters(worker: WorkerSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _parameters_cached(worker.seed, worker.input_dim, worker.hidden_dim)


def parameter_digest(worker: WorkerSpec) -> str:
    header = {
        "algorithm": PARAMETER_ALGORITHM,
        "seed": worker.seed,
        "input_dim": worker.input_dim,
        "hidden_dim": worker.hidden_dim,
    }
    digest = hashlib.sha256(canonical_bytes(header))
    for name, array in zip(
        ("weight_1", "bias_1", "weight_2", "bias_2"), parameters(worker)
    ):
        stable = np.asarray(array, dtype="<f8", order="C")
        digest.update(name.encode("ascii"))
        digest.update(struct.pack("<I", stable.ndim))
        for extent in stable.shape:
            digest.update(struct.pack("<Q", extent))
        digest.update(stable.tobytes(order="C"))
    return digest.hexdigest()


def parameter_snapshot(worker: WorkerSpec) -> dict[str, object]:
    return {
        "algorithm": PARAMETER_ALGORITHM,
        "worker_id": worker.worker_id,
        "param_version_seq": worker.param_version_seq,
        "seed": worker.seed,
        "input_dim": worker.input_dim,
        "hidden_dim": worker.hidden_dim,
        "param_digest": parameter_digest(worker),
    }


def forward(worker: WorkerSpec, vector: tuple[float, ...] | list[float]) -> np.ndarray:
    weight_1, bias_1, weight_2, bias_2 = parameters(worker)
    x = np.asarray(vector, dtype=np.float64)
    hidden = np.tanh(x @ weight_1 + bias_1)
    return hidden @ weight_2 + bias_2


def output_digest(worker: WorkerSpec, vector: tuple[float, ...] | list[float]) -> str:
    output = np.asarray(forward(worker, vector), dtype="<f8", order="C")
    digest = hashlib.sha256()
    digest.update(OUTPUT_ALGORITHM.encode("ascii"))
    digest.update(output.tobytes(order="C"))
    return digest.hexdigest()


def batch_output_digest(
    request_outputs: dict[str, str],
    *,
    param_version_seq: int,
    param_digest: str,
) -> str:
    payload = {
        "binding": "request-set+param-version+param-digest+outputs",
        "request_ids": sorted(request_outputs),
        "param_version_seq": param_version_seq,
        "param_digest": param_digest,
        "outputs": [
            {"request_id": request_id, "output_digest": request_outputs[request_id]}
            for request_id in sorted(request_outputs)
        ],
    }
    return sha256_bytes(canonical_bytes(payload))
