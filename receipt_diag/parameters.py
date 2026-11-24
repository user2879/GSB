"""Deterministic worker parameters and forward-computation digests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .canonical import array_digest, digest_json, sha256_bytes
from .specs import WorkerSpec


PARAM_SCHEMA = "receipt-diag-params-v1"
OUTPUT_SCHEMA = "receipt-diag-output-v1"


@dataclass(frozen=True)
class WorkerParameters:
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray


def _seed_entropy(seed: int, param_version: int, model_dim: int, hidden_dim: int) -> list[int]:
    unsigned = seed & ((1 << 64) - 1)
    return [
        unsigned & 0xFFFFFFFF,
        (unsigned >> 32) & 0xFFFFFFFF,
        param_version & 0xFFFFFFFF,
        model_dim,
        hidden_dim,
    ]


def generate_parameters(
    *,
    seed: int,
    param_version: int,
    model_dim: int,
    hidden_dim: int,
) -> WorkerParameters:
    rng = np.random.default_rng(
        np.random.SeedSequence(_seed_entropy(seed, param_version, model_dim, hidden_dim))
    )

    def integers(shape: tuple[int, ...]) -> np.ndarray:
        return rng.integers(-100, 101, size=shape, dtype=np.int64).astype(np.float64) / 100.0

    return WorkerParameters(
        w1=integers((hidden_dim, model_dim)),
        b1=integers((hidden_dim,)),
        w2=integers((model_dim, hidden_dim)),
        b2=integers((model_dim,)),
    )


def parameters_for_worker(worker: WorkerSpec) -> WorkerParameters:
    return generate_parameters(
        seed=worker.seed,
        param_version=worker.param_version,
        model_dim=worker.model_dim,
        hidden_dim=worker.hidden_dim,
    )


def parameter_digest(worker: WorkerSpec, parameters: WorkerParameters) -> str:
    metadata = {
        "schema": PARAM_SCHEMA,
        "worker_id": worker.worker_id,
        "seed": worker.seed,
        "param_version": worker.param_version,
        "model_dim": worker.model_dim,
        "hidden_dim": worker.hidden_dim,
        "w1": array_digest(parameters.w1),
        "b1": array_digest(parameters.b1),
        "w2": array_digest(parameters.w2),
        "b2": array_digest(parameters.b2),
    }
    return digest_json(metadata)


def forward(parameters: WorkerParameters, inputs: np.ndarray) -> np.ndarray:
    hidden = np.tanh(inputs @ parameters.w1.T + parameters.b1)
    return hidden @ parameters.w2.T + parameters.b2


def vector_digest(request_id: str, vector: np.ndarray) -> str:
    return sha256_bytes(
        b"request-vector-v1\x00"
        + request_id.encode("utf-8")
        + b"\x00"
        + array_digest(vector).encode("ascii")
    )


def batch_output_digest(
    *,
    batch_id: str,
    request_ids: Iterable[str],
    param_version: int,
    param_digest: str,
    vector_digests: Sequence[tuple[str, str]],
) -> str:
    ordered = sorted((request_id, digest) for request_id, digest in vector_digests)
    return digest_json(
        {
            "schema": OUTPUT_SCHEMA,
            "batch_id": batch_id,
            "request_ids": sorted(request_ids),
            "param_version": param_version,
            "param_digest": param_digest,
            "vectors": [
                {"request_id": request_id, "vector_digest": digest}
                for request_id, digest in ordered
            ],
        }
    )


def parameter_summary(worker: WorkerSpec, param_digest: str) -> dict[str, object]:
    return {
        "worker_id": worker.worker_id,
        "seed": worker.seed,
        "param_version": worker.param_version,
        "model_dim": worker.model_dim,
        "hidden_dim": worker.hidden_dim,
        "param_digest": param_digest,
    }
