from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .canonical import canonical_digest, sha256_hex
from .models import Batch, RequestSpec, WorkerSpec


FIXED_POINT_SCALE = 4
FIXED_POINT_MOD = 1_000_003


def parameter_matrices(worker: WorkerSpec) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(int(worker.seed))
    first = rng.randint(
        -3,
        4,
        size=(worker.model_dim, worker.hidden_dim),
        dtype=np.int64,
    )
    second = rng.randint(
        -3,
        4,
        size=(worker.hidden_dim, worker.model_dim),
        dtype=np.int64,
    )
    return first, second


def param_digest(worker: WorkerSpec) -> str:
    first, second = parameter_matrices(worker)
    material = {
        "worker_id": worker.worker_id,
        "seed": worker.seed,
        "model_dim": worker.model_dim,
        "hidden_dim": worker.hidden_dim,
        "param_version": worker.param_version,
        "fixed_point_scale": FIXED_POINT_SCALE,
        "first_matrix": first.tolist(),
        "second_matrix": second.tolist(),
    }
    return canonical_digest(material)


def _input_matrix(batch: Batch, request_lookup: dict[str, RequestSpec]) -> np.ndarray:
    rows = []
    for request_id in batch.request_ids:
        request = request_lookup[request_id]
        rows.append([int(value) for value in request.vector])
    return np.asarray(rows, dtype=np.int64)


def forward_batch(
    worker: WorkerSpec,
    batch: Batch,
    request_lookup: dict[str, RequestSpec],
    *,
    elapsed_ms: int = 0,
) -> dict[str, Any]:
    """Run integer fixed-point forward computation.

    Rows are independent; the digest is built after sorting per-request
    outputs by request ID, so caller iteration order cannot alter it.
    """

    inputs = _input_matrix(batch, request_lookup)
    first, second = parameter_matrices(worker)
    hidden = np.maximum(inputs @ first, 0)
    hidden = (hidden // FIXED_POINT_SCALE) % FIXED_POINT_MOD
    outputs = ((hidden @ second) // FIXED_POINT_SCALE) % FIXED_POINT_MOD

    per_request = []
    for index, request_id in enumerate(batch.request_ids):
        per_request.append(
            {
                "request_id": request_id,
                "arrival_seq": batch.arrival_seqs[index],
                "rank": batch.stable_ranks[request_id],
                "output_vector": [int(value) for value in outputs[index].tolist()],
            }
        )
    per_request.sort(key=lambda item: item["request_id"])
    p_digest = param_digest(worker)
    binding = {
        "batch_id": batch.batch_id,
        "worker_id": worker.worker_id,
        "param_version": worker.param_version,
        "param_digest": p_digest,
        "request_ids": sorted(batch.request_ids),
        "per_request": per_request,
    }
    output_digest = canonical_digest(binding)
    return {
        "batch_id": batch.batch_id,
        "worker_id": worker.worker_id,
        "param_version": worker.param_version,
        "param_digest": p_digest,
        "output_digest": output_digest,
        "request_ids": list(batch.request_ids),
        "per_request": per_request,
        "elapsed_ms": int(elapsed_ms),
        "algorithm": "numpy-int-fixed-point-v1",
    }


def digest_text(text: str | bytes) -> str:
    data = text.encode("utf-8") if isinstance(text, str) else text
    return sha256_hex(data)


def deterministic_vectors(seed: int, count: int, dim: int) -> list[tuple[int, ...]]:
    rng = np.random.RandomState(int(seed))
    matrix = rng.randint(-9, 10, size=(count, dim), dtype=np.int64)
    return [tuple(int(value) for value in row) for row in matrix.tolist()]
