"""Deterministic NumPy forward computation and output summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .canonical import sha256_hex
from .planning import Batch
from .spec import RequestSpec, SystemSpec, WorkerSpec


@dataclass(frozen=True)
class WorkerParameters:
    worker_id: str
    param_version: int
    input_dim: int
    hidden_dim: int
    param_digest: str
    weight: np.ndarray
    bias: np.ndarray

    def to_ledger_dict(self) -> dict:
        return {
            "worker_id": self.worker_id,
            "param_version": self.param_version,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "param_digest": self.param_digest,
        }


def build_parameters(worker: WorkerSpec) -> WorkerParameters:
    """Derive the parameter matrices deterministically from the worker seed."""

    rng = np.random.default_rng(worker.seed)
    weight = rng.standard_normal((worker.hidden_dim, worker.input_dim))
    bias = rng.standard_normal(worker.hidden_dim)
    param_digest = sha256_hex(
        {
            "worker_id": worker.worker_id,
            "seed": worker.seed,
            "param_version": worker.param_version,
            "input_dim": worker.input_dim,
            "hidden_dim": worker.hidden_dim,
            "weight_shape": list(weight.shape),
            "weight_first": float(weight.reshape(-1)[0]),
            "bias_first": float(bias.reshape(-1)[0]),
        }
    )
    return WorkerParameters(
        worker_id=worker.worker_id,
        param_version=worker.param_version,
        input_dim=worker.input_dim,
        hidden_dim=worker.hidden_dim,
        param_digest=param_digest,
        weight=weight,
        bias=bias,
    )


def _request_vector(request: RequestSpec, input_dim: int) -> np.ndarray:
    rng = np.random.default_rng(request.seed)
    return rng.standard_normal(input_dim)


@dataclass(frozen=True)
class ForwardResult:
    batch_id: str
    worker_id: str
    param_version: int
    param_digest: str
    intermediate_dim: int
    output_digest: str
    per_request: dict[str, str]
    rows: np.ndarray


def forward_batch(
    spec: SystemSpec,
    worker: WorkerSpec,
    batch: Batch,
    parameters: WorkerParameters | None = None,
) -> ForwardResult:
    """Run the deterministic forward pass for one normalized batch.

    The output digest binds the participating request-id set, parameter version
    and parameter digest. Request ids are sorted before hashing, so changing the
    in-batch processing order cannot change the summary.
    """

    params = parameters or build_parameters(worker)
    request_by_id = spec.request_by_id()
    ordered_ids = sorted(batch.request_ids)
    matrix = np.vstack(
        [_request_vector(request_by_id[request_id], worker.input_dim) for request_id in ordered_ids]
    )
    # hidden = relu(X @ W.T + b); deterministic two-layer projection.
    hidden = np.maximum(0.0, matrix @ params.weight.T + params.bias)
    rows = np.tanh(hidden)
    intermediate_dim = worker.hidden_dim

    per_request = {
        request_id: sha256_hex(
            {
                "request_id": request_id,
                "row": [float(value) for value in rows[index]],
            }
        )
        for index, request_id in enumerate(ordered_ids)
    }
    output_digest = sha256_hex(
        {
            "batch_id": batch.batch_id,
            "request_ids": ordered_ids,
            "param_version": worker.param_version,
            "param_digest": params.param_digest,
            "intermediate_dim": intermediate_dim,
            "row_digests": [per_request[request_id] for request_id in ordered_ids],
        }
    )
    return ForwardResult(
        batch_id=batch.batch_id,
        worker_id=worker.worker_id,
        param_version=worker.param_version,
        param_digest=params.param_digest,
        intermediate_dim=intermediate_dim,
        output_digest=output_digest,
        per_request=dict(sorted(per_request.items())),
        rows=rows,
    )
