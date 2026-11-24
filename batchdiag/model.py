"""Deterministic NumPy forward pass and digest binding for a batch.

The local "model" is a reproducible two-layer MLP whose weights are derived
from the worker seed and parameter version, so no network service or
external artifact is involved. Output digests bind together the request id
set, the per-request forward results, the parameter version and the
parameter digest; request processing order cannot change the digest.
"""

from dataclasses import dataclass

import numpy as np

from .canonical import canonical_float, short_digest

OUTPUT_DIGEST_BASIS = (
    "sha256:canonical{"
    "batch_id, requests=sorted(request_id), "
    "request_outputs=map(request_id->digest(forward(x))), "
    "param_version, param_digest}"
)


def _rng_seed(worker):
    return int(worker.seed) * 1_000_003 + int(worker.param_version)


def build_parameters(worker):
    rng = np.random.default_rng(_rng_seed(worker))
    params = {
        "W1": rng.standard_normal((worker.model_dim, worker.hidden_dim)),
        "b1": rng.standard_normal(worker.hidden_dim),
        "W2": rng.standard_normal((worker.hidden_dim, worker.model_dim)),
        "b2": rng.standard_normal(worker.model_dim),
    }
    return params


def parameter_descriptor(worker, params):
    return {
        "worker_id": worker.worker_id,
        "seed": worker.seed,
        "model_dim": worker.model_dim,
        "hidden_dim": worker.hidden_dim,
        "batch_size": worker.batch_size,
        "param_version": worker.param_version,
        "weights": {
            name: [[canonical_float(v) for v in row]
                   for row in np.asarray(tensor).tolist()]
            if tensor.ndim == 2
            else [canonical_float(v) for v in tensor.tolist()]
            for name, tensor in params.items()
        },
    }


def param_digest(worker, params=None):
    if params is None:
        params = build_parameters(worker)
    return short_digest(parameter_descriptor(worker, params), 32)


def forward(vector, params):
    x = np.asarray(vector, dtype=np.float64)
    hidden = np.tanh(x @ params["W1"] + params["b1"])
    output = hidden @ params["W2"] + params["b2"]
    return [canonical_float(v) for v in output.tolist()]


@dataclass(frozen=True)
class BatchResult:
    batch_id: str
    worker_id: str
    param_version: int
    param_digest: str
    rank_order: tuple
    sorted_requests: tuple
    per_request_digest: dict
    output_digest: str
    basis: str


def run_batch(worker, batch, params=None):
    """Run the forward pass for every request in *batch* and bind digests."""
    if params is None:
        params = build_parameters(worker)
    descriptor_digest = param_digest(worker, params)

    per_request = {}
    for request_id in sorted(batch.requests):
        vector = batch.vectors[request_id]
        y = forward(vector, params)
        per_request[request_id] = short_digest(
            {"request_id": request_id, "output": y}, 24)

    payload = {
        "batch_id": batch.batch_id,
        "requests": sorted(batch.requests),
        "request_outputs": {k: per_request[k]
                            for k in sorted(per_request)},
        "param_version": worker.param_version,
        "param_digest": descriptor_digest,
    }
    return BatchResult(
        batch_id=batch.batch_id,
        worker_id=worker.worker_id,
        param_version=worker.param_version,
        param_digest=descriptor_digest,
        rank_order=tuple(batch.requests),
        sorted_requests=tuple(sorted(batch.requests)),
        per_request_digest=per_request,
        output_digest=short_digest(payload, 32),
        basis=OUTPUT_DIGEST_BASIS,
    )
