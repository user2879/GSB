"""Deterministic integer fixed-point forward computation (NumPy).

All state derives from explicit integer seeds and parameter version numbers,
so any worker or recomputation run reproduces byte-identical matrices and
output digests.  The scale factor :data:`SCALE` is a shared power-of-ten
watermark used uniformly by leases, receipts and invalidation.
"""

import hashlib

import numpy as np

from .digest import canonical_json, digest

SCALE = 1000


def _seeded_rng(*parts):
    material = canonical_json(["rng-seed", *parts])
    seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return np.random.default_rng(seed)


def _bounded(rng, shape, low, high):
    return rng.integers(low=low, high=high, size=shape, dtype=np.int64)


def input_vector(request_id, model_dim, seed):
    """Deterministic ``model_dim`` integer input for one request."""
    rng = _seeded_rng("input", seed, request_id)
    return _bounded(rng, (model_dim,), 0, 100)


def parameter_matrices(worker_id, model_dim, hidden_dim, param_version, seed):
    """Reproducible ``(W1, b1, W2, b2)`` integer parameter set."""
    rng = _seeded_rng("param", seed, worker_id, param_version)
    w1 = _bounded(rng, (model_dim, hidden_dim), -8, 9)
    b1 = _bounded(rng, (hidden_dim,), -4, 5)
    w2 = _bounded(rng, (hidden_dim, model_dim), -8, 9)
    b2 = _bounded(rng, (model_dim,), -4, 5)
    return w1, b1, w2, b2


def parameter_digest(worker_id, model_dim, hidden_dim, param_version, seed):
    w1, b1, w2, b2 = parameter_matrices(
        worker_id, model_dim, hidden_dim, param_version, seed
    )
    return digest(
        {
            "kind": "param_snapshot",
            "worker_id": worker_id,
            "param_version": int(param_version),
            "seed": int(seed),
            "W1": w1.tolist(),
            "b1": b1.tolist(),
            "W2": w2.tolist(),
            "b2": b2.tolist(),
        }
    )


def _relu(values):
    return np.maximum(values, np.int64(0))


def forward_batch(request_ids, worker_id, model_dim, hidden_dim,
                  param_version, seed):
    """Run the integer fixed-point forward pass for a whole batch.

    Returns a dict whose ``outputs`` maps every request id to its integer
    output vector and whose ``output_digest`` is order-independent over the
    request set: requests are sorted before hashing, so reordering batch
    processing never changes the digest.
    """
    w1, b1, w2, b2 = parameter_matrices(
        worker_id, model_dim, hidden_dim, param_version, seed
    )
    rows = []
    ordered_ids = sorted(request_ids)
    for request_id in ordered_ids:
        x = input_vector(request_id, model_dim, seed)
        hidden = _relu((x @ w1) // SCALE + b1)
        out = (hidden @ w2) // SCALE + b2
        rows.append((request_id, [int(v) for v in out]))
    outputs = dict(rows)
    output_digest = digest(
        {
            "kind": "batch_output",
            "worker_id": worker_id,
            "param_version": int(param_version),
            "seed": int(seed),
            "scale": SCALE,
            "requests": ordered_ids,
            "outputs": [[request_id, outputs[request_id]]
                        for request_id in ordered_ids],
        }
    )
    return {
        "request_ids": ordered_ids,
        "outputs": outputs,
        "output_digest": output_digest,
    }
