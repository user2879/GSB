"""Deterministic local construction of parameters, inputs and output digests.

Everything is generated from integer seeds with explicit modular rules instead
of version-specific RNG APIs, so identical declarations produce bit-identical
digests across runs, platforms and Python versions. No real model files or
network services are involved.
"""

import hashlib
import json

import numpy as np

QUANT_VERSION = 1
W_SCALE = 1000
B_SCALE = 100
MOD = 9973
HIDDEN_ACTIVATION = "tanh"


def _seeded_value(seed, tag, row, col):
    key = f"{seed}|{tag}|{row}|{col}".encode("utf-8")
    return int(hashlib.sha256(key).hexdigest(), 16) % MOD


def build_parameters(seed, model_dim, hidden_dim):
    """Build a two-layer MLP as quantized integer arrays.

    Returns integer numpy arrays; the same (seed, dims) tuple always yields
    bit-identical contents.
    """
    w1 = np.empty((hidden_dim, model_dim), dtype=np.int64)
    b1 = np.empty(hidden_dim, dtype=np.int64)
    w2 = np.empty((model_dim, hidden_dim), dtype=np.int64)
    b2 = np.empty(model_dim, dtype=np.int64)
    for r in range(hidden_dim):
        for c in range(model_dim):
            w1[r, c] = _seeded_value(seed, "w1", r, c) - MOD // 2
    for r in range(model_dim):
        for c in range(hidden_dim):
            w2[r, c] = _seeded_value(seed, "w2", r, c) - MOD // 2
    for r in range(hidden_dim):
        b1[r] = _seeded_value(seed, "b1", r, 0) % 100
    for r in range(model_dim):
        b2[r] = _seeded_value(seed, "b2", r, 0) % 100
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def array_digest(array):
    return hashlib.sha256(array.astype("<i8").tobytes()).hexdigest()


def parameter_digest(backend_id, model_version, quant_version, model_dim,
                     hidden_dim, seed, params):
    body = {
        "backend_id": backend_id,
        "model_version": model_version,
        "quant_version": quant_version,
        "model_dim": model_dim,
        "hidden_dim": hidden_dim,
        "seed": seed,
        "w1": array_digest(params["w1"]),
        "b1": array_digest(params["b1"]),
        "w2": array_digest(params["w2"]),
        "b2": array_digest(params["b2"]),
    }
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def next_hotswap_seed(seed, seq):
    """Explicit rule deriving the parameter seed of hot-swap version ``seq``."""
    return (seed * 1000003 + 7919 * seq) & 0x7FFFFFFF


def build_input_vector(rule_id, sample_id, model_dim):
    """Deterministic integer input vector scaled into [-1, 1) floats."""
    vec = np.empty(model_dim, dtype=np.float64)
    for c in range(model_dim):
        key = f"{rule_id}|{sample_id}|{c}".encode("utf-8")
        vec[c] = (int(hashlib.sha256(key).hexdigest(), 16) % 2000 - 1000) / 1000.0
    return vec


def input_digest(rule_id, sample_id, model_dim, vector):
    raw = np.round(vector * 1000).astype(np.int64)
    body = {
        "rule_id": rule_id,
        "sample_id": sample_id,
        "model_dim": model_dim,
        "vector_sha256": hashlib.sha256(raw.astype("<i8").tobytes()).hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def forward(params, inputs):
    """Integer-quantized two-layer forward pass over a NumPy batch.

    ``inputs`` is a (batch, model_dim) float64 array. The computation is a
    pure function of the quantized parameters and the integer-scaled inputs.
    """
    x = np.round(inputs * 1000).astype(np.int64)
    hidden = np.tanh((x @ params["w1"].T) / W_SCALE + params["b1"] / B_SCALE)
    hidden_q = np.round(hidden * W_SCALE).astype(np.int64)
    out = (hidden_q @ params["w2"].T) / W_SCALE + params["b2"] / B_SCALE
    return out


def output_digest(rule_id, sample_ids, param_digest, outputs):
    """Digest of the deterministic parts of a batch reply."""
    scaled = np.round(outputs * 1000).astype(np.int64)
    body = {
        "rule_id": rule_id,
        "sample_ids": list(sample_ids),
        "param_digest": param_digest,
        "output_sha256": hashlib.sha256(
            scaled.astype("<i8").tobytes()
        ).hexdigest(),
        "shape": list(scaled.shape),
    }
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
