"""Deterministic backend version fingerprints.

Fingerprint inputs (explicit combination rule, in order):
    model_id | quantized params (6 decimal places, fixed order) | param length
    | config summary (host:port:c<concurrency>:b<batch_size>) | load index

The load index is the rank of the backend in the canonical ordering sorted by
model_id, so identical declarations produce identical fingerprints regardless
of file order, dict iteration order or concurrency settings.
"""

from __future__ import annotations

import hashlib

from .models import Backend, Declaration


def quantize_params(params: tuple) -> str:
    """Quantize params to 6 decimal places; fixed textual representation."""
    return ",".join(f"{p:.6f}" for p in params)


def config_summary(backend: Backend) -> str:
    return (f"{backend.host}:{backend.port}"
            f":c{backend.max_concurrency}:b{backend.batch_size}")


def backend_fingerprint(backend: Backend, load_index: int) -> str:
    payload = "|".join([
        backend.model_id,
        quantize_params(backend.params),
        str(len(backend.params)),
        config_summary(backend),
        str(load_index),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def fingerprints(decl: Declaration) -> dict:
    """Map model_id -> fingerprint using canonical (sorted) load order."""
    result = {}
    for index, backend in enumerate(decl.backends):  # already sorted by id
        result[backend.model_id] = backend_fingerprint(backend, index)
    return result
