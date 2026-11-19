"""Canonical serialization and digest primitives.

The byte-level rules in this module are part of the public audit contract.  Do
not silently change them: if a rule changes, bump ``NUMBERING_VERSION`` so old
assignments are invalidated as a whole.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .constants import (
    ARRAY_DIGEST_DOMAIN,
    ASSIGNMENT_DIGEST_DOMAIN,
    BATCH_OUTPUT_DIGEST_DOMAIN,
    INPUT_DIGEST_DOMAIN,
    MODEL_DIGEST_DOMAIN,
    SAMPLE_OUTPUT_DIGEST_DOMAIN,
)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically, without NaN or platform-dependent floats."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_float_array(array: np.ndarray) -> np.ndarray:
    """Return a C-contiguous, little-endian float64 array.

    Negative zero is normalized to positive zero before hashing.  Forward
    inference also rounds outputs to 12 decimal places before canonicalization.
    """

    result = np.ascontiguousarray(array, dtype="<f8")
    if result.size:
        result = result.copy()
        result[result == 0.0] = 0.0
    return result


def array_payload(name: str, array: np.ndarray) -> bytes:
    data = canonical_float_array(array)
    header = canonical_json_bytes(
        {
            "dtype": "<f8",
            "name": name,
            "order": "C",
            "shape": list(data.shape),
        }
    )
    return header + b"\n" + data.tobytes(order="C")


def digest_arrays(domain: bytes, arrays: Iterable[tuple[str, np.ndarray]]) -> str:
    h = hashlib.sha256()
    h.update(domain)
    for name, array in arrays:
        h.update(array_payload(name, array))
        h.update(b"\n")
    return h.hexdigest()


def digest_inputs(sample_ids: Sequence[str], inputs: np.ndarray) -> str:
    ordered_ids = sorted(str(item) for item in sample_ids)
    index = {str(item): i for i, item in enumerate(sample_ids)}
    try:
        ordered_rows = np.asarray([inputs[index[item]] for item in ordered_ids], dtype="<f8")
    except KeyError as exc:  # defensive; caller checks cardinality first
        raise ValueError(f"sample id has no input row: {exc.args[0]}") from exc
    h = hashlib.sha256()
    h.update(INPUT_DIGEST_DOMAIN)
    h.update(canonical_json_bytes({"sample_ids": ordered_ids, "shape": list(inputs.shape)}))
    h.update(b"\n")
    h.update(array_payload("inputs", ordered_rows))
    h.update(b"\n")
    return h.hexdigest()


def digest_model(model: Mapping[str, np.ndarray], order: Sequence[str]) -> str:
    return digest_arrays(MODEL_DIGEST_DOMAIN, ((name, model[name]) for name in order))


def rounded_output_vector(values: Sequence[float] | np.ndarray) -> np.ndarray:
    rounded = np.round(np.asarray(values, dtype="<f8"), 12)
    return canonical_float_array(rounded)


def digest_sample_output(sample_id: str, values: np.ndarray) -> str:
    data = rounded_output_vector(values)
    h = hashlib.sha256()
    h.update(SAMPLE_OUTPUT_DIGEST_DOMAIN)
    h.update(
        canonical_json_bytes(
            {"sample_id": sample_id, "dtype": "<f8", "shape": list(data.shape)}
        )
    )
    h.update(b"\n")
    h.update(data.tobytes(order="C"))
    h.update(b"\n")
    return h.hexdigest()


def digest_batch_output(
    members: Sequence[tuple[str, str]],
) -> str:
    """Hash ``(sample_id, sample_output_digest)`` pairs in sample-id order."""

    pairs = sorted((str(sample_id), str(digest)) for sample_id, digest in members)
    h = hashlib.sha256()
    h.update(BATCH_OUTPUT_DIGEST_DOMAIN)
    h.update(canonical_json_bytes({"members": pairs}))
    h.update(b"\n")
    return h.hexdigest()


def assignment_payload(
    *,
    numbering_version: str,
    batch_slot: int,
    rank: int,
    sample_id: str,
    param_digest: str,
) -> bytes:
    return canonical_json_bytes(
        {
            "batch_slot": int(batch_slot),
            "model_param_digest": str(param_digest),
            "numbering_version": str(numbering_version),
            "rank": int(rank),
            "sample_id": str(sample_id),
        }
    )


def assignment_identifier(
    *,
    numbering_version: str,
    batch_slot: int,
    rank: int,
    sample_id: str,
    param_digest: str,
) -> tuple[str, str]:
    payload = assignment_payload(
        numbering_version=numbering_version,
        batch_slot=batch_slot,
        rank=rank,
        sample_id=sample_id,
        param_digest=param_digest,
    )
    digest = sha256_hex(ASSIGNMENT_DIGEST_DOMAIN + payload + b"\n")
    label = f"{numbering_version}:S{int(batch_slot)}:R{int(rank):06d}:{digest[:20]}"
    return label, digest
