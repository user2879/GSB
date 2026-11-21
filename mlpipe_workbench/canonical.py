"""Canonical byte encodings and the documented fingerprint recipe.

Recipe (WBP-FP-1)
-----------------

1. Dataset and pipeline objects are serialised with sorted object keys, UTF-8,
   compact separators, and sequence order preserved. Sequence order is part of
   the contract for sources, columns, shards, and the topologically compiled
   stage chain.
2. ``dataset_digest = SHA256(b"MLPIPE-DATASET1\\0" || canonical_dataset)``.
3. ``pipeline_digest = SHA256(b"MLPIPE-PIPELINE1\\0" || canonical_pipeline)``.
4. The workspace fingerprint uses length-prefixed frames:

   ``SHA256(b"MLPIPE-SPEC1\\0" || frame("dataset", dataset_digest)
   || frame("pipeline", pipeline_digest))``.

   A frame is the 4-byte big-endian name length, name bytes, 8-byte big-endian
   value length, and value bytes. This removes ambiguity when embedding hashes
   in a composite value and is independent of dict/hash iteration order.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_bytes(prefix: bytes, value: Any) -> str:
    return sha256_hex(prefix + canonical_json(value).encode("utf-8"))


def dataset_digest(canonical_dataset: dict[str, Any]) -> str:
    return digest_bytes(b"MLPIPE-DATASET1\0", canonical_dataset)


def pipeline_digest(canonical_pipeline: dict[str, Any]) -> str:
    return digest_bytes(b"MLPIPE-PIPELINE1\0", canonical_pipeline)


def frame(name: str, value: str) -> bytes:
    name_bytes = name.encode("utf-8")
    value_bytes = value.encode("ascii")
    return (
        len(name_bytes).to_bytes(4, "big")
        + name_bytes
        + len(value_bytes).to_bytes(8, "big")
        + value_bytes
    )


def spec_fingerprint(ds_digest: str, pipe_digest: str) -> str:
    return sha256_hex(
        b"MLPIPE-SPEC1\0"
        + frame("dataset", ds_digest)
        + frame("pipeline", pipe_digest)
    )


def stage_config_digest(stage: dict[str, Any]) -> str:
    return digest_bytes(b"MLPIPE-STAGE-CONFIG1\0", stage)


def table_row_digest(
    columns: Sequence[dict[str, Any]], start: int, end: int
) -> str:
    """Digest a row interval without row-position or spec metadata.

    Each column mapping contains ``key``, ``type``, and ``values``. Values have
    already been made canonical by the caller (integer strings or None).
    """

    payload = {
        "columns": [
            {
                "key": col["key"],
                "type": col["type"],
                "values": list(col["values"][start:end]),
            }
            for col in columns
        ],
        "row_count": end - start,
    }
    return digest_bytes(b"MLPIPE-TABLE1\0", payload)


def shard_digest(
    stage_id: str,
    stage_version: int,
    start_row: int,
    end_row: int,
    rows_digest: str,
) -> str:
    return digest_bytes(
        b"MLPIPE-SHARD1\0",
        {
            "stage_id": stage_id,
            "stage_version": stage_version,
            "start_row": start_row,
            "end_row": end_row,
            "row_count": end_row - start_row,
            "rows_digest": rows_digest,
        },
    )


def framed_values(prefix: bytes, values: Iterable[bytes]) -> str:
    h = hashlib.sha256(prefix)
    for value in values:
        h.update(len(value).to_bytes(8, "big"))
        h.update(value)
    return h.hexdigest()
