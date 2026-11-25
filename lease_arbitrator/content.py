from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from . import CONTENT_SOURCE_VERSION


DIGEST_DOMAIN = "cross-node-lease-arbitrator/shard-digest/v1"


def _length_prefixed(value: str | int) -> bytes:
    raw = str(value).encode("utf-8")
    return len(raw).to_bytes(8, "big", signed=False) + raw


def deterministic_shard_bytes(
    *,
    node_seed: int,
    node_id: str,
    artifact_key: str,
    shard_index: int,
    byte_count: int,
) -> bytes:
    """Return reproducible bytes for a node/key/shard tuple.

    NumPy's PCG64 generator is seeded by a stable SeedSequence.  The key is
    first reduced with SHA-256 so Python's per-process hash seed cannot affect
    ordering or output.
    """
    if byte_count < 0:
        raise ValueError("byte_count must be non-negative")
    key_digest = hashlib.sha256(artifact_key.encode("utf-8")).digest()
    key_words = [
        int.from_bytes(key_digest[offset : offset + 4], "big")
        for offset in range(0, 32, 4)
    ]
    entropy = [
        int(CONTENT_SOURCE_VERSION),
        int(node_seed) & 0xFFFFFFFF,
        int(shard_index) & 0xFFFFFFFF,
        byte_count & 0xFFFFFFFF,
        *key_words,
    ]
    sequence = np.random.SeedSequence([value & 0xFFFFFFFF for value in entropy])
    generator = np.random.Generator(np.random.PCG64(sequence))
    values = generator.integers(0, 256, size=int(byte_count), dtype=np.uint8)
    return values.tobytes()


def shard_digest(
    *,
    node_id: str,
    artifact_key: str,
    shard_index: int,
    content: bytes,
    source_version: int = CONTENT_SOURCE_VERSION,
) -> str:
    """Hash the explicit length-prefixed combination rule."""
    digest = hashlib.sha256()
    fields: tuple[str | int, ...] = (
        DIGEST_DOMAIN,
        source_version,
        node_id,
        artifact_key,
        shard_index,
        len(content),
    )
    for field in fields:
        digest.update(_length_prefixed(field))
    digest.update(b"raw-bytes:")
    digest.update(len(content).to_bytes(8, "big", signed=False))
    digest.update(content)
    return digest.hexdigest()


def build_shard(
    *,
    node_id: str,
    node_seed: int,
    artifact_key: str,
    shard_index: int,
    byte_count: int,
) -> dict[str, Any]:
    content = deterministic_shard_bytes(
        node_seed=node_seed,
        node_id=node_id,
        artifact_key=artifact_key,
        shard_index=shard_index,
        byte_count=byte_count,
    )
    return {
        "content": content,
        "digest": shard_digest(
            node_id=node_id,
            artifact_key=artifact_key,
            shard_index=shard_index,
            content=content,
        ),
        "shard_bytes": len(content),
        "source_version": CONTENT_SOURCE_VERSION,
    }


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def chain_digest(previous_digest: str, payload: Any) -> str:
    return hashlib.sha256(
        previous_digest.encode("ascii") + canonical_json_bytes(payload)
    ).hexdigest()
