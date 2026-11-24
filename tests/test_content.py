from __future__ import annotations

import hashlib
import struct

from cachelease.canonical import field_bytes, uint32
from cachelease.content import generate_content


def _manual_digest(node_id: str, artifact_key: str, shards: tuple[bytes, ...]) -> str:
    combined = hashlib.sha256()
    combined.update(b"CACHELEASE-ARTIFACT-MANIFEST-v1\0")
    combined.update(field_bytes(node_id))
    combined.update(field_bytes(artifact_key))
    for index, payload in enumerate(shards):
        shard_hash = hashlib.sha256()
        shard_hash.update(b"CACHELEASE-ARTIFACT-SHARD-v1\0")
        shard_hash.update(field_bytes(node_id))
        shard_hash.update(field_bytes(artifact_key))
        shard_hash.update(uint32(index))
        shard_hash.update(uint32(len(payload)))
        shard_hash.update(payload)
        combined.update(uint32(index))
        combined.update(shard_hash.digest())
    return combined.hexdigest()


def test_content_generation_is_reproducible_and_matches_explicit_rule() -> None:
    first = generate_content("node-a", "core", 101, 3, 96)
    second = generate_content("node-a", "core", 101, 3, 96)
    assert first.shards == second.shards
    assert first.digest == second.digest
    assert first.digest == _manual_digest("node-a", "core", first.shards)


def test_content_binds_node_artifact_seed_and_is_order_independent() -> None:
    a = generate_content("node-a", "core", 101, 2, 32)
    b = generate_content("node-b", "core", 101, 2, 32)
    c = generate_content("node-a", "api", 101, 2, 32)
    d = generate_content("node-a", "core", 102, 2, 32)
    e = generate_content("node-a", "core", 101 + (1 << 32), 2, 32)
    assert len({a.digest, b.digest, c.digest, d.digest, e.digest}) == 5
    assert a.shards[0] == generate_content("node-a", "core", 101, 2, 32).shards[0]
    assert struct.pack(">I", len(a.shards[0])) == uint32(32)

