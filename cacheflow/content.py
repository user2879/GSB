from __future__ import annotations

import hashlib

from .spec import ArtifactSpec, NodeSpec, ShardSpec


CONTENT_DOMAIN = "cacheflow.content.v1"
DIGEST_DOMAIN = "cacheflow.shard-digest.v1"


def _length_prefixed(value: str | int | bytes) -> bytes:
    if isinstance(value, int):
        raw = str(value).encode("ascii")
    else:
        raw = value.encode("utf-8") if isinstance(value, str) else value
    return len(raw).to_bytes(8, "big", signed=False) + raw


def deterministic_shard_bytes(
    node_seed: str,
    node_id: str,
    artifact_key: str,
    shard_index: int,
    length: int,
) -> bytes:
    """Derive reproducible bytes with SHAKE-256.

    The preimage is explicit and versioned:
    domain, node seed, node id, artifact key, shard index, byte length.
    """
    if length < 0:
        raise ValueError("length must be non-negative")
    preimage = b"".join(
        [
            _length_prefixed(CONTENT_DOMAIN),
            _length_prefixed(node_seed),
            _length_prefixed(node_id),
            _length_prefixed(artifact_key),
            _length_prefixed(int(shard_index)),
            _length_prefixed(int(length)),
        ]
    )
    return hashlib.shake_256(preimage).digest(length)


def shard_digest(node_id: str, artifact_key: str, shard_index: int, content: bytes) -> str:
    """SHA-256 over domain and length-prefixed bound fields/content."""
    digest = hashlib.sha256()
    digest.update(_length_prefixed(DIGEST_DOMAIN))
    digest.update(_length_prefixed(node_id))
    digest.update(_length_prefixed(artifact_key))
    digest.update(_length_prefixed(int(shard_index)))
    digest.update(_length_prefixed(len(content)))
    digest.update(_length_prefixed(content))
    return digest.hexdigest()


def expected_shard(
    node: NodeSpec, artifact_key: str, shard: ShardSpec
) -> tuple[bytes, str]:
    content = deterministic_shard_bytes(
        node.seed,
        node.node_id,
        artifact_key,
        shard.index,
        shard.length,
    )
    return content, shard_digest(node.node_id, artifact_key, shard.index, content)


def artifact_digests(node: NodeSpec, artifact: ArtifactSpec) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for shard in artifact.shards:
        content, digest = expected_shard(node, artifact.artifact_key, shard)
        rows.append(
            {
                "shard_index": shard.index,
                "start": shard.start,
                "end": shard.end,
                "length": shard.length,
                "digest": digest,
                "content_sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    return rows
