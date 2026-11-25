"""Pure, deterministic shard generation and fixed-point verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .model import ContentSpec, Manifest, NodeSpec

_DOMAIN = b"edgecache-shard-v1\x00"
_MAX_SCALE = 1_000_000


@dataclass(frozen=True)
class ShardResult:
    data: bytes
    digest: str
    checksum: int
    sample_count: int


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def manifest_digest(manifest: Manifest) -> str:
    payload = canonical_json(manifest.to_dict()).encode("utf-8")
    return hashlib.sha256(b"edgecache-manifest-v1\x00" + payload).hexdigest()


def _derive_seed(node: NodeSpec, content: ContentSpec, shard_index: int) -> int:
    header = canonical_json(
        {
            "content_key": content.content_key,
            "content_seed": content.content_seed,
            "node_id": node.node_id,
            "node_seed": node.seed,
            "shard_index": shard_index,
        }
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(_DOMAIN + b"seed\x00" + header).digest()[:8], "little")


def _quantize(raw: np.ndarray, scale: int) -> np.ndarray:
    if not 1 <= scale <= _MAX_SCALE:
        raise ValueError(f"sample_scale must be in [1, {_MAX_SCALE}]")
    return (raw.astype(np.int64, copy=False) * scale + 16384) // 32768


def _checksum(quantized: np.ndarray) -> int:
    weights = (np.arange(quantized.size, dtype=np.int64) % 17) - 8
    return int(np.sum(quantized * weights, dtype=np.int64))


def derive_shard(
    node: NodeSpec,
    content: ContentSpec,
    shard_index: int,
    sample_scale: int,
) -> ShardResult:
    """Derive bytes, digest and fixed-point checksum from declarations only.

    Values are sampled with NumPy PCG64 using a SHA-256-derived integer seed,
    then quantized by ``(raw * sample_scale + 16384) // 32768``.  The digest
    binds the node id, content key, shard index and the exact bytes emitted.
    """

    if not 0 <= shard_index < content.shard_count:
        raise ValueError("shard_index out of range")
    byte_count = content.shard_bytes
    sample_count = max(1, (byte_count + 3) // 4)
    seed = _derive_seed(node, content, shard_index)
    rng = np.random.Generator(np.random.PCG64(seed))
    raw = rng.integers(-32768, 32769, size=sample_count, dtype=np.int64)
    quantized = _quantize(raw, sample_scale)
    data = quantized.astype("<i4", copy=False).tobytes(order="C")[:byte_count]
    header = canonical_json(
        {
            "content_key": content.content_key,
            "node_id": node.node_id,
            "shard_index": shard_index,
        }
    ).encode("utf-8")
    digest = hashlib.sha256(_DOMAIN + b"digest\x00" + header + b"\x00" + data).hexdigest()
    return ShardResult(data=data, digest=digest, checksum=_checksum(quantized), sample_count=sample_count)


def digest_for(manifest: Manifest, node_id: str, content_key: str, shard_index: int) -> ShardResult:
    return derive_shard(
        manifest.node_map[node_id],
        manifest.content_map[content_key],
        shard_index,
        manifest.config.sample_scale,
    )
