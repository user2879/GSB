from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import numpy as np

from .canonical import canonical_dumps, sha256_hex
from .spec import ContentSpec, NodeSpec


SHARD_ID_RE = re.compile(r"^shard-(?P<seq>\d{4,})$")


def shard_id_for_seq(seq: int) -> str:
    return f"shard-{seq:04d}"


def parse_shard_seq(shard_id: str) -> int:
    match = SHARD_ID_RE.match(shard_id)
    if not match:
        raise ValueError(f"unparseable shard_id {shard_id!r}")
    return int(match.group("seq"))


def content_seed(node_seed: int, manifest_seed: int, content_key: str, shard_seq: int) -> int:
    """Derive a stable unsigned 32-bit seed without Python hash randomization."""

    digest = sha256_hex(f"edge-content/v1|{node_seed}|{manifest_seed}|{content_key}|{shard_seq}")
    return int(digest[:8], 16)


def generate_shard_bytes(
    node_seed: int,
    content_key: str,
    shard_seq: int,
    size: int,
    manifest_seed: int = 0,
) -> bytes:
    """Generate bitwise-stable bytes with NumPy's stable RandomState stream."""

    if size < 0:
        raise ValueError("size must be non-negative")
    seed = content_seed(node_seed, manifest_seed, content_key, shard_seq)
    rng = np.random.RandomState(seed)
    array = rng.randint(0, 256, size=int(size), dtype=np.uint8)
    return array.tobytes()


@dataclass(frozen=True)
class ShardProof:
    schema: str
    node_id: str
    content_key: str
    shard_id: str
    shard_seq: int
    byte_start: int
    byte_end: int
    sample_bytes: int
    sample_count: int
    integer_scale: int
    weighted_sum: int
    total_weight: int
    fixed_mean_q: int
    fixed_mean_remainder: int
    content_bytes_digest: str
    digest: str
    elapsed_ms: int | None = None
    exchange_format: str = "internal-v1"

    def to_canonical(self) -> dict:
        data = asdict(self)
        data.pop("elapsed_ms", None)
        data.pop("exchange_format", None)
        return data


def fixed_point_statistics(data: bytes, sample_stride: int, integer_scale: int) -> tuple[int, int, int, int, int]:
    if sample_stride <= 0:
        raise ValueError("sample_stride must be positive")
    values = np.frombuffer(data, dtype=np.uint8)
    samples = values[::sample_stride].astype(np.int64)
    count = int(samples.size)
    if count == 0:
        return 0, 0, 1, 0, 0
    positions = np.arange(count, dtype=np.int64)
    weights = (positions % np.int64(17)) + np.int64(1)
    products = samples * weights
    weighted_sum = int(products.sum(dtype=np.int64))
    total_weight = int(weights.sum(dtype=np.int64))
    scaled = weighted_sum * int(integer_scale)
    quotient = scaled // total_weight
    remainder = scaled - quotient * total_weight
    return weighted_sum, total_weight, quotient, remainder, count


def compute_shard_proof(
    node: NodeSpec | str,
    content: ContentSpec | str,
    shard_seq: int,
    *,
    node_seed: int | None = None,
    elapsed_ms: int | None = None,
    exchange_format: str = "internal-v1",
) -> ShardProof:
    if isinstance(node, str):
        node_id = node
        resolved_node_seed = 0 if node_seed is None else node_seed
    else:
        node_id = node.node_id
        resolved_node_seed = node.seed if node_seed is None else node_seed

    if isinstance(content, str):
        content_key = content
        manifest_seed = 0
        shard_size = None
        sample_stride = 1
        integer_scale = 1
        start = 0
        end = 0
    else:
        content_key = content.content_key
        manifest_seed = content.seed
        shard_size = content.shard_size
        sample_stride = content.sample_stride
        integer_scale = content.integer_scale
        start, end = content.byte_range(shard_seq)

    if shard_size is None:
        raise ValueError("ContentSpec is required to determine shard byte length")
    if not (0 <= shard_seq < content.shard_count):
        raise ValueError(f"shard_seq {shard_seq} outside manifest range")

    data = generate_shard_bytes(
        resolved_node_seed,
        content_key,
        shard_seq,
        shard_size,
        manifest_seed,
    )
    content_digest = sha256_hex(data)
    weighted_sum, total_weight, quotient, remainder, sample_count = fixed_point_statistics(
        data, sample_stride, integer_scale
    )
    shard_id = shard_id_for_seq(shard_seq)

    binding = {
        "schema": "edge-shard-proof/v1",
        "node_id": node_id,
        "node_seed": resolved_node_seed,
        "manifest_seed": manifest_seed,
        "content_key": content_key,
        "shard_id": shard_id,
        "shard_seq": shard_seq,
        "byte_start": start,
        "byte_end": end,
        "sample_bytes": shard_size,
        "sample_count": sample_count,
        "integer_scale": integer_scale,
        "weighted_sum": weighted_sum,
        "total_weight": total_weight,
        "fixed_mean_q": quotient,
        "fixed_mean_remainder": remainder,
        "content_bytes_digest": content_digest,
    }
    digest = sha256_hex(canonical_dumps(binding))
    return ShardProof(
        schema=binding["schema"],
        node_id=node_id,
        content_key=content_key,
        shard_id=shard_id,
        shard_seq=shard_seq,
        byte_start=start,
        byte_end=end,
        sample_bytes=shard_size,
        sample_count=sample_count,
        integer_scale=integer_scale,
        weighted_sum=weighted_sum,
        total_weight=total_weight,
        fixed_mean_q=quotient,
        fixed_mean_remainder=remainder,
        content_bytes_digest=content_digest,
        digest=digest,
        elapsed_ms=elapsed_ms,
        exchange_format=exchange_format,
    )


def verify_proof(prooflike: ShardProof | dict, node: NodeSpec, content: ContentSpec) -> bool:
    if isinstance(prooflike, ShardProof):
        expected = compute_shard_proof(node, content, prooflike.shard_seq)
        return prooflike.digest == expected.digest and prooflike.content_bytes_digest == expected.content_bytes_digest
    seq = int(prooflike["shard_seq"])
    expected = compute_shard_proof(node, content, seq)
    return prooflike.get("digest") == expected.digest
