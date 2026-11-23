from __future__ import annotations

import hashlib
import struct

import numpy as np


LEAF_SIZE = 64 * 1024
CONTENT_BLOCK_SIZE = LEAF_SIZE
_TAG = b"OSSDIAG-DETERMINISTIC-CONTENT-v1\x00"


def _block_entropy(object_key: str, shard_index: int, seed: int, block_index: int) -> np.random.Generator:
    key_bytes = object_key.encode("utf-8")
    seed_bytes = str(seed).encode("ascii", "strict")
    digest = hashlib.blake2b(digest_size=16)
    digest.update(_TAG)
    digest.update(struct.pack("<I", len(key_bytes)))
    digest.update(key_bytes)
    digest.update(struct.pack("<QqQ", shard_index, seed, block_index))
    digest.update(struct.pack("<I", len(seed_bytes)))
    digest.update(seed_bytes)
    mixed_seed = int.from_bytes(digest.digest(), "little")
    return np.random.Generator(np.random.PCG64(mixed_seed))


def _content_block(object_key: str, shard_index: int, seed: int, block_index: int) -> bytes:
    generator = _block_entropy(object_key, shard_index, seed, block_index)
    words = (CONTENT_BLOCK_SIZE + 7) // 8
    raw = generator.bit_generator.random_raw(words).astype("<u8", copy=False).tobytes()
    return raw[:CONTENT_BLOCK_SIZE]


def generate_segment(
    object_key: str,
    shard_index: int,
    seed: int,
    shard_offset: int,
    length: int,
) -> bytes:
    """Generate a deterministic segment without path- or batch-dependent state.

    Content is anchored to fixed 64 KiB blocks of the shard.  A segment that
    crosses a block boundary therefore sees exactly the same bytes regardless
    of how the object is batched by any aggregation path.
    """

    if shard_offset < 0 or length < 0:
        raise ValueError("negative segment coordinate")
    if length == 0:
        return b""
    result = bytearray(length)
    source_pos = shard_offset
    destination_pos = 0
    remaining = length
    while remaining:
        block_index = source_pos // CONTENT_BLOCK_SIZE
        in_block = source_pos % CONTENT_BLOCK_SIZE
        take = min(remaining, CONTENT_BLOCK_SIZE - in_block)
        block = _content_block(object_key, shard_index, seed, block_index)
        result[destination_pos : destination_pos + take] = block[in_block : in_block + take]
        source_pos += take
        destination_pos += take
        remaining -= take
    return bytes(result)


class SegmentReader:
    """Reusable, non-caching deterministic content source for tests and tools."""

    def __init__(self, object_key: str, shard_index: int, seed: int) -> None:
        self.object_key = object_key
        self.shard_index = shard_index
        self.seed = seed

    def read(self, shard_offset: int, length: int) -> bytes:
        return generate_segment(self.object_key, self.shard_index, self.seed, shard_offset, length)
