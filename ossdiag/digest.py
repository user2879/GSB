from __future__ import annotations

import hashlib
import struct
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

from .content import LEAF_SIZE


TAG_LEAF = b"OSSDIAG-LEAF-v1\x00"
TAG_BATCH = b"OSSDIAG-BATCH-v1\x00"
TAG_NODE = b"OSSDIAG-NODE-v1\x00"
TAG_SHARD = b"OSSDIAG-SHARD-v1\x00"
TAG_OBJECT = b"OSSDIAG-OBJECT-v1\x00"


def _length_prefixed(value: bytes) -> bytes:
    return struct.pack("<I", len(value)) + value


def leaf_digest(
    object_key: str,
    shard_index: int,
    leaf_index: int,
    leaf_length: int,
    data: bytes,
) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(TAG_LEAF)
    hasher.update(_length_prefixed(object_key.encode("utf-8")))
    hasher.update(struct.pack("<QQQ", shard_index, leaf_index, leaf_length))
    hasher.update(data)
    return hasher.digest()


def batch_digest(
    object_key: str,
    shard_index: int,
    shard_offset: int,
    data: bytes,
) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(TAG_BATCH)
    hasher.update(_length_prefixed(object_key.encode("utf-8")))
    hasher.update(struct.pack("<QQQ", shard_index, shard_offset, len(data)))
    hasher.update(data)
    return hasher.digest()


def combine_nodes(left: bytes, right: bytes) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(TAG_NODE)
    hasher.update(left)
    hasher.update(right)
    return hasher.digest()


class MerkleBuilder:
    """Streaming RFC-6962-style Merkle frontier, identical to the canonical tree."""

    def __init__(self) -> None:
        self._stack: list[tuple[int, bytes]] = []
        self.leaf_count = 0

    def add(self, leaf: bytes) -> None:
        node = leaf
        level = 1
        while self._stack and self._stack[-1][0] == level:
            _, left = self._stack.pop()
            node = combine_nodes(left, node)
            level += 1
        self._stack.append((level, node))
        self.leaf_count += 1

    def extend(self, leaves: Iterable[bytes]) -> None:
        for leaf in leaves:
            self.add(leaf)

    def root(self) -> bytes:
        if not self._stack:
            return hashlib.sha256(TAG_SHARD + struct.pack("<Q", 0)).digest()
        _, root = self._stack[-1]
        for _, node in reversed(self._stack[:-1]):
            root = combine_nodes(node, root)
        return root


def _largest_power_of_two_below(value: int) -> int:
    return 1 << ((value - 1).bit_length() - 1)


def merkle_root(leaves: Sequence[bytes]) -> bytes:
    """Canonical recursive root; produces the same root as MerkleBuilder."""

    count = len(leaves)
    if count == 0:
        return MerkleBuilder().root()
    if count == 1:
        return leaves[0]
    split = _largest_power_of_two_below(count)
    return combine_nodes(merkle_root(leaves[:split]), merkle_root(leaves[split:]))


def shard_digest(leaves: Sequence[bytes]) -> bytes:
    return merkle_root(leaves)


def object_digest(
    object_key: str,
    total_bytes: int,
    shards: Sequence[tuple[int, int, int, bytes]],
) -> bytes:
    """Hash canonical shard descriptors, not traversal order or worker timing."""

    hasher = hashlib.sha256()
    hasher.update(TAG_OBJECT)
    hasher.update(_length_prefixed(object_key.encode("utf-8")))
    hasher.update(struct.pack("<QQ", total_bytes, len(shards)))
    for shard_index, start, end, root in sorted(shards, key=lambda item: item[0]):
        hasher.update(struct.pack("<QQQ", shard_index, start, end))
        hasher.update(root)
    return hasher.digest()


@dataclass(frozen=True)
class ChunkDigests:
    batch_hex: str
    leaves: tuple[tuple[int, str], ...]


def digest_chunk(
    object_key: str,
    shard_index: int,
    shard_offset: int,
    shard_length: int,
    data: bytes,
) -> ChunkDigests:
    if shard_offset % LEAF_SIZE:
        raise ValueError("chunk start must be aligned to the fixed leaf size")
    if shard_offset >= shard_length and len(data):
        raise ValueError("chunk starts outside the shard")
    if len(data) == 0:
        raise ValueError("empty chunks are not digestible")
    leaves: list[tuple[int, str]] = []
    local = 0
    while local < len(data):
        leaf_index = (shard_offset + local) // LEAF_SIZE
        leaf_length = min(LEAF_SIZE, shard_length - (shard_offset + local), len(data) - local)
        if leaf_length <= 0:
            raise ValueError("chunk has zero-length leaf")
        piece = data[local : local + leaf_length]
        leaves.append((leaf_index, leaf_digest(object_key, shard_index, leaf_index, leaf_length, piece).hex()))
        local += leaf_length
    return ChunkDigests(
        batch_hex=batch_digest(object_key, shard_index, shard_offset, data).hex(),
        leaves=tuple(leaves),
    )


def leaves_from_hex(values: Iterable[str]) -> list[bytes]:
    return [bytes.fromhex(value) for value in values]


class DirectAggregator:
    """Per-shard streaming frontier: one shard at a time, O(log leaves) state."""

    def __init__(self, initial_leaves: Iterable[bytes] = ()) -> None:
        self.builder = MerkleBuilder()
        self.builder.extend(initial_leaves)

    def add_leaf(self, digest: bytes) -> None:
        self.builder.add(digest)

    def finalize(self) -> bytes:
        return self.builder.root()

    @property
    def state_size(self) -> int:
        return len(self.builder._stack)


class SlidingWindowAggregator:
    """Bounded leaf window with ordered flushes into the canonical frontier."""

    def __init__(self, initial_leaves: Iterable[bytes] = (), window_size: int = 32) -> None:
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        self.window_size = window_size
        self.window: deque[bytes] = deque()
        self.builder = MerkleBuilder()
        for leaf in initial_leaves:
            self.add_leaf(leaf)

    def add_leaf(self, digest: bytes) -> None:
        self.window.append(digest)
        if len(self.window) >= self.window_size:
            self._flush()

    def _flush(self) -> None:
        while self.window:
            self.builder.add(self.window.popleft())

    def finalize(self) -> bytes:
        self._flush()
        return self.builder.root()

    @property
    def state_size(self) -> int:
        return len(self.window) + len(self.builder._stack)


class HierarchicalAggregator:
    """Keep all leaves, then materialize the canonical tree recursively."""

    def __init__(self, initial_leaves: Iterable[bytes] = ()) -> None:
        self.leaves: list[bytes] = list(initial_leaves)

    def add_leaf(self, digest: bytes) -> None:
        self.leaves.append(digest)

    def finalize(self) -> bytes:
        return merkle_root(self.leaves)

    @property
    def state_size(self) -> int:
        return len(self.leaves)


def make_aggregator(path: str, initial_leaves: Iterable[bytes] = ()):
    leaves = list(initial_leaves)
    if path == "direct":
        return DirectAggregator(leaves)
    if path == "sliding":
        return SlidingWindowAggregator(leaves)
    if path == "hierarchical":
        return HierarchicalAggregator(leaves)
    raise ValueError(f"unknown aggregation path: {path}")


def merkle_root_hex(leaves: Iterable[bytes]) -> str:
    return merkle_root(list(leaves)).hex()
