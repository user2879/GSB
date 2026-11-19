"""Compact bitmap marking which logical blocks changed after a snapshot."""
from __future__ import annotations

from typing import Iterable, Iterator

from .errors import DomainError, E_INPUT


def byte_length(block_count: int) -> int:
    if not isinstance(block_count, int) or block_count <= 0:
        raise DomainError(E_INPUT, "block_count must be positive", field="block_count")
    return (block_count + 7) // 8


class Bitmap:
    """Bit-per-block map; bit ``i`` set means block ``i`` was overwritten."""

    def __init__(self, block_count: int, raw: bytes | bytearray | None = None) -> None:
        self.block_count = block_count
        expected = byte_length(block_count)
        if raw is None:
            self._raw = bytearray(expected)
        else:
            if len(raw) != expected:
                raise DomainError(
                    E_INPUT,
                    f"bitmap length {len(raw)} incompatible with block_count "
                    f"{block_count} (need {expected} bytes)",
                    field="bitmap",
                )
            self._raw = bytearray(raw)

    @classmethod
    def from_blocks(cls, block_count: int, blocks: Iterable[int]) -> "Bitmap":
        bitmap = cls(block_count)
        for block_no in blocks:
            bitmap.set(block_no)
        return bitmap

    def set(self, block_no: int) -> None:
        if not 0 <= block_no < self.block_count:
            raise DomainError(E_INPUT, f"block_no {block_no} out of bitmap range", field="block_no")
        self._raw[block_no >> 3] |= 1 << (block_no & 7)

    def get(self, block_no: int) -> bool:
        if not 0 <= block_no < self.block_count:
            raise DomainError(E_INPUT, f"block_no {block_no} out of bitmap range", field="block_no")
        return bool(self._raw[block_no >> 3] & (1 << (block_no & 7)))

    def indices(self) -> Iterator[int]:
        for block_no in range(self.block_count):
            if self.get(block_no):
                yield block_no

    def changed_blocks(self) -> list[int]:
        return list(self.indices())

    def count(self) -> int:
        return sum(bin(byte).count("1") for byte in self._raw)

    def to_bytes(self) -> bytes:
        return bytes(self._raw)

    def ratio(self) -> float:
        return self.count() / self.block_count

    def union(self, other: "Bitmap") -> "Bitmap":
        if other.block_count != self.block_count:
            raise DomainError(E_INPUT, "cannot union bitmaps of different block counts", field="bitmap")
        merged = Bitmap(self.block_count)
        merged._raw = bytearray(a | b for a, b in zip(self._raw, other._raw))
        return merged

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Bitmap) and self.block_count == other.block_count and self._raw == other._raw
