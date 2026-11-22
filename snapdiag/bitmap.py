"""Dense bitmap helpers with a stable most-significant-bit-first layout."""

from __future__ import annotations

from collections.abc import Iterable

from .errors import DiagnosticError, E_FORMAT, E_GEOM


def bitmap_length(block_count: int) -> int:
    return (block_count + 7) // 8


def empty_bitmap(block_count: int) -> bytearray:
    return bytearray(bitmap_length(block_count))


def get_bit(bitmap: bytes | bytearray, index: int) -> bool:
    return bool(bitmap[index // 8] & (1 << (7 - (index % 8))))


def set_bit(bitmap: bytearray, index: int) -> None:
    bitmap[index // 8] |= 1 << (7 - (index % 8))


def clear_bit(bitmap: bytearray, index: int) -> None:
    bitmap[index // 8] &= ~(1 << (7 - (index % 8)))


def popcount(bitmap: bytes | bytearray) -> int:
    return sum(byte.bit_count() for byte in bitmap)


def set_bits(bitmap: bytes | bytearray, block_count: int | None = None) -> tuple[int, ...]:
    limit = len(bitmap) * 8 if block_count is None else block_count
    return tuple(index for index in range(limit) if get_bit(bitmap, index))


def bitmap_from_indices(block_count: int, indices: Iterable[int]) -> bytearray:
    result = empty_bitmap(block_count)
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise DiagnosticError(
                E_FORMAT,
                "bitmap block index must be an integer",
                field="bitmap",
                details={"block_index": index},
            )
        if index < 0 or index >= block_count:
            raise DiagnosticError(
                E_GEOM,
                "bitmap block index is outside the volume",
                field="block_index",
                details={"block_index": index, "block_count": block_count},
            )
        set_bit(result, index)
    return result


def bitmap_checksum(bitmap: bytes | bytearray) -> str:
    import hashlib

    return hashlib.sha256(bytes(bitmap)).hexdigest()


def validate_bitmap(
    bitmap: bytes,
    block_count: int,
    *,
    snapshot_id: str | None,
    bitmap_length_value: int | None = None,
) -> None:
    expected_length = bitmap_length(block_count)
    if bitmap_length_value is not None and bitmap_length_value != expected_length:
        raise DiagnosticError(
            E_FORMAT,
            "bitmap_length is inconsistent with volume_block_count",
            snapshot_id=snapshot_id,
            field="bitmap_length",
            details={
                "expected": expected_length,
                "actual": bitmap_length_value,
                "volume_block_count": block_count,
            },
        )
    if len(bitmap) != expected_length:
        raise DiagnosticError(
            E_FORMAT,
            "decoded bitmap length is inconsistent with volume_block_count",
            snapshot_id=snapshot_id,
            field="bitmap",
            details={"expected": expected_length, "actual": len(bitmap)},
        )
    unused = expected_length * 8 - block_count
    if unused:
        mask = (1 << unused) - 1
        if bitmap[-1] & mask:
            raise DiagnosticError(
                E_FORMAT,
                "bitmap has bits set outside the declared block range",
                snapshot_id=snapshot_id,
                field="bitmap",
                details={"unused_bits": unused, "last_byte": bitmap[-1]},
            )


def bitmap_overview(bitmap: bytes | bytearray, block_count: int, *, group_size: int = 8) -> list[str]:
    """Render blocks in ascending order for stable UI/test output."""

    rows: list[str] = []
    for start in range(0, block_count, group_size):
        end = min(block_count, start + group_size)
        cells = ["1" if get_bit(bitmap, index) else "0" for index in range(start, end)]
        rows.append(f"{start:05d}:{''.join(cells)}")
    return rows

