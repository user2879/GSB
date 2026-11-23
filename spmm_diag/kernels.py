from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .config import (
    MODE_FIXED,
    ORDER_ASCENDING,
    ORDER_B_ABS_DESC,
    ORDER_DESCENDING,
    SpMMConfig,
)
from .csr import CSRMatrix
from .fingerprint import array_bytes, digest_bytes, digest_json
from .graph import derive_dependency


@dataclass(frozen=True)
class OverflowEvent:
    block_id: int
    row: int
    column: int
    direction: str
    accumulator_bits: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "row": self.row,
            "column": self.column,
            "direction": self.direction,
            "accumulator_bits": self.accumulator_bits,
        }


@dataclass
class BlockResult:
    block_id: int
    row_start: int
    row_end: int
    values: np.ndarray
    row_fingerprints: tuple[str, ...]
    empty_rows: tuple[int, ...]
    first_overflow: OverflowEvent | None
    recompute_nnz: int
    order: str
    precision_fingerprint: str
    dependency_fingerprint: str
    output_fingerprint: str

    @property
    def rows(self) -> int:
        return self.row_end - self.row_start

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "row_start": self.row_start,
            "row_end": self.row_end,
            "values": np.asarray(self.values, dtype=np.float32).tolist(),
            "row_fingerprints": list(self.row_fingerprints),
            "empty_rows": list(self.empty_rows),
            "first_overflow": None if self.first_overflow is None else self.first_overflow.to_dict(),
            "recompute_nnz": self.recompute_nnz,
            "order": self.order,
            "precision_fingerprint": self.precision_fingerprint,
            "dependency_fingerprint": self.dependency_fingerprint,
            "output_fingerprint": self.output_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BlockResult":
        overflow = value.get("first_overflow")
        return cls(
            block_id=int(value["block_id"]),
            row_start=int(value["row_start"]),
            row_end=int(value["row_end"]),
            values=np.asarray(value["values"], dtype=np.float32).copy(),
            row_fingerprints=tuple(value["row_fingerprints"]),
            empty_rows=tuple(int(row) for row in value["empty_rows"]),
            first_overflow=None if overflow is None else OverflowEvent(**overflow),
            recompute_nnz=int(value["recompute_nnz"]),
            order=str(value["order"]),
            precision_fingerprint=str(value["precision_fingerprint"]),
            dependency_fingerprint=str(value["dependency_fingerprint"]),
            output_fingerprint=str(value["output_fingerprint"]),
        )


@dataclass(frozen=True)
class QuantizedMatrix:
    values: np.ndarray
    fractional_bits: int
    scale: int

    @classmethod
    def from_float(cls, matrix: np.ndarray, fractional_bits: int) -> "QuantizedMatrix":
        scale = 1 << fractional_bits
        scaled = np.asarray(matrix, dtype=np.float64) * float(scale)
        rounded = np.trunc(scaled + np.copysign(0.5, scaled))
        return cls(values=rounded.astype(np.int64), fractional_bits=fractional_bits, scale=scale)


def validate_dense(matrix: np.ndarray, shape: tuple[int, int], name: str) -> list[str]:
    failures: list[str] = []
    array = np.asarray(matrix)
    if array.ndim != 2:
        failures.append(f"{name} must be two-dimensional")
        return failures
    if tuple(array.shape) != shape:
        failures.append(f"{name} shape {array.shape} does not match expected {shape}")
    if array.size and not np.all(np.isfinite(array)):
        failures.append(f"{name} contains a non-finite value")
    return failures


def dense_fingerprint(matrix: np.ndarray) -> str:
    array = np.asarray(matrix, dtype=np.float64)
    return digest_json(
        {
            "shape": list(array.shape),
            "content": digest_bytes(
                str(array.dtype.str).encode("ascii"), array_bytes(array, "<f8")
            ),
        }
    )


def row_blocks(rows: int, block_size: int) -> list[tuple[int, int]]:
    if block_size <= 0:
        raise ValueError("block size must be positive")
    return [(start, min(start + block_size, rows)) for start in range(0, rows, block_size)]


def entry_order(
    csr: CSRMatrix,
    row: int,
    b_matrix: np.ndarray,
    order: str,
) -> np.ndarray:
    start = int(csr.row_offsets[row])
    end = int(csr.row_offsets[row + 1])
    local = np.arange(end - start, dtype=np.int64)
    if order == ORDER_ASCENDING:
        return local
    columns = csr.col_indices[start:end]
    if order == ORDER_DESCENDING:
        return np.argsort(-columns, kind="stable")
    if order == ORDER_B_ABS_DESC:
        if columns.size == 0:
            return local
        scores = np.max(np.abs(np.asarray(b_matrix[columns], dtype=np.float64)), axis=1)
        # Descending score, then ascending column index for a total stable order.
        return np.lexsort((columns, -scores))
    raise ValueError(f"unknown order: {order}")


def _chunks(indices: np.ndarray, chunk_size: int) -> Iterable[np.ndarray]:
    if chunk_size <= 0:
        raise ValueError("nnz_chunk_size must be positive")
    for offset in range(0, int(indices.size), chunk_size):
        yield indices[offset : offset + chunk_size]


def _row_fingerprint(
    block_id: int,
    row: int,
    values: np.ndarray,
    empty: bool,
    overflow: OverflowEvent | None,
    dependency_fingerprint: str,
) -> str:
    return digest_json(
        {
            "block_id": block_id,
            "row": row,
            "values": digest_bytes(array_bytes(np.asarray(values, dtype=np.float32), "<f4")),
            "empty": empty,
            "overflow": None if overflow is None else overflow.to_dict(),
            "dependency": dependency_fingerprint,
        }
    )


def _block_fingerprint(
    block_id: int,
    row_start: int,
    row_end: int,
    values: np.ndarray,
    row_fingerprints: Sequence[str],
    empty_rows: Sequence[int],
    first_overflow: OverflowEvent | None,
    dependency_fingerprint: str,
) -> str:
    return digest_json(
        {
            "block_id": block_id,
            "row_start": row_start,
            "row_end": row_end,
            "values": digest_bytes(array_bytes(np.asarray(values, dtype=np.float32), "<f4")),
            "rows": list(row_fingerprints),
            "empty_rows": list(empty_rows),
            "overflow": None if first_overflow is None else first_overflow.to_dict(),
            "dependency": dependency_fingerprint,
        }
    )


def _compute_float32_row(
    csr: CSRMatrix,
    row: int,
    b32: np.ndarray,
    c32: np.ndarray,
    config: SpMMConfig,
) -> np.ndarray:
    n = b32.shape[1]
    order = entry_order(csr, row, b32, config.order)
    start = int(csr.row_offsets[row])
    row_values = csr.values[start : int(csr.row_offsets[row + 1])].astype(np.float32)
    total = np.zeros(n, dtype=np.float32)
    for chunk in _chunks(order, config.nnz_chunk_size):
        partial = np.zeros(n, dtype=np.float32)
        for local_index in chunk:
            index = int(local_index)
            value = np.float32(row_values[index])
            b_row = b32[int(csr.col_indices[start + index])]
            partial = np.add(partial, np.multiply(value, b_row, dtype=np.float32), dtype=np.float32)
        total = np.add(total, partial, dtype=np.float32)
    scaled = np.multiply(total, np.float32(config.alpha), dtype=np.float32)
    base = np.multiply(c32[row], np.float32(config.beta), dtype=np.float32)
    return np.add(scaled, base, dtype=np.float32)


def _sat_add(
    accumulator: np.ndarray,
    addend: np.ndarray,
    minimum: int,
    maximum: int,
    overflow_flags: np.ndarray,
) -> None:
    positive = addend > 0
    negative = addend < 0
    upper = np.zeros_like(accumulator, dtype=bool)
    lower = np.zeros_like(accumulator, dtype=bool)
    if np.any(positive):
        upper[positive] = accumulator[positive] > maximum - addend[positive]
    if np.any(negative):
        lower[negative] = accumulator[negative] < minimum - addend[negative]
    upper |= addend > maximum
    lower |= addend < minimum
    overflow_flags |= upper | lower
    accumulator[upper] = maximum
    accumulator[lower] = minimum
    safe = ~(upper | lower)
    accumulator[safe] += addend[safe]


def _first_overflow_column(mask: np.ndarray, addend: np.ndarray) -> tuple[int, str] | None:
    positions = np.flatnonzero(mask)
    if positions.size == 0:
        return None
    column = int(positions[0])
    direction = "+" if int(addend[column]) > 0 else "-"
    return column, direction


def _compute_fixed_row(
    csr: CSRMatrix,
    row: int,
    b_int: np.ndarray,
    b_original: np.ndarray,
    c32: np.ndarray,
    config: SpMMConfig,
) -> tuple[np.ndarray, OverflowEvent | None]:
    n = b_int.shape[1]
    order = entry_order(csr, row, b_original, config.order)
    start = int(csr.row_offsets[row])
    row_values = csr.values[start : int(csr.row_offsets[row + 1])]
    quantized_a = QuantizedMatrix.from_float(row_values, config.fractional_bits).values
    minimum = -(1 << (config.accumulator_bits - 1))
    maximum = (1 << (config.accumulator_bits - 1)) - 1
    total = np.zeros(n, dtype=np.int64)
    first_local: tuple[int, str] | None = None

    for chunk in _chunks(order, config.nnz_chunk_size):
        partial = np.zeros(n, dtype=np.int64)
        for local_index in chunk:
            index = int(local_index)
            product = quantized_a[index] * b_int[int(csr.col_indices[start + index])]
            flags = np.zeros(n, dtype=bool)
            _sat_add(partial, product, minimum, maximum, flags)
            if first_local is None:
                detected = _first_overflow_column(flags, product)
                if detected is not None:
                    first_local = detected
        flags = np.zeros(n, dtype=bool)
        _sat_add(total, partial, minimum, maximum, flags)
        if first_local is None:
            detected = _first_overflow_column(flags, partial)
            if detected is not None:
                first_local = detected

    dequant_scale = float(1 << (2 * config.fractional_bits))
    total_float = (total.astype(np.float64) / dequant_scale).astype(np.float32)
    scaled = np.multiply(total_float, np.float32(config.alpha), dtype=np.float32)
    base = np.multiply(c32[row], np.float32(config.beta), dtype=np.float32)
    values = np.add(scaled, base, dtype=np.float32)
    overflow = None if first_local is None else (first_local[0], first_local[1])
    return values, overflow


def compute_block(
    csr: CSRMatrix,
    b_matrix: np.ndarray,
    c_matrix: np.ndarray,
    config: SpMMConfig,
    block_id: int,
    row_start: int,
    row_end: int,
) -> BlockResult:
    if config.precision_mode not in {"fp32", "fixed"}:
        raise ValueError(f"unknown precision mode: {config.precision_mode}")
    dependency_fingerprint = derive_dependency(
        csr, b_matrix, c_matrix, config, block_id, row_start, row_end
    ).input_fingerprint
    b32 = np.asarray(b_matrix, dtype=np.float32)
    c32 = np.asarray(c_matrix, dtype=np.float32)
    b_int = (
        QuantizedMatrix.from_float(b_matrix, config.fractional_bits).values
        if config.precision_mode == MODE_FIXED
        else None
    )
    block_values = np.empty((row_end - row_start, b32.shape[1]), dtype=np.float32)
    row_fingerprints: list[str] = []
    empty_rows: list[int] = []
    block_first_overflow: OverflowEvent | None = None

    for row in range(row_start, row_end):
        row_overflow_local: tuple[int, str] | None = None
        empty = csr.row_nnz(row) == 0
        if empty:
            block_values[row - row_start] = np.multiply(
                c32[row], np.float32(config.beta), dtype=np.float32
            )
            empty_rows.append(row)
        elif b_int is None:
            block_values[row - row_start] = _compute_float32_row(csr, row, b32, c32, config)
        else:
            values, row_overflow_local = _compute_fixed_row(
                csr, row, b_int, b_matrix, c32, config
            )
            block_values[row - row_start] = values

        row_overflow = (
            None
            if row_overflow_local is None
            else OverflowEvent(
                block_id=block_id,
                row=row,
                column=row_overflow_local[0],
                direction=row_overflow_local[1],
                accumulator_bits=config.accumulator_bits,
            )
        )
        if row_overflow is not None and block_first_overflow is None:
            block_first_overflow = row_overflow
        row_fingerprints.append(
            _row_fingerprint(
                block_id,
                row,
                block_values[row - row_start],
                empty,
                row_overflow,
                dependency_fingerprint,
            )
        )

    output_fingerprint = _block_fingerprint(
        block_id,
        row_start,
        row_end,
        block_values,
        row_fingerprints,
        empty_rows,
        block_first_overflow,
        dependency_fingerprint,
    )
    recompute_nnz = sum(csr.row_nnz(row) for row in range(row_start, row_end))
    return BlockResult(
        block_id=block_id,
        row_start=row_start,
        row_end=row_end,
        values=block_values,
        row_fingerprints=tuple(row_fingerprints),
        empty_rows=tuple(empty_rows),
        first_overflow=block_first_overflow,
        recompute_nnz=recompute_nnz,
        order=config.order,
        precision_fingerprint=config.precision_fingerprint(),
        dependency_fingerprint=dependency_fingerprint,
        output_fingerprint=output_fingerprint,
    )


def compare_float_and_fixed(
    float_blocks: Sequence[BlockResult], fixed_blocks: Sequence[BlockResult]
) -> dict[str, Any]:
    if len(float_blocks) != len(fixed_blocks):
        return {"comparable": False, "reason": "block_count_mismatch"}
    max_abs = 0.0
    element_count = 0
    for left, right in zip(float_blocks, fixed_blocks):
        if (left.row_start, left.row_end) != (right.row_start, right.row_end):
            return {"comparable": False, "reason": "block_range_mismatch"}
        left_values = np.asarray(left.values, dtype=np.float64)
        right_values = np.asarray(right.values, dtype=np.float64)
        if left_values.shape != right_values.shape:
            return {"comparable": False, "reason": "block_shape_mismatch"}
        if left_values.size:
            max_abs = max(max_abs, float(np.max(np.abs(left_values - right_values))))
        element_count += int(left_values.size)
    return {
        "comparable": True,
        "max_abs_error": max_abs,
        "element_count": element_count,
        "tolerance": 0.05,
        "within_tolerance": max_abs <= 0.05,
    }
