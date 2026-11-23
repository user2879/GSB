from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable

import numpy as np

from .diagnostics import Diagnostic, E_CSR_INVALID, E_MATRIX_NONFINITE
from .fingerprint import array_bytes, digest_bytes, digest_json, fingerprint_arrays


@dataclass(frozen=True)
class DistributionSpec:
    kind: str
    min_nnz: int = 1
    max_nnz: int = 12
    power: float = 2.0
    zero_probability: float = 0.55
    extreme_fraction: float = 0.05

    def validate(self) -> list[Diagnostic]:
        failures: list[Diagnostic] = []
        if self.kind not in {"uniform", "powerlaw", "extreme"}:
            failures.append(
                Diagnostic(
                    code=E_CSR_INVALID,
                    phase="distribution_validation",
                    message=f"unknown distribution: {self.kind}",
                    details={"kind": self.kind},
                )
            )
        if self.min_nnz < 0 or self.max_nnz < self.min_nnz:
            failures.append(
                Diagnostic(
                    code=E_CSR_INVALID,
                    phase="distribution_validation",
                    message="require 0 <= min_nnz <= max_nnz",
                    details={"min_nnz": self.min_nnz, "max_nnz": self.max_nnz},
                )
            )
        return failures


@dataclass(frozen=True)
class CSRMatrix:
    rows: int
    cols: int
    row_offsets: np.ndarray
    col_indices: np.ndarray
    values: np.ndarray
    distribution: str = "mixed"
    row_origins: tuple[int, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "row_offsets", np.asarray(self.row_offsets, dtype=np.int64))
        object.__setattr__(self, "col_indices", np.asarray(self.col_indices, dtype=np.int64))
        object.__setattr__(self, "values", np.asarray(self.values, dtype=np.float64))
        origins = tuple(self.row_origins)
        if len(origins) != self.rows:
            origins = tuple(range(self.rows))
        object.__setattr__(self, "row_origins", origins)

    @property
    def nnz(self) -> int:
        return int(self.col_indices.size)

    def row_nnz(self, row: int) -> int:
        return int(self.row_offsets[row + 1] - self.row_offsets[row])

    def row_slice(self, row: int) -> slice:
        return slice(int(self.row_offsets[row]), int(self.row_offsets[row + 1]))

    def row_columns(self, row: int) -> np.ndarray:
        return self.col_indices[self.row_slice(row)]

    def row_values(self, row: int) -> np.ndarray:
        return self.values[self.row_slice(row)]

    def row_fingerprint(self, row: int) -> str:
        section = self.row_slice(row)
        return digest_bytes(
            str(int(self.row_origins[row])).encode("ascii"),
            array_bytes(self.col_indices[section], "<i8"),
            array_bytes(self.values[section], "<f8"),
        )

    def fingerprint(self) -> str:
        return digest_json(
            {
                "distribution": self.distribution,
                "rows": self.rows,
                "cols": self.cols,
                "nnz": self.nnz,
                "row_origins": list(self.row_origins),
                "content": fingerprint_arrays(
                    self.row_offsets.astype("<i8", copy=False),
                    self.col_indices.astype("<i8", copy=False),
                    self.values.astype("<f8", copy=False),
                ),
            }
        )

    def validate(self) -> list[Diagnostic]:
        failures: list[Diagnostic] = []

        def fail(message: str, **details: object) -> None:
            failures.append(
                Diagnostic(
                    code=E_CSR_INVALID,
                    phase="csr_validation",
                    message=message,
                    details=dict(details),
                )
            )

        if self.rows < 0 or self.cols < 0:
            fail("matrix dimensions must be non-negative", rows=self.rows, cols=self.cols)
            return failures
        if self.row_offsets.ndim != 1 or self.row_offsets.size != self.rows + 1:
            fail(
                "row_offsets length must equal rows + 1",
                rows=self.rows,
                offsets_length=int(self.row_offsets.size),
            )
            return failures
        if int(self.row_offsets[0]) != 0:
            fail("row_offsets[0] must be zero", actual=int(self.row_offsets[0]))
        if self.rows and self.row_offsets[-1] != self.nnz:
            fail(
                "row_offsets[-1] must equal non-zero count",
                row_offset_end=int(self.row_offsets[-1]),
                nnz=self.nnz,
            )
        if self.rows == 0 and self.row_offsets.size == 1 and int(self.row_offsets[0]) != 0:
            fail("empty matrix must have row_offsets[0] == 0")
        if np.any(np.diff(self.row_offsets) < 0):
            fail("row_offsets must be monotonically non-decreasing")
        if self.col_indices.ndim != 1 or self.values.ndim != 1:
            fail("col_indices and values must be one-dimensional")
        elif self.col_indices.size != self.values.size:
            fail(
                "col_indices and values must have the same length",
                columns=int(self.col_indices.size),
                values=int(self.values.size),
            )
        if self.col_indices.size and (
            int(self.col_indices.min()) < 0 or int(self.col_indices.max()) >= self.cols
        ):
            fail(
                "column index out of range",
                min_column=int(self.col_indices.min()),
                max_column=int(self.col_indices.max()),
                cols=self.cols,
            )
        if not np.all(np.isfinite(self.values)):
            bad = int(np.flatnonzero(~np.isfinite(self.values))[0])
            failures.append(
                Diagnostic(
                    code=E_MATRIX_NONFINITE,
                    phase="csr_validation",
                    message="CSR value is not finite",
                    details={"value_index": bad, "value_repr": repr(float(self.values[bad]))},
                )
            )
        if not failures:
            for row in range(self.rows):
                start = int(self.row_offsets[row])
                end = int(self.row_offsets[row + 1])
                cols = self.col_indices[start:end]
                if cols.size > 1 and np.any(np.diff(cols) <= 0):
                    fail("column indices must be strictly increasing within each row", row=row)
                    break
        return failures

    def replace_row(self, row: int, columns: Iterable[int], values: Iterable[float]) -> "CSRMatrix":
        col_values = np.asarray(list(columns), dtype=np.int64)
        numeric_values = np.asarray(list(values), dtype=np.float64)
        if col_values.size != numeric_values.size:
            raise ValueError("columns and values must have equal length")
        order = np.argsort(col_values, kind="stable")
        col_values = col_values[order]
        numeric_values = numeric_values[order]
        if col_values.size > 1 and np.any(np.diff(col_values) <= 0):
            raise ValueError("columns for a replaced row must be unique")
        if self.rows == 0 or row < 0 or row >= self.rows:
            raise IndexError("row out of range")

        start = int(self.row_offsets[row])
        end = int(self.row_offsets[row + 1])
        new_offsets = self.row_offsets.copy()
        new_offsets[row + 1 :] += col_values.size - (end - start)
        new_cols = np.concatenate(
            (self.col_indices[:start], col_values, self.col_indices[end:])
        )
        new_values = np.concatenate((self.values[:start], numeric_values, self.values[end:]))
        return replace(
            self,
            row_offsets=new_offsets,
            col_indices=new_cols,
            values=new_values,
        )

    def delete_row(self, row: int) -> "CSRMatrix":
        if row < 0 or row >= self.rows:
            raise IndexError("row out of range")
        start = int(self.row_offsets[row])
        end = int(self.row_offsets[row + 1])
        new_offsets = np.concatenate(
            (self.row_offsets[: row + 1], self.row_offsets[row + 2 :] - (end - start))
        )
        new_origins = self.row_origins[:row] + self.row_origins[row + 1 :]
        return replace(
            self,
            rows=self.rows - 1,
            row_offsets=new_offsets,
            col_indices=np.concatenate((self.col_indices[:start], self.col_indices[end:])),
            values=np.concatenate((self.values[:start], self.values[end:])),
            row_origins=new_origins,
            distribution="mixed",
        )

    def append_row(
        self,
        columns: Iterable[int],
        values: Iterable[float],
        origin: int | None = None,
    ) -> "CSRMatrix":
        col_values = np.asarray(list(columns), dtype=np.int64)
        numeric_values = np.asarray(list(values), dtype=np.float64)
        if col_values.size != numeric_values.size:
            raise ValueError("columns and values must have equal length")
        order = np.argsort(col_values, kind="stable")
        col_values = col_values[order]
        numeric_values = numeric_values[order]
        if col_values.size > 1 and np.any(np.diff(col_values) <= 0):
            raise ValueError("columns for an appended row must be unique")
        next_origin = max(self.row_origins, default=-1) + 1 if origin is None else origin
        return replace(
            self,
            rows=self.rows + 1,
            row_offsets=np.concatenate((self.row_offsets, [self.nnz + col_values.size])),
            col_indices=np.concatenate((self.col_indices, col_values)),
            values=np.concatenate((self.values, numeric_values)),
            row_origins=self.row_origins + (int(next_origin),),
            distribution="mixed",
        )


def generate_csr(
    rows: int,
    cols: int,
    distribution: DistributionSpec,
    seed: int,
    *,
    value_scale: float = 2.0,
) -> CSRMatrix:
    """Generate a deterministic CSR matrix with strictly increasing row columns."""
    failures = distribution.validate()
    if failures:
        raise ValueError(failures[0].message)
    if rows < 0 or cols < 0:
        raise ValueError("rows and cols must be non-negative")
    rng = np.random.default_rng(seed)
    counts = np.zeros(rows, dtype=np.int64)
    for row in range(rows):
        if distribution.kind == "uniform":
            count = int(rng.integers(distribution.min_nnz, distribution.max_nnz + 1))
        elif distribution.kind == "powerlaw":
            draw = float(rng.random())
            span = distribution.max_nnz - distribution.min_nnz
            count = distribution.min_nnz + int(np.floor(span * (draw**distribution.power)))
        else:
            draw = float(rng.random())
            if draw < distribution.zero_probability:
                count = 0
            elif draw > 1.0 - distribution.extreme_fraction:
                count = distribution.max_nnz
            else:
                count = int(rng.integers(distribution.min_nnz, distribution.max_nnz + 1))
        counts[row] = min(max(count, 0), cols)

    offsets = np.empty(rows + 1, dtype=np.int64)
    offsets[0] = 0
    offsets[1:] = np.cumsum(counts)
    columns = np.empty(int(offsets[-1]), dtype=np.int64)
    values = np.empty(int(offsets[-1]), dtype=np.float64)
    for row, count in enumerate(counts):
        start = int(offsets[row])
        end = int(offsets[row + 1])
        if count == 0:
            continue
        row_cols = rng.choice(cols, size=int(count), replace=False)
        row_cols.sort()
        columns[start:end] = row_cols
        raw = rng.uniform(-value_scale, value_scale, size=int(count))
        # Keep every stored entry a true non-zero while retaining mixed signs.
        too_small = np.abs(raw) < 0.05
        raw[too_small] = np.copysign(0.05, raw[too_small])
        values[start:end] = raw
    return CSRMatrix(
        rows=rows,
        cols=cols,
        row_offsets=offsets,
        col_indices=columns,
        values=values,
        distribution=distribution.kind,
        row_origins=tuple(range(rows)),
    )
