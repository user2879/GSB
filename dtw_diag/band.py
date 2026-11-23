"""Sakoe-Chiba-style feasible-row band construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .fingerprints import fingerprint


@dataclass(frozen=True)
class RowBand:
    row: int
    col_start: int
    col_end: int
    empty: bool = False

    def contains(self, col: int) -> bool:
        return not self.empty and self.col_start <= col <= self.col_end

    def as_key_data(self) -> dict[str, Any]:
        return {
            "row": self.row,
            "col_start": self.col_start,
            "col_end": self.col_end,
            "empty": self.empty,
        }

    def to_dict(self) -> dict[str, Any]:
        return self.as_key_data()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RowBand":
        return cls(
            row=raw["row"],
            col_start=raw["col_start"],
            col_end=raw["col_end"],
            empty=raw.get("empty", False),
        )


@dataclass(frozen=True)
class Band:
    rows: tuple[RowBand, ...]
    row_count: int
    col_count: int
    bandwidth_permille: int
    radius: int
    fingerprint: str
    formula: str = "linear-round-half-up-v1"

    def row(self, index: int) -> RowBand:
        return self.rows[index]

    def contains(self, row: int, col: int) -> bool:
        return 0 <= row < self.row_count and self.rows[row].contains(col)

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "col_count": self.col_count,
            "bandwidth_permille": self.bandwidth_permille,
            "radius": self.radius,
            "formula": self.formula,
            "rows": [row.to_dict() for row in self.rows],
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Band":
        return cls(
            rows=tuple(RowBand.from_dict(row) for row in raw["rows"]),
            row_count=raw["row_count"],
            col_count=raw["col_count"],
            bandwidth_permille=raw["bandwidth_permille"],
            radius=raw["radius"],
            fingerprint=raw["fingerprint"],
            formula=raw.get("formula", "linear-round-half-up-v1"),
        )


def _round_half_up_ratio(numerator: int, denominator: int) -> int:
    return (2 * numerator + denominator) // (2 * denominator)


def build_band(row_count: int, col_count: int, bandwidth_permille: int) -> Band:
    """Build a band around the rounded linear diagonal.

    ``radius = ceil(max(n, m) * permille / 1000)``. For ordinary rows the
    center is ``round_half_up(i * (m-1) / (n-1))``. The two terminal cells are
    always included; degenerate one-row/one-column inputs are widened to make a
    valid endpoint path possible. Empty ranges are represented explicitly.
    """

    if row_count <= 0 or col_count <= 0:
        raise ValueError("row_count and col_count must be positive")
    radius = (max(row_count, col_count) * bandwidth_permille + 999) // 1000
    rows: list[RowBand] = []
    for i in range(row_count):
        if row_count == 1:
            lo, hi = 0, col_count - 1
        elif col_count == 1:
            lo, hi = 0, 0
        else:
            center = _round_half_up_ratio(i * (col_count - 1), row_count - 1)
            lo = max(0, center - radius)
            hi = min(col_count - 1, center + radius)
            if i == 0:
                lo = min(lo, 0)
                hi = max(hi, 0)
            if i == row_count - 1:
                lo = min(lo, col_count - 1)
                hi = max(hi, col_count - 1)
        empty = lo > hi
        rows.append(RowBand(i, lo, hi, empty))

    key_data = {
        "formula": "linear-round-half-up-v1",
        "row_count": row_count,
        "col_count": col_count,
        "bandwidth_permille": bandwidth_permille,
        "radius": radius,
        "rows": [row.as_key_data() for row in rows],
    }
    return Band(
        rows=tuple(rows),
        row_count=row_count,
        col_count=col_count,
        bandwidth_permille=bandwidth_permille,
        radius=radius,
        fingerprint=fingerprint(key_data),
    )
