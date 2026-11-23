from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from .diagnostics import (
    Diagnostic,
    E_ACC_WIDTH_INVALID,
    E_BLOCK_SIZE_INVALID,
    E_CONFIG_INVALID,
)
from .fingerprint import digest_json


ORDER_ASCENDING = "col_asc"
ORDER_DESCENDING = "col_desc"
ORDER_B_ABS_DESC = "b_abs_desc"
VALID_ORDERS = (ORDER_ASCENDING, ORDER_DESCENDING, ORDER_B_ABS_DESC)

MODE_FLOAT32 = "fp32"
MODE_FIXED = "fixed"
VALID_MODES = (MODE_FLOAT32, MODE_FIXED)


@dataclass(frozen=True)
class SpMMConfig:
    alpha: float = 1.0
    beta: float = 0.0
    row_block_size: int = 4
    nnz_chunk_size: int = 3
    order: str = ORDER_ASCENDING
    precision_mode: str = MODE_FLOAT32
    fractional_bits: int = 6
    accumulator_bits: int = 16
    max_workers: int = 4
    recompute_budget: int = 10_000
    hot_capacity: int = 64

    def with_changes(self, **changes: Any) -> "SpMMConfig":
        return replace(self, **changes)

    def precision_fingerprint(self) -> str:
        return digest_json(
            {
                "mode": self.precision_mode,
                "fractional_bits": self.fractional_bits,
                "accumulator_bits": self.accumulator_bits,
            }
        )

    def layout_fingerprint(self) -> str:
        return digest_json(
            {
                "row_block_size": self.row_block_size,
                "nnz_chunk_size": self.nnz_chunk_size,
                "order": self.order,
            }
        )

    def coefficient_fingerprint(self) -> str:
        return digest_json({"alpha": self.alpha, "beta": self.beta})

    def fingerprint(self) -> str:
        return digest_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "row_block_size": self.row_block_size,
            "nnz_chunk_size": self.nnz_chunk_size,
            "order": self.order,
            "precision_mode": self.precision_mode,
            "fractional_bits": self.fractional_bits,
            "accumulator_bits": self.accumulator_bits,
            "max_workers": self.max_workers,
            "recompute_budget": self.recompute_budget,
            "hot_capacity": self.hot_capacity,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SpMMConfig":
        return cls(**value)

    def validate(self) -> list[Diagnostic]:
        failures: list[Diagnostic] = []
        if self.row_block_size <= 0 or self.nnz_chunk_size <= 0:
            failures.append(
                Diagnostic(
                    code=E_BLOCK_SIZE_INVALID,
                    phase="config_validation",
                    message="row_block_size and nnz_chunk_size must both be positive",
                    details={
                        "row_block_size": self.row_block_size,
                        "nnz_chunk_size": self.nnz_chunk_size,
                    },
                )
            )
        if not (2 <= self.accumulator_bits <= 63):
            failures.append(
                Diagnostic(
                    code=E_ACC_WIDTH_INVALID,
                    phase="config_validation",
                    message="accumulator_bits must be in [2, 63]",
                    details={"accumulator_bits": self.accumulator_bits},
                )
            )
        if self.order not in VALID_ORDERS:
            failures.append(
                Diagnostic(
                    code=E_CONFIG_INVALID,
                    phase="config_validation",
                    message=f"unknown accumulation order: {self.order}",
                    details={"order": self.order},
                )
            )
        if self.precision_mode not in VALID_MODES:
            failures.append(
                Diagnostic(
                    code=E_CONFIG_INVALID,
                    phase="config_validation",
                    message=f"unknown precision mode: {self.precision_mode}",
                    details={"precision_mode": self.precision_mode},
                )
            )
        if self.fractional_bits < 0 or self.fractional_bits > 20:
            failures.append(
                Diagnostic(
                    code=E_CONFIG_INVALID,
                    phase="config_validation",
                    message="fractional_bits must be in [0, 20]",
                    details={"fractional_bits": self.fractional_bits},
                )
            )
        if self.max_workers <= 0:
            failures.append(
                Diagnostic(
                    code=E_CONFIG_INVALID,
                    phase="config_validation",
                    message="max_workers must be positive",
                    details={"max_workers": self.max_workers},
                )
            )
        if self.recompute_budget < 0:
            failures.append(
                Diagnostic(
                    code=E_CONFIG_INVALID,
                    phase="config_validation",
                    message="recompute_budget must be non-negative",
                    details={"recompute_budget": self.recompute_budget},
                )
            )
        if self.hot_capacity < 0:
            failures.append(
                Diagnostic(
                    code=E_CONFIG_INVALID,
                    phase="config_validation",
                    message="hot_capacity must be non-negative",
                    details={"hot_capacity": self.hot_capacity},
                )
            )
        return failures

