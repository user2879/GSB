from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence

import numpy as np

from .errors import E_BLOCK_INVALID, E_CONFIG_INVALID, E_TOLERANCE_INVALID, BusinessError


LoopOrder = Literal["ijk", "ikj", "jik"]
VALID_LOOP_ORDERS: tuple[LoopOrder, ...] = ("ijk", "ikj", "jik")


@dataclass(frozen=True)
class BlockShape:
    bm: int
    bn: int
    bk: int

    def validate(self, stage: str = "kernel_block_validation") -> None:
        values = (self.bm, self.bn, self.bk)
        if any(not isinstance(value, (int, np.integer)) or int(value) <= 0 for value in values):
            raise BusinessError(
                code=E_BLOCK_INVALID,
                stage=stage,
                message="Block dimensions must be positive integers.",
                shape={"block_shape": [self.bm, self.bn, self.bk]},
                details={"block_shape": [self.bm, self.bn, self.bk]},
            )

    def as_list(self) -> list[int]:
        return [int(self.bm), int(self.bn), int(self.bk)]

    def as_dict(self) -> dict[str, int]:
        return {"BM": int(self.bm), "BN": int(self.bn), "BK": int(self.bk)}

    @classmethod
    def from_any(cls, value: "BlockShape | Sequence[int] | dict[str, int]") -> "BlockShape":
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(int(value["BM"]), int(value["BN"]), int(value["BK"]))
        values = list(value)
        if len(values) != 3:
            raise BusinessError(
                code=E_BLOCK_INVALID,
                stage="kernel_block_validation",
                message="Block shape must contain exactly BM, BN, and BK.",
                shape={"block_shape": values},
            )
        return cls(int(values[0]), int(values[1]), int(values[2]))


@dataclass(frozen=True)
class AccuracyTolerance:
    atol: float
    rtol: float

    def validate(self, stage: str = "error_analysis") -> None:
        values = (self.atol, self.rtol)
        invalid = any(not np.isfinite(value) or float(value) < 0.0 for value in values)
        both_zero = float(self.atol) == 0.0 and float(self.rtol) == 0.0
        if invalid or both_zero:
            raise BusinessError(
                code=E_TOLERANCE_INVALID,
                stage=stage,
                message="Tolerances must be finite, non-negative, and not both zero.",
                details={"atol": self.atol, "rtol": self.rtol, "both_zero": both_zero},
            )

    def threshold(self, reference: float) -> float:
        return float(self.atol) + float(self.rtol) * abs(float(reference))

    def as_dict(self) -> dict[str, float]:
        return {"atol": float(self.atol), "rtol": float(self.rtol)}


@dataclass(frozen=True)
class SampleSpec:
    name: str
    category: str
    seed: int
    M: int
    N: int
    K: int
    batch_size: int = 1
    batched: bool | None = None
    alpha: float = 1.0
    beta: float = 0.0
    special_fraction: float = 0.0
    description: str = ""

    @property
    def is_bmm(self) -> bool:
        if self.batched is not None:
            return bool(self.batched)
        return int(self.batch_size) > 1

    def validate(self) -> None:
        if not self.name:
            raise BusinessError(E_CONFIG_INVALID, "config_validation", "Sample name is required.")
        if min(int(self.M), int(self.N), int(self.K), int(self.batch_size)) <= 0:
            raise BusinessError(
                E_CONFIG_INVALID,
                "config_validation",
                "Sample dimensions and batch size must be positive.",
                shape=self.shape_dict(),
            )
        if self.batched is False and int(self.batch_size) != 1:
            raise BusinessError(
                E_CONFIG_INVALID,
                "config_validation",
                "A non-batched sample must have batch_size=1; use batched=true for BMM.",
                shape=self.shape_dict(),
            )
        if not np.isfinite(self.alpha) or not np.isfinite(self.beta):
            raise BusinessError(
                E_CONFIG_INVALID,
                "config_validation",
                "alpha and beta must be finite.",
                details={"alpha": self.alpha, "beta": self.beta},
            )
        if not 0.0 <= float(self.special_fraction) <= 1.0:
            raise BusinessError(
                E_CONFIG_INVALID,
                "config_validation",
                "special_fraction must be in [0, 1].",
            )

    def shape_dict(self) -> dict[str, Any]:
        if self.is_bmm:
            return {
                "A": [int(self.batch_size), int(self.M), int(self.K)],
                "B": [int(self.batch_size), int(self.K), int(self.N)],
                "C": [int(self.batch_size), int(self.M), int(self.N)],
            }
        return {
            "A": [int(self.M), int(self.K)],
            "B": [int(self.K), int(self.N)],
            "C": [int(self.M), int(self.N)],
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "seed": int(self.seed),
            "M": int(self.M),
            "N": int(self.N),
            "K": int(self.K),
            "batch_size": int(self.batch_size),
            "batched": self.is_bmm,
            "alpha": float(self.alpha),
            "beta": float(self.beta),
            "special_fraction": float(self.special_fraction),
            "description": self.description,
            "shapes": self.shape_dict(),
        }


@dataclass(frozen=True)
class InjectionSpec:
    injection_id: str
    kind: Literal["nonfinite", "shape_mismatch", "block_invalid", "tolerance_invalid"]
    sample: str
    target: str | None = None
    index: tuple[int, ...] | None = None
    value: str | None = None
    block_shape: tuple[int, int, int] | None = None
    atol: float | None = None
    rtol: float | None = None
    expected_code: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "injection_id": self.injection_id,
            "kind": self.kind,
            "sample": self.sample,
            "target": self.target,
            "index": list(self.index) if self.index is not None else None,
            "value": self.value,
            "block_shape": list(self.block_shape) if self.block_shape is not None else None,
            "atol": self.atol,
            "rtol": self.rtol,
            "expected_code": self.expected_code,
        }


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    samples: tuple[SampleSpec, ...]
    block_shapes: tuple[BlockShape, ...]
    loop_orders: tuple[LoopOrder, ...]
    tolerance: AccuracyTolerance
    layer_exceed_limit: float = 0.05
    layer_count: int = 4
    top_k_contributors: int = 5
    max_workers: int = 4
    serial_workers: int = 1
    report_path: str = "artifacts/precision_report.json"
    injections: tuple[InjectionSpec, ...] = ()

    def validate(self) -> None:
        if self.seed < 0:
            raise BusinessError(E_CONFIG_INVALID, "config_validation", "seed must be non-negative.")
        if len(self.block_shapes) < 2:
            raise BusinessError(
                E_CONFIG_INVALID,
                "config_validation",
                "At least two block shapes are required for comparison.",
            )
        if len(set(self.loop_orders)) < 2:
            raise BusinessError(
                E_CONFIG_INVALID,
                "config_validation",
                "At least two distinct loop orders are required for comparison.",
            )
        if any(order not in VALID_LOOP_ORDERS for order in self.loop_orders):
            raise BusinessError(
                E_CONFIG_INVALID,
                "config_validation",
                "Loop orders must be selected from ijk, ikj, and jik.",
                details={"loop_orders": list(self.loop_orders)},
            )
        if self.max_workers <= 0 or self.serial_workers <= 0:
            raise BusinessError(E_CONFIG_INVALID, "config_validation", "Worker counts must be positive.")
        if self.top_k_contributors <= 0:
            raise BusinessError(
                E_CONFIG_INVALID,
                "config_validation",
                "top_k_contributors must be positive.",
            )
        if not np.isfinite(self.layer_exceed_limit) or self.layer_exceed_limit < 0.0:
            raise BusinessError(
                E_CONFIG_INVALID,
                "config_validation",
                "layer_exceed_limit must be finite and non-negative.",
            )
        if self.layer_count <= 0:
            raise BusinessError(E_CONFIG_INVALID, "config_validation", "layer_count must be positive.")
        for block in self.block_shapes:
            block.validate(stage="config_validation")
        self.tolerance.validate(stage="config_validation")
        names: set[str] = set()
        for sample in self.samples:
            sample.validate()
            if sample.name in names:
                raise BusinessError(
                    E_CONFIG_INVALID,
                    "config_validation",
                    f"Duplicate sample name: {sample.name}",
                )
            names.add(sample.name)
        injection_ids: set[str] = set()
        for injection in self.injections:
            if injection.injection_id in injection_ids:
                raise BusinessError(
                    E_CONFIG_INVALID,
                    "config_validation",
                    f"Duplicate injection id: {injection.injection_id}",
                )
            injection_ids.add(injection.injection_id)
            if injection.sample not in names:
                raise BusinessError(
                    E_CONFIG_INVALID,
                    "config_validation",
                    f"Injection references unknown sample: {injection.sample}",
                )

    def combo_ids(self, sample: SampleSpec) -> list[str]:
        return [
            f"{sample.name}|BM{block.bm}-BN{block.bn}-BK{block.bk}|{order}"
            for block in self.block_shapes
            for order in self.loop_orders
        ]

    def as_dict(self) -> dict[str, Any]:
        return {
            "seed": int(self.seed),
            "samples": [sample.as_dict() for sample in self.samples],
            "block_shapes": [block.as_dict() for block in self.block_shapes],
            "loop_orders": list(self.loop_orders),
            "tolerance": self.tolerance.as_dict(),
            "layer_exceed_limit": float(self.layer_exceed_limit),
            "layer_count": int(self.layer_count),
            "top_k_contributors": int(self.top_k_contributors),
            "max_workers": int(self.max_workers),
            "serial_workers": int(self.serial_workers),
            "report_path": self.report_path,
            "injections": [injection.as_dict() for injection in self.injections],
        }
