"""Versioned, JSON-serializable run configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .errors import DiagnosticError, ErrorCode

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RunConfig:
    seed: int = 20261119
    observer: str = "cie1931_2"
    normalization: str = "y100_reflector"
    reference_density_nm: float = 0.5
    quadrature: str = "trapezoid"
    sample_count: int = 12
    material_count: int = 8
    quant_bits: int = 8
    quant_range: tuple[float, float] = (0.0, 150.0)
    cond_threshold: float = 10000.0
    attribution_density_nm: float = 5.0
    attribution_drop_count: int = 1
    injections: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise DiagnosticError(
                code=ErrorCode.SNAPSHOT_VERSION,
                message="unsupported configuration schema version",
                details={
                    "expected": SCHEMA_VERSION,
                    "actual": self.schema_version,
                },
            )
        if self.sample_count < 3:
            raise DiagnosticError(
                code=ErrorCode.CONFIG_INVALID,
                message="at least three sample pairs are required",
                details={"sample_count": self.sample_count},
            )
        if self.material_count < 1:
            raise DiagnosticError(
                code=ErrorCode.CONFIG_INVALID,
                message="at least one material is required",
                details={"material_count": self.material_count},
            )
        if self.reference_density_nm <= 0 or self.attribution_density_nm <= 0:
            raise DiagnosticError(
                code=ErrorCode.CONFIG_INVALID,
                message="integration densities must be positive",
                details={
                    "reference_density_nm": self.reference_density_nm,
                    "attribution_density_nm": self.attribution_density_nm,
                },
            )
        if not 0.0 <= self.quant_range[0] < self.quant_range[1]:
            raise DiagnosticError(
                code=ErrorCode.QUANT_RANGE,
                message="quantisation range must be finite and increasing",
                details={"quant_range": list(self.quant_range)},
            )
        if not 1 <= self.quant_bits <= 24:
            raise DiagnosticError(
                code=ErrorCode.QUANT_RANGE,
                message="quantisation bit depth is outside the supported range",
                details={"quant_bits": self.quant_bits, "allowed": [1, 24]},
            )
        if self.cond_threshold <= 1.0:
            raise DiagnosticError(
                code=ErrorCode.CONFIG_INVALID,
                message="condition threshold must exceed one",
                details={"cond_threshold": self.cond_threshold},
            )
        if self.attribution_drop_count < 0:
            raise DiagnosticError(
                code=ErrorCode.CONFIG_INVALID,
                message="attribution_drop_count must be non-negative",
                details={"attribution_drop_count": self.attribution_drop_count},
            )
        if self.attribution_drop_count >= self.sample_count - 2:
            raise DiagnosticError(
                code=ErrorCode.CONFIG_INVALID,
                message="attribution distribution subset must retain at least three samples",
                details={
                    "sample_count": self.sample_count,
                    "attribution_drop_count": self.attribution_drop_count,
                },
            )

    @property
    def quant_step(self) -> float:
        levels = (1 << self.quant_bits) - 1
        return (self.quant_range[1] - self.quant_range[0]) / levels

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["quant_range"] = list(self.quant_range)
        data["injections"] = list(self.injections)
        data["quant_step"] = self.quant_step
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunConfig:
        values = dict(data)
        values.pop("quant_step", None)
        values["quant_range"] = tuple(values.get("quant_range", (0.0, 150.0)))
        values["injections"] = tuple(values.get("injections", ()))
        allowed = {
            "seed",
            "observer",
            "normalization",
            "reference_density_nm",
            "quadrature",
            "sample_count",
            "material_count",
            "quant_bits",
            "quant_range",
            "cond_threshold",
            "attribution_density_nm",
            "attribution_drop_count",
            "injections",
            "schema_version",
        }
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise DiagnosticError(
                code=ErrorCode.CONFIG_INVALID,
                message="unknown configuration keys",
                details={"keys": unknown},
            )
        return cls(**values)


NORMALIZATIONS = ("y100_reflector", "unity")
QUADRATURES = ("trapezoid",)
