"""Runtime and fixed-point configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .io_utils import fingerprint


QUANT_LEVEL_SHIFT = {0: -3, 1: 0, 2: 3}
QUANT_LEVEL_NAMES = {0: "coarse", 1: "standard", 2: "fine"}


@dataclass(frozen=True)
class QuantConfig:
    decimal_bits: int = 10
    quant_level: int = 1
    total_width: int = 32

    def __post_init__(self) -> None:
        if not isinstance(self.decimal_bits, int) or isinstance(self.decimal_bits, bool):
            raise ValueError("decimal_bits must be an integer")
        if not isinstance(self.quant_level, int) or isinstance(self.quant_level, bool):
            raise ValueError("quant_level must be an integer")
        if not isinstance(self.total_width, int) or isinstance(self.total_width, bool):
            raise ValueError("total_width must be an integer")
        if self.decimal_bits < 2 or self.decimal_bits > 20:
            raise ValueError("decimal_bits must be in [2, 20]")
        if self.quant_level not in QUANT_LEVEL_SHIFT:
            raise ValueError("quant_level must be one of 0, 1, 2")
        if self.total_width < 8 or self.total_width > 64:
            raise ValueError("total_width must be in [8, 64]")

    @property
    def effective_bits(self) -> int:
        return max(1, min(self.total_width - 2, self.decimal_bits + QUANT_LEVEL_SHIFT[self.quant_level]))

    @property
    def scale(self) -> int:
        return 1 << self.effective_bits

    @property
    def quantization_step(self) -> str:
        return f"1/2^{self.effective_bits}"

    @property
    def clamp_min(self) -> int:
        return -(1 << (self.total_width - 1))

    @property
    def clamp_max(self) -> int:
        return (1 << (self.total_width - 1)) - 1

    def fingerprint(self) -> str:
        return fingerprint(asdict(self))


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 335
    corpus_size: int = 24
    interaction_budget_ms: int = 50
    feature_dim: int = 6
    bucket_count: int = 16
    model_version: str = "ner-local-1.0.0"
    snapshot_format_version: int = 1
    quant: QuantConfig = QuantConfig()

    def fingerprint(self) -> str:
        value: dict[str, Any] = asdict(self)
        return fingerprint(value)


def make_config(data: dict[str, Any] | None = None) -> RuntimeConfig:
    data = dict(data or {})
    quant_data = data.pop("quant", {})
    if "decimal_bits" in data and "decimal_bits" not in quant_data:
        quant_data["decimal_bits"] = data.pop("decimal_bits")
    if "quant_level" in data and "quant_level" not in quant_data:
        quant_data["quant_level"] = data.pop("quant_level")
    if "total_width" in data and "total_width" not in quant_data:
        quant_data["total_width"] = data.pop("total_width")
    return RuntimeConfig(quant=QuantConfig(**quant_data), **data)
