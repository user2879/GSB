from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_EVEN, ROUND_HALF_UP, localcontext
from typing import Any

from .util import decimal_text

ALLOWED_FRACTIONAL_BITS = (8, 12, 16, 20)
ALLOWED_QUANTIZATION = (1, 2, 4, 8)
ALLOWED_ROUNDING = ("half_up", "half_even", "down")

_ROUNDING_MODES = {
    "half_up": ROUND_HALF_UP,
    "half_even": ROUND_HALF_EVEN,
    "down": ROUND_DOWN,
}


@dataclass(frozen=True)
class FixedDomain:
    fractional_bits: int
    quantization: int = 1
    rounding: str = "half_up"

    @property
    def scale(self) -> int:
        return 1 << self.fractional_bits

    @property
    def one(self) -> int:
        return self._quantize_units_decimal(Decimal(self.scale))

    def fingerprint(self) -> dict[str, Any]:
        return {
            "fractional_bits": self.fractional_bits,
            "quantization": self.quantization,
            "rounding": self.rounding,
            "scale": self.scale,
        }

    def quantize_units(self, raw_units: Decimal | int) -> int:
        return self._quantize_units_decimal(Decimal(raw_units))

    def _quantize_units_decimal(self, raw_units: Decimal) -> int:
        mode = _ROUNDING_MODES[self.rounding]
        with localcontext() as ctx:
            ctx.prec = 80
            step = Decimal(self.quantization)
            quanta = (Decimal(raw_units) / step).to_integral_value(rounding=mode)
            return int(quanta) * self.quantization

    def value_to_units(self, value: Decimal | int | str) -> int:
        with localcontext() as ctx:
            ctx.prec = 80
            raw = Decimal(value) * Decimal(self.scale)
            return self.quantize_units(raw)

    def ratio_to_units(self, value: Decimal | int | str) -> int:
        return self.value_to_units(value)

    def add(self, left_units: int, right_units: int) -> int:
        return self.quantize_units(left_units + right_units)

    def mul(self, left_units: int, right_units: int) -> int:
        with localcontext() as ctx:
            ctx.prec = 80
            raw = Decimal(left_units) * Decimal(right_units) / Decimal(self.scale)
            return self.quantize_units(raw)

    def div(self, numerator_units: int, denominator_units: int) -> int:
        if denominator_units == 0:
            return 0
        with localcontext() as ctx:
            ctx.prec = 80
            raw = (
                Decimal(numerator_units)
                * Decimal(self.scale)
                / Decimal(denominator_units)
            )
            return self.quantize_units(raw)

    def pow_ratio(self, base_units: int, exponent: int) -> int:
        if exponent < 0:
            raise ValueError("negative exponent")
        result = self.one
        factor = base_units
        power = exponent
        while power:
            if power & 1:
                result = self.mul(result, factor)
            power >>= 1
            if power:
                factor = self.mul(factor, factor)
        return result

    def units_text(self, units: int) -> str:
        return decimal_text(Decimal(units) / Decimal(self.scale))

    def describe(self) -> dict[str, Any]:
        return {
            "fractional_bits": self.fractional_bits,
            "quantization": self.quantization,
            "rounding": self.rounding,
            "scale": self.scale,
            "one_units": self.one,
        }
