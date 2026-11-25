"""Deterministic trend and harmonic basis construction."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, localcontext
from functools import lru_cache
from typing import Any, Iterable

from .errors import E_FIT_OVERFLOW, FitError
from .jsonutil import fingerprint
from .model import FitConfig, assert_width

_PI = Decimal(
    "3.1415926535897932384626433832795028841971693993751058209749445923078164"
)
_TWO_PI = _PI * 2
_PRECISION = 72


def _round_ratio_half_away(numerator: int, denominator: int) -> int:
    if denominator == 0:
        raise ZeroDivisionError
    sign = -1 if (numerator < 0) ^ (denominator < 0) else 1
    numerator = abs(numerator)
    denominator = abs(denominator)
    quotient, remainder = divmod(numerator, denominator)
    if remainder * 2 >= denominator:
        quotient += 1
    return sign * quotient


def _sin_cos_decimal(angle: Decimal) -> tuple[Decimal, Decimal]:
    with localcontext() as context:
        context.prec = _PRECISION
        value = angle % _TWO_PI
        if value > _PI:
            value -= _TWO_PI
        value_squared = value * value
        sin_value = value
        sin_term = value
        cos_value = Decimal(1)
        cos_term = Decimal(1)
        n = 1
        threshold = Decimal(1).scaleb(-(_PRECISION - 4))
        while True:
            sin_term *= -value_squared / Decimal((2 * n) * (2 * n + 1))
            cos_term *= -value_squared / Decimal((2 * n - 1) * (2 * n))
            sin_value += sin_term
            cos_value += cos_term
            n += 1
            if abs(sin_term) < threshold and abs(cos_term) < threshold:
                break
            if n > 100:
                raise ArithmeticError("trigonometric series failed to converge")
        return sin_value, cos_value


@lru_cache(maxsize=262_144)
def _scaled_trig(order: int, index: int, period: int, quantization_shift: int) -> tuple[int, int]:
    with localcontext() as context:
        context.prec = _PRECISION
        angle = (_TWO_PI * Decimal(order) * Decimal(index)) / Decimal(period)
        sine, cosine = _sin_cos_decimal(angle)
        scale = Decimal(1 << quantization_shift)
        cosine_scaled = int((cosine * scale).to_integral_value(rounding=ROUND_HALF_UP))
        sine_scaled = int((sine * scale).to_integral_value(rounding=ROUND_HALF_UP))
        return cosine_scaled, sine_scaled


@lru_cache(maxsize=262_144)
def _float_trig(order: int, index: int, period: int) -> tuple[float, float]:
    with localcontext() as context:
        context.prec = _PRECISION
        angle = (_TWO_PI * Decimal(order) * Decimal(index)) / Decimal(period)
        sine, cosine = _sin_cos_decimal(angle)
        return float(sine), float(cosine)


@dataclass(frozen=True)
class BasisResult:
    integer_matrix: tuple[tuple[int, ...], ...]
    float_matrix: tuple[tuple[float, ...], ...]
    columns: tuple[dict[str, Any], ...]
    degenerate_pure_trend: bool
    artifact_fingerprint: str
    period: int
    quantization_scale: int

    def as_payload(self) -> dict[str, Any]:
        return {
            "integer_matrix": [list(row) for row in self.integer_matrix],
            "float_matrix": [list(row) for row in self.float_matrix],
            "columns": list(self.columns),
            "degenerate_pure_trend": self.degenerate_pure_trend,
            "artifact_fingerprint": self.artifact_fingerprint,
            "period": self.period,
            "quantization_scale": self.quantization_scale,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "BasisResult":
        return cls(
            integer_matrix=tuple(tuple(int(value) for value in row) for row in payload["integer_matrix"]),
            float_matrix=tuple(tuple(float(value) for value in row) for row in payload["float_matrix"]),
            columns=tuple(dict(column) for column in payload["columns"]),
            degenerate_pure_trend=bool(payload["degenerate_pure_trend"]),
            artifact_fingerprint=str(payload["artifact_fingerprint"]),
            period=int(payload["period"]),
            quantization_scale=int(payload["quantization_scale"]),
        )


def column_metadata(config: FitConfig) -> tuple[dict[str, Any], ...]:
    columns: list[dict[str, Any]] = [
        {
            "index": 0,
            "name": "intercept",
            "kind": "trend",
            "harmonic_order": 0,
            "phase": "constant",
            "phase_radians": "0",
            "basis_formula": "Q",
        },
        {
            "index": 1,
            "name": "trend_slope",
            "kind": "trend",
            "harmonic_order": 0,
            "phase": "linear",
            "phase_radians": "n/a",
            "basis_formula": "round_half_away((2*t-(n-1))*Q/(2*period))",
        },
    ]
    for order in range(1, config.harmonic_order + 1):
        columns.append(
            {
                "index": len(columns),
                "name": f"harmonic_{order}_cos",
                "kind": "harmonic",
                "harmonic_order": order,
                "phase": "cos",
                "phase_radians": "0",
                "basis_formula": f"round_half_away(cos(2*pi*{order}*t/period)*Q)",
            }
        )
        columns.append(
            {
                "index": len(columns),
                "name": f"harmonic_{order}_sin",
                "kind": "harmonic",
                "harmonic_order": order,
                "phase": "sin",
                "phase_radians": str(_PI / 2),
                "basis_formula": f"round_half_away(sin(2*pi*{order}*t/period)*Q)",
            }
        )
    return tuple(columns)


def build_basis(length: int, config: FitConfig) -> BasisResult:
    scale = config.quantization_scale
    columns = column_metadata(config)
    integer_rows: list[tuple[int, ...]] = []
    float_rows: list[tuple[float, ...]] = []
    bound = config.fixed_point_bits

    for index in range(length):
        centered_numerator = 2 * index - (length - 1)
        trend_scaled = _round_ratio_half_away(
            centered_numerator * scale, 2 * config.period
        )
        int_row = [scale, trend_scaled]
        float_row = [1.0, centered_numerator / (2.0 * config.period)]
        for order in range(1, config.harmonic_order + 1):
            cosine, sine = _scaled_trig(order, index, config.period, config.quantization_shift)
            float_sine, float_cosine = _float_trig(order, index, config.period)
            int_row.extend((cosine, sine))
            float_row.extend((float_cosine, float_sine))
        for column, value in zip(columns, int_row):
            try:
                assert_width(value, bound, f"basis[{index},{column['name']}]")
            except FitError:
                raise FitError(
                    E_FIT_OVERFLOW,
                    f"basis value {value} exceeds signed fixed-point width {bound}",
                    row=index,
                    column=column["name"],
                    bits=bound,
                ) from None
        integer_rows.append(tuple(int_row))
        float_rows.append(tuple(float_row))

    metadata_columns = tuple(dict(column) for column in columns)
    metadata = {
        "rows": length,
        "columns": len(columns),
        "time_base": "zero_based_index_ascending",
        "period": config.period,
        "quantization_scale": scale,
        "quantization_shift": config.quantization_shift,
        "harmonic_order": config.harmonic_order,
        "degenerate_pure_trend": config.harmonic_order == 0,
        "column_order": [column["name"] for column in metadata_columns],
        "columns": list(metadata_columns),
    }
    artifact_fp = fingerprint(
        {
            "metadata": metadata,
            "integer_matrix": [list(row) for row in integer_rows],
            "float_matrix": [list(row) for row in float_rows],
        }
    )
    return BasisResult(
        integer_matrix=tuple(integer_rows),
        float_matrix=tuple(float_rows),
        columns=metadata_columns,
        degenerate_pure_trend=config.harmonic_order == 0,
        artifact_fingerprint=artifact_fp,
        period=config.period,
        quantization_scale=scale,
    )


def prefix_matrix(matrix: Iterable[Iterable[int | float]], columns: int) -> tuple[tuple[Any, ...], ...]:
    return tuple(tuple(row[:columns]) for row in matrix)


def column_names(config: FitConfig) -> tuple[str, ...]:
    return tuple(column["name"] for column in column_metadata(config))
