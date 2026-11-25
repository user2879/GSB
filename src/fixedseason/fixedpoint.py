"""Integer normal equations and deterministic fixed-point elimination."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, localcontext
from typing import Any, Sequence

from .errors import E_PIVOT_UNSTABLE, FitError
from .jsonutil import fingerprint
from .model import assert_width


def _round_div_half_away(numerator: int, denominator: int) -> tuple[int, int]:
    """Return rounded quotient and remainder n - q*d, ties away from zero."""

    if denominator == 0:
        raise ZeroDivisionError("integer division by zero")
    sign = -1 if (numerator < 0) ^ (denominator < 0) else 1
    numerator_abs = abs(numerator)
    denominator_abs = abs(denominator)
    quotient, remainder = divmod(numerator_abs, denominator_abs)
    if remainder * 2 >= denominator_abs:
        quotient += 1
    rounded = sign * quotient
    return rounded, numerator - rounded * denominator


def _round_div_power_of_two(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    return _round_div_half_away(value, 1 << shift)[0]


def _bit_length(value: int) -> int:
    return max(1, abs(value).bit_length())


def _rounding_direction(remainder: int) -> str:
    if remainder == 0:
        return "exact"
    return "down" if remainder > 0 else "up"


@dataclass(frozen=True)
class NormalEquationResult:
    matrix: tuple[tuple[int, ...], ...]
    rhs: tuple[int, ...]
    raw_matrix: tuple[tuple[int, ...], ...]
    raw_rhs: tuple[int, ...]
    regularization_integer: int
    accumulator_max_bits: int
    downshift: int
    accumulation_order: tuple[str, ...]
    artifact_fingerprint: str

    def as_payload(self) -> dict[str, Any]:
        return {
            "matrix": [list(row) for row in self.matrix],
            "rhs": list(self.rhs),
            "raw_matrix": [list(row) for row in self.raw_matrix],
            "raw_rhs": list(self.raw_rhs),
            "regularization_integer": self.regularization_integer,
            "accumulator_max_bits": self.accumulator_max_bits,
            "downshift": self.downshift,
            "accumulation_order": list(self.accumulation_order),
            "artifact_fingerprint": self.artifact_fingerprint,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "NormalEquationResult":
        return cls(
            matrix=tuple(tuple(int(value) for value in row) for row in payload["matrix"]),
            rhs=tuple(int(value) for value in payload["rhs"]),
            raw_matrix=tuple(tuple(int(value) for value in row) for row in payload["raw_matrix"]),
            raw_rhs=tuple(int(value) for value in payload["raw_rhs"]),
            regularization_integer=int(payload["regularization_integer"]),
            accumulator_max_bits=int(payload["accumulator_max_bits"]),
            downshift=int(payload["downshift"]),
            accumulation_order=tuple(payload["accumulation_order"]),
            artifact_fingerprint=str(payload["artifact_fingerprint"]),
        )


def prefix_normal(result: NormalEquationResult, dimension: int) -> NormalEquationResult:
    return NormalEquationResult(
        matrix=tuple(tuple(row[:dimension]) for row in result.matrix[:dimension]),
        rhs=tuple(result.rhs[:dimension]),
        raw_matrix=tuple(tuple(row[:dimension]) for row in result.raw_matrix[:dimension]),
        raw_rhs=tuple(result.raw_rhs[:dimension]),
        regularization_integer=result.regularization_integer,
        accumulator_max_bits=result.accumulator_max_bits,
        downshift=result.downshift,
        accumulation_order=result.accumulation_order,
        artifact_fingerprint=fingerprint(
            {
                "prefix_dimension": dimension,
                "parent": result.artifact_fingerprint,
            }
        ),
    )


def accumulate_normal_equations(
    integer_matrix: Sequence[Sequence[int]],
    rhs_values: Sequence[int],
    config: Any,
) -> NormalEquationResult:
    """Accumulate A^T A and A^T b in the mandated deterministic order.

    Raw integer accumulators use arbitrary precision.  To make the actual
    elimination respect the configured word width, all entries are rounded
    down by one common power-of-two shift when needed.  A common shift leaves
    the linear system's solution scale unchanged and is recorded in metadata.
    """

    if len(integer_matrix) != len(rhs_values):
        raise ValueError("design matrix row count does not match rhs")
    dimension = len(integer_matrix[0]) if integer_matrix else 0
    if dimension == 0:
        raise ValueError("design matrix is empty")
    matrix = [[0 for _ in range(dimension)] for _ in range(dimension)]
    rhs = [0 for _ in range(dimension)]

    for time_index in range(len(integer_matrix)):
        row = integer_matrix[time_index]
        target = rhs_values[time_index]
        for column in range(dimension):
            rhs[column] += row[column] * target
            for inner in range(column, dimension):
                product = row[column] * row[inner]
                matrix[column][inner] += product
                if inner != column:
                    matrix[inner][column] += product

    scale = int(config.quantization_scale)
    with localcontext() as context:
        context.prec = 60
        regularization_decimal = Decimal(str(config.regularization_strength)) * Decimal(scale * scale)
        regularization_integer = int(
            regularization_decimal.to_integral_value(rounding=ROUND_HALF_UP)
        )
    for index in range(1, dimension):
        matrix[index][index] += regularization_integer

    raw_matrix = tuple(tuple(value for value in row) for row in matrix)
    raw_rhs = tuple(rhs)
    maximum_bits = max(
        max(_bit_length(value) for row in matrix for value in row),
        max((_bit_length(value) for value in rhs), default=1),
        _bit_length(regularization_integer),
    )
    usable_bits = max(2, int(config.fixed_point_bits) - 2)
    downshift = max(0, maximum_bits - usable_bits)
    if downshift:
        matrix = [
            [_round_div_power_of_two(value, downshift) for value in row]
            for row in matrix
        ]
        rhs = [_round_div_power_of_two(value, downshift) for value in rhs]
    else:
        matrix = [list(row) for row in matrix]
        rhs = list(rhs)

    for row in matrix:
        for value in row:
            assert_width(value, int(config.fixed_point_bits), "normal-equation entry")
    for value in rhs:
        assert_width(value, int(config.fixed_point_bits), "normal-equation rhs")

    order = (
        "time_index_ascending",
        "column_index_ascending",
        "inner_column_index_ascending",
        "regularization_diagonal_except_intercept",
        "common_binary_downshift_if_needed",
    )
    normalized_matrix = tuple(tuple(value for value in row) for row in matrix)
    normalized_rhs = tuple(rhs)
    artifact_fp = fingerprint(
        {
            "matrix": [list(row) for row in normalized_matrix],
            "rhs": list(normalized_rhs),
            "raw_matrix": [list(row) for row in raw_matrix],
            "raw_rhs": list(raw_rhs),
            "regularization_integer": regularization_integer,
            "downshift": downshift,
            "accumulation_order": order,
        }
    )
    return NormalEquationResult(
        matrix=normalized_matrix,
        rhs=normalized_rhs,
        raw_matrix=raw_matrix,
        raw_rhs=raw_rhs,
        regularization_integer=regularization_integer,
        accumulator_max_bits=maximum_bits,
        downshift=downshift,
        accumulation_order=order,
        artifact_fingerprint=artifact_fp,
    )


@dataclass(frozen=True)
class EliminationResult:
    coefficients: tuple[int, ...]
    effective_dimension: int
    effective_harmonic_order: int
    trend_disabled: bool
    reductions: tuple[dict[str, Any], ...]
    pivots: tuple[dict[str, Any], ...]
    row_operations: tuple[dict[str, Any], ...]
    back_substitution: tuple[dict[str, Any], ...]
    upper_matrix: tuple[tuple[int, ...], ...]
    upper_rhs: tuple[int, ...]
    float_back_substitution: tuple[float, ...]
    artifact_fingerprint: str

    def as_payload(self) -> dict[str, Any]:
        return {
            "coefficients": list(self.coefficients),
            "effective_dimension": self.effective_dimension,
            "effective_harmonic_order": self.effective_harmonic_order,
            "trend_disabled": self.trend_disabled,
            "reductions": list(self.reductions),
            "pivots": list(self.pivots),
            "row_operations": list(self.row_operations),
            "back_substitution": list(self.back_substitution),
            "upper_matrix": [list(row) for row in self.upper_matrix],
            "upper_rhs": list(self.upper_rhs),
            "float_back_substitution": list(self.float_back_substitution),
            "artifact_fingerprint": self.artifact_fingerprint,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "EliminationResult":
        return cls(
            coefficients=tuple(int(value) for value in payload["coefficients"]),
            effective_dimension=int(payload["effective_dimension"]),
            effective_harmonic_order=int(payload["effective_harmonic_order"]),
            trend_disabled=bool(payload["trend_disabled"]),
            reductions=tuple(dict(item) for item in payload["reductions"]),
            pivots=tuple(dict(item) for item in payload["pivots"]),
            row_operations=tuple(dict(item) for item in payload["row_operations"]),
            back_substitution=tuple(dict(item) for item in payload["back_substitution"]),
            upper_matrix=tuple(tuple(int(value) for value in row) for row in payload["upper_matrix"]),
            upper_rhs=tuple(int(value) for value in payload["upper_rhs"]),
            float_back_substitution=tuple(float(value) for value in payload["float_back_substitution"]),
            artifact_fingerprint=str(payload["artifact_fingerprint"]),
        )


@dataclass
class _Attempt:
    success: bool
    coefficients: tuple[int, ...] = ()
    pivots: tuple[dict[str, Any], ...] = ()
    row_operations: tuple[dict[str, Any], ...] = ()
    back_substitution: tuple[dict[str, Any], ...] = ()
    upper_matrix: tuple[tuple[int, ...], ...] = ()
    upper_rhs: tuple[int, ...] = ()
    float_back_substitution: tuple[float, ...] = ()
    failed_step: int | None = None
    failed_pivot_abs: int | None = None
    failed_pivot_candidates: tuple[dict[str, Any], ...] = ()


def _float_back_substitution(
    upper: Sequence[Sequence[int]], rhs: Sequence[int]
) -> tuple[float, ...]:
    dimension = len(upper)
    result = [0.0] * dimension
    for row in range(dimension - 1, -1, -1):
        if upper[row][row] == 0:
            raise ZeroDivisionError("zero diagonal in float back-substitution")
        value = float(rhs[row])
        for column in range(row + 1, dimension):
            value -= float(upper[row][column]) * result[column]
        result[row] = value / float(upper[row][row])
    return tuple(result)


def _attempt_elimination(
    normal: NormalEquationResult,
    *,
    bits: int,
    pivot_threshold: int,
) -> _Attempt:
    dimension = len(normal.rhs)
    upper = [list(row) for row in normal.matrix]
    rhs = list(normal.rhs)
    pivots: list[dict[str, Any]] = []
    row_operations: list[dict[str, Any]] = []
    back_substitution: list[dict[str, Any]] = []

    for step in range(dimension):
        candidates = [
            {
                "row": row,
                "pivot_column": step,
                "absolute_value": abs(upper[row][step]),
                "value": upper[row][step],
            }
            for row in range(step, dimension)
        ]
        selected = max(
            range(step, dimension),
            key=lambda row: (abs(upper[row][step]), -row),
        )
        pivot_value = upper[selected][step]
        pivot_abs = abs(pivot_value)
        if pivot_abs < pivot_threshold:
            return _Attempt(
                success=False,
                failed_step=step,
                failed_pivot_abs=pivot_abs,
                pivots=tuple(pivots),
                row_operations=tuple(row_operations),
                failed_pivot_candidates=tuple(candidates),
            )
        if selected != step:
            upper[step], upper[selected] = upper[selected], upper[step]
            rhs[step], rhs[selected] = rhs[selected], rhs[step]
        sign_normalized = False
        if upper[step][step] < 0:
            upper[step] = [-value for value in upper[step]]
            rhs[step] = -rhs[step]
            sign_normalized = True
            for value in upper[step]:
                assert_width(value, bits, "elimination pivot row")
            assert_width(rhs[step], bits, "elimination pivot rhs")
        pivot_value = upper[step][step]
        pivots.append(
            {
                "step": step,
                "column": step,
                "selected_row": selected,
                "swapped": selected != step,
                "sign_normalized": sign_normalized,
                "pivot_value": pivot_value,
                "pivot_abs": abs(pivot_value),
                "threshold": pivot_threshold,
                "candidates": candidates,
            }
        )
        for row in range(step + 1, dimension):
            numerator = upper[row][step]
            quotient, remainder = _round_div_half_away(numerator, pivot_value)
            row_operations.append(
                {
                    "step": step,
                    "row": row,
                    "numerator": numerator,
                    "pivot": pivot_value,
                    "multiplier": quotient,
                    "intermediate_remainder": remainder,
                    "rounding_direction": _rounding_direction(remainder),
                }
            )
            if quotient == 0:
                continue
            for column in range(step, dimension):
                upper[row][column] -= quotient * upper[step][column]
                assert_width(upper[row][column], bits, "elimination row entry")
            rhs[row] -= quotient * rhs[step]
            assert_width(rhs[row], bits, "elimination row rhs")

    coefficients = [0] * dimension
    for row in range(dimension - 1, -1, -1):
        constant = rhs[row]
        for column in range(row + 1, dimension):
            constant -= upper[row][column] * coefficients[column]
            assert_width(constant, bits, "back-substitution partial numerator")
        quotient, remainder = _round_div_half_away(constant, upper[row][row])
        coefficients[row] = quotient
        assert_width(quotient, bits, "coefficient")
        back_substitution.append(
            {
                "row": row,
                "numerator": constant,
                "pivot": upper[row][row],
                "coefficient": quotient,
                "intermediate_remainder": remainder,
                "rounding_direction": _rounding_direction(remainder),
            }
        )
    float_coefficients = _float_back_substitution(upper, rhs)
    return _Attempt(
        success=True,
        coefficients=tuple(coefficients),
        pivots=tuple(pivots),
        row_operations=tuple(row_operations),
        back_substitution=tuple(back_substitution),
        upper_matrix=tuple(tuple(value for value in row) for row in upper),
        upper_rhs=tuple(rhs),
        float_back_substitution=float_coefficients,
    )


def solve_fixed_point(
    normal: NormalEquationResult,
    config: Any,
) -> EliminationResult:
    """Solve through integer elimination with deterministic order reduction."""

    harmonic_order = int(config.harmonic_order)
    trend_enabled = True
    reductions: list[dict[str, Any]] = []
    attempt_number = 0
    while True:
        attempt_number += 1
        dimension = 1 + (1 if trend_enabled else 0) + 2 * harmonic_order
        current = prefix_normal(normal, dimension)
        attempt = _attempt_elimination(
            current,
            bits=int(config.fixed_point_bits),
            pivot_threshold=int(config.pivot_threshold),
        )
        if attempt.success:
            artifact_fp = fingerprint(
                {
                    "upper_matrix": [list(row) for row in attempt.upper_matrix],
                    "upper_rhs": list(attempt.upper_rhs),
                    "coefficients": list(attempt.coefficients),
                    "effective_harmonic_order": harmonic_order,
                    "trend_enabled": trend_enabled,
                    "reductions": reductions,
                    "pivots": attempt.pivots,
                    "row_operations": attempt.row_operations,
                    "back_substitution": attempt.back_substitution,
                }
            )
            return EliminationResult(
                coefficients=attempt.coefficients,
                effective_dimension=dimension,
                effective_harmonic_order=harmonic_order,
                trend_disabled=not trend_enabled,
                reductions=tuple(reductions),
                pivots=attempt.pivots,
                row_operations=attempt.row_operations,
                back_substitution=attempt.back_substitution,
                upper_matrix=attempt.upper_matrix,
                upper_rhs=attempt.upper_rhs,
                float_back_substitution=attempt.float_back_substitution,
                artifact_fingerprint=artifact_fp,
            )

        if harmonic_order > 0:
            reductions.append(
                {
                    "attempt": attempt_number,
                    "reason": "PIVOT_BELOW_THRESHOLD",
                    "failed_step": attempt.failed_step,
                    "pivot_abs": attempt.failed_pivot_abs,
                    "failed_pivot_candidates": list(attempt.failed_pivot_candidates),
                    "pivot_trace_before_failure": list(attempt.pivots),
                    "row_operation_trace_before_failure": list(attempt.row_operations),
                    "threshold": int(config.pivot_threshold),
                    "previous_harmonic_order": harmonic_order,
                    "new_harmonic_order": harmonic_order - 1,
                    "rule": "drop_highest_harmonic_order_and_resolve",
                }
            )
            harmonic_order -= 1
            continue
        if trend_enabled:
            reductions.append(
                {
                    "attempt": attempt_number,
                    "reason": "PIVOT_BELOW_THRESHOLD",
                    "failed_step": attempt.failed_step,
                    "pivot_abs": attempt.failed_pivot_abs,
                    "failed_pivot_candidates": list(attempt.failed_pivot_candidates),
                    "pivot_trace_before_failure": list(attempt.pivots),
                    "row_operation_trace_before_failure": list(attempt.row_operations),
                    "threshold": int(config.pivot_threshold),
                    "previous_harmonic_order": 0,
                    "new_harmonic_order": 0,
                    "trend_before": True,
                    "trend_after": False,
                    "rule": "disable_trend_and_resolve_intercept_only",
                }
            )
            trend_enabled = False
            continue
        raise FitError(
            E_PIVOT_UNSTABLE,
            "intercept-only system still has a pivot below threshold",
            failed_step=attempt.failed_step,
            pivot_abs=attempt.failed_pivot_abs,
            threshold=int(config.pivot_threshold),
        )
