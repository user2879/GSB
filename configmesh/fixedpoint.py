"""Unified fixed-point quantization, integer MAC and error propagation.

Precision caliber
-----------------
A session selects ``grid_exp`` k so that the scaling factor is S = 10**k
(k = 0..9, guarded by E_GRID). Every real value x is quantized with explicit
*round-toward-zero* truncation into the q-unit integer::

    Q(x) = trunc(x * S)          (saturated to INT64)

All multiply-accumulate work for a derived item y = sum(c_i * x_i) is done on
integers in q-units and rescaled by S::

    q(y) = trunc( sum_i Q(c_i) * Q(x_i) / S )

Overflow saturation: every multiplication and the running sum are clamped to
the signed 64-bit range instead of wrapping.

Error model (documented and reproducible)
-----------------------------------------
Truncation gives |Q(z) - z*S| < 1 for each input and coefficient. Writing
C_i = Q(c_i), X_i = Q(x_i):

* per-term product error (q-units) <= |C_i| + |X_i| + 1
* rescale truncation of the accumulated sum (q-units) <= 1

So the conservative *integer* q-unit bound is::

    B_q = 1 + sum_i (|C_i| + |X_i| + 1)

and the real-value bound is B_real = B_q / S.  Each term's absolute-coefficient
weighted share is reported as a contribution percentage.

Two derived items a, b compare by a stable error-aware decision rule:
``gap = q(a) - q(b)`` is decisive only when |gap| exceeds the combined bound
``bound_a + bound_b + 1``; otherwise the result is INDETERMINATE and the
lexicographically smaller key wins as the documented deterministic tie break.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .errors import BusinessError, ErrorCode

INT64_MIN = -(2**63)
INT64_MAX = 2**63 - 1
MAX_GRID_EXP = 9


@dataclass(frozen=True)
class Grid:
    grid_exp: int

    @property
    def scale(self) -> int:
        return 10**self.grid_exp

    @staticmethod
    def create(grid_exp: int) -> "Grid":
        if not isinstance(grid_exp, int) or isinstance(grid_exp, bool):
            raise BusinessError(ErrorCode.E_GRID, "grid_exp must be an integer",
                                [{"grid_exp": grid_exp, "reason": "not an integer"}])
        if grid_exp < 0 or grid_exp > MAX_GRID_EXP:
            raise BusinessError(
                ErrorCode.E_GRID,
                f"grid_exp must be in [0, {MAX_GRID_EXP}] (scale 10**grid_exp)",
                [{"grid_exp": grid_exp, "reason": "out of supported range"}],
            )
        return Grid(grid_exp)


def validate_grid_param(grid_exp: object) -> Grid:
    if isinstance(grid_exp, bool) or not isinstance(grid_exp, int):
        raise BusinessError(ErrorCode.E_GRID, "illegal precision caliber parameter",
                            [{"grid_exp": grid_exp, "reason": "integer required"}])
    return Grid.create(grid_exp)


def saturate_int64(value: int) -> int:
    if value > INT64_MAX:
        return INT64_MAX
    if value < INT64_MIN:
        return INT64_MIN
    return value


def quantize(value: float, grid: Grid) -> int:
    """Round-toward-zero quantization with INT64 saturation."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BusinessError(ErrorCode.E_MANIFEST,
                            "numeric value required for quantization",
                            [{"value": value, "reason": "unsupported type"}])
    scaled = value * grid.scale
    truncated = int(scaled)  # int() truncates toward zero for floats
    return saturate_int64(truncated)


def integer_mul(a: int, b: int) -> int:
    return saturate_int64(a * b)


def derive_qvalue(quantized_coeffs: Mapping[str, int],
                  quantized_inputs: Mapping[str, int],
                  grid: Grid) -> int:
    """q(y) = trunc(sum(Q(c_i)*Q(x_i)) / S), all arithmetic integer-based."""
    total = 0
    for key in sorted(quantized_coeffs):
        product = integer_mul(quantized_coeffs[key], quantized_inputs[key])
        total = saturate_int64(total + product)
    rescaled = int(total / grid.scale)  # truncation toward zero
    return saturate_int64(rescaled)


def error_bound(quantized_coeffs: Mapping[str, int],
                quantized_inputs: Mapping[str, int],
                grid: Grid) -> dict:
    """Absolute-coefficient-weighted quantization error upper bound."""
    terms: list[dict] = []
    bound_q = 1  # rescale truncation contribution
    for key in sorted(quantized_coeffs):
        coeff = quantized_coeffs[key]
        qinput = quantized_inputs[key]
        term_bound = abs(coeff) + abs(qinput) + 1
        bound_q += term_bound
        terms.append({
            "key": key,
            "q_coeff": coeff,
            "q_input": qinput,
            "term_bound_q": term_bound,
        })
    total_weight = sum(t["term_bound_q"] for t in terms) or 1
    for term in terms:
        term["contribution_pct"] = round(100.0 * term["term_bound_q"] / total_weight, 4)
    return {
        "bound_q": saturate_int64(bound_q),
        "bound_real": bound_q / grid.scale,
        "truncation_rule": "round_toward_zero",
        "saturation": "int64_clamp",
        "rescale_truncation_q": 1,
        "terms": terms,
    }


def compare_derived(key_a: str, qvalue_a: int, bound_a: int,
                    key_b: str, qvalue_b: int, bound_b: int) -> dict:
    """Stable error-aware comparison between two derived items."""
    gap = qvalue_a - qvalue_b
    combined = bound_a + bound_b + 1
    if abs(gap) > combined:
        verdict = "GT" if gap > 0 else "LT"
        winner = key_a if gap > 0 else key_b
        decisive = True
    else:
        verdict = "INDETERMINATE"
        winner = min(key_a, key_b)
        decisive = False
    return {
        "a": key_a,
        "b": key_b,
        "gap_q": gap,
        "combined_bound_q": combined,
        "decisive": decisive,
        "verdict": verdict,
        "tie_break_winner": winner,
        "tie_break_rule": "lexicographic_min_key_when_indeterminate",
    }
