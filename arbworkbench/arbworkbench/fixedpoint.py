"""Deterministic fixed-point integer arithmetic.

Session scale
-------------
The session scale factor is ``S = 10 ** scale_exponent`` (a power of ten,
exponent range [-6, 6]). Quantization always produces an integer in *q-units*
where one unit represents ``10**-scale_exponent`` real units:

    q = round_half_away_from_zero(real_value * 10**scale_exponent)

For negative exponents the Decimal representation keeps the power of ten exact
and the result is still a plain Python integer (e.g. exponent -3 maps the real
value 1500 to q=2 because 1500 * 10**-3 = 1.5 -> 2). Every downstream
computation is integer-only, independent of host floating-point behaviour.

Rounding / truncation rules (explicit)
--------------------------------------
* Quantization rounds half away from zero (``|frac| >= 0.5`` rounds away).
* Integer division that loses precision uses *floor division* ``//``.
* The interval midpoint is ``(lo + hi) // 2`` with explicit floor semantics,
  i.e. for an odd integer span the lower of the two candidates is chosen.
* All arithmetic saturates at ``SAT_HIGH`` / ``SAT_LOW`` instead of raising
  ``OverflowError``. Saturation events are countable for diagnostics.

Error budget (in real units, ``u = 10**-scale_exponent`` is one quantum)
------------------------------------------------------------------------
* Single-observation quantization error: ``|e_i| <= u / 2``.
* Weighted-mean synthetic intermediate term (absolute accumulation):
  ``A = (u/2) * (sum of weights)`` grows *linearly* with the number of
  participating environments and their weights. After normalization by the
  total weight it collapses to ``u/2``; adding the integer division
  truncation term gives the decision bound ``B_majority = 2u``.
* Latest-wins reuses one quantized observation: ``B_latest = u``.
* Interval intersection midpoint combines two endpoints: ``B_interval = 2u``.
"""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, localcontext

from .errors import GridError

MIN_EXPONENT = -6
MAX_EXPONENT = 6
SAT_HIGH = 2**63 - 1
SAT_LOW = -(2**63)

# Decision bounds expressed in fixed-point integer units.
B_LATEST_Q = 1
B_MAJORITY_Q = 2
B_INTERVAL_Q = 2


def _sat(value: int) -> int:
    if value > SAT_HIGH:
        return SAT_HIGH
    if value < SAT_LOW:
        return SAT_LOW
    return value


def validate_exponent(scale_exponent: int) -> None:
    if (
        not isinstance(scale_exponent, int)
        or isinstance(scale_exponent, bool)
        or not MIN_EXPONENT <= scale_exponent <= MAX_EXPONENT
    ):
        raise GridError(
            f"scale_exponent must be an integer in [{MIN_EXPONENT}, {MAX_EXPONENT}]",
            location="scale_exponent",
            detail={"value": scale_exponent},
        )


class FixedPoint:
    """Session-scaled fixed-point context."""

    def __init__(self, scale_exponent: int):
        validate_exponent(scale_exponent)
        self.scale_exponent = scale_exponent
        # Exact Decimal power of ten; integer q-unit conversion happens in
        # quantize() so no float ever enters the arithmetic path.
        self._decimal_scale = Decimal(10) ** scale_exponent
        self.quantum_q = 1
        self.quantum_real = float(Decimal(1) / self._decimal_scale)
        self.saturation_events = 0

    # ---- quantization ----------------------------------------------------
    def quantize(self, value) -> int:
        """Quantize an int / float / Decimal / numeric string to q-units."""
        if isinstance(value, bool):
            raise GridError("boolean is not a valid observation value", location="value")
        if isinstance(value, int):
            dec = Decimal(value)
        elif isinstance(value, Decimal):
            dec = value
        else:
            dec = Decimal(str(value))
        with localcontext() as ctx:
            ctx.prec = 50
            scaled = dec * self._decimal_scale
            rounded = int(scaled.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        result = _sat(rounded)
        if result != rounded:
            self.saturation_events += 1
        return result

    # ---- integer arithmetic with saturation ------------------------------
    def add(self, a: int, b: int) -> int:
        raw = a + b
        result = _sat(raw)
        if result != raw:
            self.saturation_events += 1
        return result

    def mul(self, a: int, b: int) -> int:
        raw = a * b
        result = _sat(raw)
        if result != raw:
            self.saturation_events += 1
        return result

    def div_floor(self, numerator: int, denominator: int) -> int:
        if denominator == 0:
            raise GridError("division by zero in fixed-point arithmetic", location="denominator")
        quotient, _remainder = divmod(numerator, denominator)
        result = _sat(quotient)
        if result != quotient:
            self.saturation_events += 1
        return result

    def weighted_mean(self, pairs: list[tuple[int, int]]) -> int:
        """Return floor(sum(q*w) / sum(w)); weights are non-negative ints."""
        total_weight = sum(w for _, w in pairs)
        if total_weight <= 0:
            raise GridError("total arbitration weight must be positive", location="weights")
        accum = 0
        for q, w in pairs:
            accum = self.add(accum, self.mul(q, w))
        return self.div_floor(accum, total_weight)

    def midpoint_floor(self, lo_q: int, hi_q: int) -> int:
        """Floor midpoint of an inclusive integer interval (lower on ties)."""
        return self.div_floor(lo_q + hi_q, 2)

    # ---- conversion back to real ----------------------------------------
    def to_real(self, q: int) -> float:
        return float(Decimal(q) / self._decimal_scale)

    def describe_bounds(self) -> dict:
        u = self.quantum_real
        return {
            "quantum_real": u,
            "single_observation_quantization_error": u / 2,
            "bound_latest_real": B_LATEST_Q * u,
            "bound_majority_real": B_MAJORITY_Q * u,
            "bound_interval_real": B_INTERVAL_Q * u,
            "note": (
                "majority absolute intermediate term grows linearly with the sum "
                "of weights; after normalization the decision bound stays at 2u."
            ),
        }
