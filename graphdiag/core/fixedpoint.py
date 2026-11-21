"""Integer fixed-point quantization, saturation and error-bound arithmetic.

Conventions (explicit and documented, never implicit floats)
-----------------------------------------------------------
* Raw edge weights and caps are non-negative integer *micro units*
  (``1_000_000`` micro units == 1.0 logical unit).
* The session scale factor is ``S = 10 ** exponent`` with ``exponent`` in
  ``0..6``.  A weight is quantized to fixed-point (fp) integers by

      q = floor(weight_micro * S / 1_000_000)

  i.e. truncation toward zero (weights are non-negative, so this is floor).
* One fp unit (ulp) therefore represents ``10 ** (6 - exponent)`` micro units.
* All distance accumulation is plain integer addition.  Overflow is handled
  by saturating addition at the signed 64-bit maximum ``FP_MAX``.  A scale
  that could truncate any reachable cumulative distance is rejected up front
  with ``E_SCALE`` so saturation is a defensive guard, not a silent result.
* Single-edge truncation error is strictly below one ulp.  The guaranteed
  conservative bound for an ``m``-edge path is ``m`` ulps; the exact micro
  remainder sum is tracked alongside it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import ScaleError
from .models import MICRO

FP_MAX = (1 << 63) - 1  # signed 64-bit maximum, the declared integer domain
VALID_EXPONENTS = range(0, 7)


@dataclass(frozen=True)
class Quantizer:
    exponent: int

    def __post_init__(self) -> None:
        if self.exponent not in VALID_EXPONENTS:
            raise ScaleError(
                f"scale exponent {self.exponent} outside 0..6",
                detail={"exponent": self.exponent},
            )

    @property
    def scale(self) -> int:
        return 10 ** self.exponent

    @property
    def micro_per_ulp(self) -> int:
        return MICRO // self.scale

    def quantize(self, weight_micro: int) -> int:
        """Truncate a micro-unit weight to a fixed-point integer."""
        return (weight_micro * self.scale) // MICRO

    def fp_to_micro(self, value_fp: int) -> int:
        return value_fp * self.micro_per_ulp

    def check_domain(self, max_weight_micro: int, max_cap_micro: int) -> None:
        """Reject sessions whose scale would truncate cumulative distances.

        Bounded traversal never settles a node beyond the cap, but the
        relaxation that crosses the cap still performs one integer addition.
        The largest integer the engine can legitimately form is therefore
        ``quantize(cap) + quantize(max edge weight)``.
        """
        max_weight_fp = self.quantize(max_weight_micro)
        cap_fp = self.quantize(max_cap_micro)
        if max_weight_fp >= FP_MAX:
            raise ScaleError(
                "scale factor makes an edge weight reach the fixed-point ceiling",
                detail={
                    "exponent": self.exponent,
                    "max_weight_micro": max_weight_micro,
                    "quantized": max_weight_fp,
                    "fp_max": FP_MAX,
                },
            )
        if cap_fp >= FP_MAX or cap_fp + max_weight_fp > FP_MAX:
            raise ScaleError(
                "scale factor makes cumulative fixed-point distances truncate",
                detail={
                    "exponent": self.exponent,
                    "cap_fp": cap_fp,
                    "max_weight_fp": max_weight_fp,
                    "reachable_max_fp": cap_fp + max_weight_fp,
                    "fp_max": FP_MAX,
                },
            )


def saturating_add(left: int, right: int) -> tuple[int, bool]:
    """Integer addition clamped at :data:`FP_MAX`.

    Returns ``(result, saturated)``.  Non-negative inputs only.
    """
    total = left + right
    if total > FP_MAX or total < left:  # second clause guards theoretical wrap
        return FP_MAX, True
    return total, False


@dataclass(frozen=True)
class EdgeErrorTerm:
    """Running error bookkeeping at one position along a path."""

    edge_index: int
    source: int
    target: int
    weight_micro: int
    quantized_fp: int
    edge_remainder_micro: int
    running_edges: int
    conservative_bound_ulp: int       # running edge count (each edge < 1 ulp)
    conservative_bound_micro: int     # conservative bound expressed in micro
    exact_bound_micro: int            # running exact remainder sum


def build_path_terms(edge_micros: list[tuple[int, int, int]], quantizer: Quantizer) -> list[EdgeErrorTerm]:
    """Build per-edge running error terms for a path.

    ``edge_micros`` is an ordered list of ``(source, target, weight_micro)``.
    """
    mpu = quantizer.micro_per_ulp
    terms: list[EdgeErrorTerm] = []
    exact_total = 0
    for index, (source, target, weight_micro) in enumerate(edge_micros):
        quantized = quantizer.quantize(weight_micro)
        remainder = weight_micro - quantized * mpu
        exact_total += remainder
        running_edges = index + 1
        terms.append(
            EdgeErrorTerm(
                edge_index=index,
                source=source,
                target=target,
                weight_micro=weight_micro,
                quantized_fp=quantized,
                edge_remainder_micro=remainder,
                running_edges=running_edges,
                conservative_bound_ulp=running_edges,
                conservative_bound_micro=running_edges * mpu,
                exact_bound_micro=exact_total,
            )
        )
    return terms


def dominant_term(terms: list[EdgeErrorTerm]) -> EdgeErrorTerm | None:
    """Stable dominant error segment: largest exact remainder, earliest wins."""
    if not terms:
        return None
    best = terms[0]
    for term in terms[1:]:
        if term.edge_remainder_micro > best.edge_remainder_micro:
            best = term
    return best


def format_micro(micro: int) -> str:
    """Render micro units as a fixed 6-decimal logical value."""
    sign = "-" if micro < 0 else ""
    micro = abs(micro)
    return f"{sign}{micro // MICRO}.{micro % MICRO:06d}"
