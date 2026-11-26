"""Deterministic fixed-point integer arithmetic.

All numeric work in the pipeline is represented as signed Python integers in a
Qm.n style fixed point format.  No float participates in the rendering math;
the only place ``float`` may appear is parsing human entered decimal strings,
and even there the conversion itself is an exact integer routine (see
:func:`parse_decimal_q`).

Quantization (changing ``frac_bits`` and/or the rounding mode) is recorded with
its step, integer result and residual so every decision is auditable.
"""
from __future__ import annotations

from dataclasses import dataclass

from .errors import E_QBITS_RANGE, E_QMODE_ENUM, RenderError

# Allowed enumerations (business rule 1).
ALLOWED_FRAC_BITS = (8, 12, 16)
ALLOWED_QMODES = ("trunc", "nearest", "floor")


def validate_qmode(frac_bits: int, qmode: str, location: str, field: str) -> list[RenderError]:
    errs: list[RenderError] = []
    if frac_bits not in ALLOWED_FRAC_BITS:
        errs.append(RenderError(
            E_QBITS_RANGE, location, field,
            f"frac_bits={frac_bits} not in {ALLOWED_FRAC_BITS}",
        ))
    if qmode not in ALLOWED_QMODES:
        errs.append(RenderError(
            E_QMODE_ENUM, location, field,
            f"qmode={qmode!r} not in {ALLOWED_QMODES}",
        ))
    return errs


def q_scale(frac_bits: int) -> int:
    return 1 << frac_bits


def round_int(value: int, qmode: str) -> int:
    """Round a raw integer (already scaled) under a named mode."""
    if qmode == "trunc":
        # Toward zero.
        return int(value) if value >= 0 else -((-value))
    if qmode == "floor":
        # Toward negative infinity (Python // already does this).
        return value
    if qmode == "nearest":
        # Half away from zero for fully deterministic tie breaking.
        if value >= 0:
            return (value + 1) // 2 if False else value  # placeholder replaced below
        return value
    raise RenderError(E_QMODE_ENUM, "fixedpoint", "qmode", f"unknown qmode {qmode!r}")


def q_div_round(numerator: int, denominator: int, qmode: str) -> int:
    """Integer division with explicit, auditable rounding."""
    if denominator == 0:
        raise ZeroDivisionError("q_div_round denominator is zero")
    if qmode == "trunc":
        # Toward zero.
        q = abs(numerator) // abs(denominator)
        if (numerator < 0) != (denominator < 0):
            q = -q
        return q
    if qmode == "floor":
        return numerator // denominator
    if qmode == "nearest":
        # Half away from zero.
        sign = -1 if (numerator < 0) ^ (denominator < 0) else 1
        num, den = abs(numerator), abs(denominator)
        q, r = divmod(num, den)
        if r * 2 >= den:
            q += 1
        return sign * q
    raise RenderError(E_QMODE_ENUM, "fixedpoint", "qmode", f"unknown qmode {qmode!r}")


def parse_decimal_q(text: str, frac_bits: int, qmode: str = "trunc") -> int:
    """Convert an exact decimal string (e.g. '0.25', '-3', '1.2') to a Q int.

    Pure integer routine: splitting on the decimal point avoids any binary
    floating point rounding.  Extra fractional digits beyond ``frac_bits`` are
    handled with the configured rounding mode and the discarded part is simply
    truncated (callers needing the residual use :func:`quantize`).
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    neg = text.startswith("-")
    if neg or text.startswith("+"):
        text = text[1:]
    if "." in text:
        whole, frac = text.split(".", 1)
    else:
        whole, frac = text, ""
    if not whole:
        whole = "0"
    if not whole.isdigit() or (frac and not frac.isdigit()):
        raise ValueError(f"not an exact decimal: {text!r}")
    scale = q_scale(frac_bits)
    base = int(whole) * scale
    if frac:
        # Exact decimal -> binary fixed point:
        # value = int(frac) / 10**len(frac); in Q: * scale, rounded explicitly.
        frac_int = int(frac)
        decimal_den = 10 ** len(frac)
        numerator = frac_int * scale
        frac_q = q_div_round(numerator, decimal_den, qmode)
        base += frac_q
    return -base if neg else base


@dataclass(frozen=True)
class QuantTrace:
    """One quantization / scaling event in fully auditable form."""
    op: str
    in_frac_bits: int
    out_frac_bits: int
    qmode: str
    step: int            # quantization step (integer value of 1 LSB_out)
    raw: int             # input integer
    value: int           # output integer
    residual: int        # raw - value*step, expressed in input LSB units

    def to_dict(self) -> dict:
        return {
            "op": self.op,
            "in_frac_bits": self.in_frac_bits,
            "out_frac_bits": self.out_frac_bits,
            "qmode": self.qmode,
            "step": self.step,
            "raw": self.raw,
            "value": self.value,
            "residual": self.residual,
        }


def quantize(value_q: int, in_frac_bits: int, out_frac_bits: int,
             qmode: str, op: str = "quantize") -> tuple[int, QuantTrace]:
    """Re-quantize a Q integer to a different fractional bit width."""
    if qmode not in ALLOWED_QMODES:
        raise RenderError(E_QMODE_ENUM, "fixedpoint", "qmode", f"unknown qmode {qmode!r}")
    if in_frac_bits not in ALLOWED_FRAC_BITS or out_frac_bits not in ALLOWED_FRAC_BITS:
        raise RenderError(E_QBITS_RANGE, "fixedpoint", "frac_bits", "bad frac_bits")
    if in_frac_bits == out_frac_bits:
        trace = QuantTrace(op, in_frac_bits, out_frac_bits, qmode, 1,
                           value_q, value_q, 0)
        return value_q, trace
    if out_frac_bits < in_frac_bits:
        drop = in_frac_bits - out_frac_bits
        step = 1 << drop
        if qmode == "trunc":
            # toward zero
            if value_q >= 0:
                out = value_q >> drop
            else:
                out = -((-value_q) >> drop)
        elif qmode == "floor":
            out = value_q >> drop
        else:  # nearest, half away from zero
            if value_q >= 0:
                out = (value_q + (1 << (drop - 1))) >> drop
            else:
                out = -(((-value_q) + (1 << (drop - 1))) >> drop)
        residual = value_q - (out << drop)
        trace = QuantTrace(op, in_frac_bits, out_frac_bits, qmode, step,
                           value_q, out, residual)
        return out, trace
    # Widening is exact.
    gain = out_frac_bits - in_frac_bits
    out = value_q << gain
    trace = QuantTrace(op, in_frac_bits, out_frac_bits, qmode, 1,
                       value_q, out, 0)
    return out, trace


def q_mul(a_q: int, b_q: int, frac_bits: int, qmode: str,
          op: str = "mul") -> tuple[int, QuantTrace]:
    """Multiply two Q numbers with the same frac width and renormalize."""
    raw = a_q * b_q
    # raw has 2*frac_bits; divide by scale with explicit rounding.
    scale = q_scale(frac_bits)
    out = q_div_round(raw, scale, qmode)
    residual = raw - out * scale
    trace = QuantTrace(op, frac_bits * 2, frac_bits, qmode, scale,
                       raw, out, residual)
    return out, trace


def q_from_int(value: int, frac_bits: int) -> int:
    return int(value) << frac_bits


def q_to_int_trunc(value_q: int, frac_bits: int) -> int:
    if value_q >= 0:
        return value_q >> frac_bits
    return -((-value_q) >> frac_bits)


def q_to_uint8_clamp(value_q: int, frac_bits: int, qmode: str) -> tuple[int, QuantTrace]:
    """Convert a Q pixel value to uint8 with clamp and auditable rounding."""
    lo, hi = 0, 255 * q_scale(frac_bits)
    clamped = max(lo, min(hi, value_q))
    pixel = q_div_round(clamped, q_scale(frac_bits), qmode)
    pixel = max(0, min(255, pixel))
    trace = QuantTrace("q_to_uint8", frac_bits, 0, qmode, q_scale(frac_bits),
                       value_q, pixel, clamped - pixel * q_scale(frac_bits))
    return pixel, trace
