"""Fixed-point integer arithmetic with an explicit residual ledger.

All values are stored as int64 arrays with ``frac_bits`` fractional bits.
Every operation that discards bits records the rounding direction and the
residual (in output quanta) into a :class:`Ledger`.  No runtime global
random state is used anywhere; results are bit-exact across runs.

Overflow policy: raw products are computed in arbitrary precision and
clipped to +/-2**62 before rounding (a saturation event visible in the
ledger); stored values are further clamped to the STORAGE_BAND by
:func:`saturate`.  This keeps the int64 domain safe for adversarial inputs.
"""

from __future__ import annotations

import numpy as np

# Allowed quantization tiers: tier index -> fractional bit width.
TIERS = {0: 8, 1: 12, 2: 16, 3: 20}
# Extra bits used when diagnostics need one tier above the top tier.
DIAG_EXTRA_BITS = 8
# Guard bits carried by merge deltas so that per-adapter rounding is exact.
MERGE_GUARD_BITS = 4
# Saturation band for stored fixed-point values (storage-format ceiling;
# far above any healthy merged weight, reachable by adversarial inputs).
STORAGE_BAND = np.int64(1 << 40)
# Clip bound for raw (pre-rounding) products.
RAW_CLIP = 1 << 62


def tier_frac_bits(tier: int) -> int:
    if tier in TIERS:
        return TIERS[tier]
    if tier == max(TIERS) + 1:
        return TIERS[max(TIERS)] + DIAG_EXTRA_BITS
    raise ValueError(f"unknown quant tier: {tier}")


def higher_frac_bits(frac_bits: int) -> int:
    ordered = sorted(TIERS.values())
    for bits in ordered:
        if bits > frac_bits:
            return bits
    return frac_bits + DIAG_EXTRA_BITS


class Ledger:
    """Append-only residual ledger; one entry per recorded operation."""

    def __init__(self) -> None:
        self.entries: list[dict] = []

    def record(self, op: str, direction: np.ndarray, residual: np.ndarray) -> None:
        direction = np.asarray(direction)
        residual = np.asarray(residual, dtype=np.float64)
        self.entries.append(
            {
                "op": op,
                "count": int(direction.size),
                "round_up": int(np.sum(direction > 0)),
                "round_down": int(np.sum(direction < 0)),
                "exact": int(np.sum(direction == 0)),
                "max_abs_residual_quanta": float(np.max(np.abs(residual))) if residual.size else 0.0,
            }
        )


def _safe_product(a, b):
    """Exact arbitrary-precision elementwise product, clipped to +/-2**62."""
    raw = np.asarray(a, dtype=np.int64).astype(object) * np.asarray(b, dtype=np.int64).astype(object)
    clipped = np.where(raw > RAW_CLIP, RAW_CLIP, np.where(raw < -RAW_CLIP, -RAW_CLIP, raw))
    return clipped.astype(np.int64)


def _safe_matmul(a, b):
    raw = np.asarray(a, dtype=np.int64).astype(object) @ np.asarray(b, dtype=np.int64).astype(object)
    clipped = np.where(raw > RAW_CLIP, RAW_CLIP, np.where(raw < -RAW_CLIP, -RAW_CLIP, raw))
    return clipped.astype(np.int64)


def _round_shift_raw(raw: np.ndarray, frac_bits: int):
    """Divide int64 ``raw`` by 2**frac_bits, rounding half away from zero."""
    raw = np.asarray(raw, dtype=np.int64)
    half = np.int64(1) << np.int64(frac_bits - 1)
    mag = np.abs(raw)
    res_mag = (mag + half) >> np.int64(frac_bits)
    res = np.where(raw >= 0, res_mag, -res_mag).astype(np.int64)
    diff = res * (np.int64(1) << np.int64(frac_bits)) - raw
    direction = np.sign(diff).astype(np.int64)
    residual = diff.astype(np.float64) / float(1 << frac_bits)
    return res, direction, residual


def quantize(values, frac_bits: int, ledger: Ledger | None = None, op: str = "quantize"):
    """Float array -> fixed-point int64, half away from zero."""
    scaled = np.asarray(values, dtype=np.float64) * float(1 << frac_bits)
    scaled = np.clip(scaled, -float(RAW_CLIP), float(RAW_CLIP))
    res = np.sign(scaled) * np.floor(np.abs(scaled) + 0.5)
    res = res.astype(np.int64)
    if ledger is not None:
        diff = res.astype(np.float64) - scaled
        ledger.record(op, np.sign(diff).astype(np.int64), diff)
    return res


def dequant(values, frac_bits: int) -> np.ndarray:
    return np.asarray(values, dtype=np.int64).astype(np.float64) / float(1 << frac_bits)


def round_shift(raw, frac_bits: int, ledger: Ledger | None = None, op: str = "round"):
    res, direction, residual = _round_shift_raw(raw, frac_bits)
    if ledger is not None:
        ledger.record(op, direction, residual)
    return res


def round_div(raw, divisor: int, ledger: Ledger | None = None, op: str = "div"):
    """Integer division by positive ``divisor``, half away from zero."""
    raw = np.asarray(raw, dtype=np.int64)
    d = int(divisor)
    if d <= 0:
        raise ValueError("divisor must be positive")
    mag = np.abs(raw)
    q = (mag + d // 2) // d
    res = np.where(raw >= 0, q, -q).astype(np.int64)
    if ledger is not None:
        diff = res * d - raw
        ledger.record(op, np.sign(diff).astype(np.int64), diff.astype(np.float64) / d)
    return res


def mul_fixed(a, b, frac_bits: int, ledger: Ledger | None = None, op: str = "mul"):
    raw = _safe_product(a, b)
    return round_shift(raw, frac_bits, ledger, op)


def matmul_fixed(a, b, frac_bits: int, ledger: Ledger | None = None, op: str = "matmul"):
    """Exact accumulation, single rounding at the end (fixed order)."""
    raw = _safe_matmul(a, b)
    return round_shift(raw, frac_bits, ledger, op)


def saturate(values, ledger: Ledger | None = None, op: str = "saturate"):
    """Clamp to the storage band; returns (clamped, clamped_mask)."""
    v = np.asarray(values, dtype=np.int64)
    mask = np.abs(v) > STORAGE_BAND
    if not np.any(mask):
        return v, mask
    clamped = np.clip(v, -STORAGE_BAND, STORAGE_BAND)
    if ledger is not None:
        diff = clamped - v
        ledger.record(op, np.sign(diff).astype(np.int64), diff.astype(np.float64))
    return clamped, mask
