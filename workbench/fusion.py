"""Fixed-point fusion chain (requirement 3).

Pipeline per turn, executed in a fixed operation order:

1. normalize    - min-max normalisation of each channel (batch statistics,
                  accumulated in ascending turn-id order), rounded to the
                  configured significant digits.
2. quantize     - normalised values -> fixed-point integers using the
                  configured fractional bit width and quantum step.
3. weighted_sum - integer weighted sum with fixed-point weights, divided by
                  the fusion temperature, plus the bias.
4. clip         - clip to the confidence interval and emit the fixed-point
                  integer confidence with deterministic round-half-up.

All rational arithmetic uses fractions.Fraction so results are exact and
platform independent.  Abnormal conditions emit business codes together
with the raw (pre-normalisation) values; nothing is silently defaulted.
"""

from __future__ import annotations

from fractions import Fraction

from . import codes

CHANNELS = ("text", "acoustic", "tool")
STAGES = ("normalize", "quantize", "weighted_sum", "clip")


def round_half_up(frac: Fraction) -> int:
    """Deterministic round half away from zero for exact rationals."""
    if frac >= 0:
        return (frac.numerator * 2 + frac.denominator) // (2 * frac.denominator)
    return -round_half_up(-frac)


def to_fraction(value) -> Fraction:
    """Exact rational from a float/int via its shortest decimal repr."""
    if isinstance(value, Fraction):
        return value
    return Fraction(str(value))


def round_sig(value: float, digits: int) -> float:
    """Round to the configured number of significant digits, deterministically."""
    if value == 0:
        return 0.0
    return float(f"{value:.{digits}g}")


def normalization_stats(raw_scores):
    """Min/max per channel over the batch.

    raw_scores: list of {turn_id, raw: {channel: score}} sorted by ascending
    turn_id, so accumulation order is fixed.
    """
    stats = {}
    for channel in CHANNELS:
        values = [row["raw"][channel] for row in raw_scores
                  if row["raw"].get(channel) is not None]
        stats[channel] = {"min": min(values), "max": max(values)} if values else None
    return stats


def normalize_value(raw, stat, sig_digits):
    """Min-max normalise one raw value; returns None when undefined."""
    if raw is None or stat is None:
        return None
    span = stat["max"] - stat["min"]
    if span == 0:
        return round_sig(0.0, sig_digits)
    return round_sig((raw - stat["min"]) / span, sig_digits)


def quantize(value: float, frac_bits: int, quantum: int) -> int:
    """Normalised value -> fixed-point integer multiple of quantum."""
    scale = 1 << frac_bits
    scaled = to_fraction(value) * scale
    return round_half_up(scaled / quantum) * quantum


def fuse_turn(normalized, config, frac_bits, quantum):
    """Run stages 2-4 for one turn.

    normalized: {channel: normalised float or None}
    Returns a dict with per-stage products, codes and raw values.
    """
    weights = config["weights"]
    temperature = to_fraction(config["temperature"])
    bias = to_fraction(config.get("bias", 0))
    lo, hi = (to_fraction(v) for v in config["confidence_interval"])
    scale = 1 << frac_bits

    result = {
        "codes": [],
        "quantized": {},
        "weighted_sum": None,
        "confidence_int": None,
        "confidence": None,
        "clipped": False,
    }

    weight_ints = {ch: round_half_up(to_fraction(weights[ch]) * scale) for ch in CHANNELS}
    weight_sum = sum(weight_ints[ch] for ch in CHANNELS)
    if weight_sum == 0:
        result["codes"].append(codes.E_FUSE_WEIGHT_SUM_ZERO)
        return result

    missing = [ch for ch in CHANNELS if normalized.get(ch) is None]
    if missing:
        result["codes"].append(codes.E_FUSE_EVIDENCE_MISSING)
        result["missing"] = missing
        return result

    for ch in CHANNELS:
        result["quantized"][ch] = quantize(normalized[ch], frac_bits, quantum)

    acc = sum(weight_ints[ch] * result["quantized"][ch] for ch in CHANNELS)
    summed = Fraction(acc, scale) / temperature / weight_sum * scale
    summed += bias * scale
    result["weighted_sum"] = {
        "accumulator": acc,
        "pre_clip": str(summed),
        "pre_clip_float": float(summed / scale),
    }

    lo_i = lo * scale
    hi_i = hi * scale
    clipped = summed
    if clipped < lo_i:
        clipped = lo_i
        result["clipped"] = True
    elif clipped > hi_i:
        clipped = hi_i
        result["clipped"] = True
    if result["clipped"]:
        result["codes"].append(codes.E_FUSE_CLIPPED)
    conf_int = round_half_up(clipped)
    result["confidence_int"] = conf_int
    result["confidence"] = float(Fraction(conf_int, scale))
    return result


def classify(confidence, threshold):
    if confidence is None:
        return None
    return "pos" if confidence >= threshold else "neg"
