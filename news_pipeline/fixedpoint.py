"""Deterministic fixed-point quantization helpers.

Scores live in the signed 32-bit integer world at runtime.  A Qm.f format is
derived from the configured decimal width / quantization tier; all arithmetic
is done in a single fixed evaluation order so two runs over identical inputs
always produce byte-identical integers.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QFormat:
    width: int          # total decimal width declared in config
    tier: str           # quantization tier name
    frac_bits: int      # fractional bits (f in Qm.f)
    scale: int          # 2 ** frac_bits

    def quantize(self, value):
        """Round half away from zero and return a plain Python int."""
        scaled = np.float64(value) * np.float64(self.scale)
        rounded = np.floor(np.abs(scaled) + np.float64("0.5"))
        return int(np.sign(scaled) * rounded)

    def dequantize(self, q_value):
        return float(np.float64(int(q_value)) / np.float64(self.scale))


# Allowed (decimal width, tier) combinations.  Higher tiers spend more bits on
# the fractional part, which is exactly what makes boundary inclusion flip
# when the tier is switched.
_TIER_FRAC_BITS = {"coarse": 4, "standard": 8, "fine": 12, "precision": 16}


def validate_fixedpoint(width, tier):
    if not isinstance(width, int) or isinstance(width, bool) or width < 1 or width > 10:
        return "fixedpoint width must be an integer in [1, 10]"
    if tier not in _TIER_FRAC_BITS:
        return "fixedpoint tier must be one of %s" % sorted(_TIER_FRAC_BITS)
    frac_bits = _TIER_FRAC_BITS[tier]
    integer_bits = 31 - frac_bits
    if width > integer_bits + frac_bits:
        return "fixedpoint width %d exceeds Q-format capacity for tier %s" % (width, tier)
    return None


def make_qformat(width, tier):
    problem = validate_fixedpoint(width, tier)
    if problem is not None:
        raise ValueError(problem)
    frac_bits = _TIER_FRAC_BITS[tier]
    return QFormat(width=width, tier=tier, frac_bits=frac_bits, scale=1 << frac_bits)
