"""Decoding configuration: allowed enums, ranges and canonical fingerprints."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

# Allowed fixed-point fractional widths (number of decimal digits).
FIXED_WIDTHS = (3, 4, 5, 6)

# Allowed quantization tiers. ROUND_HALF_UP and FLOOR are integer-domain
# deterministic rules; REF is the extra-width unquantized reference path.
QUANT_TIERS = ("ROUND_HALF_UP", "FLOOR", "REF")

# Hard ranges for sampling parameters.
TOPK_MIN, TOPK_MAX = 1, 50
TOPP_MIN, TOPP_MAX = 1, 1000          # per-mille threshold, i.e. 0.001 .. 1.0
TEMPERATURE_MIN_EXCLUSIVE = 0.0
TEMPERATURE_MAX_INCLUSIVE = 10.0

# Known source decoding channels.
CHANNELS = ("GREEDY", "SAMPLE", "BEAM")

# Tolerance is expressed in units of the active fixed point scale.
DEFAULT_TOLERANCE_UNITS = 2


def canonical_json(obj) -> str:
    """Stable JSON text used for fingerprinting (sorted keys, no whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=_json_default)


def _json_default(value):
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"not serializable: {type(value)!r}")


@dataclass(frozen=True)
class DecodeConfig:
    """Global decoding configuration.

    width         fractional digits of the working fixed-point integers
    tier          quantization rule (see QUANT_TIERS)
    top_k         hard TopK cap
    top_p_mille   nucleus threshold in per-mille (1..1000)
    tolerance_units  max allowed integer error before attribution flags it
    """

    width: int = 4
    tier: str = "ROUND_HALF_UP"
    top_k: int = 8
    top_p_mille: int = 900
    tolerance_units: int = DEFAULT_TOLERANCE_UNITS

    def validate(self) -> list:
        from .errors import BizError, BizCode
        errs: list[BizError] = []
        if self.width not in FIXED_WIDTHS:
            errs.append(BizError(BizCode.INVALID_WIDTH, "config.width",
                                 f"width={self.width!r} not in {FIXED_WIDTHS}"))
        if self.tier not in QUANT_TIERS:
            errs.append(BizError(BizCode.INVALID_TIER, "config.tier",
                                 f"tier={self.tier!r} not in {QUANT_TIERS}"))
        if not (TOPK_MIN <= self.top_k <= TOPK_MAX):
            errs.append(BizError(BizCode.TOPK_OUT_OF_RANGE, "config.top_k",
                                 f"top_k={self.top_k} outside "
                                 f"[{TOPK_MIN},{TOPK_MAX}]"))
        if not (TOPP_MIN <= self.top_p_mille <= TOPP_MAX):
            errs.append(BizError(BizCode.TOPP_OUT_OF_RANGE, "config.top_p_mille",
                                 f"top_p_mille={self.top_p_mille} outside "
                                 f"[{TOPP_MIN},{TOPP_MAX}]"))
        if self.tolerance_units < 0:
            errs.append(BizError(BizCode.TOLERANCE_EXCEEDED,
                                 "config.tolerance_units",
                                 "tolerance units must be non-negative"))
        return errs

    @property
    def scale(self) -> int:
        return 10 ** self.width

    @property
    def ref_width(self) -> int:
        """The 'one digit wider' reference path width."""
        return min(self.width + 1, max(FIXED_WIDTHS))

    @property
    def half_step(self) -> int:
        # Half of one quantization step in integer units.
        return self.scale // 20

    def with_replacements(self, **changes) -> "DecodeConfig":
        return replace(self, **changes)

    def fingerprint(self) -> str:
        payload = {
            "width": self.width,
            "tier": self.tier,
            "top_k": self.top_k,
            "top_p_mille": self.top_p_mille,
        }
        return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
