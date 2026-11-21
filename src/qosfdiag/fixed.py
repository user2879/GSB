"""Declared integer fixed-point arithmetic for rates, tokens and time.

The single session scale is ``S = 10**scale`` (0 <= scale <= 6):

* one token unit is exactly 1/S byte;
* one time tick is exactly 1/S second, or 10**(9-scale) ns;
* a rate represented as X token-units/tick therefore denotes X bytes/s.

Because committed rates are integral kbps and 1 kbps = 125 bytes/s, the rate
representation is exact. Truncation is toward zero (floor for non-negative
input) for elapsed time, and ceiling for queue-release delay so that a queued
packet is never released before enough tokens exist. Arithmetic saturates at
the token-bucket capacity and at Python's arbitrary-precision integers, so no
machine-word overflow can occur.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


NS_PER_SECOND = 1_000_000_000
BYTES_PER_SECOND_PER_KBPS = 125


def ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -((-numerator) // denominator)


@dataclass(frozen=True)
class FixedPointSpec:
    scale: int

    def __post_init__(self) -> None:
        if isinstance(self.scale, bool) or not isinstance(self.scale, int):
            raise ValueError("scale must be an integer")
        if not 0 <= self.scale <= 6:
            raise ValueError("scale must be in [0, 6]")

    @property
    def token_units_per_byte(self) -> int:
        return 10**self.scale

    @property
    def ticks_per_second(self) -> int:
        return 10**self.scale

    @property
    def tick_ns(self) -> int:
        return NS_PER_SECOND // self.ticks_per_second

    def quantize_time_ns(self, time_ns: int) -> int:
        if time_ns < 0:
            raise ValueError("time_ns must be non-negative")
        return time_ns // self.tick_ns

    def time_tick_to_ns(self, tick: int) -> int:
        return tick * self.tick_ns

    def bytes_to_units(self, byte_count: int) -> int:
        return byte_count * self.token_units_per_byte

    def units_to_bytes_floor(self, units: int) -> int:
        return units // self.token_units_per_byte

    def units_to_bytes_ceil(self, units: int) -> int:
        return ceil_div(units, self.token_units_per_byte)

    def rate_units_per_tick(self, committed_rate_kbps: int) -> int:
        return committed_rate_kbps * BYTES_PER_SECOND_PER_KBPS

    def queue_delay_ticks(self, deficit_units: int, rate_units_per_tick: int) -> int:
        if deficit_units <= 0:
            return 0
        if rate_units_per_tick <= 0:
            raise ValueError("rate_units_per_tick must be positive")
        return ceil_div(deficit_units, rate_units_per_tick)

    def rate_quantization_analysis(self, committed_rate_kbps: int) -> dict[str, int]:
        represented_units = self.rate_units_per_tick(committed_rate_kbps)
        represented_bytes_per_s = represented_units
        exact_bytes_per_s = committed_rate_kbps * BYTES_PER_SECOND_PER_KBPS
        error = abs(represented_bytes_per_s - exact_bytes_per_s)
        return {
            "scale": self.scale,
            "token_units_per_byte": self.token_units_per_byte,
            "tick_ns": self.tick_ns,
            "committed_rate_kbps": committed_rate_kbps,
            "exact_bytes_per_s": exact_bytes_per_s,
            "represented_units_per_tick": represented_units,
            "represented_bytes_per_s": represented_bytes_per_s,
            "rate_quantization_error_bytes_per_s": error,
            "one_time_tick_error_ns": self.tick_ns,
            "refill_error_units_per_boundary": represented_units,
        }

    def delay_quantization_analysis(self) -> dict[str, int]:
        return {
            "scale": self.scale,
            "tick_ns": self.tick_ns,
            "arrival_truncation_error_ns_upper_exclusive": self.tick_ns,
            "release_ceiling_error_ns_upper_exclusive": self.tick_ns,
            "queue_delay_error_bound_ns": 2 * self.tick_ns,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "token_units_per_byte": self.token_units_per_byte,
            "ticks_per_second": self.ticks_per_second,
            "tick_ns": self.tick_ns,
            "time_truncation": "floor toward zero for non-negative virtual time",
            "delay_rounding": "ceiling to the next tick",
            "token_saturation": "clamp to burst capacity",
            "overflow_policy": "arbitrary-precision integers; capacity clamp",
        }
