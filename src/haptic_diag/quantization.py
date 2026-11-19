"""Quantization profiles and error bounds for device encoding."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import floor

from .constants import (
    AMPLITUDE_CODE_MAX,
    AMPLITUDE_STEP,
    FREQUENCY_CODE_MAX,
    FREQUENCY_STEP,
    MAX_FREQUENCY_HZ,
)


def _round_half_up(value: float) -> int:
    # floor(x + 0.5) is stable and avoids Python's bankers rounding.
    return int(floor(value + 0.5))


@dataclass(frozen=True)
class QuantProfile:
    amplitude_step: float = AMPLITUDE_STEP
    frequency_step: float = FREQUENCY_STEP
    amplitude_code_max: int = AMPLITUDE_CODE_MAX
    frequency_code_max: int = FREQUENCY_CODE_MAX
    max_frequency_hz: float = MAX_FREQUENCY_HZ

    @property
    def amplitude_max_error(self) -> float:
        return self.amplitude_step / 2.0

    @property
    def frequency_max_error(self) -> float:
        return self.frequency_step / 2.0

    def encode_amplitude(self, value: float) -> int:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"amplitude out of range [0,1]: {value!r}")
        return min(self.amplitude_code_max, max(0, _round_half_up(value / self.amplitude_step)))

    def decode_amplitude(self, code: int) -> float:
        if not 0 <= int(code) <= self.amplitude_code_max:
            raise ValueError(f"amplitude code out of range: {code!r}")
        return int(code) * self.amplitude_step

    def encode_frequency(self, value: float) -> int:
        if not 0.0 <= value <= self.max_frequency_hz:
            raise ValueError(
                f"frequency out of range [0,{self.max_frequency_hz}]: {value!r}"
            )
        return min(self.frequency_code_max, max(0, _round_half_up(value / self.frequency_step)))

    def decode_frequency(self, code: int) -> float:
        if not 0 <= int(code) <= self.frequency_code_max:
            raise ValueError(f"frequency code out of range: {code!r}")
        return int(code) * self.frequency_step

    def amplitude_error(self, value: float) -> float:
        return abs(self.decode_amplitude(self.encode_amplitude(value)) - value)

    def frequency_error(self, value: float) -> float:
        return abs(self.decode_frequency(self.encode_frequency(value)) - value)

    def same_amplitude_code(self, left: float, right: float) -> bool:
        return self.encode_amplitude(left) == self.encode_amplitude(right)

    def same_frequency_code(self, left: float, right: float) -> bool:
        return self.encode_frequency(left) == self.encode_frequency(right)

    def to_dict(self) -> dict[str, float | int]:
        payload = asdict(self)
        payload["amplitude_max_error"] = self.amplitude_max_error
        payload["frequency_max_error"] = self.frequency_max_error
        return payload


DEFAULT_QUANT_PROFILE = QuantProfile()

