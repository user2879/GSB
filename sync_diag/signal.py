from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .hashing import sha256_json


@dataclass(frozen=True)
class ChannelClock:
    index: int
    delay_samples: float
    base_ppm: float
    temp_ppm_per_c: float
    temp_mean_c: float
    temp_amplitude_c: float
    temp_period_s: float
    temp_phase_rad: float
    sample_rate_hz: int

    @classmethod
    def from_config(cls, channel: dict[str, Any], sample_rate_hz: int) -> "ChannelClock":
        temp = channel["temperature_c"]
        return cls(
            index=int(channel["index"]),
            delay_samples=float(channel["delay_samples"]),
            base_ppm=float(channel["base_ppm"]),
            temp_ppm_per_c=float(channel["temp_ppm_per_c"]),
            temp_mean_c=float(temp["mean"]),
            temp_amplitude_c=float(temp["amplitude_c"]),
            temp_period_s=float(temp["period_s"]),
            temp_phase_rad=float(temp["phase_rad"]),
            sample_rate_hz=sample_rate_hz,
        )

    def ppm_at(self, samples: np.ndarray | float) -> np.ndarray:
        values = np.asarray(samples, dtype=np.float64)
        omega = 2.0 * np.pi / (float(self.sample_rate_hz) * max(self.temp_period_s, 1.0e-9))
        return self.base_ppm + self.temp_ppm_per_c * self.temp_amplitude_c * np.sin(omega * values + self.temp_phase_rad)

    def drift_samples(self, samples: np.ndarray | float) -> np.ndarray:
        values = np.asarray(samples, dtype=np.float64)
        omega = 2.0 * np.pi / (float(self.sample_rate_hz) * max(self.temp_period_s, 1.0e-9))
        integral_ppm = self.base_ppm * values
        if abs(self.temp_ppm_per_c * self.temp_amplitude_c) > 0.0:
            scale = self.temp_ppm_per_c * self.temp_amplitude_c / omega
            integral_ppm = integral_ppm + scale * (
                math.cos(self.temp_phase_rad) - np.cos(omega * values + self.temp_phase_rad)
            )
        return integral_ppm * 1.0e-6

    def source_coordinate(self, samples: np.ndarray | float) -> np.ndarray:
        values = np.asarray(samples, dtype=np.float64)
        return values + self.delay_samples + self.drift_samples(values)

    def observed_sample_for_reference(self, reference_sample: float) -> float:
        span = 8192.0 + 0.02 * abs(float(reference_sample))
        low = float(reference_sample) - self.delay_samples - span
        high = float(reference_sample) - self.delay_samples + span
        low = max(low, -span)
        for _ in range(64):
            middle = 0.5 * (low + high)
            value = float(self.source_coordinate(np.asarray(middle))) - float(reference_sample)
            if value < 0.0:
                low = middle
            else:
                high = middle
        return 0.5 * (low + high)


class TestSignal:
    def __init__(self, config: dict[str, Any]) -> None:
        signal = config["signal"]
        self.sample_rate_hz = int(config["sample_rate_hz"])
        self.tones_hz = [float(value) for value in signal["tones_hz"]]
        self.tone_amplitudes = [float(value) for value in signal["tone_amplitudes"]]
        self.tone_phases = [float(value) for value in signal["tone_phase_rad"]]
        self.pulse_period_samples = float(signal["pulse_period_s"]) * self.sample_rate_hz
        self.pulse_offset_samples = float(signal["pulse_offset_s"]) * self.sample_rate_hz
        self.pulse_sigma_samples = max(float(signal["pulse_sigma_s"]) * self.sample_rate_hz, 1.0)
        self.pulse_amplitude = float(signal["pulse_amplitude"])
        self.signature = sha256_json(
            {
                "tones_hz": self.tones_hz,
                "tone_amplitudes": self.tone_amplitudes,
                "tone_phase_rad": self.tone_phases,
                "pulse_period_samples": self.pulse_period_samples,
                "pulse_offset_samples": self.pulse_offset_samples,
                "pulse_sigma_samples": self.pulse_sigma_samples,
                "pulse_amplitude": self.pulse_amplitude,
            }
        )

    def sample_indices(self, start: int, count: int) -> np.ndarray:
        indices = np.arange(start, start + count, dtype=np.float64)
        phases = indices * (2.0 * np.pi / float(self.sample_rate_hz))
        values = np.zeros(count, dtype=np.float64)
        for frequency, amplitude, phase in zip(self.tones_hz, self.tone_amplitudes, self.tone_phases):
            values += amplitude * np.sin(phases * frequency + phase)
        mod = np.mod(indices - self.pulse_offset_samples, self.pulse_period_samples)
        distance = np.minimum(mod, self.pulse_period_samples - mod)
        values += self.pulse_amplitude * np.exp(-0.5 * np.square(distance / self.pulse_sigma_samples))
        return values

    def reference_pulse_samples(self, frames: int) -> list[int]:
        positions: list[int] = []
        first = int(math.ceil(-self.pulse_offset_samples / self.pulse_period_samples))
        position = self.pulse_offset_samples + first * self.pulse_period_samples
        while position < frames:
            positions.append(int(round(position)))
            position += self.pulse_period_samples
        return positions

    def truth_at_reference_samples(self, reference_samples: np.ndarray, clocks: list[ChannelClock]) -> list[np.ndarray]:
        return [np.asarray([clock.observed_sample_for_reference(value) for value in reference_samples], dtype=np.float64) for clock in clocks]
