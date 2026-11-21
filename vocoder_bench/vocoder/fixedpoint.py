"""Fixed-point phase accumulation.

Phase is stored as an *unsigned* integer on the ring ``[0, 2**bits)`` mapping
linearly to ``[0, 2*pi)``.  Every accumulation and wrap operation is integer
arithmetic (no floating-point modulo).  Float64 only appears when converting a
physical phase into a quantized integer number of grid units.

The fixed-point path quantizes the *absolute* target phase; therefore the only
error versus the float64 path is one quantization grid per frame (no error
accumulation).  Lowering ``phase_bits`` / raising ``phase_quant_step`` coarsens
that grid and genuinely changes the synthesized waveform.
"""
from __future__ import annotations

import math

import numpy as np

from .hashing import hash_mix, splitmix64

TWO_PI = 2.0 * math.pi
MASK64 = (1 << 64) - 1


def _seed_units(utter_id: str, frame_fp: str, harmonic: int, modulus: int) -> int:
    seed = int(hash_mix(utter_id, frame_fp, f"h{harmonic}"), 16)
    return splitmix64(seed) % modulus


class FixedPointPhase:
    def __init__(self, bits: int, quant_step: int):
        if not (4 <= bits <= 48):
            raise ValueError("phase bits out of range")
        if quant_step < 1:
            raise ValueError("quant_step must be >= 1")
        self.bits = bits
        self.quant_step = int(quant_step)
        self.modulus = 1 << bits
        self.mask = self.modulus - 1

    def start_phase(self, utter_id: str, frame_fp: str, harmonic: int) -> int:
        # Seed on the shared virtual grid, then quantize onto this bit-width
        # grid so the float64 control path and high-bit fixed point coincide.
        virtual = _seed_units(
            utter_id, frame_fp, harmonic, 1 << Float64Phase.VIRTUAL_BITS
        )
        scaled = virtual >> (Float64Phase.VIRTUAL_BITS - self.bits)
        return ((scaled // self.quant_step) * self.quant_step) & self.mask

    def quantize_absolute(self, phase_radians: float) -> int:
        units = math.fmod(phase_radians, TWO_PI)
        if units < 0:
            units += TWO_PI
        grid = int(math.floor(units / TWO_PI * self.modulus / self.quant_step + 0.5))
        return (grid * self.quant_step) & self.mask

    def accumulate(self, frames: list[dict], hop: int, sample_rate: int,
                   utter_id: str, frame_fps: list[str], harmonic_count: int) -> np.ndarray:
        n_frames = len(frames)
        phases = np.zeros((n_frames, harmonic_count), dtype=np.int64)
        current = np.array(
            [self.start_phase(utter_id, frame_fps[0], h + 1)
             for h in range(harmonic_count)],
            dtype=np.int64,
        )
        phases[0, :] = current
        float_phase = (current.astype(np.float64) / self.modulus) * TWO_PI
        for idx in range(1, n_frames):
            f0 = float(frames[idx - 1]["f0"])
            for h in range(harmonic_count):
                float_phase[h] += TWO_PI * (h + 1) * f0 * hop / sample_rate
                # Integer accumulation of the quantized absolute target phase;
                # the wrap itself stays entirely in fixed-point integers.
                target = self.quantize_absolute(float_phase[h])
                phases[idx, h] = target
                current[h] = target
        return phases

    def to_radians(self, phase_units: np.ndarray) -> np.ndarray:
        return phase_units.astype(np.float64) / self.modulus * TWO_PI


class Float64Phase:
    """Control path: accumulation and 2*pi wrap both in float64.

    The starting phase uses the same content-derived seed as the fixed-point
    path (mapped through a virtual 48 bit grid), so any difference between the
    two paths is attributable purely to fixed-point phase quantization.
    """

    VIRTUAL_BITS = 48

    def __init__(self):
        self.bits = None
        self.quant_step = None

    def start_phase(self, utter_id: str, frame_fp: str, harmonic: int) -> float:
        units = _seed_units(utter_id, frame_fp, harmonic, 1 << self.VIRTUAL_BITS)
        return units / (1 << self.VIRTUAL_BITS) * TWO_PI

    def accumulate(self, frames: list[dict], hop: int, sample_rate: int,
                   utter_id: str, frame_fps: list[str], harmonic_count: int) -> np.ndarray:
        n_frames = len(frames)
        phases = np.zeros((n_frames, harmonic_count), dtype=np.float64)
        current = np.array(
            [self.start_phase(utter_id, frame_fps[0], h + 1)
             for h in range(harmonic_count)],
            dtype=np.float64,
        )
        phases[0, :] = current
        for idx in range(1, n_frames):
            f0 = float(frames[idx - 1]["f0"])
            for h in range(harmonic_count):
                delta = TWO_PI * (h + 1) * f0 * hop / sample_rate
                current[h] = math.fmod(current[h] + delta, TWO_PI)
            phases[idx, :] = current
        return phases

    def to_radians(self, phase_radians: np.ndarray) -> np.ndarray:
        return phase_radians
