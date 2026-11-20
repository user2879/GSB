"""8-bit quantization with declared step and dither switch.

Quantization step: 1/255 (code value 255 == normalized 1.0).
Dither off (default and the only deterministic mode): round-half-to-even
via numpy ``rint`` on the scaled float32 value, then cast to uint8.
Dither on: symmetric triangular dither from a seeded RNG (non-deterministic
across pipelines unless a seed is supplied); never used by the deterministic
runs and kept behind an explicit flag.
"""

import numpy as np

from .config import QUANT_STEP


def quantize_array(x, state, *, rng=None):
    clipped = np.clip(x, np.float32(0.0), np.float32(1.0))
    scaled = clipped * np.float32(255.0)
    if state.dither:
        if rng is None:
            raise ValueError("dither=True requires an explicit seeded rng")
        # Triangular dither in (-1, 1), one LSB amplitude.
        dither = rng.random(scaled.shape, dtype=np.float32) + rng.random(
            scaled.shape, dtype=np.float32
        ) - np.float32(1.0)
        scaled = scaled + dither
    codes = np.rint(scaled).astype(np.uint8)
    return codes


def quantization_step():
    return QUANT_STEP
