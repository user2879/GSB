"""Tone mapping operators with frozen formulas and domain handling.

Reinhard (extended, white-point corrected):
    f(x) = x * (1 + x / W^2) / (1 + x),  W = tonemap_white
ACES fitted curve (Narkowicz 2015 RRTAndODTFit):
    f(x) = (x*(A*x+B)) / (x*(C*x+D)+E)

Domain policy (deterministic, no undefined behavior):
  * non-finite input -> E_TONEMAP_DOMAIN
  * x < 0            -> clamped to 0 before evaluation
  * x > TONEMAP_INPUT_MAX (64.0) -> clamped to 64.0 (HDR overflow / highlight
    saturation); the operator output is then clamped to [0, 1].
"""

import numpy as np

from .config import (
    ACES_A, ACES_B, ACES_C, ACES_D, ACES_E,
    TONEMAP_INPUT_MAX, TONEMAP_OUTPUT_MAX, TONEMAP_OUTPUT_MIN,
)
from .errors import TonemapDomainError


def _check_finite(x, frame_index, stage):
    if not np.all(np.isfinite(x)):
        bad = int(np.count_nonzero(~np.isfinite(x)))
        raise TonemapDomainError(
            "non-finite value reached tone mapping",
            details={"frame": frame_index, "stage": stage,
                     "non_finite_pixels": bad},
        )


def reinhard_scalar(x, white):
    x = max(0.0, min(float(TONEMAP_INPUT_MAX), float(x)))
    w2 = float(white) * float(white)
    y = x * (1.0 + x / w2) / (1.0 + x)
    return max(TONEMAP_OUTPUT_MIN, min(TONEMAP_OUTPUT_MAX, y))


def aces_scalar(x, a, b, c, d, e):
    x = max(0.0, min(float(TONEMAP_INPUT_MAX), float(x)))
    y = (x * (a * x + b)) / (x * (c * x + d) + e)
    return max(TONEMAP_OUTPUT_MIN, min(TONEMAP_OUTPUT_MAX, y))


def tonemap_array(x, state, frame_index):
    """Vectorized, fixed-order float32 tone mapping."""
    _check_finite(x, frame_index, "tonemap")
    clipped = np.clip(x, np.float32(0.0), np.float32(TONEMAP_INPUT_MAX))
    if state.tonemap_operator == "reinhard":
        white = np.float32(state.tonemap_white)
        w2 = white * white
        out = clipped * (np.float32(1.0) + clipped / w2) / (
            np.float32(1.0) + clipped
        )
    elif state.tonemap_operator == "aces":
        a = np.float32(state.tonemap_a)
        b = np.float32(state.tonemap_b)
        c = np.float32(state.tonemap_c)
        d = np.float32(state.tonemap_d)
        e = np.float32(state.tonemap_e)
        numerator = clipped * (a * clipped + b)
        denominator = clipped * (c * clipped + d) + e
        out = numerator / denominator
    else:  # validated earlier, defensive guard
        from .errors import InjectBoundsError

        raise InjectBoundsError(
            f"illegal tonemap operator {state.tonemap_operator!r}")
    return np.clip(
        out, np.float32(TONEMAP_OUTPUT_MIN), np.float32(TONEMAP_OUTPUT_MAX)
    )


def default_aces_constants():
    return {"a": ACES_A, "b": ACES_B, "c": ACES_C, "d": ACES_D, "e": ACES_E}
