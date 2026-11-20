"""Output transfer function (OETF) encoding to [0,1] float32.

srgb (IEC 61966-2-1), linear segment below 0.0031308:
    C <= 0.0031308 -> 12.92 * C
    otherwise      -> 1.055 * C**(1/2.4) - 0.055
gamma:
    C ** (1/gamma), gamma legal in [0.1, 10.0] (E_INJECT_BOUNDS otherwise).

Inputs are clamped to [0,1] (the tone mapped domain); exponentiation is done
in float32 with the fixed constants below.
"""

import numpy as np

from .config import GAMMA_MAX, GAMMA_MIN
from .errors import InjectBoundsError

SRGB_K0 = np.float32(0.0031308)
SRGB_PHI = np.float32(12.92)
SRGB_ALPHA = np.float32(1.055)
SRGB_GAMMA = np.float32(2.4)
SRGB_OFFSET = np.float32(0.055)


def srgb_scalar(c):
    c = max(0.0, min(1.0, float(c)))
    if c <= float(SRGB_K0):
        return float(SRGB_PHI) * c
    return float(SRGB_ALPHA) * c ** (1.0 / float(SRGB_GAMMA)) - float(
        SRGB_OFFSET
    )


def gamma_scalar(c, gamma):
    if not GAMMA_MIN <= float(gamma) <= GAMMA_MAX:
        raise InjectBoundsError(
            f"transfer gamma {gamma} outside [{GAMMA_MIN}, {GAMMA_MAX}]",
            details={"gamma": gamma},
        )
    c = max(0.0, min(1.0, float(c)))
    return c ** (1.0 / float(gamma))


def encode_array(x, state):
    clipped = np.clip(x, np.float32(0.0), np.float32(1.0))
    if state.transfer == "srgb":
        low = clipped <= SRGB_K0
        out = np.empty_like(clipped)
        out[low] = SRGB_PHI * clipped[low]
        high = ~low
        out[high] = (
            SRGB_ALPHA * np.power(clipped[high], np.float32(1.0 / 2.4))
            - SRGB_OFFSET
        )
        return np.clip(out, np.float32(0.0), np.float32(1.0))
    if state.transfer == "gamma":
        gamma = float(state.transfer_gamma)
        if not GAMMA_MIN <= gamma <= GAMMA_MAX:
            raise InjectBoundsError(
                f"transfer gamma {gamma} outside [{GAMMA_MIN}, {GAMMA_MAX}]",
                details={"gamma": gamma},
            )
        out = np.power(clipped, np.float32(1.0 / gamma))
        return np.clip(out, np.float32(0.0), np.float32(1.0))
    raise InjectBoundsError(
        f"illegal transfer {state.transfer!r}",
        details={"transfer": state.transfer},
    )
