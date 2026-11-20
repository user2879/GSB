"""Fixed RGB primary matrices and luma coefficients.

The synthetic generator authors frames in BT.709/sRGB linear primaries.
Targeting BT.709 is therefore the identity; targeting BT.2020 applies the
fixed BT.709 -> BT.2020 linear RGB matrix.

Operation order is fixed and explicit: for every pixel, each output channel
is the float32 dot product of one matrix row and the input RGB triple,
accumulated left-to-right in float32 (no BLAS call) so the result is
byte-identical across platforms.
"""

import numpy as np

# BT.709 -> BT.2020 linear RGB matrix (fixed constants, float32).
_BT709_TO_BT2020 = (
    (0.62740389, 0.32928304, 0.04331307),
    (0.06909729, 0.91954039, 0.01136232),
    (0.01639144, 0.08801331, 0.89559525),
)

LUMA_BT709 = (0.21263901, 0.71516868, 0.07219232)
LUMA_BT2020 = (0.26270021, 0.67799807, 0.05930172)


def matrix_for(colorspace):
    if colorspace == "bt709":
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    if colorspace == "bt2020":
        return _BT709_TO_BT2020
    raise ValueError(f"unknown colorspace {colorspace!r}")


def luma_coeffs(colorspace):
    if colorspace == "bt709":
        return LUMA_BT709
    if colorspace == "bt2020":
        return LUMA_BT2020
    raise ValueError(f"unknown colorspace {colorspace!r}")


def apply_matrix(rgb, colorspace):
    """Apply the fixed matrix with explicit float32 multiply-add order.

    rgb: float32 HxWx3. Returns float32 HxWx3.
    """
    matrix = matrix_for(colorspace)
    m = np.array(matrix, dtype=np.float32)
    out = np.empty_like(rgb)
    for channel in range(3):
        acc = rgb[..., 0] * np.float32(m[channel, 0])
        acc = acc + rgb[..., 1] * np.float32(m[channel, 1])
        acc = acc + rgb[..., 2] * np.float32(m[channel, 2])
        out[..., channel] = acc
    return out


def luma(rgb, colorspace):
    """Luminance as explicit float32 dot product with fixed coefficients."""
    kr, kg, kb = (np.float32(c) for c in luma_coeffs(colorspace))
    y = rgb[..., 0] * kr
    y = y + rgb[..., 1] * kg
    y = y + rgb[..., 2] * kb
    return y
