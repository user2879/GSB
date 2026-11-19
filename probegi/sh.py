"""Real orthonormal spherical harmonics (order 1..3), projection and
reconstruction.

Normalisation contract (see also sampling.py):

* Basis: real, orthonormal over the sphere (``int Y_lm Y_l'm' dw = delta``).
* Projection: ``c_lm = sum_i w_i Y_lm(d_i) L(d_i)`` with ``sum_i w_i = 4*pi``.
  The coefficient of band ``l`` does not depend on the expansion order, so
  low-order coefficients are invariant under order changes.
* Radiance reconstruction: ``L(d) ~= sum_lm c_lm Y_lm(d)``.
* Irradiance (Lambertian convolution): ``E(n) = sum_lm A_l c_lm Y_lm(n)``
  with ``A = [pi, 2*pi/3, pi/4, 0]`` (A_3 vanishes analytically; order-3
  terms still contribute to radiance reconstruction).
"""
import numpy as np

from .errors import SHOrderError
from .sampling import check_direction_count, quadrature_weights

MIN_ORDER = 1
MAX_ORDER = 3

# Lambertian convolution kernel per band.
AZIMUTHAL_KERNEL = (np.pi, 2.0 * np.pi / 3.0, np.pi / 4.0, 0.0)


def check_order(order: int) -> int:
    order = int(order)
    if order < MIN_ORDER or order > MAX_ORDER:
        raise SHOrderError("SH order out of range", order=order,
                           min_order=MIN_ORDER, max_order=MAX_ORDER)
    return order


def num_coeffs(order: int) -> int:
    return (order + 1) ** 2


def sh_basis(directions: np.ndarray, order: int) -> np.ndarray:
    """Real SH basis values, directions (N,3) unit -> (N, (order+1)^2)."""
    order = check_order(order)
    d = np.asarray(directions, dtype=np.float64)
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    cols = [np.full_like(z, 0.28209479177387814)]  # Y00
    if order >= 1:
        c1 = 0.4886025119029199
        cols += [c1 * y, c1 * z, c1 * x]
    if order >= 2:
        cols += [
            1.0925484305920792 * x * y,
            1.0925484305920792 * y * z,
            0.31539156525352009 * (3.0 * z * z - 1.0),
            1.0925484305920792 * x * z,
            0.5462742152960396 * (x * x - y * y),
        ]
    if order >= 3:
        zz1 = 5.0 * z * z - 1.0
        cols += [
            0.5900435899266435 * y * (3.0 * x * x - y * y),
            2.890611442640554 * x * y * z,
            0.4570457994644658 * y * zz1,
            0.3731763325901154 * z * (5.0 * z * z - 3.0),
            0.4570457994644658 * x * zz1,
            1.445305721320277 * z * (x * x - y * y),
            0.5900435899266435 * x * (x * x - 3.0 * y * y),
        ]
    return np.stack(cols, axis=1)


def kernel_scale(order: int) -> np.ndarray:
    """Per-coefficient Lambertian kernel scale, shape ((order+1)^2,)."""
    order = check_order(order)
    scale = np.empty(num_coeffs(order), dtype=np.float64)
    idx = 0
    for l in range(order + 1):
        for _m in range(-l, l + 1):
            scale[idx] = AZIMUTHAL_KERNEL[l]
            idx += 1
    return scale


def project(radiance: np.ndarray, directions: np.ndarray, order: int,
            chunk: int = 4096) -> np.ndarray:
    """Project incident radiance samples to SH coefficients.

    radiance: (P, D, 3) samples at ``directions`` (D, 3).
    Returns coefficients (P, (order+1)^2, 3) float64.  Deterministic for a
    fixed (radiance, directions, order, chunk) tuple.
    """
    order = check_order(order)
    radiance = np.asarray(radiance)
    directions = np.asarray(directions, dtype=np.float64)
    check_direction_count(directions.shape[0], order)
    w = quadrature_weights(directions.shape[0])
    yw = sh_basis(directions, order) * w[:, None]  # (D, K)
    p_count = radiance.shape[0]
    out = np.empty((p_count, yw.shape[1], 3), dtype=np.float64)
    for start in range(0, p_count, chunk):
        block = np.asarray(radiance[start:start + chunk], dtype=np.float64)
        out[start:start + chunk] = np.einsum("pdc,dk->pkc", block, yw)
    return out


def reconstruct_radiance(coeffs: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """coeffs (..., K, 3), directions (N, 3) -> radiance (..., N, 3)."""
    coeffs = np.asarray(coeffs, dtype=np.float64)
    order = int(np.sqrt(coeffs.shape[-2])) - 1
    y = sh_basis(np.asarray(directions, dtype=np.float64), order)
    return np.einsum("...kc,nk->...nc", coeffs, y)


def irradiance(coeffs: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Lambertian irradiance. coeffs (..., K, 3), normals (..., 3) -> (..., 3).

    E(n) = sum_lm A_l c_lm Y_lm(n); the band kernel A_l is folded into the
    basis values.
    """
    coeffs = np.asarray(coeffs, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    order = int(np.sqrt(coeffs.shape[-2])) - 1
    flat = normals.reshape(-1, 3)
    y = sh_basis(flat, order) * kernel_scale(order)[None, :]
    y = y.reshape(normals.shape[:-1] + (y.shape[1],))
    return np.einsum("...kc,...k->...c", coeffs, y)
