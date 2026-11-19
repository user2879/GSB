"""Deterministic sample-direction generation and quadrature weights.

Declaration of the sampling / weighting convention (normalisation contract):

* Directions are a Fibonacci lattice on the unit sphere, generated from the
  golden angle.  For a fixed count ``n`` the lattice is exactly reproducible.
* Every direction carries the uniform quadrature weight ``w_i = 4*pi / n`` so
  that ``sum_i w_i = 4*pi`` exactly (up to float64 summation order, which is
  fixed because the lattice order is fixed).
* Spherical-harmonic projection uses the Monte-Carlo / quadrature form
  ``c_lm = sum_i w_i Y_lm(d_i) L(d_i)``.  Because the weight of a basis
  function does not depend on the expansion order, the low-order coefficients
  of an order-3 projection are bit-identical to those of an order-1 or
  order-2 projection of the same samples (order truncation is semantics
  preserving).
"""
import numpy as np

from .errors import SampleDirectionError

GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))

# Minimum directions required per SH order: at least 4x the coefficient count
# so the quadrature matrix is comfortably over-determined.
MIN_DIRECTIONS_FACTOR = 4


def min_directions(order: int) -> int:
    return MIN_DIRECTIONS_FACTOR * (order + 1) ** 2


def fibonacci_sphere(n: int) -> np.ndarray:
    """n deterministic unit directions, z-up, shape (n, 3) float64."""
    if n < 1:
        raise SampleDirectionError("direction count must be positive", n_directions=n)
    i = np.arange(n, dtype=np.float64)
    z = 1.0 - 2.0 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = i * GOLDEN_ANGLE
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def quadrature_weights(n: int) -> np.ndarray:
    return np.full(n, 4.0 * np.pi / n, dtype=np.float64)


def check_direction_count(n: int, order: int) -> None:
    if n < min_directions(order):
        raise SampleDirectionError(
            "insufficient sample directions for SH order",
            n_directions=n,
            order=order,
            required=min_directions(order),
        )
