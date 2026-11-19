from __future__ import annotations

import numpy as np
import pytest

from spatial_diag.sh import component_descriptor, component_index, real_sh
from spatial_diag.sphere import fibonacci_sphere, spherical_weights


def test_real_sh_is_orthonormal_under_fibonacci_quadrature():
    directions = fibonacci_sphere(12000)
    basis = real_sh(directions, 4)
    gram = (basis.T @ basis) / len(directions) * (4.0 * np.pi)
    assert gram.shape == (25, 25)
    assert np.max(np.abs(gram - np.eye(25))) < 0.02


def test_acn_component_order_round_trip():
    for order_l in range(5):
        for order_m in range(-order_l, order_l + 1):
            index = component_index(order_l, order_m)
            descriptor = component_descriptor(index)
            assert descriptor["l"] == order_l
            assert descriptor["m"] == order_m


def test_fibonacci_layout_is_equal_area_and_deterministic():
    left = fibonacci_sphere(200)
    right = fibonacci_sphere(200)
    np.testing.assert_array_equal(left, right)
    np.testing.assert_allclose(np.linalg.norm(left, axis=1), 1.0, atol=1.0e-12)
    weights = spherical_weights(200)
    assert weights.shape == (200,)
    assert np.allclose(weights, 4.0 * np.pi / 200.0)


def test_unsupported_order_is_rejected_by_basis():
    with pytest.raises(ValueError):
        real_sh(fibonacci_sphere(4), -1)

