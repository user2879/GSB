import numpy as np
import pytest

from probegi import probes, sh
from probegi.errors import SHOrderError, SampleDirectionError
from probegi.sampling import fibonacci_sphere


@pytest.fixture(scope="module")
def probe_data():
    positions, directions, radiance, _ = probes.generate(8, 99, 64)
    return directions, radiance


def test_projection_byte_deterministic(probe_data):
    directions, radiance = probe_data
    c1 = sh.project(radiance, directions, 2)
    c2 = sh.project(radiance, directions, 2)
    assert c1.tobytes() == c2.tobytes()


def test_low_order_coefficients_invariant_across_orders(probe_data):
    directions, radiance = probe_data
    c1 = sh.project(radiance, directions, 1)
    c2 = sh.project(radiance, directions, 2)
    c3 = sh.project(radiance, directions, 3)
    assert np.array_equal(c1, c2[:, :4])
    assert np.array_equal(c2, c3[:, :9])
    assert np.array_equal(c1, c3[:, :4])


def test_order_validation(probe_data):
    directions, radiance = probe_data
    with pytest.raises(SHOrderError) as exc:
        sh.project(radiance, directions, 4)
    assert exc.value.code == "E_SH_ORDER"
    with pytest.raises(SHOrderError):
        sh.project(radiance, directions, 0)


def test_insufficient_directions(probe_data):
    directions, radiance = probe_data
    with pytest.raises(SampleDirectionError) as exc:
        sh.project(radiance[:, :16], directions[:16], 3)
    assert exc.value.code == "E_SAMPLE_DIR"


def test_dc_term_recovers_mean_radiance(probe_data):
    directions, radiance = probe_data
    coeffs = sh.project(radiance, directions, 1)
    # c00 = (1/sqrt(4pi)) * integral L dw  ->  mean radiance = c00 / sqrt(4pi)
    mean_rad = radiance.astype(np.float64).mean(axis=1)
    assert np.allclose(coeffs[:, 0, :] / np.sqrt(4 * np.pi), mean_rad,
                       rtol=1e-9, atol=1e-9)


def test_irradiance_positive_and_smooth(probe_data):
    directions, radiance = probe_data
    coeffs = sh.project(radiance, directions, 2)
    normals = fibonacci_sphere(32)
    e = sh.irradiance(coeffs[0], normals)  # one probe, many normals
    assert e.shape == (32, 3)
    assert (e > 0).all()
    e_all = sh.irradiance(coeffs, np.tile([0.0, 0.0, 1.0], (512, 1)))
    assert e_all.shape == (512, 3)
    assert (e_all > 0).all()
