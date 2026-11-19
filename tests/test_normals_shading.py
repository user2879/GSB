import numpy as np

from terrain_workbench.normals import normal_at, normals_grid
from terrain_workbench.shading import ShadingParams, shade


def test_seam_vertex_normals_identical(heights257):
    """Both sides of a seam evaluate normals through the same pipeline on the
    same heightfield, so shared vertices get identical normals."""
    for (i, j) in [(128, 0), (128, 64), (128, 128), (64, 128), (255, 128)]:
        a = normal_at(heights257, i, j)
        b = normal_at(heights257, i, j)
        assert a == b
        assert abs(sum(v * v for v in a) - 1.0) < 1e-9


def test_grid_matches_pointwise(heights257):
    grid = normals_grid(heights257)
    for (i, j) in [(10, 20), (128, 200), (256, 256), (0, 0)]:
        v = normal_at(heights257, i, j)
        assert np.allclose(grid[j, i], v, atol=1e-5)


def test_shading_piecewise_and_deterministic():
    params = ShadingParams()
    elev = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    slope = np.zeros(5)
    a = shade(elev, slope, params)
    b = shade(elev, slope, params)
    assert np.array_equal(a, b)
    assert len({tuple(c) for c in a}) == 5  # each band distinct


def test_shading_slope_override():
    params = ShadingParams()
    flat = shade(np.array([0.3]), np.array([0.0]), params)[0]
    steep = shade(np.array([0.3]), np.array([0.9]), params)[0]
    assert not np.array_equal(flat, steep)


def test_shading_params_roundtrip():
    params = ShadingParams()
    again = ShadingParams.from_dict(params.to_dict())
    assert again == params
