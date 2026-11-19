import numpy as np
import pytest

from terrain_workbench.errors import E_HEIGHT_RANGE, TerrainError
from terrain_workbench.heightfield import (
    generate_heightfield, heightfield_hash, validate_elevation_range,
)


def test_determinism_byte_identical():
    a = generate_heightfield(257, 777)
    b = generate_heightfield(257, 777)
    assert a.dtype == np.uint16 and a.shape == (257, 257)
    assert heightfield_hash(a) == heightfield_hash(b)
    assert (a == b).all()


def test_seed_changes_output():
    a = generate_heightfield(257, 1)
    b = generate_heightfield(257, 2)
    assert heightfield_hash(a) != heightfield_hash(b)


def test_range_within_requested_bounds():
    h = generate_heightfield(257, 5, hmin=100, hmax=9000)
    assert h.min() >= 100 and h.max() <= 9000


@pytest.mark.parametrize("hmin,hmax", [(-1, 100), (100, 100), (5000, 100), (0, 70000)])
def test_illegal_elevation_range(hmin, hmax):
    with pytest.raises(TerrainError) as exc:
        generate_heightfield(257, 1, hmin=hmin, hmax=hmax)
    assert exc.value.code == E_HEIGHT_RANGE


def test_illegal_size():
    with pytest.raises(TerrainError) as exc:
        generate_heightfield(258, 1)
    assert exc.value.code == E_HEIGHT_RANGE
