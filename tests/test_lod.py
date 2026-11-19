import pytest

from terrain_workbench.camera import Camera
from terrain_workbench.errors import E_LOD_LEVEL, TerrainError
from terrain_workbench.quadtree import density_cells
from terrain_workbench.selection import geom_bound_world, select_lod

CAM = Camera(0, (32.0, 8.0, 87.5), (56.0, 62.0, 11.0), 60.0, (1280, 720))


def test_selection_deterministic(quadtree257, pyramid257):
    _, meta = pyramid257
    a = select_lod(quadtree257, meta["cumulative_dev"], CAM, tau=4.0)
    b = select_lod(quadtree257, meta["cumulative_dev"], CAM, tau=4.0)
    assert [x.key.as_tuple() for x in a] == [x.key.as_tuple() for x in b]
    assert [x.manifest_entry() for x in a] == [x.manifest_entry() for x in b]


def test_error_bound_monotonic_in_density(quadtree257, pyramid257):
    _, meta = pyramid257
    bounds = [geom_bound_world(quadtree257, meta["cumulative_dev"], 1, 0, 0, g)
              for g in range(6)]
    assert all(bounds[i] <= bounds[i + 1] for i in range(5))


def test_screen_error_within_tau(quadtree257, pyramid257):
    _, meta = pyramid257
    blocks = select_lod(quadtree257, meta["cumulative_dev"], CAM, tau=4.0)
    assert blocks, "expected visible blocks"
    assert max(b.screen_error for b in blocks) <= 4.0


def test_density_level_validation():
    with pytest.raises(TerrainError) as exc:
        density_cells(6)
    assert exc.value.code == E_LOD_LEVEL


def test_block_validation(quadtree257):
    with pytest.raises(TerrainError) as exc:
        quadtree257.validate_block(99, 0, 0)
    assert exc.value.code == E_LOD_LEVEL
    with pytest.raises(TerrainError) as exc:
        quadtree257.validate_block(1, 5, 0)
    assert exc.value.code == E_LOD_LEVEL
