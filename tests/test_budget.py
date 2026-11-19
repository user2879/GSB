import pytest

from terrain_workbench.budget import enforce_triangle_budget
from terrain_workbench.camera import Camera
from terrain_workbench.errors import E_BUDGET_DENIED, TerrainError
from terrain_workbench.selection import select_lod

CAM = Camera(0, (32.0, 8.0, 87.5), (56.0, 62.0, 11.0), 60.0, (1280, 720))


def _select(qt, meta):
    return select_lod(qt, meta["cumulative_dev"], CAM, tau=4.0)


def test_degradation_deterministic_and_recorded(quadtree257, pyramid257):
    _, meta = pyramid257
    blocks = _select(quadtree257, meta)
    budget = sum(b.triangles for b in blocks) // 2
    out1, deg1 = enforce_triangle_budget(blocks, quadtree257, meta["cumulative_dev"], CAM, budget)
    out2, deg2 = enforce_triangle_budget(
        _select(quadtree257, meta), quadtree257, meta["cumulative_dev"], CAM, budget)
    assert [d.to_dict() for d in deg1] == [d.to_dict() for d in deg2]
    assert len(deg1) > 0
    assert sum(b.triangles for b in out1) <= budget
    # no visible block is ever dropped
    assert len(out1) == len(blocks)
    # every degradation records reason and error before/after
    for d in deg1:
        assert d.reason == "triangle_budget"
        assert d.to_density == d.from_density + 1
        assert d.error_after >= d.error_before >= 0.0


def test_budget_unsatisfiable_raises(quadtree257, pyramid257):
    _, meta = pyramid257
    blocks = _select(quadtree257, meta)
    with pytest.raises(TerrainError) as exc:
        enforce_triangle_budget(blocks, quadtree257, meta["cumulative_dev"], CAM, 1)
    assert exc.value.code == E_BUDGET_DENIED
    assert exc.value.block is not None


def test_budget_already_met_is_noop(quadtree257, pyramid257):
    _, meta = pyramid257
    blocks = _select(quadtree257, meta)
    out, deg = enforce_triangle_budget(
        blocks, quadtree257, meta["cumulative_dev"], CAM, 10 ** 9)
    assert deg == [] and out == blocks
