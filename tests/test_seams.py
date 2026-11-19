import numpy as np
import pytest

import terrain_workbench.mesh as mesh
from terrain_workbench.camera import Camera
from terrain_workbench.errors import E_CRACK_GAP, TerrainError
from terrain_workbench.mesh import (
    block_vertex_heights, build_neighbor_map, check_cracks, neighbor_stride_map,
)
from terrain_workbench.selection import BlockKey, SelectedBlock, select_lod

CAM = Camera(0, (32.0, 8.0, 87.5), (56.0, 62.0, 11.0), 60.0, (1280, 720))


def _block(qt, depth, bx, by, g):
    key = BlockKey(depth, bx, by, g)
    return SelectedBlock(key, 0.0, 0.0, 1.0, 2 * (32 >> g) ** 2)


def _pair(qt, g_fine, g_coarse):
    """Two adjacent blocks at depth 1 with a controlled stride difference."""
    return [
        _block(qt, 1, 0, 0, g_fine),
        _block(qt, 1, 1, 0, g_coarse),
    ]


@pytest.mark.parametrize("diff", [1, 2, 3])
def test_no_cracks_for_level_differences(quadtree257, heights257, diff):
    blocks = _pair(quadtree257, 0, diff)
    result = check_cracks(quadtree257, heights257, blocks)
    assert result["edges_checked"] == 1
    assert result["max_gap"] == 0.0


def test_shared_boundary_vertices_identical(quadtree257, heights257):
    qt = quadtree257
    blocks = _pair(qt, 0, 3)
    neighbors, shared, regions = build_neighbor_map(qt, blocks)
    ka, kb = blocks[0].key, blocks[1].key
    ga, ia, ja = block_vertex_heights(qt, heights257, ka, neighbor_stride_map(qt, ka, neighbors))
    gb, ib, jb = block_vertex_heights(qt, heights257, kb, neighbor_stride_map(qt, kb, neighbors))
    # A's east edge and B's west edge: sample B's (coarser) vertex positions on both
    pos_b = jb  # global y of B's vertices
    a_east = np.interp(pos_b, ja, ga[:, -1])
    b_west = gb[:, 0]
    assert np.array_equal(a_east, b_west)


def test_real_selection_has_no_cracks(quadtree257, pyramid257, heights257):
    _, meta = pyramid257
    blocks = select_lod(quadtree257, meta["cumulative_dev"], CAM, tau=4.0)
    result = check_cracks(quadtree257, heights257, blocks)
    assert result["edges_checked"] > 0
    assert result["max_gap"] == 0.0


def test_unsnapped_would_crack(quadtree257, heights257):
    """Sanity: without edge snapping the fine edge would NOT lie on the coarse
    polyline, i.e. the detector is checking something real."""
    qt = quadtree257
    heights = heights257
    x_fixed = 128  # shared edge column between the two depth-1 blocks
    coarse = heights[0:257:128, x_fixed].astype(np.float64)  # B at g=3: stride 32... use analytic
    # fine block A at g=0 stride 4: raw heights along the edge
    fine_pos = np.arange(0, 129, 4)
    fine_raw = heights[fine_pos, x_fixed].astype(np.float64)
    coarse_poly = np.interp(fine_pos, [0, 128], [heights[0, x_fixed], heights[128, x_fixed]])
    assert np.abs(fine_raw - coarse_poly).max() > mesh.CRACK_TOLERANCE


def test_crack_detection_raises_with_block_coord(quadtree257, heights257, monkeypatch):
    blocks = _pair(quadtree257, 0, 0)
    monkeypatch.setattr(mesh, "CRACK_TOLERANCE", -1.0)  # force any edge to fail
    with pytest.raises(TerrainError) as exc:
        check_cracks(quadtree257, heights257, blocks)
    assert exc.value.code == E_CRACK_GAP
    assert exc.value.block is not None
