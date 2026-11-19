"""Block mesh boundary conforming and crack detection.

Declared stitching scheme: EDGE SNAPPING TO THE COARSER NEIGHBOR.
Every block edge is sampled at the vertex stride of the coarser of the two
blocks sharing it. Concretely, for an edge with own stride s and neighbor
stride sN, the effective edge stride is s_edge = max(s, sN); non-corner
boundary vertices whose global position is not a multiple of s_edge are
snapped onto the piecewise-linear polyline through the heightfield values at
multiples of s_edge. Corner vertices always take the exact heightfield value.

Because both blocks evaluate the SAME analytic edge profile (global
coordinates, same heightfield, same integer strides), the shared boundary is
vertex-identical on both sides for any level difference (1..3 and beyond):
no cracks, no T-junctions, no overlapping triangles (block interiors are
disjoint in XY by quadtree construction).

Crack detection samples every shared edge at the finer stride and compares
each side's boundary vertices against the analytic profile bit-exactly
(tolerance 1e-9 height units). Any violation raises E_CRACK_GAP with the
first offending block coordinate.
"""

from __future__ import annotations

import numpy as np

from .errors import E_CRACK_GAP, TerrainError
from .quadtree import Quadtree
from .selection import BlockKey, SelectedBlock

EDGES = ("W", "E", "S", "N")
CRACK_TOLERANCE = 1e-9


def block_stride(qt: Quadtree, depth: int, g: int) -> int:
    span = qt.max_depth - depth
    return 1 << (span + g)


def block_region_cells(qt: Quadtree, depth: int, bx: int, by: int):
    span = qt.block_cells(depth)
    return bx * span, by * span, span


def edge_profile_positions(y0: int, span: int, stride: int) -> np.ndarray:
    return y0 + np.arange(span // stride + 1, dtype=np.int64) * stride


def snapped_edge_values(heights: np.ndarray, fixed: int, positions: np.ndarray,
                        s_edge: int, axis: str) -> np.ndarray:
    """Snapped boundary heights at the given global positions (stride s_edge)."""
    base = (positions // s_edge) * s_edge
    nxt = np.minimum(base + s_edge, heights.shape[0] - 1)
    t = (positions - base).astype(np.float64) / s_edge
    if axis == "x":
        h0 = heights[fixed, base].astype(np.float64)
        h1 = heights[fixed, nxt].astype(np.float64)
    else:
        h0 = heights[base, fixed].astype(np.float64)
        h1 = heights[nxt, fixed].astype(np.float64)
    out = h0 * (1.0 - t) + h1 * t
    # positions that are exact multiples keep the exact heightfield value
    exact = (positions % s_edge) == 0
    if axis == "x":
        out[exact] = heights[fixed, positions[exact]]
    else:
        out[exact] = heights[positions[exact], fixed]
    return out


def block_vertex_heights(qt: Quadtree, heights: np.ndarray, key: BlockKey,
                         neighbor_strides: dict | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full (n+1, n+1) vertex height grid for a block, edges snapped.

    neighbor_strides: mapping edge -> neighbor stride (only edges with a
    coarser neighbor matter). Returns (heights_f64, global_i, global_j).
    """
    s = block_stride(qt, key.depth, key.density)
    x0, y0, span = block_region_cells(qt, key.depth, key.bx, key.by)
    ii = edge_profile_positions(x0, span, s)
    jj = edge_profile_positions(y0, span, s)
    grid = heights[np.ix_(jj, ii)].astype(np.float64)
    neighbor_strides = neighbor_strides or {}
    # corners keep exact heightfield values: snap only interior of each edge
    if "W" in neighbor_strides:
        grid[1:-1, 0] = snapped_edge_values(heights, x0, jj[1:-1],
                                            max(s, neighbor_strides["W"]), "y")
    if "E" in neighbor_strides:
        grid[1:-1, -1] = snapped_edge_values(heights, x0 + span, jj[1:-1],
                                             max(s, neighbor_strides["E"]), "y")
    if "S" in neighbor_strides:
        grid[0, 1:-1] = snapped_edge_values(heights, y0, ii[1:-1],
                                            max(s, neighbor_strides["S"]), "x")
    if "N" in neighbor_strides:
        grid[-1, 1:-1] = snapped_edge_values(heights, y0 + span, ii[1:-1],
                                             max(s, neighbor_strides["N"]), "x")
    return grid, ii, jj


def build_neighbor_map(qt: Quadtree, blocks: list[SelectedBlock]):
    """For every block edge, list the selected blocks sharing it.

    Returns dict (block_key, edge) -> list of neighbor BlockKey, plus a list
    of unique shared-edge records for crack detection:
    (key_a, key_b, edge_of_a).
    """
    regions = {}
    for b in blocks:
        k = b.key
        x0, y0, span = block_region_cells(qt, k.depth, k.bx, k.by)
        regions[k] = (x0, y0, span)

    by_x0: dict[int, list] = {}
    by_y0: dict[int, list] = {}
    for k, (x0, y0, span) in regions.items():
        by_x0.setdefault(x0, []).append((y0, span, k))
        by_y0.setdefault(y0, []).append((x0, span, k))
    for lst in by_x0.values():
        lst.sort()
    for lst in by_y0.values():
        lst.sort()

    neighbors: dict = {(b.key, e): [] for b in blocks for e in EDGES}
    shared = []
    for b in blocks:
        k = b.key
        x0, y0, span = regions[k]
        x1, y1 = x0 + span, y0 + span
        # east edge of k == west edge of blocks with x0 == x1
        for (ny0, nspan, nk) in by_x0.get(x1, []):
            if ny0 < y1 and ny0 + nspan > y0:
                neighbors[(k, "E")].append(nk)
                neighbors[(nk, "W")].append(k)
                shared.append((k, nk, "E"))
        # north edge of k == south edge of blocks with y0 == y1
        for (nx0, nspan, nk) in by_y0.get(y1, []):
            if nx0 < x1 and nx0 + nspan > x0:
                neighbors[(k, "N")].append(nk)
                neighbors[(nk, "S")].append(k)
                shared.append((k, nk, "N"))
    return neighbors, shared, regions


def neighbor_stride_map(qt: Quadtree, key: BlockKey, neighbors) -> dict:
    """Edge -> max neighbor stride, only where the neighbor is coarser."""
    own = block_stride(qt, key.depth, key.density)
    out = {}
    for edge in EDGES:
        strides = [block_stride(qt, nk.depth, nk.density)
                   for nk in neighbors.get((key, edge), [])]
        if strides and max(strides) > own:
            out[edge] = max(strides)
    return out


def _boundary_polyline(qt, heights, key, edge, neighbors, regions, cache):
    """Boundary polyline of a block edge: (global_positions, heights_f64)."""
    if key not in cache:
        ns = neighbor_stride_map(qt, key, neighbors)
        cache[key] = block_vertex_heights(qt, heights, key, ns)
    grid, ii, jj = cache[key]
    if edge == "E":
        return jj.copy(), grid[:, -1]
    if edge == "W":
        return jj.copy(), grid[:, 0]
    if edge == "N":
        return ii.copy(), grid[-1, :]
    return ii.copy(), grid[0, :]


_OPPOSITE = {"E": "W", "W": "E", "N": "S", "S": "N"}


def check_cracks(qt: Quadtree, heights: np.ndarray, blocks: list[SelectedBlock],
                 neighbors=None, shared=None, regions=None) -> dict:
    """Per-edge sampling comparison of every shared boundary.

    Each side's rendered boundary polyline (after edge snapping) is evaluated
    at the union of both sides' vertex positions; the two polylines must
    coincide within CRACK_TOLERANCE everywhere. Raises E_CRACK_GAP with the
    first violating block coordinate.
    """
    if neighbors is None or shared is None or regions is None:
        neighbors, shared, regions = build_neighbor_map(qt, blocks)

    cache: dict = {}
    max_gap = 0.0
    for (ka, kb, edge) in shared:
        pos_a, h_a = _boundary_polyline(qt, heights, ka, edge, neighbors, regions, cache)
        pos_b, h_b = _boundary_polyline(qt, heights, kb, _OPPOSITE[edge], neighbors,
                                        regions, cache)
        lo = max(int(pos_a[0]), int(pos_b[0]))
        hi = min(int(pos_a[-1]), int(pos_b[-1]))
        union = np.union1d(pos_a, pos_b)
        union = union[(union >= lo) & (union <= hi)]
        if union.size == 0:
            continue
        eval_a = np.interp(union, pos_a, h_a)
        eval_b = np.interp(union, pos_b, h_b)
        gap = float(np.abs(eval_a - eval_b).max())
        max_gap = max(max_gap, gap)
        if gap > CRACK_TOLERANCE:
            raise TerrainError(
                E_CRACK_GAP,
                f"boundary gap {gap:.3e} height units on shared edge {edge}",
                block=ka.as_tuple(),
                other=kb.as_tuple(),
                edge=edge,
                gap=gap,
            )
    return {"edges_checked": len(shared), "max_gap": max_gap}
