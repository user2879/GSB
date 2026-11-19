"""LOD selection: camera-distance + geometry-error driven, fully deterministic.

Error model (documented upper bound):
- A block rendered at (depth d, density g) samples the heightfield at stride
  2**(span+g) level-0 units, where span = max_depth - d. Its vertex error vs.
  the level-0 surface is bounded by
      geom_bound = min(cum_dev[span+g], block_range/2)   [uint16 units]
  cum_dev[k] is the pyramid's cumulative max deviation bound for level k
  (rigorous for bilinear reconstruction); block_range/2 bounds any
  approximation staying inside the block's elevation range.
- Screen-space error upper bound:
      rho = geom_bound * HEIGHT_SCALE * K / dist_min     [pixels]
  with K = viewport_h / (2*tan(fov_y/2)) and dist_min the closest distance
  from the camera to the block AABB.

Selection rule: keep the coarsest (d, g) whose rho <= tau; if even the
finest density exceeds tau, descend to children. Traversal order is fixed
(row-major children), so the emitted list is deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .camera import Camera
from .quadtree import MAX_DENSITY, Quadtree

# world-unit scales (fixed, part of the deterministic spec)
SAMPLE_SPACING = 1.0
HEIGHT_SCALE = 0.02


@dataclass(frozen=True)
class BlockKey:
    depth: int
    bx: int
    by: int
    density: int

    def as_tuple(self):
        return (self.depth, self.bx, self.by, self.density)


@dataclass
class SelectedBlock:
    key: BlockKey
    screen_error: float   # rho at chosen density [pixels]
    geom_bound: float     # world units
    dist_min: float       # world units
    triangles: int

    def manifest_entry(self):
        return {
            "block": list(self.key.as_tuple()),
            "screen_error": round(self.screen_error, 6),
            "geom_bound": round(self.geom_bound, 6),
            "dist_min": round(self.dist_min, 6),
            "triangles": self.triangles,
        }


def block_aabb(qt: Quadtree, depth: int, bx: int, by: int):
    span = qt.block_cells(depth)
    x0, y0 = bx * span * SAMPLE_SPACING, by * span * SAMPLE_SPACING
    x1, y1 = x0 + span * SAMPLE_SPACING, y0 + span * SAMPLE_SPACING
    hmin, hmax = qt.block_range(depth, bx, by)
    return (x0, y0, hmin * HEIGHT_SCALE), (x1, y1, hmax * HEIGHT_SCALE)


def aabb_dist_min(lo, hi, point) -> float:
    d = 0.0
    for a, b, p in zip(lo, hi, point):
        if p < a:
            d += (a - p) ** 2
        elif p > b:
            d += (p - b) ** 2
    return math.sqrt(d)


def geom_bound_world(qt: Quadtree, cum_dev, depth: int, bx: int, by: int, g: int) -> float:
    span = qt.max_depth - depth
    k = min(span + g, len(cum_dev) - 1)
    hmin, hmax = qt.block_range(depth, bx, by)
    bound_units = min(cum_dev[k], (hmax - hmin) / 2.0)
    return bound_units * HEIGHT_SCALE


def screen_error(geom_world: float, dist_min: float, cam: Camera) -> float:
    return geom_world * cam.projection_k / max(dist_min, 1e-3)


def triangles_for(g: int) -> int:
    cells = 32 >> g
    return 2 * cells * cells


def _in_frustum(cam: Camera, lo, hi) -> bool:
    """Conservative sphere-vs-view-cone test (deterministic)."""
    center = np.array([(lo[i] + hi[i]) / 2.0 for i in range(3)])
    radius = math.sqrt(sum(((hi[i] - lo[i]) / 2.0) ** 2 for i in range(3)))
    to_c = center - np.asarray(cam.pos)
    dist = float(np.linalg.norm(to_c))
    if dist <= radius:
        return True
    view = cam.view_dir
    cosang = float(np.dot(to_c, view) / dist)
    if cosang < 0.0 and dist > radius:
        return False
    # diagonal half-angle of the frustum plus the sphere's angular radius
    fov_y = math.radians(cam.fov_y_deg)
    aspect = cam.viewport[0] / cam.viewport[1]
    half_diag = math.atan(math.tan(fov_y / 2.0) * math.sqrt(1.0 + aspect * aspect))
    margin = math.asin(min(radius / dist, 1.0))
    return math.acos(max(min(cosang, 1.0), -1.0)) <= half_diag + margin


def select_lod(qt: Quadtree, cum_dev, cam: Camera, tau: float) -> list[SelectedBlock]:
    """Top-down deterministic LOD selection. Returns blocks in traversal order."""
    out: list[SelectedBlock] = []

    def best_density(depth, bx, by, dist_min):
        """Coarsest g with rho <= tau; None if even g=0 fails."""
        chosen = None
        for g in range(MAX_DENSITY, -1, -1):
            geom = geom_bound_world(qt, cum_dev, depth, bx, by, g)
            rho = screen_error(geom, dist_min, cam)
            if rho <= tau:
                chosen = (g, geom, rho)
                break
        return chosen

    def visit(depth, bx, by):
        lo, hi = block_aabb(qt, depth, bx, by)
        if not _in_frustum(cam, lo, hi):
            return
        dist_min = aabb_dist_min(lo, hi, cam.pos)
        pick = best_density(depth, bx, by, dist_min)
        if pick is None and depth < qt.max_depth:
            for child in qt.children(depth, bx, by):
                visit(*child)
            return
        if pick is None:
            g = 0
            geom = geom_bound_world(qt, cum_dev, depth, bx, by, g)
            pick = (g, geom, screen_error(geom, dist_min, cam))
        g, geom, rho = pick
        out.append(
            SelectedBlock(BlockKey(depth, bx, by, g), rho, geom, dist_min, triangles_for(g))
        )

    visit(0, 0, 0)
    return out
