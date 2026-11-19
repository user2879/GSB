"""Budget decision + deterministic degradation path.

Triangle budget: if the LOD selection exceeds the triangle budget, blocks are
degraded (density +1, i.e. coarser) in a fixed priority order - ascending
screen error, then descending distance, then deeper depth, then block
coordinates - one step per block per pass, until the budget is met. Every
degradation is recorded with reason and error bounds before/after. No visible
block is ever dropped; if the budget is unsatisfiable even with every block
at the coarsest density, E_BUDGET_DENIED is raised with the first block in
priority order.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import E_BUDGET_DENIED, TerrainError
from .quadtree import MAX_DENSITY
from .selection import SelectedBlock, geom_bound_world, screen_error, triangles_for


@dataclass
class Degradation:
    block: tuple
    from_density: int
    to_density: int
    reason: str
    error_before: float
    error_after: float

    def to_dict(self):
        return {
            "block": list(self.block),
            "from_density": self.from_density,
            "to_density": self.to_density,
            "reason": self.reason,
            "error_before": round(self.error_before, 6),
            "error_after": round(self.error_after, 6),
        }


def _priority_key(block: SelectedBlock):
    k = block.key
    return (
        round(block.screen_error, 9),
        -round(block.dist_min, 6),
        -k.depth,
        k.bx,
        k.by,
    )


def enforce_triangle_budget(
    blocks: list[SelectedBlock], qt, cum_dev, cam, tri_budget: int
) -> tuple[list[SelectedBlock], list[Degradation]]:
    """Degrade in fixed priority order until under budget. Deterministic."""
    degradations: list[Degradation] = []
    total = sum(b.triangles for b in blocks)
    if total <= tri_budget:
        return blocks, degradations

    order = sorted(range(len(blocks)), key=lambda i: _priority_key(blocks[i]))
    densities = [b.key.density for b in blocks]
    cur_tris = [b.triangles for b in blocks]

    def retotal():
        return sum(triangles_for(densities[i]) for i in range(len(blocks)))

    while total > tri_budget:
        progressed = False
        for i in order:
            if total <= tri_budget:
                break
            if densities[i] >= MAX_DENSITY:
                continue
            b = blocks[i]
            k = b.key
            err_before = screen_error(
                geom_bound_world(qt, cum_dev, k.depth, k.bx, k.by, densities[i]),
                b.dist_min,
                cam,
            )
            densities[i] += 1
            err_after = screen_error(
                geom_bound_world(qt, cum_dev, k.depth, k.bx, k.by, densities[i]),
                b.dist_min,
                cam,
            )
            degradations.append(
                Degradation(
                    (k.depth, k.bx, k.by, densities[i] - 1),
                    densities[i] - 1,
                    densities[i],
                    "triangle_budget",
                    err_before,
                    err_after,
                )
            )
            total -= cur_tris[i] - triangles_for(densities[i])
            cur_tris[i] = triangles_for(densities[i])
            progressed = True
        if not progressed:
            first = blocks[order[0]].key if order else None
            raise TerrainError(
                E_BUDGET_DENIED,
                f"triangle budget {tri_budget} unsatisfiable "
                f"(min possible {retotal()}, blocks {len(blocks)})",
                block=first.as_tuple() if first else None,
                tri_budget=tri_budget,
                min_triangles=retotal(),
            )

    out = []
    for i, b in enumerate(blocks):
        k = b.key
        g = densities[i]
        geom = geom_bound_world(qt, cum_dev, k.depth, k.bx, k.by, g)
        out.append(
            SelectedBlock(
                type(k)(k.depth, k.bx, k.by, g),
                screen_error(geom, b.dist_min, cam),
                geom,
                b.dist_min,
                triangles_for(g),
            )
        )
    return out, degradations
