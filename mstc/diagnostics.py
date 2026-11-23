"""Boundary diagnostics: float64 reference voting vs fixed-point verdicts.

For every overlap pixel whose top-vs-second weighted margin is within half a
quantization step we recompute the vote in double precision, compare it against
the committed fixed-point winner and record the difference direction.  A real
disagreement is reported as ``E_BOUNDARY_FLIP`` with an attribution explaining
which side/feature quantization drove the change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .classifier import TileResult
from .fixedpoint import FixedConfig
from .fusion import FusionResult
from .model import CLASSES


@dataclass
class BoundaryRecord:
    pair_key: str
    global_row: int
    global_col: int
    fixed_winner: int
    reference_winner: int
    fixed_margin: int
    reference_margin: float
    half_step: int
    difference_direction: str
    flipped: bool
    attribution: str


def _reference_weights(strategy: str, da: int, db: int, pa: float, pb: float):
    if strategy == "FV1":
        return (da + 1) * (pa + 1.0), (db + 1) * (pb + 1.0)
    wa = (da + 1) ** 2 * (pa + 1.0) + pa
    wb = (db + 1) ** 2 * (pb + 1.0) + pb
    return wa, wb


def evaluate_boundary(
    fusion: FusionResult,
    tile_a: TileResult,
    tile_b: TileResult,
    fcfg: FixedConfig,
    strategy: str,
    overlap: int,
    unit: int,
) -> List[BoundaryRecord]:
    # Vote totals live in (weight x probability) units, i.e. roughly unit**2.
    # Express the quantization half-step at that same scale so the boundary
    # test is a real, config-derived fraction of the vote magnitude.
    half_step = max(1, fcfg.half_step * unit)
    records: List[BoundaryRecord] = []
    geom = fusion.geometry
    depth, span = geom["depth"], geom["span"]

    for vote in fusion.votes:
        if vote.margin > half_step:
            continue
        d = None
        # recover overlap depth index from global coordinates / geometry
        idx = fusion.votes.index(vote)
        s = idx // depth
        d = idx % depth
        if fusion.direction == "right":
            ar, ac = geom["a_r0"] + s, geom["a_c0"] + d
            br, bc = geom["b_r0"] + s, geom["b_c0"] + d
        else:
            ar, ac = geom["a_r0"] + d, geom["a_c0"] + s
            br, bc = geom["b_r0"] + d, geom["b_c0"] + s

        ref_totals = [0.0] * len(CLASSES)
        for k in range(len(CLASSES)):
            pa = tile_a.prob_raster[ar, ac, k] / unit
            pb = tile_b.prob_raster[br, bc, k] / unit
            wa, wb = _reference_weights(strategy, d, overlap - 1 - d, pa, pb)
            ref_totals[k] += wa * pa + wb * pb

        ref_winner = 0
        for k in range(1, len(CLASSES)):
            if ref_totals[k] > ref_totals[ref_winner]:
                ref_winner = k
        ordered = sorted(ref_totals, reverse=True)
        ref_margin = ordered[0] - ordered[1]

        fixed_winner = vote.winner_class
        if ref_totals[ref_winner] > ref_totals[fixed_winner]:
            direction = "fixed_underestimates_reference_class"
        elif ref_totals[ref_winner] < ref_totals[fixed_winner]:
            direction = "fixed_overestimates_winner_class"
        else:
            direction = "equal"
        flipped = ref_winner != fixed_winner
        if flipped:
            attribution = (
                f"quantization half-step={half_step} collapsed margin; "
                f"side A/B probabilities rounded to 1/{unit} shifted class "
                f"{CLASSES[ref_winner]} below {CLASSES[fixed_winner]}"
            )
        else:
            attribution = "sub-half-step margin but both verdicts agree"
        records.append(
            BoundaryRecord(
                pair_key=fusion.pair_key,
                global_row=vote.global_row,
                global_col=vote.global_col,
                fixed_winner=int(fixed_winner),
                reference_winner=int(ref_winner),
                fixed_margin=int(vote.margin),
                reference_margin=float(ref_margin),
                half_step=int(half_step),
                difference_direction=direction,
                flipped=flipped,
                attribution=attribution,
            )
        )
    return records
