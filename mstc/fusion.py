"""Overlap-band weighted-vote fusion between adjacent tiles.

Only ``right`` (A is left of B) and ``down`` (A is above B) neighbours are
fused, so every adjacent pair is processed exactly once.  Each overlap pixel
casts one vote per side.  The vote weight combines:

* the geometric distance of the pixel to the tile border on that side, and
* the side's class probability for that pixel.

Two fusion strategy versions exist and produce genuinely different weights:

* ``FV1``: linear distance weight ``d+1``;
* ``FV2``: squared distance weight ``(d+1)^2`` plus a probability dominance term.

Ties resolve deterministically by row/column lexicographic order then lowest
class code.  All weights are integers.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .classifier import TileResult
from .model import CLASSES


@dataclass
class VoteDetail:
    global_row: int
    global_col: int
    side_a_tile: str
    side_b_tile: str
    votes: List[Dict[str, Any]]          # per-side per-class weighted votes
    winner_class: int
    margin: int                           # top - second top weighted score
    tie_broken: bool


@dataclass
class FusionResult:
    pair_key: str
    tile_a: str
    tile_b: str
    direction: str
    fingerprint: str
    votes: List[VoteDetail] = field(default_factory=list)
    winner_raster: Optional[np.ndarray] = None  # overlap local coords -> class
    geometry: Dict[str, int] = field(default_factory=dict)

    def to_cache_dict(self) -> dict:
        return {
            "pair_key": self.pair_key,
            "tile_a": self.tile_a,
            "tile_b": self.tile_b,
            "direction": self.direction,
            "fingerprint": self.fingerprint,
            "votes": [v.__dict__ for v in self.votes],
            "winner_raster": self.winner_raster.tolist() if self.winner_raster is not None else None,
            "geometry": self.geometry,
        }

    @classmethod
    def from_cache_dict(cls, doc: dict) -> "FusionResult":
        votes = [VoteDetail(**v) for v in doc["votes"]]
        wr = doc["winner_raster"]
        return cls(
            pair_key=doc["pair_key"],
            tile_a=doc["tile_a"],
            tile_b=doc["tile_b"],
            direction=doc["direction"],
            fingerprint=doc["fingerprint"],
            votes=votes,
            winner_raster=np.array(wr, dtype=np.int8) if wr is not None else None,
            geometry=dict(doc["geometry"]),
        )


def pair_direction(a: TileResult, b: TileResult, step: int) -> Optional[str]:
    if a.row == b.row and b.col - a.col == 1:
        return "right"
    if a.col == b.col and b.row - a.row == 1:
        return "down"
    if a.row == b.row and a.col - b.col == 1:
        return "left"
    if a.col == b.col and a.row - b.row == 1:
        return "up"
    return None


def overlap_geometry(a: TileResult, b: TileResult, edge: int, overlap: int,
                     direction: str) -> Dict[str, int]:
    """Return local coordinate ranges on each tile for the shared overlap."""
    if direction == "right":
        return {
            "a_r0": 0, "a_r1": edge, "a_c0": edge - overlap, "a_c1": edge,
            "b_r0": 0, "b_r1": edge, "b_c0": 0, "b_c1": overlap,
            "depth": overlap, "span": edge,
        }
    if direction == "down":
        return {
            "a_r0": edge - overlap, "a_r1": edge, "a_c0": 0, "a_c1": edge,
            "b_r0": 0, "b_r1": overlap, "b_c0": 0, "b_c1": edge,
            "depth": overlap, "span": edge,
        }
    raise ValueError(f"cannot derive geometry for direction {direction}")


def fusion_fingerprint(
    a: TileResult, b: TileResult, geometry: Dict[str, int], strategy: str
) -> str:
    payload = {
        "a_key": a.cache_key,
        "b_key": b.cache_key,
        "a_prob": a.prob_summary,
        "b_prob": b.prob_summary,
        "geometry": geometry,
        "strategy": strategy,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _vote_weights(strategy: str, da: int, db: int, pa: int, pb: int
                  ) -> Tuple[int, int]:
    if strategy == "FV1":
        # geometric distance and class probability weighted equally; the tile
        # nearer the pixel gets the stronger voice
        wa = (da + 1) * (pa + 1)
        wb = (db + 1) * (pb + 1)
        return wa, wb
    # FV2: confidence-dominant rule.  Geometry is deliberately neutralised
    # (constant unit factor) and the squared class-probability confidence alone
    # decides, so the more confident tile wins regardless of border distance.
    # Relative to FV1, where forward geometric distance dominates, this reverses
    # the decision whenever the farther tile is the more confident one.
    wa = (pa + 1) * (pa + 1)
    wb = (pb + 1) * (pb + 1)
    return wa, wb


def fuse_pair(
    a: TileResult,
    b: TileResult,
    edge: int,
    overlap: int,
    strategy: str,
) -> FusionResult:
    direction = pair_direction(a, b, edge - overlap)
    if direction in (None, "left", "up"):
        raise ValueError("fuse_pair expects A before B in row/col order")
    geom = overlap_geometry(a, b, edge, overlap, direction)
    fp = fusion_fingerprint(a, b, geom, strategy)
    result = FusionResult(
        pair_key=f"{a.tile_id}|{b.tile_id}",
        tile_a=a.tile_id,
        tile_b=b.tile_id,
        direction=direction,
        fingerprint=fp,
        geometry=geom,
    )
    depth, span = geom["depth"], geom["span"]
    winner = np.full((depth, span), -1, dtype=np.int8)
    votes: List[VoteDetail] = []

    for s in range(span):
        for d in range(depth):
            if direction == "right":
                ar, ac = geom["a_r0"] + s, geom["a_c0"] + d
                br, bc = geom["b_r0"] + s, geom["b_c0"] + d
            else:
                ar, ac = geom["a_r0"] + d, geom["a_c0"] + s
                br, bc = geom["b_r0"] + d, geom["b_c0"] + s

            totals = [0] * len(CLASSES)
            vote_log: List[Dict[str, Any]] = []
            probs_a = a.prob_raster[ar, ac]
            probs_b = b.prob_raster[br, bc]
            dist_a, dist_b = d, overlap - 1 - d
            side_specs = (
                ("A", probs_a, dist_a),
                ("B", probs_b, dist_b),
            )
            for k in range(len(CLASSES)):
                pa_k = int(probs_a[k])
                pb_k = int(probs_b[k])
                wa, wb = _vote_weights(strategy, dist_a, dist_b, pa_k, pb_k)
                for side_name, probs, dist_self, weight in (
                    ("A", probs_a, dist_a, wa),
                    ("B", probs_b, dist_b, wb),
                ):
                    pk = int(probs[k])
                    weighted = weight * pk
                    totals[k] += weighted
                    vote_log.append(
                        {
                            "side": side_name,
                            "class": k,
                            "prob": pk,
                            "distance": dist_self,
                            "weight": weight,
                            "weighted": weighted,
                        }
                    )

            # deterministic winner: highest total; ties -> lexicographic scan
            # order then lowest class code (argmax first occurrence)
            top_class = 0
            top_value = totals[0]
            tie = False
            for k in range(1, len(CLASSES)):
                if totals[k] > top_value:
                    top_value = totals[k]
                    top_class = k
            sorted_totals = sorted(totals, reverse=True)
            if len(sorted_totals) > 1 and sorted_totals[0] == sorted_totals[1]:
                tie = True
            second = sorted_totals[1] if len(sorted_totals) > 1 else 0
            winner[d, s] = top_class

            gr, gc = _global_coord(a, b, direction, edge, overlap, d, s)
            votes.append(
                VoteDetail(
                    global_row=gr,
                    global_col=gc,
                    side_a_tile=a.tile_id,
                    side_b_tile=b.tile_id,
                    votes=vote_log,
                    winner_class=int(top_class),
                    margin=int(top_value - second),
                    tie_broken=tie,
                )
            )

    result.votes = votes
    result.winner_raster = winner
    return result


def _global_coord(a: TileResult, b: TileResult, direction: str, edge: int,
                  overlap: int, d: int, s: int) -> Tuple[int, int]:
    step = edge - overlap
    if direction == "right":
        return a.row * step + s, a.col * step + (edge - overlap) + d
    return a.row * step + (edge - overlap) + d, a.col * step + s
