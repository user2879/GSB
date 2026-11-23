"""Pixel-level deterministic classification with centre-region ownership.

Windows tile the image with the configured stride.  A window only *owns* its
centre region (``stride x stride``), so pixels on window borders are never
counted twice; windows that would spill past the bottom/right edge are shifted
inward (their centre region is clipped) until every pixel is covered exactly
once.  Scores are accumulated in a fixed feature order and ties resolve to the
lowest class code.  All arithmetic stays in fixed-point integers.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .config import Config
from .features import FeatureRecord, extract_window_features, pad_bands
from .fixedpoint import FixedConfig, argmax_tie_low, q_mul, weight_to_fixed
from .model import CLASSES, Tile


def window_origins(edge: int, size: int, stride: int) -> List[Tuple[int, int, int, int]]:
    """Return (top, left, centre_r0, centre_c0) for every window.

    Centre regions are the disjoint ``stride x stride`` cells that tile the
    image (the last cell is clipped at the edge).  Each window is centred on
    its region and clamped to the image, with replication padding applied at
    feature extraction, so every pixel is owned exactly once.
    """
    half = size // 2
    centre_starts = list(range(0, edge, stride))
    origins: List[Tuple[int, int, int, int]] = []
    for cr0 in centre_starts:
        for cc0 in centre_starts:
            top = min(max(cr0 + stride // 2 - half, 0), edge - size)
            left = min(max(cc0 + stride // 2 - half, 0), edge - size)
            origins.append((top, left, cr0, cc0))
    return origins


def _score_classes(feat: FeatureRecord, weight_q: Dict[str, int], cfg: FixedConfig) -> List[int]:
    """Fixed order scoring: WATER, VEGETATION, BUILT, BARE."""
    # Normalized indices live in [-scale, +scale]; shift by scale to make a
    # non-negative "evidence" quantity before weighting.
    one = cfg.scale >> cfg.gear_shift
    water_ev, _, _ = q_mul(feat.ndwi + one, weight_q["water"], cfg)
    veg_ev, _, _ = q_mul(feat.ndvi + one, weight_q["vegetation"], cfg)
    built_ev, _, _ = q_mul(feat.texture, weight_q["texture"], cfg)
    bare_ev, _, _ = q_mul(feat.slope, weight_q["slope"], cfg)
    return [water_ev, veg_ev, built_ev, bare_ev]


def _softmax_probabilities(scores: List[int], cfg: FixedConfig) -> List[int]:
    """Integer pseudo-probabilities summing to the fixed-point unit (1.0).

    Uses a deterministic exponential-free normalization: shift scores to be
    non-negative, then distribute units proportionally with a fixed largest
    remainder allocation, guaranteeing an exact sum and stable results.
    """
    if not scores:
        return []
    lowest = min(scores)
    shifted = [s - lowest for s in scores]
    total = sum(shifted)
    unit = cfg.scale >> cfg.gear_shift
    if total == 0:
        base, rem = divmod(unit, len(scores))
        probs = [base] * len(scores)
        for i in range(rem):
            probs[i] += 1
        return probs
    quotas = [(s * unit) // total for s in shifted]
    used = sum(quotas)
    remainder = unit - used
    # largest remainder; ties by lowest class code (already ascending scan)
    leftovers = sorted(
        range(len(scores)), key=lambda i: (-((shifted[i] * unit) % total), i)
    )
    for i in leftovers[:remainder]:
        quotas[i] += 1
    return quotas


@dataclass
class TileResult:
    tile_id: str
    row: int
    col: int
    class_raster: np.ndarray          # int8 class codes
    score_raster: np.ndarray          # int64 [edge,edge,num_classes]
    prob_raster: np.ndarray           # int64 [edge,edge,num_classes]
    prob_summary: List[int]           # summed probs per class (fixed units)
    score_summary: List[int]          # summed winning score per class
    residual_raster: Dict[str, np.ndarray]
    rounding_raster: Dict[str, np.ndarray]
    cache_key: str
    pixels_classified: int
    band_content_hash: str
    weight_fingerprint: str

    def to_cache_dict(self) -> dict:
        return {
            "tile_id": self.tile_id,
            "row": self.row,
            "col": self.col,
            "class_raster": self.class_raster.tolist(),
            "score_raster": self.score_raster.tolist(),
            "prob_raster": self.prob_raster.tolist(),
            "prob_summary": self.prob_summary,
            "score_summary": self.score_summary,
            "residual_raster": {k: v.tolist() for k, v in self.residual_raster.items()},
            "rounding_raster": {k: v.tolist() for k, v in self.rounding_raster.items()},
            "cache_key": self.cache_key,
            "pixels_classified": self.pixels_classified,
            "band_content_hash": self.band_content_hash,
            "weight_fingerprint": self.weight_fingerprint,
        }

    @classmethod
    def from_cache_dict(cls, doc: dict) -> "TileResult":
        return cls(
            tile_id=doc["tile_id"],
            row=doc["row"],
            col=doc["col"],
            class_raster=np.array(doc["class_raster"], dtype=np.int8),
            score_raster=np.array(doc["score_raster"], dtype=np.int64),
            prob_raster=np.array(doc["prob_raster"], dtype=np.int64),
            prob_summary=list(doc["prob_summary"]),
            score_summary=list(doc["score_summary"]),
            residual_raster={k: np.array(v, dtype=np.int64) for k, v in doc["residual_raster"].items()},
            rounding_raster={k: np.array(v, dtype=np.int8) for k, v in doc["rounding_raster"].items()},
            cache_key=doc["cache_key"],
            pixels_classified=doc["pixels_classified"],
            band_content_hash=doc["band_content_hash"],
            weight_fingerprint=doc["weight_fingerprint"],
        )


def band_content_hash(tile: Tile) -> str:
    payload = []
    for name in sorted(tile.data.keys()):
        arr = tile.data[name]
        payload.append(name.encode())
        payload.append(np.ascontiguousarray(arr).tobytes())
    return hashlib.sha256(b"|".join(payload)).hexdigest()


def weight_fingerprint(config: Config) -> str:
    fp = {
        "weights": config.weights.as_dict(),
        "fractional_bits": config.fixed.fractional_bits,
        "gear": config.fixed.gear.value,
        "window": (config.window.size, config.window.stride, config.window.context_overlap),
    }
    return hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()


def make_cache_key(tile: Tile, config: Config) -> Tuple[str, str, str]:
    content = band_content_hash(tile)
    wfp = weight_fingerprint(config)
    raw = f"{content}|{tile.spec.bandset}|{wfp}"
    return hashlib.sha256(raw.encode()).hexdigest(), content, wfp


def classify_tile(tile: Tile, config: Config) -> TileResult:
    edge = tile.edge
    size = config.window.size
    stride = config.window.stride
    fcfg = FixedConfig(config.fixed.fractional_bits, config.fixed.gear)
    weight_q = {name: weight_to_fixed(getattr(config.weights, name), fcfg) for name in
                ("vegetation", "water", "texture", "slope")}

    class_raster = np.full((edge, edge), -1, dtype=np.int8)
    score_raster = np.zeros((edge, edge, len(CLASSES)), dtype=np.int64)
    prob_raster = np.zeros((edge, edge, len(CLASSES)), dtype=np.int64)
    residual_raster = {
        k: np.zeros((edge, edge), dtype=np.int64) for k in ("ndvi", "ndwi", "texture", "slope")
    }
    rounding_raster = {
        k: np.zeros((edge, edge), dtype=np.int8) for k in ("ndvi", "ndwi", "texture", "slope")
    }

    half = size // 2
    padded = pad_bands(tile.data, half)
    owned = np.zeros((edge, edge), dtype=bool)
    for top, left, cr0, cc0 in window_origins(edge, size, stride):
        r1 = min(cr0 + stride, edge)
        c1 = min(cc0 + stride, edge)
        # Features describe the whole window, so they are computed once and
        # shared by every pixel in the disjoint centre region it owns.
        feat = extract_window_features(tile.data, padded, top, left, size, fcfg)
        scores = _score_classes(feat, weight_q, fcfg)
        probs = _softmax_probabilities(scores, fcfg)
        label = argmax_tie_low(scores)
        for r in range(cr0, r1):
            for c in range(cc0, c1):
                if owned[r, c]:
                    continue
                class_raster[r, c] = label
                score_raster[r, c] = scores
                prob_raster[r, c] = probs
                for key in residual_raster:
                    residual_raster[key][r, c] = feat.residuals[key]
                    rounding_raster[key][r, c] = feat.rounding[key]
                owned[r, c] = True

    if not owned.all():
        raise RuntimeError(f"centre ownership left {int((~owned).sum())} pixels unclassified")

    prob_summary = prob_raster.sum(axis=(0, 1)).tolist()
    score_summary = [
        int(score_raster[class_raster == k, k].sum()) for k in range(len(CLASSES))
    ]
    cache_key, content, wfp = make_cache_key(tile, config)
    return TileResult(
        tile_id=tile.tile_id,
        row=tile.spec.row,
        col=tile.spec.col,
        class_raster=class_raster,
        score_raster=score_raster,
        prob_raster=prob_raster,
        prob_summary=prob_summary,
        score_summary=score_summary,
        residual_raster=residual_raster,
        rounding_raster=rounding_raster,
        cache_key=cache_key,
        pixels_classified=int(owned.sum()),
        band_content_hash=content,
        weight_fingerprint=wfp,
    )
