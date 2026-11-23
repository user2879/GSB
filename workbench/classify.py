"""Windowed pixel-level deterministic classification (fixed point)."""
from __future__ import annotations

import hashlib

import numpy as np

from .config import CLASSES, FEATURES
from .features import compute_features


def iter_windows(tile_size: int, stride: int, context: int):
    """Yield (core_slice, ctx_slice, core_offset) windows.

    Core regions partition the tile exactly once; the context window extends
    each core by ``context`` pixels on every side (clipped to the tile).
    """
    for r0 in range(0, tile_size, stride):
        for c0 in range(0, tile_size, stride):
            r1 = min(r0 + stride, tile_size)
            c1 = min(c0 + stride, tile_size)
            ctx_r0 = max(r0 - context, 0)
            ctx_c0 = max(c0 - context, 0)
            ctx_r1 = min(r1 + context, tile_size)
            ctx_c1 = min(c1 + context, tile_size)
            core = (slice(r0, r1), slice(c0, c1))
            ctx = (slice(ctx_r0, ctx_r1), slice(ctx_c0, ctx_c1))
            offset = (r0 - ctx_r0, c0 - ctx_c0)
            yield core, ctx, offset


def classify_tile(bands: dict, config, fp) -> dict:
    """Classify one tile. Returns classes, probs, summaries (all deterministic)."""
    tile_size = next(iter(bands.values())).shape[0]
    classes = np.zeros((tile_size, tile_size), dtype=np.uint8)
    probs = np.zeros((tile_size, tile_size, len(CLASSES)), dtype=np.int64)
    score_sum = np.zeros(len(CLASSES), dtype=np.int64)
    quantized_feats = None
    for core, ctx, offset in iter_windows(tile_size, config.stride, config.context_overlap):
        ctx_bands = {name: arr[ctx] for name, arr in bands.items()}
        feats = compute_features(ctx_bands)
        quantized_feats = [fp.quantize(feats[name]) for name in FEATURES]
        scores = []
        for cls in CLASSES:
            scores.append(fp.score(config.weight_vector(cls), quantized_feats))
        stacked = np.stack(scores, axis=-1)  # H x W x C fixed-point ints
        off_r, off_c = offset
        core_scores = stacked[off_r:off_r + config.stride, off_c:off_c + config.stride]
        core_scores = core_scores[:core[0].stop - core[0].start,
                                  :core[1].stop - core[1].start]
        # argmax with ties resolved to the lowest class code
        best = core_scores[:, :, 0].copy()
        winner = np.zeros(core_scores.shape[:2], dtype=np.uint8)
        for code in range(1, len(CLASSES)):
            better = core_scores[:, :, code] > best
            winner[better] = code
            best[better] = core_scores[:, :, code][better]
        classes[core] = winner
        # fixed-point pseudo-probabilities: shifted scores normalised to SCALE
        min_score = core_scores.min(axis=-1, keepdims=True)
        shifted = core_scores - min_score
        denom = shifted.sum(axis=-1, keepdims=True) + 1
        prob = (shifted * fp.scale + denom // 2) // denom
        probs[core] = prob
        score_sum += core_scores.reshape(-1, len(CLASSES)).sum(axis=0)
    counts = [int((classes == code).sum()) for code in range(len(CLASSES))]
    total_px = tile_size * tile_size
    score_summary = {
        "class_pixel_counts": counts,
        "mean_scores": [int(score_sum[i] // total_px) for i in range(len(CLASSES))],
    }
    prob_summary = {
        "mean_probs": [int(probs[:, :, i].sum() // total_px) for i in range(len(CLASSES))],
        "hash": hashlib.sha256(probs.tobytes()).hexdigest()[:16],
    }
    return {
        "classes": classes,
        "probs": probs,
        "score_summary": score_summary,
        "prob_summary": prob_summary,
    }
