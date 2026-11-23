"""Overlap-band weighted voting fusion and small-component post-processing."""
from __future__ import annotations

from collections import deque

import numpy as np

from . import errors
from .config import CLASSES


def overlap_geometry(spec_a, spec_b, tile_size: int, overlap: int) -> dict:
    """Geometry of the shared band between two adjacent tiles.

    Returns orientation plus, for each side, the local slice of the overlap
    strip and the global origin of the strip.
    """
    if spec_a.row == spec_b.row and spec_b.col == spec_a.col + 1:
        orientation = "horizontal"
        strip = (slice(0, tile_size), slice(tile_size - overlap, tile_size))
        strip_b = (slice(0, tile_size), slice(0, overlap))
    elif spec_a.col == spec_b.col and spec_b.row == spec_a.row + 1:
        orientation = "vertical"
        strip = (slice(tile_size - overlap, tile_size), slice(0, tile_size))
        strip_b = (slice(0, overlap), slice(0, tile_size))
    else:
        raise ValueError(f"tiles {spec_a.tile_id} and {spec_b.tile_id} are not adjacent")
    origin_a = tile_global_origin(spec_a, tile_size, overlap)
    origin_b = tile_global_origin(spec_b, tile_size, overlap)
    if orientation == "horizontal":
        global_origin = (origin_a[0], origin_a[1] + tile_size - overlap)
    else:
        global_origin = (origin_a[0] + tile_size - overlap, origin_a[1])
    return {
        "orientation": orientation,
        "overlap_px": overlap,
        "tile_size": tile_size,
        "strip_a": strip,
        "strip_b": strip_b,
        "global_origin": global_origin,
        "origin_a": origin_a,
        "origin_b": origin_b,
    }


def tile_global_origin(spec, tile_size: int, overlap: int):
    step = tile_size - overlap
    return (spec.row * step, spec.col * step)


def _distance_weights(overlap: int, version: str, fp) -> tuple:
    """Per-offset fixed-point weights (w_a, w_b) for the two sides."""
    offsets = np.arange(overlap, dtype=np.int64)
    dist_a = overlap - offsets  # 1..overlap, larger near side A interior
    dist_b = offsets + 1
    if version == "fs-v1":
        num_a, num_b, den = dist_a, dist_b, overlap
    elif version == "fs-v2":
        num_a, num_b, den = dist_a * dist_a, dist_b * dist_b, overlap * overlap
    else:
        raise ValueError(f"unknown fusion strategy {version}")
    w_a = fp.snap((num_a * fp.scale) // den)
    w_b = fp.snap((num_b * fp.scale) // den)
    return w_a, w_b


def fuse_pair(result_a: dict, result_b: dict, spec_a, spec_b,
              manifest, config, fp, version_a: str, version_b: str) -> dict:
    """Weighted vote over the overlap band of one adjacent pair.

    Every overlap pixel yields a traceable vote record; boundary-sensitive
    pixels (top-1/top-2 gap within half a quantum) carry a double-precision
    reference comparison.
    """
    tile_size = manifest.tile_size
    overlap = manifest.overlap_px
    geom = overlap_geometry(spec_a, spec_b, tile_size, overlap)
    w_a, w_b = _distance_weights(overlap, config.fusion_strategy_version, fp)
    n_classes = len(CLASSES)
    probs_a = result_a["probs"][geom["strip_a"]]
    probs_b = result_b["probs"][geom["strip_b"]]
    classes_a = result_a["classes"][geom["strip_a"]]
    classes_b = result_b["classes"][geom["strip_b"]]
    if geom["orientation"] == "horizontal":
        w_a = w_a.reshape(1, overlap, 1)
        w_b = w_b.reshape(1, overlap, 1)
    else:
        w_a = w_a.reshape(overlap, 1, 1)
        w_b = w_b.reshape(overlap, 1, 1)
    votes = (w_a * probs_a + w_b * probs_b) >> fp.frac_bits
    votes = fp.snap(votes)
    # winner: highest vote; tie -> class of the tile with smaller (row, col)
    side_a_wins = (spec_a.sort_key() <= spec_b.sort_key())
    fallback = classes_a if side_a_wins else classes_b
    best = votes[:, :, 0].copy()
    winner = np.zeros(votes.shape[:2], dtype=np.uint8)
    tied = votes[:, :, 0] == votes[:, :, 0]  # all False start; filled below
    tied = np.zeros(votes.shape[:2], dtype=bool)
    for code in range(1, n_classes):
        better = votes[:, :, code] > best
        equal = votes[:, :, code] == best
        winner[better] = code
        tied[better] = False
        tied[equal] = True
        best = np.maximum(best, votes[:, :, code])
    fused = winner.copy()
    fused[tied] = fallback[tied]
    # per-pixel vote records + boundary diagnostics
    records = []
    boundary_events = []
    origin_r, origin_c = geom["global_origin"]
    sorted_idx = np.argsort(votes, axis=-1)
    top1 = np.take_along_axis(votes, sorted_idx[:, :, -1:], axis=-1)[:, :, 0]
    top2 = np.take_along_axis(votes, sorted_idx[:, :, -2:-1], axis=-1)[:, :, 0]
    gap = top1 - top2
    half_q = fp.quantum // 2
    sensitive_mask = gap <= half_q
    # double precision reference votes
    ref = None
    if sensitive_mask.any():
        wfa, wfb = _distance_weights_float(overlap, config.fusion_strategy_version)
        if geom["orientation"] == "horizontal":
            wfa = wfa.reshape(1, overlap, 1)
            wfb = wfb.reshape(1, overlap, 1)
        else:
            wfa = wfa.reshape(overlap, 1, 1)
            wfb = wfb.reshape(overlap, 1, 1)
        ref = wfa * (probs_a / fp.scale) + wfb * (probs_b / fp.scale)
    for lr in range(votes.shape[0]):
        for lc in range(votes.shape[1]):
            gr, gc = origin_r + lr, origin_c + lc
            record = {
                "global_row": int(gr), "global_col": int(gc),
                "final_class": int(fused[lr, lc]),
                "tiles": [spec_a.tile_id, spec_b.tile_id],
                "weights": {spec_a.tile_id: int(w_a.flat[_w_index(geom, lr, lc, overlap)]),
                            spec_b.tile_id: int(w_b.flat[_w_index(geom, lr, lc, overlap)])},
                "probs": {spec_a.tile_id: [int(x) for x in probs_a[lr, lc]],
                          spec_b.tile_id: [int(x) for x in probs_b[lr, lc]]},
                "prob_versions": {spec_a.tile_id: version_a, spec_b.tile_id: version_b},
                "votes": [int(x) for x in votes[lr, lc]],
                "tie_broken": bool(tied[lr, lc]),
            }
            records.append(record)
            if sensitive_mask[lr, lc]:
                ref_winner = int(np.argmax(ref[lr, lc]))
                fixed_winner = int(fused[lr, lc])
                event = {
                    "code": errors.E_BOUNDARY_SENSITIVITY,
                    "global_row": int(gr), "global_col": int(gc),
                    "fixed_point_class": fixed_winner,
                    "reference_class": ref_winner,
                    "gap_fixed": int(gap[lr, lc]),
                    "direction": "reference_higher" if ref_winner != fixed_winner else "agree",
                    "flip_pair": None,
                }
                if ref_winner != fixed_winner:
                    event["direction"] = "flipped"
                    event["flip_pair"] = sorted([fixed_winner, ref_winner])
                boundary_events.append(event)
    return {
        "geometry": {k: v for k, v in geom.items() if k in
                     ("orientation", "overlap_px", "tile_size", "global_origin")},
        "fused": fused,
        "records": records,
        "boundary_events": boundary_events,
    }


def _w_index(geom, lr, lc, overlap):
    return lc if geom["orientation"] == "horizontal" else lr


def _distance_weights_float(overlap: int, version: str):
    offsets = np.arange(overlap, dtype=np.float64)
    dist_a = overlap - offsets
    dist_b = offsets + 1.0
    if version == "fs-v1":
        return dist_a / overlap, dist_b / overlap
    return (dist_a ** 2) / (overlap ** 2), (dist_b ** 2) / (overlap ** 2)


def assemble_global(manifest, valid_ids, tile_results: dict, fusion_results: dict,
                    pairs) -> np.ndarray:
    """Assemble the global class raster; fused bands overwrite raw tile classes.

    Pairs are applied in canonical sorted order; where horizontal and vertical
    seams cross, the later pair in sorted order wins (both votes stay recorded
    in the fusion artifacts).
    """
    step = manifest.tile_size - manifest.overlap_px
    if not valid_ids:
        return np.zeros((0, 0), dtype=np.uint8)
    max_r = max(manifest.tiles[t].row for t in valid_ids)
    max_c = max(manifest.tiles[t].col for t in valid_ids)
    size_r = max_r * step + manifest.tile_size
    size_c = max_c * step + manifest.tile_size
    global_class = np.zeros((size_r, size_c), dtype=np.uint8)
    for tid in sorted(valid_ids, key=lambda t: manifest.tiles[t].sort_key()):
        spec = manifest.tiles[tid]
        r0, c0 = tile_global_origin(spec, manifest.tile_size, manifest.overlap_px)
        global_class[r0:r0 + manifest.tile_size, c0:c0 + manifest.tile_size] = \
            tile_results[tid]["classes"]
    from .loader import pair_key
    for id_a, id_b in pairs:
        pk = pair_key(manifest, id_a, id_b)
        if pk not in fusion_results:
            continue  # retained for a later batch
        fused = fusion_results[pk]["fused"]
        spec_a, spec_b = manifest.tiles[id_a], manifest.tiles[id_b]
        geom = overlap_geometry(spec_a, spec_b, manifest.tile_size, manifest.overlap_px)
        gr, gc = geom["global_origin"]
        global_class[gr:gr + fused.shape[0], gc:gc + fused.shape[1]] = fused
    return global_class


def postprocess(global_class: np.ndarray, min_connected: int):
    """Remove connected components smaller than min_connected and refill them
    with the majority class of their 4-neighbourhood. Returns (cleaned, refills).
    """
    cleaned = global_class.copy()
    refills = []
    if min_connected <= 0:
        return cleaned, refills
    n_rows, n_cols = cleaned.shape
    labels = -np.ones_like(cleaned, dtype=np.int64)
    components = []
    next_label = 0
    for r in range(n_rows):
        for c in range(n_cols):
            if labels[r, c] != -1:
                continue
            cls = cleaned[r, c]
            pixels = []
            queue = deque([(r, c)])
            labels[r, c] = next_label
            while queue:
                pr, pc = queue.popleft()
                pixels.append((pr, pc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = pr + dr, pc + dc
                    if 0 <= nr < n_rows and 0 <= nc < n_cols and \
                            labels[nr, nc] == -1 and cleaned[nr, nc] == cls:
                        labels[nr, nc] = next_label
                        queue.append((nr, nc))
            components.append({"label": next_label, "class": int(cls), "pixels": pixels})
            next_label += 1
    # stable order: by (min_row, min_col, class)
    components.sort(key=lambda comp: (min(p[0] for p in comp["pixels"]),
                                      min(p[1] for p in comp["pixels"]),
                                      comp["class"]))
    for idx, comp in enumerate(components):
        size = len(comp["pixels"])
        if size >= min_connected:
            continue
        pixel_set = set(comp["pixels"])
        neighbour_counts = {}
        for (pr, pc) in comp["pixels"]:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = pr + dr, pc + dc
                if 0 <= nr < n_rows and 0 <= nc < n_cols and (nr, nc) not in pixel_set:
                    ncls = int(cleaned[nr, nc])
                    neighbour_counts[ncls] = neighbour_counts.get(ncls, 0) + 1
        if neighbour_counts:
            best_count = max(neighbour_counts.values())
            candidates = sorted(c for c, n in neighbour_counts.items() if n == best_count)
            new_class = candidates[0]  # tie -> lowest class code
        else:
            new_class = comp["class"]
        for (pr, pc) in comp["pixels"]:
            cleaned[pr, pc] = new_class
        rows = [p[0] for p in comp["pixels"]]
        cols = [p[1] for p in comp["pixels"]]
        refills.append({
            "component_id": idx,
            "old_class": comp["class"],
            "new_class": int(new_class),
            "size": size,
            "bbox": [min(rows), min(cols), max(rows), max(cols)],
            "reason": "below_min_connected",
            "neighbor_class_counts": {str(k): neighbour_counts[k]
                                      for k in sorted(neighbour_counts)},
        })
    return cleaned, refills
