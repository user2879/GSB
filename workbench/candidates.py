"""Per-frame primitive-level event candidate detection with deterministic ordering."""

import numpy as np


def _connected_components(changed_pixels):
    """Union-find connected components (4-connectivity) over changed pixel list."""
    pixel_set = set(map(tuple, changed_pixels))
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for y, x in sorted(pixel_set):
        find((y, x))
        for nb in ((y - 1, x), (y, x - 1)):
            if nb in pixel_set:
                union((y, x), nb)
    groups = {}
    for px in pixel_set:
        groups.setdefault(find(px), []).append(px)
    comps = [sorted(v) for v in groups.values()]
    comps.sort(key=lambda c: (c[0][0], c[0][1]))
    return comps


def _hist_distance(h1, h2):
    return round(sum(abs(a - b) for a, b in zip(h1, h2)), 6)


def _mean_color(frame, pixels):
    sel = frame[[p[0] for p in pixels], [p[1] for p in pixels]]
    return [int(v) for v in sel.mean(axis=0).round()]


def _color_close(c1, c2, tol):
    return all(abs(a - b) <= tol for a, b in zip(c1, c2))


def _classify(prev_frame, curr_frame, pixels, params, bg_color):
    curr_color = _mean_color(curr_frame, pixels)
    prev_color = _mean_color(prev_frame, pixels)
    curr_is_bg = _color_close(curr_color, bg_color, params.color_match_tolerance)
    prev_is_bg = _color_close(prev_color, bg_color, params.color_match_tolerance)
    if prev_is_bg and not curr_is_bg:
        return "appear"
    if curr_is_bg and not prev_is_bg:
        return "disappear"
    if not _color_close(curr_color, prev_color, params.color_match_tolerance):
        return "color_switch"
    return "translate"


def detect_candidates(seq_name, frame_index, prev_frame, curr_frame, evidence, params):
    """Build primitive-level candidates for one frame; deterministic ordering."""
    if evidence["change_ratio"] < params.change_pixel_ratio_threshold:
        return [], []
    bg_color = [int(v) for v in np.median(
        curr_frame.reshape(-1, 3), axis=0).astype(np.int64)]
    candidates = []
    for rank, comp in enumerate(_connected_components(evidence["changed_pixels"])):
        ys = [p[0] for p in comp]
        xs = [p[1] for p in comp]
        bbox = [min(ys), min(xs), max(ys), max(xs)]  # y0, x0, y1, x1 inclusive
        category = _classify(prev_frame, curr_frame, comp, params, bg_color)
        if category not in params.event_categories:
            category = params.event_categories[0]
        candidates.append({
            "candidate_id": f"{seq_name}:f{frame_index:04d}:c{rank:02d}",
            "frame_index": frame_index,
            "category": category,
            "bbox": [int(v) for v in bbox],
            "area": len(comp),
            "color_signature": _mean_color(curr_frame, comp),
            "histogram": evidence["histogram"],
            "hist_distance": _hist_distance(evidence["histogram"],
                                            _background_hist(curr_frame, params)),
            "suppressed_by": None,
            "suppress_reason": None,
        })
    return _suppress_overlaps(candidates, params)


def _background_hist(frame, params):
    from .evidence import color_histogram
    pixels = frame.reshape(-1, 3)
    return color_histogram(pixels, params.hist_bins)


def _intersects(b1, b2):
    return not (b1[2] < b2[0] or b2[2] < b1[0] or b1[3] < b2[1] or b2[3] < b1[1])


def _suppress_overlaps(candidates, params):
    """Resolve overlapping candidates by (hist_distance, -area, category priority)."""
    priority = {cat: i for i, cat in enumerate(params.event_categories)}
    order = sorted(
        range(len(candidates)),
        key=lambda i: (
            candidates[i]["hist_distance"],
            -candidates[i]["area"],
            priority.get(candidates[i]["category"], len(priority)),
            candidates[i]["bbox"][0],
            candidates[i]["bbox"][1],
        ),
    )
    kept = []
    for idx in order:
        cand = candidates[idx]
        overlapped = False
        for kept_idx in kept:
            if _intersects(cand["bbox"], candidates[kept_idx]["bbox"]):
                cand["suppressed_by"] = candidates[kept_idx]["candidate_id"]
                cand["suppress_reason"] = "overlap_lower_priority"
                overlapped = True
                break
        if not overlapped:
            kept.append(idx)
    survivors = [candidates[i] for i in kept]
    survivors.sort(key=lambda c: (c["frame_index"], c["bbox"][0], c["bbox"][1]))
    suppressed = [c for i, c in enumerate(candidates) if i not in kept]
    suppressed.sort(key=lambda c: (c["frame_index"], c["bbox"][0], c["bbox"][1]))
    return survivors, suppressed
