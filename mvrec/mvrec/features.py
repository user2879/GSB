"""Deterministic feature detection and descriptor construction.

The synthetic samples render bright square markers on a dark background, so a
threshold plus 4-connected component labelling is enough to recover the
markers exactly.  Every operation uses raster-scan order and pure Python /
NumPy loops, which makes the output bit-for-bit reproducible.
"""

from collections import deque

import numpy as np

from .errors import E_FEATURE_NONFINITE, PipelineError

PATCH_HALF = 4
PATCH_SIZE = PATCH_HALF * 2 + 1
DESC_DIM = PATCH_SIZE * PATCH_SIZE


def extract_features(view, threshold, injections=None):
    """Return a list of feature records sorted by the (y, x) stable key."""
    inject = (injections or {}).get(str(view["view_id"]),
                                    (injections or {}).get(view["view_id"], {}))
    gray = view["gray"]
    height, width = gray.shape
    mask = gray >= threshold
    visited = np.zeros_like(mask, dtype=bool)
    components = []
    for start_y in range(height):
        for start_x in range(width):
            if not mask[start_y, start_x] or visited[start_y, start_x]:
                continue
            queue = deque([(start_y, start_x)])
            visited[start_y, start_x] = True
            pixels = []
            while queue:
                py, px = queue.popleft()
                pixels.append((py, px))
                for ny, nx in ((py - 1, px), (py + 1, px), (py, px - 1), (py, px + 1)):
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        queue.append((ny, nx))
            components.append(pixels)

    records = []
    for pixels in components:
        ys = np.array([p[0] for p in pixels], dtype=np.float64)
        xs = np.array([p[1] for p in pixels], dtype=np.float64)
        cy, cx = float(ys.mean()), float(xs.mean())
        top, left = int(round(cy)) - PATCH_HALF, int(round(cx)) - PATCH_HALF
        if top < 0 or left < 0 or top + PATCH_SIZE > height or left + PATCH_SIZE > width:
            continue
        window = gray[top : top + PATCH_SIZE, left : left + PATCH_SIZE].astype(np.float64)
        norm = np.linalg.norm(window)
        descriptor = (window / norm).reshape(-1) if norm > 0 else window.reshape(-1)
        records.append({
            "view_id": view["view_id"],
            "x": cx,
            "y": cy,
            "descriptor": descriptor,
            "stable_key": (cy, cx),
        })

    records.sort(key=lambda item: (item["stable_key"][0], item["stable_key"][1]))
    for index, record in enumerate(records):
        record["feature_id"] = index

    if inject.get("nonfinite_feature") and records:
        target = records[0]
        target["descriptor"] = target["descriptor"].copy()
        target["descriptor"][0] = np.nan
    for record in records:
        if not np.all(np.isfinite(record["descriptor"])):
            raise PipelineError(
                E_FEATURE_NONFINITE,
                "feature descriptor contains non-finite values",
                {"view_id": view["view_id"], "feature_id": record["feature_id"]},
            )
    return records
