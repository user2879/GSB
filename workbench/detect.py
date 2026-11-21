"""Programmatic part detection from images.

Pure function of (image array, detection config): never reads descriptions
or label lists. Deterministic: identical inputs yield identical instance
bboxes, color distances and instance order on every run.
"""

import numpy as np

from .fingerprint import sha256_bytes


def _classify(fill, config):
    if fill >= config.shape_fill["square_min"]:
        return "square"
    if fill >= config.shape_fill["circle_min"]:
        return "circle"
    return "triangle"


def _components(mask):
    """Row-major BFS connected components over a boolean mask."""
    height, width = mask.shape
    visited = np.zeros((height, width), dtype=bool)
    comps = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            points = []
            while stack:
                cy, cx = stack.pop()
                points.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(points)
    return comps


def _bbox_gap(a, b):
    gap_x = max(0, max(a[0] - b[2], b[0] - a[2]))
    gap_y = max(0, max(a[1] - b[3], b[1] - a[3]))
    return max(gap_x, gap_y)


def detect_sample(sample_id, arr, config):
    """Detect part instances in one image array.

    Returns a JSON-serializable artifact carrying the image content
    fingerprint and the detection-parameter fingerprint.
    """
    image_fp = sha256_bytes(arr.tobytes())
    params_fp = config.detection_params_fingerprint()
    arrf = arr.astype(np.float64)
    names = list(config.colors.keys())
    dist_stack = np.stack(
        [np.sqrt(((arrf - np.array(config.colors[n], dtype=np.float64)) ** 2).sum(-1)) for n in names]
    )
    best = dist_stack.argmin(0)
    best_dist = dist_stack.min(0)
    tolerance = config.color_tolerance
    min_area = config.min_area
    aspect_lo, aspect_hi = config.aspect_ratio_range

    fragments = []
    for color_idx, color_name in enumerate(names):
        mask = (best == color_idx) & (best_dist <= tolerance)
        for points in _components(mask):
            ys = [p[0] for p in points]
            xs = [p[1] for p in points]
            y0, y1 = min(ys), max(ys) + 1
            x0, x1 = min(xs), max(xs) + 1
            area = len(points)
            if area < min_area:
                continue
            bw, bh = x1 - x0, y1 - y0
            aspect = bw / bh
            if not (aspect_lo <= aspect <= aspect_hi):
                continue
            fill = area / (bw * bh)
            shape = _classify(fill, config)
            pys = np.array(ys, dtype=np.int64)
            pxs = np.array(xs, dtype=np.int64)
            color_distance = float(dist_stack[color_idx][pys, pxs].mean())
            fragments.append(
                {
                    "color": color_name,
                    "shape": shape,
                    "bbox": [x0, y0, x1, y1],
                    "area": area,
                    "color_distance": round(color_distance, 6),
                }
            )

    # Merge same-color same-shape fragments split by occlusion.
    count = len(fragments)
    parent = list(range(count))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)

    for i in range(count):
        for j in range(i + 1, count):
            a, b = fragments[i], fragments[j]
            if a["color"] != b["color"] or a["shape"] != b["shape"]:
                continue
            if _bbox_gap(a["bbox"], b["bbox"]) <= config.merge_gap:
                union(i, j)

    groups = {}
    for i in range(count):
        groups.setdefault(find(i), []).append(i)

    instances = []
    for root in sorted(groups, key=lambda r: min(groups[r])):
        members = [fragments[i] for i in sorted(groups[root])]
        x0 = min(m["bbox"][0] for m in members)
        y0 = min(m["bbox"][1] for m in members)
        x1 = max(m["bbox"][2] for m in members)
        y1 = max(m["bbox"][3] for m in members)
        area = sum(m["area"] for m in members)
        color_distance = round(
            sum(m["color_distance"] * m["area"] for m in members) / area, 6
        )
        instances.append(
            {
                "category": members[0]["shape"],
                "color": members[0]["color"],
                "bbox": [x0, y0, x1, y1],
                "area": area,
                "color_distance": color_distance,
                "merged": len(members) > 1,
                "fragments": [
                    {
                        "bbox": m["bbox"],
                        "area": m["area"],
                        "color_distance": m["color_distance"],
                    }
                    for m in members
                ],
            }
        )

    instances.sort(
        key=lambda t: (
            t["bbox"][1],
            t["bbox"][0],
            t["bbox"][3],
            t["bbox"][2],
            t["color"],
            t["category"],
        )
    )
    for index, instance in enumerate(instances):
        instance["instance_id"] = f"{sample_id}#det{index:03d}"

    return {
        "sample_id": sample_id,
        "image_fingerprint": image_fp,
        "params_fingerprint": params_fp,
        "instances": instances,
    }
