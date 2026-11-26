from __future__ import annotations

from typing import Any

import numpy as np

from .config import WorkbenchConfig
from .vision import (
    compute_window,
    connected_components,
    merge_components,
    split_domain_by_valley,
    window_rects,
)


def compute_image_windows(sample: dict[str, Any], config: WorkbenchConfig) -> list[dict[str, Any]]:
    pixels = sample["pixels"]
    rects = window_rects(config.width, config.height, config.window_size, config.stride)
    return [compute_window(sample["image_id"], pixels, rect, config) for rect in rects]


def prepare_column(
    sample: dict[str, Any],
    column: dict[str, Any],
    windows: list[dict[str, Any]],
    config: WorkbenchConfig,
) -> dict[str, Any]:
    image_id = sample["image_id"]
    pixels = sample["pixels"]
    rank = int(column["column_rank"])
    x_min, x_max = int(column["x_min"]), int(column["x_max"])
    strip_width = x_max - x_min
    strip_mask = np.zeros((config.height, strip_width), dtype=bool)
    used_windows: list[dict[str, Any]] = []

    for window in windows:
        if not window["candidate"]:
            continue
        wx0, wy = window["x"], window["y"]
        wx1 = wx0 + window["w"]
        wy1 = wy + window["h"]
        if wx1 <= x_min or wx0 >= x_max:
            continue
        used_windows.append(window)
        cx0 = max(wx0, x_min)
        cx1 = min(wx1, x_max)
        patch = pixels[wy:wy1, cx0:cx1]
        strip_mask[wy:wy1, cx0 - x_min : cx1 - x_min] |= patch < config.ink_threshold

    raw_components = connected_components(strip_mask)
    for component in raw_components:
        component["x_min"] += x_min
        component["x_max"] += x_min
        component["sum_x"] += component["area"] * x_min
        component["coords"] = [(py, px + x_min) for py, px in component["coords"]]

    return {
        "image_id": image_id,
        "column_rank": rank,
        "x_min": x_min,
        "x_max": x_max,
        "raw_components": raw_components,
        "used_window_ids": [w["window_id"] for w in used_windows],
        "used_window_keys": [w.get("cache_key") for w in used_windows if w.get("cache_key")],
        "comparison_demand": len(raw_components) * (len(raw_components) - 1) // 2,
    }


def merge_prepared_column(
    prepared: dict[str, Any], config: WorkbenchConfig
) -> dict[str, Any]:
    domains, merge_records, comparisons = merge_components(
        prepared["raw_components"], prepared["used_window_ids"], config
    )
    split_domains: list[dict[str, Any]] = []
    split_records: list[dict[str, Any]] = []
    for domain in domains:
        pieces = split_domain_by_valley(domain, config)
        split_domains.extend(pieces)
        for piece in pieces:
            split_records.extend(piece.get("split_records", []))

    split_domains.sort(key=lambda d: (d["y_min"], d["x_min"]))
    public_domains = []
    accepted_domains = []
    for index, domain in enumerate(split_domains, start=1):
        domain_id = f"{prepared['image_id']}:c{prepared['column_rank']}:d{index:02d}"
        domain["domain_id"] = domain_id
        accepted = domain["area"] >= config.min_component_area
        if accepted:
            accepted_domains.append(domain)
        public_domains.append(
            {
                "domain_id": domain_id,
                "image_id": prepared["image_id"],
                "column_rank": prepared["column_rank"],
                "accepted": accepted,
                "area": domain["area"],
                "bbox": [domain["x_min"], domain["y_min"], domain["x_max"], domain["y_max"]],
                "source_component_ids": domain["source_component_ids"],
                "merge_records": domain["merge_records"],
                "split_records": domain["split_records"],
                "area_boundary_equal": domain["area_boundary_equal"],
                "boundary_flag": domain["boundary_flag"] or not accepted,
            }
        )
    return {
        "image_id": prepared["image_id"],
        "column_rank": prepared["column_rank"],
        "domains": accepted_domains,
        "public_domains": public_domains,
        "merge_records": merge_records,
        "split_records": split_records,
        "comparisons": comparisons,
    }


def _round_fixed_half_up(value_q: int, frac_bits: int) -> int:
    scale = 1 << frac_bits
    return (int(value_q) + scale // 2) // scale


def _quantize_grid(value: int, step: int, maximum: int) -> int:
    value = ((int(value) + step // 2) // step) * step
    return max(0, min(maximum, value))


def regress_domain(
    sample: dict[str, Any],
    domain: dict[str, Any],
    config: WorkbenchConfig,
) -> dict[str, Any]:
    pixels = sample["pixels"]
    frac_bits = config.fixed_frac_bits
    scale = 1 << frac_bits
    total_weight = 0
    weighted_x = 0
    weighted_y = 0
    for py, px in domain["coords"]:
        gray = int(pixels[py, px])
        weight = max(1, 255 - gray)
        total_weight += weight
        weighted_x += weight * px
        weighted_y += weight * py

    centroid_x_q = weighted_x * scale // max(1, total_weight)
    centroid_y_q = weighted_y * scale // max(1, total_weight)

    x0, y0, x1, y1 = domain["x_min"], domain["y_min"], domain["x_max"], domain["y_max"]
    expanded_center_x_q = max(0, x0 - 1 + x1) * scale // 2
    expanded_center_y_q = max(0, y0 - 1 + y1) * scale // 2
    x_q = (3 * centroid_x_q + expanded_center_x_q) // 4
    y_q = (3 * centroid_y_q + expanded_center_y_q) // 4

    x = _round_fixed_half_up(x_q, frac_bits)
    y = _round_fixed_half_up(y_q, frac_bits)
    x = _quantize_grid(x, config.quant_step, config.width - 1)
    y = _quantize_grid(y, config.quant_step, config.height - 1)

    bbox_area = max(1, (x1 - x0) * (y1 - y0))
    fill_bp = min(10_000, domain["area"] * 10_000 // bbox_area)
    confidence_bp = 6_200 + fill_bp // 5
    confidence_bp -= 140 * len(domain.get("merge_records", []))
    confidence_bp -= 220 * len(domain.get("split_records", []))
    if domain.get("area_boundary_equal"):
        confidence_bp -= 180
    confidence_bp = max(3_500, min(9_800, confidence_bp))

    return {
        "keypoint_id": domain["domain_id"].replace(":d", ":kp"),
        "domain_id": domain["domain_id"],
        "column_rank": int(domain.get("column_rank", 0)),
        "x": int(x),
        "y": int(y),
        "x_q": int(x_q),
        "y_q": int(y_q),
        "frac_bits": frac_bits,
        "quant_gear": config.quant_gear,
        "quant_step": config.quant_step,
        "confidence_bp": int(confidence_bp),
        "regression_rule": "3/4_ink_centroid_plus_1/4_expanded_bbox_half_up",
    }


def assign_reading_and_conflicts(
    image_id: str,
    keypoints: list[dict[str, Any]],
    truth: dict[str, Any],
) -> dict[str, Any]:
    ordered = sorted(keypoints, key=lambda p: (int(p["column_rank"]), int(p["y"]), int(p["x"])))
    for index, point in enumerate(ordered, start=1):
        point["reading_order"] = index

    truth_points = {p["reading_order"]: p for p in truth["keypoints"]}
    truth_by_id = {p["keypoint_id"]: p for p in truth["keypoints"]}

    pairs = []
    for pi, point in enumerate(ordered):
        for truth_order, truth_point in truth_points.items():
            dx = int(point["x"]) - int(truth_point["x"])
            dy = int(point["y"]) - int(truth_point["y"])
            distance_sq = dx * dx + dy * dy
            if distance_sq <= 48 * 48:
                pairs.append((distance_sq, pi, truth_order))
    pairs.sort(key=lambda item: (item[0], item[1], item[2]))
    matched_predictions: set[int] = set()
    matched_truth_orders: set[int] = set()
    for _distance, pi, truth_order in pairs:
        if pi in matched_predictions or truth_order in matched_truth_orders:
            continue
        matched_predictions.add(pi)
        matched_truth_orders.add(truth_order)
        truth_point = truth_points[truth_order]
        ordered[pi]["matched_truth_id"] = truth_point["keypoint_id"]
        ordered[pi]["truth_reading_order"] = truth_order
        ordered[pi]["truth_column_rank"] = int(truth_point["column_rank"])
        ordered[pi]["match_distance_sq"] = int(_distance)

    conflicts: list[dict[str, Any]] = []

    def add_conflict(
        kind: str,
        priority: int,
        predicted_ids: list[str],
        truth_orders: list[int | None],
        reasons: list[str],
        missing_order: int | None = None,
        pair_index: int | None = None,
    ) -> None:
        conflicts.append(
            {
                "image_id": image_id,
                "pair_index": pair_index,
                "type": kind,
                "priority": priority,
                "predicted_keypoint_ids": predicted_ids,
                "truth_reading_orders": truth_orders,
                "missing_reading_order": missing_order,
                "reasons": reasons,
                "decision_priority": "cross_column(3)>missing_number(2)>wrong_order(1)",
            }
        )

    for truth_order in sorted(set(truth_points) - matched_truth_orders):
        truth_point = truth_points[truth_order]
        add_conflict(
            "缺号",
            2,
            [],
            [truth_order],
            [f"truth reading order {truth_order} has no matched predicted keypoint"],
            missing_order=truth_order,
        )

    for index, point in enumerate(ordered):
        if index not in matched_predictions:
            add_conflict(
                "错序",
                1,
                [point["keypoint_id"]],
                [None],
                ["predicted keypoint has no truth match within 48px"],
                pair_index=index,
            )

    for pair_index in range(len(ordered) - 1):
        a = ordered[pair_index]
        b = ordered[pair_index + 1]
        ta = a.get("truth_reading_order")
        tb = b.get("truth_reading_order")
        candidate_types: list[tuple[str, int, list[str]]] = []
        if ta is None or tb is None:
            candidate_types.append(("错序", 1, ["adjacent pair contains an unmatched keypoint"]))
        else:
            truth_a = truth_points[int(ta)]
            truth_b = truth_points[int(tb)]
            cross_reasons = []
            if int(a["column_rank"]) != int(truth_a["column_rank"]):
                cross_reasons.append("first point column differs from truth")
            if int(b["column_rank"]) != int(truth_b["column_rank"]):
                cross_reasons.append("second point column differs from truth")
            if int(truth_a["column_rank"]) != int(truth_b["column_rank"]) and int(a["column_rank"]) == int(b["column_rank"]):
                cross_reasons.append("truth crosses columns but prediction stays in one column")
            if int(truth_a["column_rank"]) == int(truth_b["column_rank"]) and int(a["column_rank"]) != int(b["column_rank"]):
                cross_reasons.append("prediction crosses columns but truth stays in one column")
            if cross_reasons:
                candidate_types.append(("跨栏", 3, cross_reasons))
            if int(tb) - int(ta) > 1:
                candidate_types.append(
                    ("缺号", 2, [f"truth order gap between {ta} and {tb}"])
                )
            elif int(tb) <= int(ta):
                candidate_types.append(
                    ("错序", 1, [f"truth order does not increase: {ta} -> {tb}"])
                )
        if candidate_types:
            chosen = max(candidate_types, key=lambda item: item[1])
            add_conflict(
                chosen[0],
                chosen[1],
                [a["keypoint_id"], b["keypoint_id"]],
                [ta if ta is not None else None, tb if tb is not None else None],
                chosen[2] + [f"candidate_types={[item[0] for item in candidate_types]}"],
                pair_index=pair_index,
            )

    conflicts.sort(
        key=lambda c: (
            c.get("missing_reading_order") is None,
            c.get("missing_reading_order") or 0,
            c.get("pair_index") if c.get("pair_index") is not None else -1,
            -c["priority"],
            c["type"],
        )
    )
    for index, conflict in enumerate(conflicts, start=1):
        conflict["conflict_id"] = f"{image_id}:conflict-{index:03d}"

    sequence = [
        {
            "reading_order": point["reading_order"],
            "keypoint_id": point["keypoint_id"],
            "column_rank": point["column_rank"],
            "x": point["x"],
            "y": point["y"],
            "confidence_bp": point["confidence_bp"],
            "truth_reading_order": point.get("truth_reading_order"),
        }
        for point in ordered
    ]
    return {"keypoints": ordered, "reading_sequence": sequence, "conflicts": conflicts}
