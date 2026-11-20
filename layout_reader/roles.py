"""Boundary-inclusive, priority-tie-broken block role classification."""

from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any

from .models import Block, round_metric


ROLE_STAGE_VERSION = 1
ALLOWED_ROLES = ("image", "title", "body", "header_footer")


def assign_roles(
    blocks: list[Block], page_height: int, config: dict[str, Any]
) -> tuple[list[Block], dict[str, Any]]:
    role_config = config["role_detection"]
    image_density = float(role_config["image_min_ink_density"])
    image_periodicity = float(role_config["image_max_projection_periodicity"])
    margin_ratio = float(role_config["header_footer_margin_ratio"])
    ratio_band = role_config["title_body_height_ratio"]
    ratio_min = float(ratio_band["min"])
    ratio_max = float(ratio_band["max"])
    priority = list(role_config["role_priority"])

    top_margin = page_height * margin_ratio
    bottom_margin = page_height * (1.0 - margin_ratio)

    text_like_heights: list[int] = []
    for block in blocks:
        center_y = block.bbox.center_y
        in_margin = center_y <= top_margin or center_y >= bottom_margin
        image_like = (
            block.ink_density >= image_density
            and block.projection_periodicity <= image_periodicity
        )
        if not in_margin and not image_like:
            text_like_heights.append(block.bbox.height)

    if text_like_heights:
        ordered_heights = sorted(text_like_heights)
        lower_half = ordered_heights[: max(1, len(ordered_heights) // 2)]
        body_median_height = float(median(lower_half))
    elif blocks:
        body_median_height = float(median(sorted(block.bbox.height for block in blocks)))
    else:
        body_median_height = 0.0
    if body_median_height <= 0:
        body_median_height = 1.0

    role_counts: Counter[str] = Counter()
    for block in blocks:
        height_ratio = block.bbox.height / body_median_height
        center_y = block.bbox.center_y
        in_top_margin = center_y <= top_margin
        in_bottom_margin = center_y >= bottom_margin
        image_periodic_ok = block.projection_periodicity <= image_periodicity
        title_ratio_ok = ratio_min <= height_ratio <= ratio_max

        criteria = {
            "height_ratio_in_title_band": title_ratio_ok,
            "ink_density_ge_image_min": block.ink_density >= image_density,
            "projection_periodicity_le_image_max": image_periodic_ok,
            "center_y_in_header_margin": in_top_margin,
            "center_y_in_footer_margin": in_bottom_margin,
        }
        eligible: list[str] = []
        if criteria["ink_density_ge_image_min"] and image_periodic_ok:
            eligible.append("image")
        if in_top_margin or in_bottom_margin:
            eligible.append("header_footer")
        if title_ratio_ok and block.ink_density < image_density:
            eligible.append("title")
        eligible.append("body")

        eligible_set = set(eligible)
        selected = next(role for role in priority if role in eligible_set)
        rule = selected
        if selected == "title" and height_ratio == ratio_min:
            rule = "title_height_ratio_boundary_inclusive"
        elif selected == "image" and block.ink_density == image_density:
            rule = "image_density_boundary_inclusive"
        elif len(eligible) > 1:
            rule = "role_priority_after_multiple_eligibility"

        block.role = selected
        block.role_evidence = {
            "height": block.bbox.height,
            "body_median_height": round_metric(body_median_height),
            "height_ratio": round_metric(height_ratio),
            "ink_density": block.ink_density,
            "projection_periodicity": block.projection_periodicity,
            "criteria": criteria,
            "eligible_roles": eligible,
            "role_priority": priority,
            "tie_break_applied": len(eligible) > 1,
            "decision_rule": rule,
        }
        role_counts[selected] += 1

    diagnostics = {
        "body_median_height": round_metric(body_median_height),
        "body_median_estimation": "median_of_lower_half_of_non_margin_text_blocks",
        "title_body_height_ratio_band": {"min": ratio_min, "max": ratio_max},
        "header_margin": [0.0, round_metric(top_margin)],
        "footer_margin": [round_metric(bottom_margin), float(page_height)],
        "role_priority": priority,
        "role_counts": {role: int(role_counts.get(role, 0)) for role in ALLOWED_ROLES},
    }
    return blocks, diagnostics
