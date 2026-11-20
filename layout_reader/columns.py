"""Column-band inference with explicit boundary conflict resolution."""

from __future__ import annotations

from statistics import median
from typing import Any

from .errors import E_COLUMN_GAP_INVALID, WorkbenchError
from .models import Block, round_metric


COLUMNS_STAGE_VERSION = 1


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def _interval_bands(
    intervals: list[tuple[int, int]], gap_threshold: int
) -> list[tuple[int, int]]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    bands: list[list[int]] = [[ordered[0][0], ordered[0][1]]]
    for x0, x1 in ordered[1:]:
        current = bands[-1]
        if x0 - current[1] <= gap_threshold:
            current[1] = max(current[1], x1)
        else:
            bands.append([x0, x1])
    return [(int(x0), int(x1)) for x0, x1 in bands]


def infer_columns(
    blocks: list[Block], page_width: int, page_height: int, config: dict[str, Any]
) -> tuple[list[Block], dict[str, Any]]:
    del page_width  # retained in the public signature for stage symmetry
    gap_threshold = int(config["column_gap_threshold"])
    if gap_threshold <= 0:
        raise WorkbenchError(
            E_COLUMN_GAP_INVALID,
            "column_gap_threshold must be positive",
            "columns",
            details={"column_gap_threshold": gap_threshold},
        )

    if not blocks:
        return blocks, {
            "column_gap_threshold": gap_threshold,
            "column_count": 0,
            "column_layout": "empty",
            "bands": [],
            "spanning_blocks": [],
            "conflict_count": 0,
            "conflicts": [],
        }

    content_blocks = [block for block in blocks if block.role != "header_footer"]
    content_source = content_blocks or blocks
    content_x0 = min(block.bbox.x0 for block in content_source)
    content_x1 = max(block.bbox.x1 for block in content_source)
    content_width = max(1, content_x1 - content_x0)
    candidate_max_width = content_width * float(
        config["column_candidate_max_width_ratio"]
    )
    candidates = [
        block
        for block in content_blocks
        if block.bbox.width <= candidate_max_width
    ]

    if candidates:
        bands = _interval_bands(
            [(block.bbox.x0, block.bbox.x1) for block in candidates],
            gap_threshold,
        )
    else:
        # A single-column page may contain only blocks wider than the candidate
        # cutoff.  In that case the observed content extent itself is the band.
        bands = [(int(content_x0), int(content_x1))]
    if not bands:
        bands = [(int(content_x0), int(content_x1))]

    span_ratio = float(config["column_span_ratio"])
    span_min_overlap = float(config["column_span_min_overlap_ratio"])
    min_overlap_ratio = float(config["column_min_overlap_ratio"])
    band_widths = [x1 - x0 for x0, x1 in bands]
    median_band_width = float(median(band_widths)) if band_widths else 1.0
    block_counts = [0 for _ in bands]
    assigned: dict[str, int] = {}
    spanning_ids: list[str] = []
    conflicts: list[dict[str, Any]] = []

    ordered_blocks = sorted(
        blocks, key=lambda block: (block.bbox.y0, block.bbox.x0, block.block_id)
    )
    for block in ordered_blocks:
        overlaps = [
            _overlap(block.bbox.x0, block.bbox.x1, band_x0, band_x1)
            for band_x0, band_x1 in bands
        ]
        significant = [
            index
            for index, overlap in enumerate(overlaps)
            if overlap >= float(block.bbox.width) * span_min_overlap
        ]
        is_spanning = len(bands) > 1 and (
            len(significant) >= 2
            or block.bbox.width >= content_width * span_ratio
            or block.role == "header_footer"
        )
        if is_spanning:
            block.column_id = None
            block.is_spanning = True
            spanning_ids.append(block.block_id)
            continue

        positive = [index for index, overlap in enumerate(overlaps) if overlap > 0]
        if not positive:
            # No horizontal overlap normally means a tiny outlier; choose the
            # nearest band centre and retain the decision in the conflict log.
            center = (block.bbox.x0 + block.bbox.x1) / 2.0
            distances = [
                abs(center - ((band_x0 + band_x1) / 2.0))
                for band_x0, band_x1 in bands
            ]
            best = min(range(len(bands)), key=lambda index: (distances[index], index))
            conflicts.append(
                {
                    "block_id": block.block_id,
                    "reason": "no_horizontal_overlap",
                    "candidate_columns": list(range(len(bands))),
                    "chosen_column": best,
                    "decision_order": ["nearest_band_centre", "leftmost_tie_break"],
                }
            )
        else:
            best = min(positive, key=lambda index: (-overlaps[index], index))
            tied = [index for index in positive if overlaps[index] == overlaps[best]]
            if len(tied) > 1:
                def conflict_key(index: int) -> tuple[float, int, int]:
                    deviation = abs(band_widths[index] - median_band_width)
                    stability = 1.0 / (1.0 + deviation)
                    return (-stability, -block_counts[index], index)

                best = min(tied, key=conflict_key)
                conflicts.append(
                    {
                        "block_id": block.block_id,
                        "reason": "equal_overlap_between_columns",
                        "candidate_columns": tied,
                        "overlap_pixels": {str(index): overlaps[index] for index in tied},
                        "chosen_column": best,
                        "decision_order": [
                            "greater_width_stability",
                            "greater_existing_block_count",
                            "leftmost_tie_break",
                        ],
                    }
                )
            elif len(positive) > 1:
                conflicts.append(
                    {
                        "block_id": block.block_id,
                        "reason": "multi_column_boundary_overlap",
                        "candidate_columns": positive,
                        "overlap_pixels": {
                            str(index): overlaps[index] for index in positive
                        },
                        "chosen_column": best,
                        "decision_order": [
                            "greatest_horizontal_overlap",
                            "leftmost_tie_break",
                        ],
                    }
                )
            if overlaps[best] < float(block.bbox.width) * min_overlap_ratio:
                conflicts.append(
                    {
                        "block_id": block.block_id,
                        "reason": "weak_column_overlap",
                        "candidate_columns": positive,
                        "chosen_column": best,
                        "decision_order": ["greatest_horizontal_overlap"],
                    }
                )
        block.column_id = best
        block.is_spanning = False
        assigned[block.block_id] = best
        block_counts[best] += 1

    groups: list[dict[str, Any]] = []
    for index, (band_x0, band_x1) in enumerate(bands):
        members = [block for block in blocks if assigned.get(block.block_id) == index]
        if members:
            observed_x0 = min(block.bbox.x0 for block in members)
            observed_x1 = max(block.bbox.x1 for block in members)
        else:
            observed_x0, observed_x1 = band_x0, band_x1
        width = observed_x1 - observed_x0
        groups.append(
            {
                "column_id": index,
                "band_x0": int(band_x0),
                "band_x1": int(band_x1),
                "x0": int(observed_x0),
                "x1": int(observed_x1),
                "width": int(width),
                "block_count": len(members),
                "block_ids": [block.block_id for block in members],
                "width_stability": round_metric(
                    1.0 / (1.0 + abs(width - median_band_width))
                ),
            }
        )

    column_count = len(groups)
    if column_count == 1:
        layout_type = "single"
    elif column_count == 2:
        layout_type = "double"
    elif column_count == 3:
        layout_type = "triple"
    else:
        layout_type = "multi"

    diagnostics = {
        "column_gap_threshold": gap_threshold,
        "content_x_range": [int(content_x0), int(content_x1)],
        "column_count": column_count,
        "column_layout": layout_type,
        "layout_mixed": bool(column_count > 1 and spanning_ids),
        "bands": [
            {"band_x0": int(x0), "band_x1": int(x1), "width": int(x1 - x0)}
            for x0, x1 in bands
        ],
        "spanning_blocks": spanning_ids,
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
        "column_groups": groups,
    }
    return blocks, diagnostics
