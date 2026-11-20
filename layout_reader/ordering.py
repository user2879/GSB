"""Reading-order rules, spanning-block insertion and deterministic validation."""

from __future__ import annotations

from typing import Any

from .errors import E_ORDER_DUPLICATE, E_ORDER_INVALID, WorkbenchError
from .models import Block


ORDER_STAGE_VERSION = 1
RULE_COLUMN_FIRST = "column_first"
RULE_REGION_FIRST = "region_first"


def _column_key(block: Block) -> tuple[int, int, int, str]:
    column_id = block.column_id if block.column_id is not None else -1
    return (column_id, block.bbox.y0, block.bbox.x0, block.block_id)


def _region_bands(
    blocks: list[Block], gap_threshold: int
) -> list[list[Block]]:
    if not blocks:
        return []
    ordered = sorted(
        blocks, key=lambda block: (block.bbox.y0, block.bbox.x0, block.block_id)
    )
    bands: list[list[Block]] = [[ordered[0]]]
    band_bottom = ordered[0].bbox.y1
    for block in ordered[1:]:
        if block.bbox.y0 - band_bottom > gap_threshold:
            bands.append([block])
            band_bottom = block.bbox.y1
        else:
            bands[-1].append(block)
            band_bottom = max(band_bottom, block.bbox.y1)
    return bands


def _region_units(blocks: list[Block], gap_threshold: int) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for band_index, band in enumerate(_region_bands(blocks, gap_threshold), start=1):
        ordered = sorted(
            band, key=lambda block: (block.bbox.x0, block.bbox.y0, block.block_id)
        )
        units.append(
            {
                "kind": "band",
                "band_index": band_index,
                "y0": min(block.bbox.y0 for block in ordered),
                "y1": max(block.bbox.y1 for block in ordered),
                "blocks": ordered,
                "block_ids": [block.block_id for block in ordered],
            }
        )
    return units


def _column_units(blocks: list[Block]) -> list[dict[str, Any]]:
    return [
        {
            "kind": "column",
            "column_id": block.column_id if block.column_id is not None else -1,
            "y0": block.bbox.y0,
            "y1": block.bbox.y1,
            "blocks": [block],
            "block_ids": [block.block_id],
        }
        for block in sorted(blocks, key=_column_key)
    ]


def _covered_columns(block: Block, column_groups: list[dict[str, Any]]) -> list[int]:
    covered: list[int] = []
    block_center = (block.bbox.x0 + block.bbox.x1) / 2.0
    for group in column_groups:
        left = int(group.get("band_x0", group.get("x0", 0)))
        right = int(group.get("band_x1", group.get("x1", 0)))
        overlap = max(0, min(block.bbox.x1, right) - max(block.bbox.x0, left))
        if overlap > 0 or left <= block_center <= right:
            covered.append(int(group["column_id"]))
    return covered


def sort_reading_order(
    blocks: list[Block],
    config: dict[str, Any],
    column_diagnostics: dict[str, Any],
) -> tuple[list[Block], dict[str, Any]]:
    rule = str(config["reading_order_rule"])
    if rule not in {RULE_COLUMN_FIRST, RULE_REGION_FIRST}:
        raise WorkbenchError(
            E_ORDER_INVALID,
            f"unsupported reading_order_rule: {rule}",
            "order",
        )
    region_gap = int(config["region_band_gap"])
    spanning = sorted(
        [block for block in blocks if block.is_spanning],
        key=lambda block: (block.bbox.y0, block.bbox.y1, block.bbox.x0, block.block_id),
    )
    nonspanning = [block for block in blocks if not block.is_spanning]
    column_groups = list(column_diagnostics.get("column_groups", []))

    if rule == RULE_COLUMN_FIRST:
        units = _column_units(nonspanning)
        base_diagnostics: dict[str, Any] = {
            "base_order": "column_id,y0,x0,block_id",
            "column_count": column_diagnostics.get("column_count", 0),
        }
    else:
        units = _region_units(nonspanning, region_gap)
        base_diagnostics = {
            "base_order": "vertical_band,x0,y0,block_id",
            "region_band_gap": region_gap,
            "region_band_count": len(units),
        }

    ordered_blocks: list[Block] = []
    separators: list[dict[str, Any]] = []
    for span in spanning:
        above = [
            unit
            for unit in units
            if max(block.bbox.center_y for block in unit["blocks"]) < span.bbox.y0
        ]
        if not above:
            above = [unit for unit in units if unit["blocks"][0].bbox.y0 < span.bbox.y0]
        # Units can contain more than one block (region bands); use object
        # identity below rather than relying on the first id alone.
        selected_units = [unit for unit in units if unit in above]
        units = [unit for unit in units if unit not in selected_units]
        before_count = len(ordered_blocks)
        for unit in selected_units:
            ordered_blocks.extend(unit["blocks"])
        ordered_blocks.append(span)
        separators.append(
            {
                "spanning_block_id": span.block_id,
                "y_interval": [span.bbox.y0, span.bbox.y1],
                "covered_column_ids": _covered_columns(span, column_groups),
                "blocks_inserted_before": len(ordered_blocks) - before_count - 1,
                "order_position": len(ordered_blocks),
            }
        )

    for unit in units:
        ordered_blocks.extend(unit["blocks"])

    for sequence, block in enumerate(ordered_blocks, start=1):
        block.reading_order = sequence
        block.reading_order_rule = rule

    validate_reading_order(blocks)
    diagnostics = {
        "reading_order_rule": rule,
        **base_diagnostics,
        "spanning_separator_count": len(separators),
        "spanning_separators": separators,
        "reading_sequence": [
            block.block_id
            for block in sorted(blocks, key=lambda item: int(item.reading_order or 0))
        ],
    }
    return blocks, diagnostics


def validate_reading_order(blocks: list[Block]) -> None:
    if not blocks:
        return
    orders = [block.reading_order for block in blocks]
    if any(order is None for order in orders):
        raise WorkbenchError(
            E_ORDER_INVALID,
            "not every block has a reading order",
            "order",
        )
    integer_orders = [int(order) for order in orders if order is not None]
    duplicates = sorted(
        {order for order in integer_orders if integer_orders.count(order) > 1}
    )
    if duplicates:
        raise WorkbenchError(
            E_ORDER_DUPLICATE,
            "duplicate reading order values",
            "order",
            details={"duplicate_orders": duplicates},
        )
    expected = list(range(1, len(blocks) + 1))
    if sorted(integer_orders) != expected:
        raise WorkbenchError(
            E_ORDER_INVALID,
            "reading order is not a contiguous permutation",
            "order",
            details={
                "actual": sorted(integer_orders),
                "expected": expected,
            },
        )
