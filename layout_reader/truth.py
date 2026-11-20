"""Truth comparison for the deterministic synthetic sample set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import ConfigError
from .models import BBox


def load_truth(path: str | Path) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            truth = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"unable to read truth JSON: {exc}") from exc
    if not isinstance(truth, dict) or not isinstance(truth.get("pages"), dict):
        raise ConfigError("truth JSON must contain an object-valued 'pages' field")
    return truth


def _truth_blocks(truth_page: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = truth_page.get("blocks", [])
    if not isinstance(blocks, list):
        raise ConfigError("truth page blocks must be an array")
    return [block for block in blocks if isinstance(block, dict)]


def _bbox_key(value: Any) -> tuple[int, int, int, int]:
    if isinstance(value, dict):
        return tuple(BBox.from_any(value).to_list())  # type: ignore[return-value]
    return tuple(BBox.from_any(value).to_list())  # type: ignore[return-value]


def compare_page(
    page: str, analysis: dict[str, Any], truth_page: dict[str, Any], rule: str
) -> dict[str, Any]:
    detected = analysis.get("blocks", [])
    truth_blocks = _truth_blocks(truth_page)
    detected_by_bbox = {_bbox_key(block["bbox"]): block for block in detected}
    truth_by_bbox = {_bbox_key(block["bbox"]): block for block in truth_blocks}
    missing = sorted(set(truth_by_bbox) - set(detected_by_bbox))
    extra = sorted(set(detected_by_bbox) - set(truth_by_bbox))
    mismatches: list[dict[str, Any]] = []

    role_mismatches = 0
    column_mismatches = 0
    order_mismatches = 0
    for bbox in sorted(set(truth_by_bbox) & set(detected_by_bbox)):
        expected = truth_by_bbox[bbox]
        actual = detected_by_bbox[bbox]
        if expected.get("role") != actual.get("role"):
            role_mismatches += 1
            mismatches.append(
                {
                    "bbox": list(bbox),
                    "field": "role",
                    "expected": expected.get("role"),
                    "actual": actual.get("role"),
                }
            )

        expected_spanning = bool(expected.get("is_spanning", False))
        expected_column = -1 if expected_spanning else expected.get("column_group")
        actual_column = -1 if actual.get("is_spanning") else actual.get("column_id")
        if expected_column != actual_column:
            column_mismatches += 1
            mismatches.append(
                {
                    "bbox": list(bbox),
                    "field": "column",
                    "expected": expected_column,
                    "actual": actual_column,
                }
            )

        if rule == "region_first":
            expected_order = expected.get(
                "reading_order_region_first", expected.get("reading_order")
            )
        else:
            expected_order = expected.get("reading_order")
        if expected_order != actual.get("reading_order"):
            order_mismatches += 1
            mismatches.append(
                {
                    "bbox": list(bbox),
                    "field": "reading_order",
                    "expected": expected_order,
                    "actual": actual.get("reading_order"),
                }
            )

    for bbox in missing:
        mismatches.append({"bbox": list(bbox), "field": "missing_block"})
    for bbox in extra:
        mismatches.append({"bbox": list(bbox), "field": "extra_block"})

    return {
        "page": page,
        "passed": not missing and not extra and not mismatches,
        "truth_block_count": len(truth_blocks),
        "detected_block_count": len(detected),
        "matched_block_count": len(set(truth_by_bbox) & set(detected_by_bbox)),
        "missing_blocks": [list(bbox) for bbox in missing],
        "extra_blocks": [list(bbox) for bbox in extra],
        "role_mismatch_count": role_mismatches,
        "column_mismatch_count": column_mismatches,
        "reading_order_mismatch_count": order_mismatches,
        "mismatches": mismatches,
    }


def compare_truth(
    pages: list[dict[str, Any]], truth: dict[str, Any], rule: str
) -> dict[str, Any]:
    truth_pages = truth.get("pages", {})
    comparisons: list[dict[str, Any]] = []
    for page in pages:
        page_name = page["page"]
        truth_page = truth_pages.get(page_name)
        if truth_page is None:
            comparisons.append(
                {
                    "page": page_name,
                    "passed": False,
                    "error": "page missing from truth JSON",
                }
            )
            continue
        comparisons.append(compare_page(page_name, page, truth_page, rule))
    return {
        "available": True,
        "reading_order_rule": rule,
        "pages_checked": len(comparisons),
        "pages_passed": sum(1 for item in comparisons if item.get("passed")),
        "passed": all(item.get("passed") for item in comparisons) if comparisons else True,
        "pages": comparisons,
    }

