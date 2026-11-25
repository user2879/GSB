"""Reading-order serialization with explicit, deterministic conflict resolution."""

from __future__ import annotations

from typing import Any

from .constants import CONFLICT_TYPES
from .models import ImageTruth


_TYPE_ORDER = {name: index for index, name in enumerate(CONFLICT_TYPES)}
_READING_PRIORITY = ["column_right_to_left", "top_to_bottom", "x_desc", "stable_keypoint_id"]


def index_all_keypoints(keypoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign global 1-based keypoint indexes in stable image and reading order."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for point in keypoints:
        grouped.setdefault(point["image"], []).append(point)
    result: list[dict[str, Any]] = []
    index = 1
    for image_name in sorted(grouped):
        ordered = sorted(
            grouped[image_name],
            key=lambda point: (
                point.get("column_order", 1_000_000),
                point["y_q"],
                point["x_q"],
                point["keypoint_id"],
            ),
        )
        for point in ordered:
            point = dict(point)
            point["keypoint_index"] = index
            result.append(point)
            index += 1
    return result


def run_reading_stage(
    image_name: str,
    truth: ImageTruth,
    image_keypoints: list[dict[str, Any]],
    *,
    processed_columns: set[str] | None = None,
    truth_limits: dict[str, int] | None = None,
) -> dict[str, Any]:
    geometry_columns = sorted(truth.columns, key=lambda column: (-column.center_x, column.id))
    column_order = {column.id: index for index, column in enumerate(geometry_columns)}
    column_by_id = {column.id: column for column in geometry_columns}
    processed = processed_columns if processed_columns is not None else set(column_order)
    ordered = sorted(
        image_keypoints,
        key=lambda point: (
            0 if point.get("column") in column_order else 1,
            column_order.get(point.get("column"), 1_000_000),
            point["y_q"],
            -point["x_q"],
            point["keypoint_id"],
        ),
    )
    truth_by_column: dict[str, list[Any]] = {}
    for point in truth.reading_sorted():
        truth_by_column.setdefault(point.column, []).append(point)
    if truth_limits is not None:
        truth_by_column = {
            column: points[: max(0, int(truth_limits.get(column, len(points))))]
            for column, points in truth_by_column.items()
        }
    visible_truth = [
        point
        for point in truth.reading_sorted()
        if point.column in processed
        and point in truth_by_column.get(point.column, [])
    ]

    ordinal_by_column: dict[str, int] = {}
    sequence: list[dict[str, Any]] = []
    expected_pairs: list[tuple[int, int]] = []
    for serial, point in enumerate(ordered, start=1):
        column_id = point.get("column")
        ordinal = ordinal_by_column.get(column_id, 0)
        ordinal_by_column[column_id] = ordinal + 1
        expected_list = truth_by_column.get(column_id, [])
        expected = expected_list[ordinal] if ordinal < len(expected_list) else None
        expected_index = expected.reading_index if expected is not None else None
        sequence.append(
            {
                "reading_serial": serial,
                "keypoint_index": point["keypoint_index"],
                "keypoint_id": point["keypoint_id"],
                "column": column_id,
                "x_q": point["x_q"],
                "y_q": point["y_q"],
                "truth_reading_index": expected_index,
            }
        )
        expected_pairs.append((point["keypoint_index"], expected_index if expected_index is not None else -1))

    conflicts: list[dict[str, Any]] = []

    def add_conflict(
        keypoint_index: int,
        keypoint_id: str,
        conflict_type: str,
        detail: dict[str, Any],
    ) -> None:
        conflicts.append(
            {
                "keypoint_index": keypoint_index,
                "keypoint_id": keypoint_id,
                "image": image_name,
                "type": conflict_type,
                "type_order": _TYPE_ORDER[conflict_type],
                "code": {
                    "cross_column": "E_READ_CROSS_COLUMN",
                    "wrong_order": "E_READ_WRONG_ORDER",
                    "missing_number": "E_READ_MISSING_NUMBER",
                }[conflict_type],
                "resolution_priority": _READING_PRIORITY,
                "detail": detail,
            }
        )

    for position, point in enumerate(ordered):
        serial = position + 1
        column_id = point.get("column")
        expected_list = truth_by_column.get(column_id, [])
        ordinal = sum(1 for previous in ordered[:position] if previous.get("column") == column_id)
        expected = expected_list[ordinal] if ordinal < len(expected_list) else None
        if column_id not in column_by_id:
            add_conflict(
                point["keypoint_index"],
                point["keypoint_id"],
                "cross_column",
                {"column": column_id, "reason": "column_not_in_truth", "serial": serial},
            )
        elif expected is None:
            add_conflict(
                point["keypoint_index"],
                point["keypoint_id"],
                "missing_number",
                {"column": column_id, "detected_ordinal": ordinal, "truth_count": len(expected_list)},
            )
        if position == 0:
            continue
        previous = ordered[position - 1]
        previous_expected_list = truth_by_column.get(previous.get("column"), [])
        previous_ordinal = sum(
            1 for earlier in ordered[: position - 1] if earlier.get("column") == previous.get("column")
        )
        previous_expected = (
            previous_expected_list[previous_ordinal]
            if previous_ordinal < len(previous_expected_list)
            else None
        )
        current_expected_index = expected.reading_index if expected is not None else None
        previous_expected_index = previous_expected.reading_index if previous_expected is not None else None
        if current_expected_index is not None and previous_expected_index is not None:
            difference = current_expected_index - previous_expected_index
            if difference <= 0:
                add_conflict(
                    point["keypoint_index"],
                    point["keypoint_id"],
                    "wrong_order",
                    {
                        "previous_keypoint_index": previous["keypoint_index"],
                        "previous_keypoint_id": previous["keypoint_id"],
                        "previous_truth_reading_index": previous_expected_index,
                        "truth_reading_index": current_expected_index,
                        "difference": difference,
                    },
                )
            elif difference > 1:
                add_conflict(
                    point["keypoint_index"],
                    point["keypoint_id"],
                    "missing_number",
                    {
                        "previous_truth_reading_index": previous_expected_index,
                        "truth_reading_index": current_expected_index,
                        "missing_count": difference - 1,
                    },
                )
        if current_expected_index is not None and previous_expected_index is not None:
            previous_column = column_by_id.get(previous.get("column"))
            current_column = column_by_id.get(column_id)
            if previous_column is not None and current_column is not None and previous_column.id != current_column.id:
                expected_transition = current_column.order == previous_column.order + 1
                if not expected_transition:
                    add_conflict(
                        point["keypoint_index"],
                        point["keypoint_id"],
                        "cross_column",
                        {
                            "previous_column": previous_column.id,
                            "column": current_column.id,
                            "previous_order": previous_column.order,
                            "current_order": current_column.order,
                        },
                    )

    for column in truth.columns:
        if column.id not in processed:
            continue
        detected = ordinal_by_column.get(column.id, 0)
        truth_count = len(truth_by_column.get(column.id, []))
        if detected < truth_count:
            candidates = [point for point in ordered if point.get("column") == column.id]
            anchor = candidates[-1]["keypoint_index"] if candidates else (ordered[-1]["keypoint_index"] if ordered else 0)
            anchor_id = candidates[-1]["keypoint_id"] if candidates else (ordered[-1]["keypoint_id"] if ordered else "")
            add_conflict(
                anchor,
                anchor_id,
                "missing_number",
                {"column": column.id, "detected_count": detected, "truth_count": truth_count},
            )

    conflicts.sort(key=lambda item: (item["keypoint_index"], item["type_order"], item["keypoint_id"]))
    for index, conflict in enumerate(conflicts, start=1):
        conflict["seq"] = index
        del conflict["type_order"]

    observed_indices = [expected for _, expected in expected_pairs if expected >= 0]
    return {
        "image": image_name,
        "sequence": sequence,
        "conflicts": conflicts,
        "processed_columns": sorted(processed, key=lambda item: (column_order.get(item, 1_000_000), item)),
        "visible_truth_points": len(visible_truth),
        "detected_points": len(ordered),
        "priority": _READING_PRIORITY,
        "continuity": {
            "observed_truth_indices": observed_indices,
            "is_continuous": observed_indices == list(range(min(observed_indices), max(observed_indices) + 1))
            if observed_indices
            else True,
        },
    }
