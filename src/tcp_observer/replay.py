from __future__ import annotations

from typing import Any

from .errors import ReplayError


def first_difference(expected: Any, actual: Any, path: str = "$") -> str | None:
    if type(expected) is not type(actual):
        # ints and bools are distinct intentionally; dict values remain JSON types.
        if not (isinstance(expected, (int, float)) and isinstance(actual, (int, float)) and not isinstance(expected, bool)):
            return f"{path}:type:{type(expected).__name__}!={type(actual).__name__}"
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return f"{path}:type:object!={type(actual).__name__}"
        keys = sorted(set(expected) | set(actual))
        for key in keys:
            child = f"{path}.{key}"
            if key not in actual:
                return f"{child}:missing_actual"
            if key not in expected:
                return f"{child}:unexpected_actual"
            difference = first_difference(expected[key], actual[key], child)
            if difference:
                return difference
        return None
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return f"{path}:type:array!={type(actual).__name__}"
        if len(expected) != len(actual):
            return f"{path}:length:{len(expected)}!={len(actual)}"
        for index, (left, right) in enumerate(zip(expected, actual)):
            difference = first_difference(left, right, f"{path}[{index}]")
            if difference:
                return difference
        return None
    if expected != actual:
        return f"{path}:value:{expected!r}!={actual!r}"
    return None


def compare_items(expected_result: dict[str, Any], actual_result: dict[str, Any]) -> dict[str, Any]:
    expected_by_seq = {int(item["event_seq"]): item for item in expected_result["items"]}
    actual_by_seq = {int(item["event_seq"]): item for item in actual_result["items"]}
    for event_seq in sorted(expected_by_seq):
        expected_item = expected_by_seq[event_seq]
        actual_item = actual_by_seq.get(event_seq)
        if actual_item is None:
            raise ReplayError(
                "replay is missing observation item",
                field_name="item",
                event_seq=event_seq,
                details={"item_id": expected_item["item_id"]},
            )
        expected_core = {key: value for key, value in expected_item.items() if key != "content_digest"}
        actual_core = {key: value for key, value in actual_item.items() if key != "content_digest"}
        difference = first_difference(expected_core, actual_core)
        if difference:
            field_path = difference.split(":", 1)[0]
            raise ReplayError(
                "replayed observation item differs from uninterrupted run",
                field_name=field_path,
                event_seq=event_seq,
                details={"item_id": expected_item["item_id"], "difference": difference},
            )
        if expected_item.get("content_digest") != actual_item.get("content_digest"):
            raise ReplayError(
                "replayed item content digest differs despite equal core",
                field_name="content_digest",
                event_seq=event_seq,
                details={"item_id": expected_item["item_id"]},
            )
    return {"matching_item_count": len(expected_by_seq), "first_difference": None}


def compare_edit_items(expected_result: dict[str, Any], actual_result: dict[str, Any], changed_seq: int) -> dict[str, Any]:
    """Compare an incremental edit while allowing reused prefix provenance stamps."""

    expected_by_seq = {int(item["event_seq"]): item for item in expected_result["items"]}
    actual_by_seq = {int(item["event_seq"]): item for item in actual_result["items"]}
    for event_seq in sorted(expected_by_seq):
        expected_item = expected_by_seq[event_seq]
        actual_item = actual_by_seq[event_seq]
        ignored = {"content_digest"}
        if event_seq < changed_seq:
            ignored.add("script_digest")
        left = {key: value for key, value in expected_item.items() if key not in ignored}
        right = {key: value for key, value in actual_item.items() if key not in ignored}
        difference = first_difference(left, right)
        if difference:
            raise ReplayError(
                "incrementally recomputed item differs from full recomputation",
                field_name=difference.split(":", 1)[0],
                event_seq=event_seq,
                details={"item_id": expected_item["item_id"], "difference": difference},
            )
    return {"matching_item_count": len(expected_by_seq), "changed_seq": changed_seq, "first_difference": None}


def compare_batches(expected_result: dict[str, Any], actual_result: dict[str, Any]) -> dict[str, Any]:
    expected_by_index = {int(batch["batch_index"]): batch for batch in expected_result["batches"]}
    actual_by_index = {int(batch["batch_index"]): batch for batch in actual_result["batches"]}
    for index in sorted(expected_by_index):
        expected = expected_by_index[index]
        actual = actual_by_index.get(index)
        if actual is None:
            raise ReplayError(f"missing replayed batch {index}", field_name="batch", details={"batch_index": index})
        difference = first_difference(expected, actual)
        if difference:
            raise ReplayError(
                "replayed batch differs",
                field_name=difference.split(":", 1)[0],
                event_seq=int(expected["start_seq"]),
                details={"batch_index": index, "difference": difference},
            )
    return {"matching_batch_count": len(expected_by_index), "first_difference": None}
