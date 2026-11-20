from __future__ import annotations

from typing import Any

from .canonical import canonical_json_bytes, digest_json


VIEW_KEYS = (
    "manifest_id",
    "global_step",
    "shard_count",
    "partition_mode",
    "shards",
    "rng_state",
    "parent_manifest",
)
SHARD_KEYS = ("shard_index", "parameter_digest", "momentum_digest", "data_cursor")


def canonical_view(view: dict[str, Any]) -> dict[str, Any]:
    """Create a fresh, complete v3 view without merging into any old object."""
    result: dict[str, Any] = {
        "manifest_id": view["manifest_id"],
        "global_step": view["global_step"],
        "shard_count": view["shard_count"],
        "partition_mode": view["partition_mode"],
        "shards": [
            {
                "shard_index": shard["shard_index"],
                "parameter_digest": shard["parameter_digest"],
                "momentum_digest": shard["momentum_digest"],
                "data_cursor": dict(shard["data_cursor"]),
            }
            for shard in sorted(view["shards"], key=lambda item: item["shard_index"])
        ],
        "rng_state": dict(view["rng_state"]),
        "parent_manifest": (
            None
            if view["parent_manifest"] is None
            else dict(view["parent_manifest"])
        ),
    }
    return result


def view_bytes(view: dict[str, Any]) -> bytes:
    return canonical_json_bytes(canonical_view(view))


def view_digest(view: dict[str, Any]) -> str:
    return digest_json(canonical_view(view))


def _first_nested_difference(
    left: Any,
    right: Any,
    *,
    prefix: str,
    field_prefix: str,
    shard_index: int | None,
    global_step: int,
) -> dict[str, Any] | None:
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                return {
                    "shard_index": shard_index,
                    "field": f"{field_prefix}.{key}" if field_prefix else key,
                    "field_path": f"{prefix}.{key}" if prefix else key,
                    "global_step": global_step,
                    "left": left.get(key),
                    "right": right.get(key),
                }
            mismatch = _first_nested_difference(
                left[key],
                right[key],
                prefix=f"{prefix}.{key}" if prefix else key,
                field_prefix=f"{field_prefix}.{key}" if field_prefix else key,
                shard_index=shard_index,
                global_step=global_step,
            )
            if mismatch is not None:
                return mismatch
        return None
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return {
                "shard_index": shard_index,
                "field": field_prefix,
                "field_path": prefix,
                "global_step": global_step,
                "left": left,
                "right": right,
            }
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            mismatch = _first_nested_difference(
                left_item,
                right_item,
                prefix=f"{prefix}[{index}]",
                field_prefix=field_prefix,
                shard_index=shard_index,
                global_step=global_step,
            )
            if mismatch is not None:
                return mismatch
        return None
    if left != right:
        return {
            "shard_index": shard_index,
            "field": field_prefix,
            "field_path": prefix,
            "global_step": global_step,
            "left": left,
            "right": right,
        }
    return None


def first_difference(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any] | None:
    step = int(left.get("global_step", right.get("global_step", -1)))
    top_fields = ("manifest_id", "global_step", "shard_count", "partition_mode")
    for field in top_fields:
        mismatch = _first_nested_difference(
            left.get(field),
            right.get(field),
            prefix=field,
            field_prefix=field,
            shard_index=None,
            global_step=step,
        )
        if mismatch is not None:
            return mismatch

    left_shards = {int(shard["shard_index"]): shard for shard in left.get("shards", [])}
    right_shards = {int(shard["shard_index"]): shard for shard in right.get("shards", [])}
    for shard_index in sorted(set(left_shards) | set(right_shards)):
        if shard_index not in left_shards or shard_index not in right_shards:
            return {
                "shard_index": shard_index,
                "field": "shard_index",
                "field_path": f"shards[{shard_index}]",
                "global_step": step,
                "left": left_shards.get(shard_index),
                "right": right_shards.get(shard_index),
            }
        for field in ("parameter_digest", "momentum_digest", "data_cursor"):
            mismatch = _first_nested_difference(
                left_shards[shard_index].get(field),
                right_shards[shard_index].get(field),
                prefix=f"shards[{shard_index}].{field}",
                field_prefix=field,
                shard_index=shard_index,
                global_step=step,
            )
            if mismatch is not None:
                return mismatch

    for field in ("rng_state", "parent_manifest"):
        mismatch = _first_nested_difference(
            left.get(field),
            right.get(field),
            prefix=field,
            field_prefix=field,
            shard_index=None,
            global_step=step,
        )
        if mismatch is not None:
            return mismatch
    return None
