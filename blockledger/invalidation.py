from __future__ import annotations

import copy
from typing import Any

from .canonical import canonical_bytes, sha256_hex
from .engine import replay_spec
from .spec import expand_intervals, normalize_spec, normalize_intervals


BlockKey = tuple[str, int]


def projection_from_replay(replay: Any) -> dict[str, Any]:
    snapshots: dict[str, Any] = {}
    for snapshot_id in sorted(replay.snapshots):
        snapshot = replay.snapshots[snapshot_id]
        snapshots[snapshot_id] = {
            "volume_id": snapshot.volume_id,
            "version": snapshot.version,
            "captured_seq": snapshot.captured_seq,
            "status": snapshot.status,
            "deleted_seq": snapshot.deleted_seq,
            "digests": {str(k): v for k, v in sorted(snapshot.digests.items())},
            "baselines": {str(k): v for k, v in sorted(snapshot.baselines.items())},
            "block_versions": {str(k): v for k, v in sorted(snapshot.block_versions.items())},
        }
    rollbacks: dict[str, Any] = {}
    for plan in sorted(replay.rollback_plans, key=lambda item: item["rollback_id"]):
        rollbacks[plan["rollback_id"]] = {
            "event_seq": plan["event_seq"],
            "volume_id": plan["volume_id"],
            "snapshot_id": plan["snapshot_id"],
            "baseline_after": plan["baseline_after"],
            "blocks": {
                str(block["block_index"]): {
                    "expected_digest": block["expected_digest"],
                    "baseline_version": block["baseline_version"],
                    "block_version": block["replay_block_version"],
                }
                for block in sorted(plan["blocks"], key=lambda item: item["block_index"])
            },
        }
    return {"snapshots": snapshots, "rollbacks": rollbacks}


def _snapshot_block_set(spec: dict[str, Any], snapshot_id: str) -> set[int]:
    snapshot = next(item for item in spec["snapshots"] if item["id"] == snapshot_id)
    return set(expand_intervals(snapshot["intervals"]))


def _events_for_volume(spec: dict[str, Any], volume_id: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    snapshot_volume = {
        snapshot["id"]: snapshot["volume_id"] for snapshot in spec["snapshots"]
    }
    for event in spec["events"]:
        if event["type"] == "write" and event["volume_id"] == volume_id:
            result.append(event)
        elif event["type"] == "snapshot_create" and snapshot_volume.get(event["snapshot_id"]) == volume_id:
            result.append(event)
        elif event["type"] == "rollback":
            snapshot_id = event["snapshot_id"]
            if snapshot_volume.get(snapshot_id) == volume_id:
                result.append(event)
        elif event["type"] == "snapshot_delete" and snapshot_volume.get(event["snapshot_id"]) == volume_id:
            result.append(event)
    return result


def recompute_block_timeline(spec: dict[str, Any], volume_id: str, block_index: int) -> dict[str, Any]:
    """Recompute one block through arrival-order events.

    Block byte derivation and version updates are block-local.  Volume baseline
    increments from rollbacks are still observed because later writes bind the
    then-current baseline.
    """

    volume = next(item for item in spec["volumes"] if item["id"] == volume_id)
    from .blocks import initial_block_bytes, write_block_bytes
    from .canonical import content_digest

    content = initial_block_bytes(volume, block_index)
    volume_baseline = int(volume["baseline_version"])
    block_baseline = volume_baseline
    block_version = 1
    digest = content_digest(volume_id, block_index, block_baseline, content)

    snapshot_blocks = {
        snapshot["id"]: set(expand_intervals(snapshot["intervals"]))
        for snapshot in spec["snapshots"]
        if snapshot["volume_id"] == volume_id
    }
    snapshot_values: dict[str, Any] = {}
    rollback_values: dict[str, Any] = {}

    for event in sorted(_events_for_volume(spec, volume_id), key=lambda item: item["seq"]):
        event_type = event["type"]
        if event_type == "write":
            if block_index in set(expand_intervals(event["intervals"])):
                content = write_block_bytes(volume, event["seq"], event["seed"], block_index)
                block_version += 1
                block_baseline = volume_baseline
                digest = content_digest(volume_id, block_index, block_baseline, content)
        elif event_type == "snapshot_create":
            snapshot_id = event["snapshot_id"]
            if block_index in snapshot_blocks[snapshot_id]:
                snapshot_values[snapshot_id] = {
                    "digest": digest,
                    "baseline_version": block_baseline,
                    "block_version": block_version,
                    "captured_seq": event["seq"],
                }
        elif event_type == "rollback":
            snapshot_id = event["snapshot_id"]
            if block_index in snapshot_blocks[snapshot_id] and snapshot_id in snapshot_values:
                captured = snapshot_values[snapshot_id]
                digest = captured["digest"]
                block_baseline = int(captured["baseline_version"])
                block_version += 1
                rollback_values[event["rollback_id"]] = {
                    "snapshot_id": snapshot_id,
                    "expected_digest": digest,
                    "baseline_version": block_baseline,
                    "block_version": block_version,
                    "event_seq": event["seq"],
                }
            volume_baseline += 1

    return {
        "volume_id": volume_id,
        "block_index": block_index,
        "snapshots": snapshot_values,
        "rollbacks": rollback_values,
        "final": {
            "digest": digest,
            "baseline_version": block_baseline,
            "block_version": block_version,
            "volume_baseline": volume_baseline,
        },
    }


def _first_mismatch(expected: Any, actual: Any, path: str = "$") -> dict[str, str] | None:
    if type(expected) is not type(actual):
        return {"path": path, "expected": repr(expected), "actual": repr(actual)}
    if isinstance(expected, dict):
        for key in sorted(set(expected) | set(actual)):
            next_path = f"{path}.{key}"
            if key not in expected or key not in actual:
                return {"path": next_path, "expected": "<missing>", "actual": "<missing>"}
            mismatch = _first_mismatch(expected[key], actual[key], next_path)
            if mismatch:
                return mismatch
    elif isinstance(expected, list):
        if len(expected) != len(actual):
            return {"path": path + ".length", "expected": str(len(expected)), "actual": str(len(actual))}
        for index, (left, right) in enumerate(zip(expected, actual)):
            mismatch = _first_mismatch(left, right, f"{path}[{index}]")
            if mismatch:
                return mismatch
    elif expected != actual:
        return {"path": path, "expected": repr(expected), "actual": repr(actual)}
    return None


def _changed_spec_for_write(original: dict[str, Any], change: dict[str, Any]) -> tuple[dict[str, Any], set[BlockKey], dict[str, Any]]:
    write = change.get("write", change.get("write_change", change))
    seq = int(write["seq"])
    volume_id = str(write.get("volume_id", write.get("volume")))
    volume = next(item for item in original["volumes"] if item["id"] == volume_id)
    new_intervals = normalize_intervals(write.get("intervals", write.get("ranges")), volume["block_count"], entry_id=f"event:{seq}")
    new_blocks = set(expand_intervals(new_intervals))

    raw = copy.deepcopy(original)
    old_event = next((event for event in raw["events"] if event["seq"] == seq), None)
    old_blocks: set[int] = set()
    if old_event is not None and old_event["type"] == "write":
        old_blocks = set(expand_intervals(old_event["intervals"]))
        raw["events"].remove(old_event)
    replacement = {
        "seq": seq,
        "type": "write",
        "volume_id": volume_id,
        "intervals": new_intervals,
        "seed": int(write.get("seed", write.get("content_seed", 0))),
    }
    if "event_id" in write:
        replacement["event_id"] = str(write["event_id"])
    raw["events"].append(replacement)
    raw["events"].sort(key=lambda event: event["seq"])
    changed = normalize_spec(raw)
    seed = {(volume_id, index) for index in sorted(old_blocks | new_blocks)}
    detail = {"seq": seq, "old_blocks": sorted(old_blocks), "new_blocks": sorted(new_blocks)}
    return changed, seed, detail


def _changed_spec_for_delete(original: dict[str, Any], change: dict[str, Any]) -> tuple[dict[str, Any], set[BlockKey], dict[str, Any]]:
    snapshot_id = str(change.get("delete_snapshot", change.get("snapshot_id")))
    snapshot = next(item for item in original["snapshots"] if item["id"] == snapshot_id)
    raw = copy.deepcopy(original)
    next_seq = max(int(event["seq"]) for event in raw["events"]) + 1
    delete_seq = int(change.get("seq", change.get("delete_seq", next_seq)))
    raw["events"].append({"seq": delete_seq, "type": "snapshot_delete", "snapshot_id": snapshot_id})
    raw["events"].sort(key=lambda event: event["seq"])
    changed = normalize_spec(raw)
    seed = {(snapshot["volume_id"], index) for index in expand_intervals(snapshot["intervals"])}
    return changed, seed, {"snapshot_id": snapshot_id, "delete_seq": delete_seq}


def analyze_change(
    original_spec: dict[str, Any],
    change: dict[str, Any],
    *,
    baseline_replay: Any | None = None,
) -> dict[str, Any]:
    change_id = str(change.get("change_id", sha256_hex(change)[:16]))
    if "delete_snapshot" in change or ("snapshot_id" in change and change.get("action") == "delete"):
        changed_spec, seed_blocks, mutation = _changed_spec_for_delete(original_spec, change)
        mutation_type = "snapshot_delete"
    else:
        changed_spec, seed_blocks, mutation = _changed_spec_for_write(original_spec, change)
        mutation_type = "write_change"

    if baseline_replay is None:
        baseline_replay = replay_spec(original_spec, verify_declared_digests=True)
    changed_replay = replay_spec(changed_spec, verify_declared_digests=False)
    baseline_projection = projection_from_replay(baseline_replay)
    full_projection = projection_from_replay(changed_replay)
    incremental_projection = copy.deepcopy(baseline_projection)

    snapshot_to_blocks: dict[str, set[int]] = {}
    snapshot_volume: dict[str, str] = {}
    for snapshot in original_spec["snapshots"]:
        snapshot_to_blocks[snapshot["id"]] = set(snapshot["blocks"])
        snapshot_volume[snapshot["id"]] = snapshot["volume_id"]

    rollback_to_blocks: dict[str, set[BlockKey]] = {}
    rollback_snapshot: dict[str, str] = {}
    for plan in baseline_replay.rollback_plans:
        rollback_to_blocks[plan["rollback_id"]] = {
            (plan["volume_id"], block["block_index"]) for block in plan["blocks"]
        }
        rollback_snapshot[plan["rollback_id"]] = plan["snapshot_id"]

    affected_snapshot_ids: set[str] = set()
    affected_rollback_ids: set[str] = set()
    for volume_id, block_index in sorted(seed_blocks):
        for snapshot_id, blocks in snapshot_to_blocks.items():
            if snapshot_volume[snapshot_id] == volume_id and block_index in blocks:
                affected_snapshot_ids.add(snapshot_id)
        for rollback_id, keys in rollback_to_blocks.items():
            if (volume_id, block_index) in keys:
                affected_rollback_ids.add(rollback_id)

    propagation_steps: list[dict[str, Any]] = []
    affected_set: list[dict[str, Any]] = []
    changed_snapshot_fields: list[dict[str, Any]] = []
    changed_rollback_fields: list[dict[str, Any]] = []

    if mutation_type == "write_change":
        propagation_steps.append(
            {
                "order": 0,
                "kind": "write_seed",
                "volume_id": mutation.get("volume_id") or next(iter(seed_blocks))[0],
                "reason": "write event intervals define direct seed block set",
                "mutation": mutation,
            }
        )
    else:
        propagation_steps.append(
            {
                "order": 0,
                "kind": "snapshot_delete_seed",
                "snapshot_id": mutation["snapshot_id"],
                "volume_id": next(iter(seed_blocks))[0] if seed_blocks else None,
                "reason": "deleted snapshot interval defines direct invalidation seed set",
                "mutation": mutation,
            }
        )

    timelines: dict[BlockKey, dict[str, Any]] = {}
    for step_index, (volume_id, block_index) in enumerate(sorted(seed_blocks), start=1):
        timeline = recompute_block_timeline(changed_spec, volume_id, block_index)
        timelines[(volume_id, block_index)] = timeline
        affected_set.append(
            {"kind": "block", "volume_id": volume_id, "block_index": block_index, "status": "seed"}
        )
        propagation_steps.append(
            {
                "order": step_index,
                "kind": "recompute_block_timeline",
                "volume_id": volume_id,
                "block_index": block_index,
                "reason": "recompute initial/write/snapshot/rollback state for only this block",
                "snapshot_count": len(timeline["snapshots"]),
                "rollback_count": len(timeline["rollbacks"]),
            }
        )

        for snapshot_id, values in timeline["snapshots"].items():
            old = baseline_projection["snapshots"].get(snapshot_id, {}).get("digests", {}).get(str(block_index))
            new = values["digest"]
            inc_snapshot = incremental_projection["snapshots"][snapshot_id]
            inc_snapshot["digests"][str(block_index)] = new
            inc_snapshot["baselines"][str(block_index)] = values["baseline_version"]
            inc_snapshot["block_versions"][str(block_index)] = values["block_version"]
            if old != new:
                changed_snapshot_fields.append(
                    {
                        "snapshot_id": snapshot_id,
                        "volume_id": volume_id,
                        "block_index": block_index,
                        "field": "digests",
                        "old": old,
                        "new": new,
                    }
                )
                affected_set.append(
                    {
                        "kind": "snapshot_digest",
                        "snapshot_id": snapshot_id,
                        "volume_id": volume_id,
                        "block_index": block_index,
                        "changed": True,
                    }
                )
                propagation_steps.append(
                    {
                        "kind": "block_to_snapshot_expected_digest_changed",
                        "snapshot_id": snapshot_id,
                        "volume_id": volume_id,
                        "block_index": block_index,
                        "reason": "write arrives before snapshot capture and intersects its interval",
                        "old_digest": old,
                        "new_digest": new,
                    }
                )
            else:
                propagation_steps.append(
                    {
                        "kind": "snapshot_digest_unchanged_due_to_freeze_or_overwrite",
                        "snapshot_id": snapshot_id,
                        "volume_id": volume_id,
                        "block_index": block_index,
                        "reason": "arrival ordering means frozen digest is identical",
                        "digest": old,
                    }
                )

        for rollback_id, values in timeline["rollbacks"].items():
            old_entry = baseline_projection["rollbacks"].get(rollback_id, {}).get("blocks", {}).get(str(block_index), {})
            new_entry = {
                "expected_digest": values["expected_digest"],
                "baseline_version": values["baseline_version"],
                "block_version": values["block_version"],
            }
            incremental_projection["rollbacks"][rollback_id]["blocks"][str(block_index)] = new_entry
            changed_fields = [
                field
                for field in ("expected_digest", "baseline_version", "block_version")
                if old_entry.get(field) != new_entry[field]
            ]
            if changed_fields:
                changed_rollback_fields.append(
                    {
                        "rollback_id": rollback_id,
                        "snapshot_id": values["snapshot_id"],
                        "volume_id": volume_id,
                        "block_index": block_index,
                        "fields": changed_fields,
                        "old": old_entry,
                        "new": new_entry,
                    }
                )
                affected_rollback_ids.add(rollback_id)
                affected_set.append(
                    {
                        "kind": "rollback_block_set",
                        "rollback_id": rollback_id,
                        "snapshot_id": values["snapshot_id"],
                        "volume_id": volume_id,
                        "block_index": block_index,
                        "changed_fields": changed_fields,
                    }
                )
                propagation_steps.append(
                    {
                        "kind": "snapshot_to_rollback_block_changed",
                        "rollback_id": rollback_id,
                        "snapshot_id": values["snapshot_id"],
                        "volume_id": volume_id,
                        "block_index": block_index,
                        "reason": "rollback resolves expected digest from invalidated snapshot",
                        "changed_fields": changed_fields,
                    }
                )

    if mutation_type == "snapshot_delete":
        snapshot_id = mutation["snapshot_id"]
        if snapshot_id in incremental_projection["snapshots"]:
            incremental_projection["snapshots"][snapshot_id]["status"] = "deleted"
            incremental_projection["snapshots"][snapshot_id]["deleted_seq"] = mutation["delete_seq"]
        for rollback_id, target_snapshot in rollback_snapshot.items():
            if target_snapshot == snapshot_id:
                affected_rollback_ids.add(rollback_id)
                plan = next(plan for plan in baseline_replay.rollback_plans if plan["rollback_id"] == rollback_id)
                for block in plan["blocks"]:
                    key = (plan["volume_id"], block["block_index"])
                    affected_set.append(
                        {
                            "kind": "ledger_conclusion",
                            "rollback_id": rollback_id,
                            "snapshot_id": snapshot_id,
                            "volume_id": key[0],
                            "block_index": key[1],
                            "reason": "target_snapshot_deleted",
                        }
                    )
                    propagation_steps.append(
                        {
                            "kind": "rollback_to_ledger_conclusion_invalidated",
                            "rollback_id": rollback_id,
                            "snapshot_id": snapshot_id,
                            "volume_id": key[0],
                            "block_index": key[1],
                            "reason": "delete propagates to block outcome, cursor and quota history",
                        }
                    )

    # Stable ordering requirement: volume id then block index.
    affected_set.sort(
        key=lambda item: (
            item.get("volume_id") or "",
            item.get("block_index") if item.get("block_index") is not None else -1,
            item.get("kind", ""),
            item.get("snapshot_id", item.get("rollback_id", "")),
        )
    )

    mismatch = _first_mismatch(full_projection, incremental_projection)
    comparison_fields = ["snapshots.*.digests", "snapshots.*.status", "rollbacks.*.blocks"]
    report = {
        "change_id": change_id,
        "mutation_type": mutation_type,
        "mutation": mutation,
        "seed_blocks": [
            {"volume_id": volume_id, "block_index": index}
            for volume_id, index in sorted(seed_blocks)
        ],
        "affected_snapshot_ids": sorted(affected_snapshot_ids),
        "affected_rollback_ids": sorted(affected_rollback_ids),
        "changed_snapshot_fields": changed_snapshot_fields,
        "changed_rollback_fields": changed_rollback_fields,
        "affected_set": affected_set,
        "propagation_steps": propagation_steps,
        "ledger_keys": _ledger_keys_from_set(affected_set),
        "incremental_projection_digest": sha256_hex(canonical_bytes(incremental_projection)),
        "full_rebuild_projection_digest": sha256_hex(canonical_bytes(full_projection)),
        "full_rebuild_consistency": {
            "match": mismatch is None,
            "compared_fields": comparison_fields,
            "first_mismatch": mismatch,
            "untouched_entries_preserved": mismatch is None,
        },
    }
    if mismatch is not None:
        raise AssertionError(f"incremental rebuild differs from full rebuild: {mismatch}")
    return report


def _ledger_keys_from_set(affected_set: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for item in affected_set:
        rollback_id = item.get("rollback_id")
        if not rollback_id or "block_index" not in item:
            continue
        key = (rollback_id, item["volume_id"], int(item["block_index"]))
        if key in seen:
            continue
        seen.add(key)
        keys.append(
            {
                "rollback_id": rollback_id,
                "volume_id": item["volume_id"],
                "block_index": int(item["block_index"]),
                "reason": item.get("reason", "incremental_invalidation"),
            }
        )
    keys.sort(key=lambda item: (item["volume_id"], item["block_index"], item["rollback_id"]))
    return keys


def deletion_events_to_ledger_keys(replay: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    affected: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    order = 0
    for effect in replay.effects:
        if effect["type"] != "snapshot_delete":
            continue
        snapshot_id = effect["snapshot_id"]
        for rollback_id in effect.get("dependent_rollbacks", []):
            plan = next(plan for plan in replay.rollback_plans if plan["rollback_id"] == rollback_id)
            for block in sorted(plan["blocks"], key=lambda item: item["block_index"]):
                order += 1
                item = {
                    "rollback_id": rollback_id,
                    "snapshot_id": snapshot_id,
                    "volume_id": plan["volume_id"],
                    "block_index": block["block_index"],
                    "reason": f"snapshot {snapshot_id} deleted at seq {effect['seq']}",
                }
                affected.append(item)
                steps.append({"order": order, **item, "kind": "event_delete_propagation"})
    affected.sort(key=lambda item: (item["volume_id"], item["block_index"], item["rollback_id"]))
    return affected, steps
