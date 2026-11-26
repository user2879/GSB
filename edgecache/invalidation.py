"""Revision invalidation propagation and field-by-field full rebuild checks."""

from __future__ import annotations

from typing import Any

from .canonical import canonical_bytes, content_digest, digest_bytes
from .sampling import compute_sampling, deterministic_sampling_payload
from .spec import EdgeSpec, FragmentEvent, RATE_SCALE


ZERO_SAMPLING_COUNTS = {
    "fragment_candidates": 0,
    "fragment_selected": 0,
    "block_candidates": 0,
    "block_selected": 0,
    "block_uncovered": 0,
}


def _record_payload(spec: EdgeSpec, fragment: FragmentEvent) -> dict[str, Any]:
    node = spec.node_by_id()[fragment.node_id]
    revision = spec.revision_by_id()[fragment.quota_revision]
    return deterministic_sampling_payload(compute_sampling(node, fragment, revision).payload)


def build_sample_records(spec: EdgeSpec) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for fragment in spec.ordered_fragments():
        payload = _record_payload(spec, fragment)
        records[fragment.fragment_id] = {
            "fragment_id": fragment.fragment_id,
            "node_id": fragment.node_id,
            "sequence": fragment.sequence,
            "payload": payload,
            "bytes": canonical_bytes(payload),
            "digest": content_digest(payload),
        }
    return records


def _chain_changes(spec: EdgeSpec) -> list[dict[str, Any]]:
    revision_by_id = spec.revision_by_id()
    changes: list[dict[str, Any]] = []
    for revision in spec.revisions:
        if revision.parent_revision is None:
            continue
        parent = revision_by_id[revision.parent_revision]
        changed_fields = []
        for field_name in ("quota", "fragment_rate_ppm", "block_rate_ppm"):
            old_value = getattr(parent, field_name)
            new_value = getattr(revision, field_name)
            if old_value != new_value:
                changed_fields.append(
                    {
                        "field": field_name,
                        "previous": old_value,
                        "current": new_value,
                    }
                )
        if changed_fields:
            changes.append(
                {
                    "node_id": revision.node_id,
                    "revision_id": revision.revision_id,
                    "parent_revision": revision.parent_revision,
                    "changed_fields": changed_fields,
                }
            )
    return sorted(changes, key=lambda item: (item["node_id"], item["revision_id"]))


def plan_current_invalidations(spec: EdgeSpec) -> dict[str, Any]:
    changes = _chain_changes(spec)
    changed_nodes = {change["node_id"] for change in changes}
    affected: list[dict[str, Any]] = []
    unaffected: list[dict[str, Any]] = []
    for fragment in spec.ordered_fragments():
        item = {
            "fragment_id": fragment.fragment_id,
            "node_id": fragment.node_id,
            "sequence": fragment.sequence,
            "budget_id": fragment.batch_id,
        }
        if fragment.node_id in changed_nodes:
            item["cleared_sampling_counts"] = dict(ZERO_SAMPLING_COUNTS)
            affected.append(item)
        else:
            unaffected.append(item)
    result = {
        "format_version": spec.format_version,
        "policy": "node_revision_change_invalidates_all_node_fragments",
        "chain_changes": changes,
        "affected": affected,
        "unaffected": unaffected,
    }
    result["content_digest"] = content_digest(result)
    return result


def plan_spec_invalidation(before: EdgeSpec, after: EdgeSpec) -> dict[str, Any]:
    before_latest = before.latest_revision_for_node()
    after_latest = after.latest_revision_for_node()
    changed_nodes: set[str] = set()
    node_changes: list[dict[str, Any]] = []
    all_node_ids = sorted(set(before_latest) | set(after_latest))
    for node_id in all_node_ids:
        previous = before_latest.get(node_id)
        current = after_latest.get(node_id)
        if previous is None or current is None:
            changed_nodes.add(node_id)
            node_changes.append(
                {
                    "node_id": node_id,
                    "reason": "node_revision_presence_changed",
                    "before": None if previous is None else previous.to_dict(),
                    "after": None if current is None else current.to_dict(),
                }
            )
            continue
        fields: list[dict[str, Any]] = []
        for field_name in ("quota", "fragment_rate_ppm", "block_rate_ppm"):
            old_value = getattr(previous, field_name)
            new_value = getattr(current, field_name)
            if old_value != new_value:
                fields.append({"field": field_name, "before": old_value, "after": new_value})
        if fields:
            changed_nodes.add(node_id)
            node_changes.append(
                {
                    "node_id": node_id,
                    "reason": "revision_fields_changed",
                    "before_revision": previous.revision_id,
                    "after_revision": current.revision_id,
                    "fields": fields,
                }
            )
    affected: list[dict[str, Any]] = []
    unaffected: list[dict[str, Any]] = []
    for fragment in after.ordered_fragments():
        item = {
            "fragment_id": fragment.fragment_id,
            "node_id": fragment.node_id,
            "sequence": fragment.sequence,
            "budget_id": fragment.batch_id,
        }
        if fragment.node_id in changed_nodes:
            item["cleared_sampling_counts"] = dict(ZERO_SAMPLING_COUNTS)
            item["cleared_sampling_counts_ppm"] = 0
            affected.append(item)
        else:
            unaffected.append(item)
    result = {
        "format_version": after.format_version,
        "policy": "node_latest_revision_quota_or_rate_change_invalidates_node",
        "node_changes": node_changes,
        "affected": affected,
        "unaffected": unaffected,
    }
    result["content_digest"] = content_digest(result)
    return result


def first_field_difference(expected: Any, actual: Any, path: str = "$") -> dict[str, Any] | None:
    if type(expected) is not type(actual):
        return {"field": path, "expected": expected, "actual": actual}
    if isinstance(expected, dict):
        expected_keys = sorted(expected)
        actual_keys = sorted(actual)
        for key in expected_keys:
            if key not in actual:
                return {"field": f"{path}.{key}", "expected": expected[key], "actual": None}
            difference = first_field_difference(expected[key], actual[key], f"{path}.{key}")
            if difference is not None:
                return difference
        for key in actual_keys:
            if key not in expected:
                return {"field": f"{path}.{key}", "expected": None, "actual": actual[key]}
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return {"field": f"{path}.length", "expected": len(expected), "actual": len(actual)}
        for index, (left, right) in enumerate(zip(expected, actual)):
            difference = first_field_difference(left, right, f"{path}[{index}]")
            if difference is not None:
                return difference
        return None
    if expected != actual:
        return {"field": path, "expected": expected, "actual": actual}
    return None


def propagate_invalidation(
    before: EdgeSpec,
    after: EdgeSpec,
    before_records: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    old_records = before_records or build_sample_records(before)
    plan = plan_spec_invalidation(before, after)
    records: dict[str, dict[str, Any]] = {}
    byte_identical = True
    for item in plan["affected"]:
        fragment = after.fragment_by_id()[item["fragment_id"]]
        payload = _record_payload(after, fragment)
        cleared = {
            "fragment_id": fragment.fragment_id,
            "node_id": fragment.node_id,
            "sequence": fragment.sequence,
            "status": "invalidated",
            "sampling_counts": dict(ZERO_SAMPLING_COUNTS),
            "cleared_sampling_counts": dict(ZERO_SAMPLING_COUNTS),
            "recomputed_payload": payload,
            "recomputed_digest": content_digest(payload),
        }
        records[fragment.fragment_id] = cleared
    for item in plan["unaffected"]:
        fragment_id = item["fragment_id"]
        record = old_records.get(fragment_id)
        if record is None:
            byte_identical = False
            continue
        existing_bytes = bytes(record["bytes"])
        stored_digest = digest_bytes(existing_bytes)
        if stored_digest != record["digest"]:
            byte_identical = False
        records[fragment_id] = record
    rebuilt_records = build_sample_records(after)
    first_difference: dict[str, Any] | None = None
    for fragment in after.ordered_fragments():
        fragment_id = fragment.fragment_id
        item = next(
            (entry for entry in plan["affected"] if entry["fragment_id"] == fragment_id),
            None,
        )
        target = records.get(fragment_id, {}).get("recomputed_payload") if item is not None else records.get(fragment_id, {}).get("payload")
        difference = first_field_difference(rebuilt_records[fragment_id]["payload"], target or {})
        if difference is not None:
            first_difference = {"fragment_id": fragment_id, **difference}
            break
    return {
        "format_version": after.format_version,
        "plan": plan,
        "records": records,
        "unaffected_byte_identical": byte_identical,
        "full_rebuild_consistent": first_difference is None,
        "first_rebuild_difference": first_difference,
        "rate_scale": RATE_SCALE,
    }


def full_rebuild_check(spec: EdgeSpec, records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    expected = build_sample_records(spec)
    for fragment in spec.ordered_fragments():
        fragment_id = fragment.fragment_id
        actual_record = records.get(fragment_id)
        if actual_record is None:
            return {
                "consistent": False,
                "fragment_id": fragment_id,
                "field": "__presence__",
            }
        actual_payload = actual_record.get("payload", actual_record.get("recomputed_payload"))
        difference = first_field_difference(expected[fragment_id]["payload"], actual_payload)
        if difference is not None:
            return {
                "consistent": False,
                "fragment_id": fragment_id,
                **difference,
            }
        if bytes(actual_record.get("bytes", canonical_bytes(actual_payload))) != expected[fragment_id]["bytes"]:
            return {
                "consistent": False,
                "fragment_id": fragment_id,
                "field": "$.canonical_bytes",
            }
    return {"consistent": True, "fragment_id": None, "field": None}

