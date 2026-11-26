from __future__ import annotations

import json
from typing import Any

from .canonical import content_digest
from .errors import ReplayError
from .ledger import Ledger


def _parse_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    return json.loads(value)


def first_json_difference(left: Any, right: Any, prefix: str = "") -> str | None:
    if type(left) is not type(right):
        if isinstance(left, int) and isinstance(right, int) and not isinstance(left, bool):
            pass
        else:
            return prefix or "$"
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            field = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                return field
            difference = first_json_difference(left[key], right[key], field)
            if difference:
                return difference
        return None
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return f"{prefix}.length"
        for index, (item_left, item_right) in enumerate(zip(left, right)):
            difference = first_json_difference(item_left, item_right, f"{prefix}[{index}]")
            if difference:
                return difference
        return None
    if left != right:
        return prefix or "$"
    return None


def _index_view(view: dict[str, list[dict[str, Any]]], table: str, key: str) -> dict[Any, dict[str, Any]]:
    return {row[key]: row for row in view.get(table, [])}


def verify_ledger_replay(ledger: Ledger, topo_order: list[str] | tuple[str, ...]) -> dict[str, Any]:
    view = ledger.export_view()
    outcomes = _index_view(view, "outcome_commits", "shard_id")
    shards = _index_view(view, "shards", "shard_id")
    errors = _index_view(view, "error_attribution", "shard_id")
    checked: list[str] = []

    for shard_id in topo_order:
        outcome = outcomes.get(shard_id)
        shard = shards.get(shard_id)
        if shard is None or shard.get("status") != "completed":
            continue
        if outcome is None:
            raise ReplayError("completed shard lacks atomic outcome", shard_id=shard_id, field="outcome_commits")
        payload = _parse_json(outcome["payload_json"])
        if content_digest(payload) != outcome["content_digest"]:
            raise ReplayError("outcome digest mismatch", shard_id=shard_id, field="outcome_commits.content_digest")
        conclusion_digest = payload.get("conclusion_digest")
        if conclusion_digest and conclusion_digest != shard.get("content_digest"):
            raise ReplayError("shard/conclusion digest mismatch", shard_id=shard_id, field="shards.content_digest")
        summary = _parse_json(shard.get("summary_json"))
        if content_digest({k: v for k, v in summary.items() if k != "summary_digest"}) + "" == "":
            pass
        expected_summary_digest = content_digest(
            {key: value for key, value in summary.items() if key != "summary_digest"}
        )
        if expected_summary_digest != summary.get("summary_digest"):
            raise ReplayError("summary digest mismatch", shard_id=shard_id, field="summary.summary_digest")
        if shard.get("summary_digest") != summary.get("summary_digest"):
            raise ReplayError("stored summary digest mismatch", shard_id=shard_id, field="shards.summary_digest")
        error_row = errors.get(shard_id)
        if error_row is None:
            raise ReplayError("missing error attribution", shard_id=shard_id, field="error_attribution")
        error_record = _parse_json(error_row["record_json"])
        if content_digest(error_record) != error_row["content_digest"]:
            raise ReplayError("error attribution digest mismatch", shard_id=shard_id, field="error_attribution.content_digest")
        checked.append(shard_id)

    active_leases = [row for row in view["leases"] if row["state"] == "active"]
    active_resources = sum(int(row["resource_units"]) for row in view["resource_occupancy"] if row["state"] == "active")
    cursor_rows = view.get("progress_cursor", [])
    if cursor_rows:
        cursor = cursor_rows[0]
        if int(cursor["active_lease_count"]) != len(active_leases):
            raise ReplayError(
                "cursor active lease count mismatch",
                shard_id="$cursor",
                field="progress_cursor.active_lease_count",
            )
        if int(cursor["active_resource_units"]) != active_resources:
            raise ReplayError(
                "cursor active resource mismatch",
                shard_id="$cursor",
                field="progress_cursor.active_resource_units",
            )
    return {"replayed_shards": checked, "active_leases": len(active_leases), "active_resource_units": active_resources}


SHARD_SCALAR_FIELDS = [
    "status",
    "generation",
    "node_id",
    "lease_id",
    "sample_rows",
    "participating_rows",
    "capacity_version",
    "summary_digest",
    "wait_started_ms",
    "wait_ended_ms",
    "wait_ms",
    "error_code",
    "format_version",
    "content_digest",
]

LEASE_FIELDS = [
    "lease_id",
    "shard_id",
    "node_id",
    "generation",
    "granted_at_ms",
    "expires_at_ms",
    "renew_count",
    "lease_version",
    "capacity_version",
    "state",
    "reap_reason",
    "format_version",
    "content_digest",
]

OCCUPANCY_FIELDS = [
    "shard_id",
    "generation",
    "lease_id",
    "node_id",
    "resource_units",
    "global_budget",
    "quota_slots",
    "lease_slots",
    "state",
    "started_at_ms",
    "ended_at_ms",
    "scale_exponent",
    "scaled_resource_units",
    "saturated",
    "format_version",
    "content_digest",
]


def compare_conclusion_views(
    left_view: dict[str, list[dict[str, Any]]],
    right_view: dict[str, list[dict[str, Any]]],
    topo_order: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    left_shards = _index_view(left_view, "shards", "shard_id")
    right_shards = _index_view(right_view, "shards", "shard_id")
    left_leases = _index_view(left_view, "leases", "lease_id")
    right_leases = _index_view(right_view, "leases", "lease_id")
    left_occ = {(r["shard_id"], r["generation"]): r for r in left_view.get("resource_occupancy", [])}
    right_occ = {(r["shard_id"], r["generation"]): r for r in right_view.get("resource_occupancy", [])}
    left_errors = _index_view(left_view, "error_attribution", "shard_id")
    right_errors = _index_view(right_view, "error_attribution", "shard_id")

    for shard_id in topo_order:
        left = left_shards.get(shard_id)
        right = right_shards.get(shard_id)
        if (left is None) != (right is None):
            raise ReplayError("shard presence mismatch", shard_id=shard_id, field="shards.shard_id")
        if left is None:
            continue
        for field in SHARD_SCALAR_FIELDS:
                if left.get(field) != right.get(field):
                    raise ReplayError(
                        f"shard field mismatch: {shard_id}.{field}",
                        shard_id=shard_id,
                        field=f"shards.{field}",
                    left=left.get(field),
                    right=right.get(field),
                )
        left_summary = _parse_json(left.get("summary_json"))
        right_summary = _parse_json(right.get("summary_json"))
        difference = first_json_difference(left_summary, right_summary, "summary")
        if difference:
            raise ReplayError("summary field mismatch", shard_id=shard_id, field=difference)

        lease_id = left.get("lease_id")
        if lease_id:
            left_lease = left_leases.get(lease_id)
            right_lease = right_leases.get(lease_id)
            if (left_lease is None) != (right_lease is None):
                raise ReplayError("lease presence mismatch", shard_id=shard_id, field="leases.lease_id")
            if left_lease:
                for field in LEASE_FIELDS:
                    if left_lease.get(field) != right_lease.get(field):
                        raise ReplayError(
                            "lease field mismatch",
                            shard_id=shard_id,
                            field=f"leases.{field}",
                            left=left_lease.get(field),
                            right=right_lease.get(field),
                        )

        generation = int(left.get("generation", 1))
        left_occ_row = left_occ.get((shard_id, generation))
        right_occ_row = right_occ.get((shard_id, generation))
        if (left_occ_row is None) != (right_occ_row is None):
            raise ReplayError("occupancy presence mismatch", shard_id=shard_id, field="resource_occupancy")
        if left_occ_row:
            for field in OCCUPANCY_FIELDS:
                if left_occ_row.get(field) != right_occ_row.get(field):
                    raise ReplayError(
                        "occupancy field mismatch",
                        shard_id=shard_id,
                        field=f"resource_occupancy.{field}",
                        left=left_occ_row.get(field),
                        right=right_occ_row.get(field),
                    )

        left_error = left_errors.get(shard_id)
        right_error = right_errors.get(shard_id)
        if (left_error is None) != (right_error is None):
            raise ReplayError("error attribution presence mismatch", shard_id=shard_id, field="error_attribution")
        if left_error:
            left_record = _parse_json(left_error["record_json"])
            right_record = _parse_json(right_error["record_json"])
            difference = first_json_difference(left_record, right_record, "error_attribution")
            if difference:
                raise ReplayError("error attribution mismatch", shard_id=shard_id, field=difference)

    left_water = _index_view(left_view, "water_records", "record_key")
    right_water = _index_view(right_view, "water_records", "record_key")
    for key in sorted(set(left_water) | set(right_water)):
        if key not in left_water or key not in right_water:
            raise ReplayError("water record presence mismatch", shard_id="$water", field=f"water_records.{key}")
        left_record = _parse_json(left_water[key]["record_json"])
        right_record = _parse_json(right_water[key]["record_json"])
        difference = first_json_difference(left_record, right_record, f"water_records.{key}")
        if difference:
            shard_hint = left_record.get("shard_id") if isinstance(left_record, dict) else "$water"
            raise ReplayError("water record mismatch", shard_id=shard_hint or "$water", field=difference)

    return {"identical": True, "compared_shards": list(topo_order)}


def unchanged_byte_records(
    before_view: dict[str, list[dict[str, Any]]],
    after_view: dict[str, list[dict[str, Any]]],
    shard_ids: list[str] | tuple[str, ...],
) -> dict[str, Any]:
    before_shards = _index_view(before_view, "shards", "shard_id")
    after_shards = _index_view(after_view, "shards", "shard_id")
    fields = ["summary_json", "content_digest", "node_id", "lease_id", "capacity_version"]
    for shard_id in shard_ids:
        before = before_shards.get(shard_id)
        after = after_shards.get(shard_id)
        if before is None or after is None:
            raise ReplayError("unchanged shard missing", shard_id=shard_id, field="shards.shard_id")
        for field in fields:
            if before.get(field) != after.get(field):
                raise ReplayError(
                    "unchanged record byte changed",
                    shard_id=shard_id,
                    field=f"shards.{field}",
                )
    return {"byte_identical_shards": list(shard_ids)}
