from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .aggregate import FIXED_POINT_VERSION, FixedPointAggregator
from .cache import PageCache
from .canonical import canonical_bytes, canonical_json, sha256_hex
from .errors import EngineError, ErrorCode


SNAPSHOT_SCHEMA_VERSION = 1


def make_snapshot(
    *,
    cache: PageCache,
    aggregator: FixedPointAggregator,
    scan_order: list[str] | tuple[str, ...],
    position: int,
    reference_trace: list[dict[str, Any]],
    reference_final_output: dict[str, Any],
) -> dict[str, Any]:
    if not 0 <= position <= len(scan_order):
        raise EngineError(ErrorCode.E_REPLAY, "snapshot position is outside the scan order", path="$snapshot.position")
    if len(reference_trace) != len(scan_order):
        raise EngineError(
            ErrorCode.E_REPLAY,
            "reference trace length differs from scan order",
            path="$snapshot.reference_trace",
            details={"trace_records": len(reference_trace), "scan_records": len(scan_order)},
        )
    output_bytes = canonical_bytes(reference_final_output)
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "position": position,
        "scan_order": list(scan_order),
        "manifest_format_versions": list(cache.manifest.format_versions),
        "manifest_header_version": cache.manifest.header_version,
        "fixed_point": {
            "version": FIXED_POINT_VERSION,
            "scale": aggregator.scale,
            "group_column": aggregator.group_column,
            "value_column": aggregator.value_column,
        },
        "cache": cache.state(),
        "aggregation": aggregator.state(),
        "disk_state": cache.disk_inventory(),
        "reference_trace": reference_trace,
        "reference_final_output": reference_final_output,
        "reference_final_output_sha256": sha256_hex(output_bytes),
    }


def save_snapshot(path: str | Path, snapshot: dict[str, Any]) -> bytes:
    data = canonical_bytes(snapshot)
    from .canonical import atomic_write_bytes

    atomic_write_bytes(path, data)
    return data


def load_snapshot(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EngineError(ErrorCode.E_REPLAY, f"cannot load snapshot: {exc}", path=str(target)) from exc
    if not isinstance(value, dict):
        raise EngineError(ErrorCode.E_REPLAY, "snapshot root must be an object", path=str(target))
    if value.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise EngineError(
            ErrorCode.E_REPLAY,
            "unsupported snapshot schema version",
            path="$snapshot.schema_version",
            details={"version": value.get("schema_version"), "supported": SNAPSHOT_SCHEMA_VERSION},
        )
    required = {
        "position",
        "scan_order",
        "manifest_format_versions",
        "manifest_header_version",
        "fixed_point",
        "cache",
        "aggregation",
        "disk_state",
        "reference_trace",
        "reference_final_output",
        "reference_final_output_sha256",
    }
    missing = sorted(required - set(value))
    if missing:
        raise EngineError(
            ErrorCode.E_REPLAY,
            "snapshot is missing required fields",
            path="$snapshot",
            details={"missing_fields": missing},
        )
    return value


def compare_disk_state(snapshot_disk: dict[str, Any], current_disk: dict[str, Any]) -> dict[str, Any]:
    snapshot_versions = sorted({int(value) for value in snapshot_disk.get("manifest_versions", [])})
    current_versions = sorted({int(value) for value in current_disk.get("manifest_versions", [])})
    snapshot_segments = {
        str(row.get("name")): row for row in snapshot_disk.get("segments", []) if isinstance(row, dict) and row.get("name")
    }
    current_segments = {
        str(row.get("name")): row for row in current_disk.get("segments", []) if isinstance(row, dict) and row.get("name")
    }
    snapshot_names = set(snapshot_segments)
    current_names = set(current_segments)
    lost = sorted(snapshot_names - current_names)
    new = sorted(current_names - snapshot_names)
    changed: list[str] = []
    for name in sorted(snapshot_names & current_names):
        snapshot_row = snapshot_segments[name]
        current_row = current_segments[name]
        if snapshot_row.get("file_sha256") != current_row.get("file_sha256") or snapshot_row.get("status") != current_row.get("status"):
            changed.append(name)
    manifest_versions_changed = snapshot_versions != current_versions
    header_changed = int(snapshot_disk.get("manifest_header_version", 0)) != int(current_disk.get("manifest_header_version", 0))
    manifest_content_changed = snapshot_disk.get("manifest_sha256") != current_disk.get("manifest_sha256")
    return {
        "compatible": not manifest_versions_changed
        and not header_changed
        and not manifest_content_changed
        and not lost
        and not changed,
        "manifest_versions_changed": manifest_versions_changed,
        "manifest_header_changed": header_changed,
        "manifest_content_changed": manifest_content_changed,
        "snapshot_manifest_versions": snapshot_versions,
        "current_manifest_versions": current_versions,
        "lost_overflow_segments": lost,
        "new_overflow_segments": new,
        "changed_overflow_segments": changed,
    }


def restore_snapshot(
    snapshot: dict[str, Any],
    *,
    cache: PageCache,
    aggregator: FixedPointAggregator,
    exact_disk: bool,
) -> dict[str, Any]:
    current_disk = cache.disk_inventory()
    diff = compare_disk_state(snapshot["disk_state"], current_disk)
    snapshot_versions = sorted({int(value) for value in snapshot.get("manifest_format_versions", [])})
    current_versions = sorted({int(value) for value in current_disk.get("manifest_versions", [])})
    if snapshot_versions != current_versions:
        diff["manifest_versions_changed"] = True
        diff["compatible"] = False
        diff["snapshot_manifest_versions"] = snapshot_versions
        diff["current_manifest_versions"] = current_versions
    snapshot_header = int(snapshot.get("manifest_header_version", 0))
    current_header = int(current_disk.get("manifest_header_version", 0))
    if snapshot_header != current_header:
        diff["manifest_header_changed"] = True
        diff["compatible"] = False
    fixed_point = snapshot.get("fixed_point", {})
    if fixed_point.get("version") != FIXED_POINT_VERSION or int(fixed_point.get("scale", -1)) != aggregator.scale:
        raise EngineError(
            ErrorCode.E_REPLAY,
            "fixed-point semantics changed; snapshot cannot be replayed",
            path="$snapshot.fixed_point.version",
            details={
                "snapshot_version": fixed_point.get("version"),
                "current_version": FIXED_POINT_VERSION,
                "snapshot_scale": fixed_point.get("scale"),
                "current_scale": aggregator.scale,
                "basis": "fixed_point_version_changed",
            },
        )
    if fixed_point.get("group_column") != aggregator.group_column or fixed_point.get("value_column") != aggregator.value_column:
        raise EngineError(
            ErrorCode.E_REPLAY,
            "aggregation column semantics changed; snapshot cannot be replayed",
            path="$snapshot.fixed_point",
            details={
                "snapshot": fixed_point,
                "current_group_column": aggregator.group_column,
                "current_value_column": aggregator.value_column,
                "basis": "aggregation_columns_changed",
            },
        )
    if exact_disk:
        if diff["manifest_versions_changed"] or diff["manifest_header_changed"]:
            first_page = _first_affected_page(snapshot, diff)
            raise EngineError(
                ErrorCode.E_REPLAY,
                "manifest format versions changed after the snapshot",
                path="$snapshot.manifest_format_versions",
                page_id=first_page,
                details={**diff, "basis": "manifest_format_version_changed"},
            )
        if diff["manifest_content_changed"]:
            first_page = _first_affected_page(snapshot, diff)
            raise EngineError(
                ErrorCode.E_REPLAY,
                "manifest content changed after the snapshot",
                path="$snapshot.manifest_format_versions",
                page_id=first_page,
                details={**diff, "basis": "manifest_state_changed"},
            )
        if diff["lost_overflow_segments"] or diff["changed_overflow_segments"]:
            affected = sorted(set(diff["lost_overflow_segments"]) | set(diff["changed_overflow_segments"]))
            first_page = _page_for_segment(snapshot, affected[0]) if affected else None
            raise EngineError(
                ErrorCode.E_REPLAY,
                "overflow segment state changed after the snapshot",
                path="$snapshot.disk_state.segments",
                page_id=first_page,
                details={**diff, "basis": "overflow_segment_missing_or_changed"},
            )

    # Both state containers are replaced wholesale. No current resident page,
    # counter, aggregation group, or segment index participates in a merge.
    cache.restore_state(snapshot["cache"])
    aggregator.load_state(snapshot["aggregation"])
    return {
        "status": "restored",
        "position": int(snapshot["position"]),
        "disk_diff": diff,
        "replacement_mode": "wholesale",
        "restored_cache_pages": len(snapshot["cache"].get("resident_pages", []))
        + len(snapshot["cache"].get("overflow_pages", [])),
        "restored_aggregation_groups": len(snapshot["aggregation"].get("groups", [])),
    }


def snapshot_digest(snapshot: dict[str, Any]) -> str:
    return sha256_hex(canonical_json(snapshot).encode("utf-8"))


def _page_for_segment(snapshot: dict[str, Any], segment_name: str) -> str | None:
    for row in snapshot.get("cache", {}).get("overflow_pages", []):
        if row.get("overflow_ref") == segment_name:
            return str(row.get("page_id"))
    for row in snapshot.get("disk_state", {}).get("segments", []):
        if row.get("name") == segment_name and row.get("page_id"):
            return str(row["page_id"])
    return None


def _first_affected_page(snapshot: dict[str, Any], diff: dict[str, Any]) -> str | None:
    for row in snapshot.get("cache", {}).get("overflow_pages", []):
        return str(row.get("page_id"))
    for name in diff.get("lost_overflow_segments", []):
        page_id = _page_for_segment(snapshot, name)
        if page_id:
            return page_id
    return None
