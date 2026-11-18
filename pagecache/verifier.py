from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from .canonical import canonical_bytes, canonical_json
from .manifest import read_manifest
from .pagefile import page_path, read_page_file
from .segment import segment_inventory
from .snapshot import compare_disk_state, load_snapshot


CHECK_ORDER = (
    "snapshot_schema",
    "fixed_point_semantics",
    "cache_page_indices",
    "resident_page_bytes",
    "manifest_membership",
    "manifest_versions",
    "overflow_segment_index",
    "aggregation_group_order",
    "mean_derivation",
    "sum_error_bound",
    "mean_error_bound",
    "reference_output",
)


def verify_artifacts(
    *,
    snapshot_path: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    pages_dir: str | Path,
    overflow_dir: str | Path,
) -> dict[str, Any]:
    checks: dict[str, dict[str, Any]] = {}

    def run_check(name: str, function: Callable[[], dict[str, Any]]) -> None:
        try:
            details = function()
            checks[name] = {"status": "PASS", "details": details}
        except Exception as exc:  # validator must produce a stable report for malformed inputs
            if hasattr(exc, "to_dict"):
                error = exc.to_dict()  # type: ignore[union-attr]
            else:
                error = {"type": type(exc).__name__, "message": str(exc)}
            checks[name] = {"status": "FAIL", "details": error}

    def snapshot_schema() -> dict[str, Any]:
        nonlocal snapshot, output
        snapshot = load_snapshot(snapshot_path)
        output = _read_json_object(output_path, label="output")
        return {
            "snapshot_schema_version": snapshot["schema_version"],
            "snapshot_position": snapshot["position"],
            "output_groups": len(output.get("groups", [])),
        }

    snapshot: dict[str, Any] = {}
    output: dict[str, Any] = {}
    run_check("snapshot_schema", snapshot_schema)

    def fixed_point_semantics() -> dict[str, Any]:
        snapshot_fixed = snapshot["fixed_point"]
        output_fixed = output["fixed_point"]
        if snapshot_fixed["version"] != output_fixed["version"]:
            raise ValueError("snapshot and output fixed-point versions differ")
        if int(snapshot_fixed["scale"]) != int(output_fixed["scale"]):
            raise ValueError("snapshot and output fixed-point scales differ")
        scale = int(output_fixed["scale"])
        if Fraction(output_fixed["value_abs_error_upper_bound"]["numerator"], output_fixed["value_abs_error_upper_bound"]["denominator"]) != Fraction(1, 2 * scale):
            raise ValueError("value quantization bound is inconsistent with the scale")
        if Fraction(output_fixed["mean_abs_error_upper_bound"]["numerator"], output_fixed["mean_abs_error_upper_bound"]["denominator"]) != Fraction(3, 2 * scale):
            raise ValueError("mean error bound is inconsistent with the scale")
        return {
            "version": output_fixed["version"],
            "scale": scale,
            "mean_rounding": output_fixed["mean_rounding"],
            "value_quantization": output_fixed["value_quantization"],
        }

    run_check("fixed_point_semantics", fixed_point_semantics)

    def cache_page_indices() -> dict[str, Any]:
        cache = snapshot["cache"]
        resident = [str(row["page_id"]) for row in cache["resident_pages"]]
        overflow = [str(row["page_id"]) for row in cache["overflow_pages"]]
        if resident != sorted(resident) or overflow != sorted(overflow):
            raise ValueError("cache page indices are not in stable sorted order")
        if len(set(resident)) != len(resident) or len(set(overflow)) != len(overflow):
            raise ValueError("cache page index contains duplicates")
        if set(resident) & set(overflow):
            raise ValueError("resident and overflow page indices overlap")
        if len(resident) > int(cache["capacity_frames"]):
            raise ValueError("resident page count exceeds the declared capacity")
        last_access = [str(row["page_id"]) for row in cache["last_access"]]
        if last_access != sorted(last_access) or set(last_access) != set(resident):
            raise ValueError("last_access index does not correspond to resident pages")
        return {
            "capacity_frames": int(cache["capacity_frames"]),
            "resident_count": len(resident),
            "overflow_count": len(overflow),
            "resident_pages": resident,
            "overflow_pages": overflow,
        }

    run_check("cache_page_indices", cache_page_indices)

    def resident_page_bytes() -> dict[str, Any]:
        rows = snapshot["cache"]["resident_pages"]
        checked: list[dict[str, str]] = []
        for index, row in enumerate(rows):
            page_id = str(row["page_id"])
            page = read_page_file(page_path(pages_dir, page_id), expected_page_id=page_id)
            actual = hashlib.sha256(page.raw_bytes).hexdigest()
            if actual != str(row["raw_sha256"]):
                raise ValueError(f"resident page {page_id} bytes do not match the snapshot")
            checked.append({"page_id": page_id, "raw_sha256": actual})
        return {"checked_pages": checked}

    run_check("resident_page_bytes", resident_page_bytes)

    def manifest_membership() -> dict[str, Any]:
        manifest = read_manifest(manifest_path)
        manifest_ids = {entry.page_id for entry in manifest.records}
        cache = snapshot["cache"]
        indexed = {str(row["page_id"]) for row in cache["resident_pages"]} | {
            str(row["page_id"]) for row in cache["overflow_pages"]
        }
        missing = sorted(indexed - manifest_ids)
        if missing:
            raise ValueError(f"cache references pages outside the manifest: {missing}")
        return {
            "manifest_pages": sorted(manifest_ids),
            "indexed_pages": sorted(indexed),
            "unloaded_pages": sorted(manifest_ids - indexed),
        }

    run_check("manifest_membership", manifest_membership)

    def manifest_versions() -> dict[str, Any]:
        manifest = read_manifest(manifest_path)
        snapshot_versions = sorted(int(value) for value in snapshot["manifest_format_versions"])
        current_versions = list(manifest.format_versions)
        if snapshot_versions != current_versions:
            raise ValueError(f"manifest versions differ: snapshot={snapshot_versions}, current={current_versions}")
        if int(snapshot["manifest_header_version"]) != manifest.header_version:
            raise ValueError("manifest header version differs from the snapshot")
        return {"snapshot_versions": snapshot_versions, "current_versions": current_versions, "header_version": manifest.header_version}

    run_check("manifest_versions", manifest_versions)

    def overflow_segment_index() -> dict[str, Any]:
        current = {
            "manifest_versions": list(read_manifest(manifest_path).format_versions),
            "manifest_header_version": read_manifest(manifest_path).header_version,
            "segments": segment_inventory(overflow_dir),
        }
        diff = compare_disk_state(snapshot["disk_state"], current)
        current_by_name = {str(row["name"]): row for row in current["segments"] if row.get("name")}
        checked: list[dict[str, Any]] = []
        for row in snapshot["cache"]["overflow_pages"]:
            name = str(row["overflow_ref"])
            segment_row = current_by_name.get(name)
            if segment_row is None:
                raise ValueError(f"overflow segment {name} is missing")
            if segment_row.get("status") != "ok":
                raise ValueError(f"overflow segment {name} is invalid")
            if str(segment_row.get("page_id")) != str(row["page_id"]):
                raise ValueError(f"overflow segment {name} has the wrong page identifier")
            if bool(segment_row.get("compressed")) != bool(row.get("compressed")):
                raise ValueError(f"overflow segment {name} compression flag differs from the snapshot")
            checked.append(
                {
                    "page_id": row["page_id"],
                    "overflow_ref": name,
                    "compressed": bool(row.get("compressed")),
                    "file_sha256": segment_row.get("file_sha256"),
                }
            )
        return {"checked_segments": checked, "disk_diff": diff, "lost": diff["lost_overflow_segments"], "new": diff["new_overflow_segments"]}

    run_check("overflow_segment_index", overflow_segment_index)

    def aggregation_group_order() -> dict[str, Any]:
        groups = output["groups"]
        keys = [int(row["key"]) for row in groups]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError("aggregation groups are not in strict numeric order")
        for row in groups:
            if int(row["count"]) <= 0:
                raise ValueError("aggregation group count must be positive")
        snapshot_groups = snapshot["aggregation"]["groups"]
        snapshot_keys = [int(row["key"]) for row in snapshot_groups]
        if snapshot_keys != sorted(snapshot_keys):
            raise ValueError("snapshot aggregation groups are not sorted")
        return {"keys": keys, "group_count": len(groups), "snapshot_group_count": len(snapshot_groups)}

    run_check("aggregation_group_order", aggregation_group_order)

    def mean_derivation() -> dict[str, Any]:
        checked: list[dict[str, Any]] = []
        for row in output["groups"]:
            count = int(row["count"])
            total = int(row["sum_scaled"])
            expected = total // count if total >= 0 else -((-total) // count)
            if int(row["mean_scaled"]) != expected:
                raise ValueError(f"group {row['key']} mean does not truncate toward zero")
            checked.append({"key": row["key"], "count": count, "sum_scaled": total, "mean_scaled": expected})
        return {"checked_groups": checked, "rounding": "truncate_toward_zero"}

    run_check("mean_derivation", mean_derivation)

    def sum_error_bound() -> dict[str, Any]:
        scale = int(output["fixed_point"]["scale"])
        checked: list[dict[str, Any]] = []
        for row in output["groups"]:
            bound = row["sum_abs_error_upper_bound"]
            expected = Fraction(int(row["count"]), 2 * scale)
            actual = Fraction(int(bound["numerator"]), int(bound["denominator"]))
            if actual != expected:
                raise ValueError(f"group {row['key']} sum error bound is incorrect")
            checked.append({"key": row["key"], "numerator": int(bound["numerator"]), "denominator": int(bound["denominator"])})
        return {"checked_groups": checked}

    run_check("sum_error_bound", sum_error_bound)

    def mean_error_bound() -> dict[str, Any]:
        scale = int(output["fixed_point"]["scale"])
        bound = output["fixed_point"]["mean_abs_error_upper_bound"]
        actual = Fraction(int(bound["numerator"]), int(bound["denominator"]))
        expected = Fraction(3, 2 * scale)
        if actual != expected:
            raise ValueError("mean error bound is not 1.5 quantization units")
        return {"numerator": int(bound["numerator"]), "denominator": int(bound["denominator"]), "value": f"{actual.numerator}/{actual.denominator}"}

    run_check("mean_error_bound", mean_error_bound)

    def reference_output() -> dict[str, Any]:
        reference = snapshot["reference_final_output"]
        if canonical_json(output) != canonical_json(reference):
            raise ValueError("output does not match the snapshot reference output")
        expected_sha = str(snapshot["reference_final_output_sha256"])
        actual_sha = hashlib.sha256(canonical_bytes(output)).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError("output SHA-256 does not match the snapshot")
        return {"output_sha256": actual_sha, "groups": len(output["groups"])}

    run_check("reference_output", reference_output)

    ordered_checks = [{"name": name, **checks[name]} for name in CHECK_ORDER]
    passed = sum(1 for check in ordered_checks if check["status"] == "PASS")
    failed = len(ordered_checks) - passed
    return {
        "status": "PASS" if failed == 0 else "FAIL",
        "summary": {
            "checks_total": len(ordered_checks),
            "checks_passed": passed,
            "checks_failed": failed,
            "snapshot_position": snapshot.get("position") if snapshot else None,
            "output_groups": len(output.get("groups", [])) if output else 0,
        },
        "checks": ordered_checks,
    }


def _read_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    target = Path(path)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} JSON root must be an object")
    return value
