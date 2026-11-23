"""Versioned session snapshots with content validation and replacement."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import DiagnosticError, snapshot_error
from .schema import Plan, parse_plan, plan_diff
from .util import (
    atomic_write_json,
    digest_value,
    digest_without_field,
    sha256_file,
    wall_time_ms,
)


SNAPSHOT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class SnapshotLoadResult:
    path: str
    plan: Plan
    sample_count: int
    dataset_digest: str | None
    lease_records: tuple[dict[str, Any], ...]
    layer_summaries: dict[str, Any]
    processed_batch_cursor: dict[str, Any] | None
    ledger_content_digest: str | None
    differences: tuple[dict[str, Any], ...]
    replaced_whole_state: bool = True
    content_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "path": self.path,
            "plan_version": self.plan.version,
            "quantization_scale_version": self.plan.quantization_scale_version,
            "sample_count": self.sample_count,
            "dataset_digest": self.dataset_digest,
            "lease_records": list(self.lease_records),
            "layer_summaries": self.layer_summaries,
            "processed_batch_cursor": self.processed_batch_cursor,
            "ledger_content_digest": self.ledger_content_digest,
            "differences": list(self.differences),
            "replaced_whole_state": self.replaced_whole_state,
            "content_digest": self.content_digest,
        }


def _sorted_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(record) for record in sorted(records, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))]


def build_snapshot(
    plan: Plan,
    *,
    sample_count: int,
    dataset_digest: str | None,
    lease_records: Sequence[Mapping[str, Any]],
    layer_summaries: Mapping[str, Any],
    processed_batch_cursor: Mapping[str, Any] | None,
    ledger_content_digest: str | None,
    created_ms: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": SNAPSHOT_FORMAT_VERSION,
        "created_ms": wall_time_ms() if created_ms is None else created_ms,
        "plan": plan.to_dict(),
        "plan_version": plan.version,
        "quantization_scale_version": plan.quantization_scale_version,
        "sample_count": sample_count,
        "dataset_digest": dataset_digest,
        "lease_records": _sorted_records(lease_records),
        "layer_summaries": {key: layer_summaries[key] for key in sorted(layer_summaries)},
        "processed_batch_cursor": dict(processed_batch_cursor) if processed_batch_cursor else None,
        "ledger_content_digest": ledger_content_digest,
    }
    payload["content_digest"] = digest_value(payload)
    return payload


def save_snapshot(
    directory: str | Path,
    snapshot: Mapping[str, Any],
) -> str:
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    digest = str(snapshot["content_digest"])
    version = int(snapshot["plan_version"])
    path = target_dir / f"snapshot-v{version:06d}-{digest[:16]}.json"
    atomic_write_json(path, snapshot)
    verify = load_snapshot_payload(path)
    if verify["content_digest"] != digest:
        raise snapshot_error("snapshot verification failed after write", path=str(path))
    return str(path)


def load_snapshot_payload(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    try:
        with open(target, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise snapshot_error(
            f"invalid snapshot JSON at line {exc.lineno} column {exc.colno}",
            path=str(target),
        ) from exc
    except OSError as exc:
        raise DiagnosticError(
            "E_IO", f"cannot read snapshot: {exc}", field="snapshot_path"
        ) from exc
    if not isinstance(payload, dict):
        raise snapshot_error("snapshot must be a JSON object", path=str(target))
    if payload.get("format_version") != SNAPSHOT_FORMAT_VERSION:
        raise snapshot_error(
            "snapshot format version cannot be aligned",
            expected=SNAPSHOT_FORMAT_VERSION,
            actual=payload.get("format_version"),
            path=str(target),
        )
    stored_digest = payload.get("content_digest")
    if not isinstance(stored_digest, str):
        raise snapshot_error("snapshot has no content_digest", path=str(target))
    actual_digest = digest_without_field(payload, "content_digest")
    if actual_digest != stored_digest:
        raise snapshot_error(
            "snapshot content digest mismatch",
            expected=stored_digest,
            actual=actual_digest,
            path=str(target),
        )
    return payload


def load_snapshot(
    path: str | Path,
    *,
    current_plan: Plan,
    current_dataset_digest: str | None = None,
    current_ledger_digest: str | None = None,
    current_lease_records: Sequence[Mapping[str, Any]] | None = None,
) -> SnapshotLoadResult:
    payload = load_snapshot_payload(path)
    snapshot_plan_raw = payload.get("plan")
    try:
        snapshot_plan = parse_plan(snapshot_plan_raw)
    except DiagnosticError as exc:
        raise snapshot_error(
            "snapshot plan is invalid and cannot be aligned",
            path=str(path),
            first_error=exc.to_dict(),
        ) from exc
    if current_plan.schema_version != snapshot_plan.schema_version:
        raise snapshot_error(
            "snapshot schema version cannot be aligned",
            expected=current_plan.schema_version,
            actual=snapshot_plan.schema_version,
            path=str(path),
        )
    if current_plan.quantization_scale_version != snapshot_plan.quantization_scale_version:
        raise snapshot_error(
            "snapshot quantization scale version cannot be aligned",
            expected=current_plan.quantization_scale_version,
            actual=snapshot_plan.quantization_scale_version,
            path=str(path),
        )
    if payload.get("plan_version") != snapshot_plan.version:
        raise snapshot_error(
            "snapshot plan version fields disagree",
            field="plan_version",
            path=str(path),
        )
    if payload.get("quantization_scale_version") != snapshot_plan.quantization_scale_version:
        raise snapshot_error(
            "snapshot quantization scale fields disagree",
            field="quantization_scale_version",
            path=str(path),
        )
    differences: list[dict[str, Any]] = plan_diff(current_plan, snapshot_plan)
    snapshot_dataset = payload.get("dataset_digest")
    if current_dataset_digest is not None and snapshot_dataset != current_dataset_digest:
        differences.append(
            {
                "item": "dataset_digest",
                "left": current_dataset_digest,
                "right": snapshot_dataset,
            }
        )
    snapshot_ledger = payload.get("ledger_content_digest")
    if current_ledger_digest is not None and snapshot_ledger != current_ledger_digest:
        differences.append(
            {
                "item": "ledger.content_digest",
                "left": current_ledger_digest,
                "right": snapshot_ledger,
            }
        )
    snapshot_leases = payload.get("lease_records", [])
    if current_lease_records is not None:
        normalized_current = _sorted_records(current_lease_records)
        normalized_snapshot = _sorted_records(snapshot_leases)
        if normalized_current != normalized_snapshot:
            differences.append(
                {
                    "item": "lease_records",
                    "left": normalized_current,
                    "right": normalized_snapshot,
                }
            )
    try:
        sample_count = int(payload.get("sample_count"))
    except (TypeError, ValueError) as exc:
        raise snapshot_error("snapshot sample_count is invalid", path=str(path)) from exc
    if sample_count <= 0:
        raise snapshot_error("snapshot sample_count must be positive", path=str(path))
    lease_records = payload.get("lease_records")
    if not isinstance(lease_records, list):
        raise snapshot_error("snapshot lease_records must be an array", path=str(path))
    summaries = payload.get("layer_summaries")
    if not isinstance(summaries, dict):
        raise snapshot_error("snapshot layer_summaries must be an object", path=str(path))
    return SnapshotLoadResult(
        path=str(path),
        plan=snapshot_plan,
        sample_count=sample_count,
        dataset_digest=snapshot_dataset,
        lease_records=tuple(_sorted_records(lease_records)),
        layer_summaries={key: summaries[key] for key in sorted(summaries)},
        processed_batch_cursor=payload.get("processed_batch_cursor"),
        ledger_content_digest=snapshot_ledger,
        differences=tuple(
            sorted(differences, key=lambda item: json.dumps(item, sort_keys=True, separators=(",", ":")))
        ),
        replaced_whole_state=True,
        content_digest=str(payload["content_digest"]),
    )


def list_snapshots(directory: str | Path) -> list[dict[str, Any]]:
    target = Path(directory)
    if not target.exists():
        return []
    result: list[dict[str, Any]] = []
    for path in sorted(target.glob("snapshot-v*.json")):
        try:
            payload = load_snapshot_payload(path)
            result.append(
                {
                    "path": str(path),
                    "filename": path.name,
                    "plan_version": payload.get("plan_version"),
                    "quantization_scale_version": payload.get("quantization_scale_version"),
                    "created_ms": payload.get("created_ms"),
                    "content_digest": payload.get("content_digest"),
                    "file_size": path.stat().st_size,
                }
            )
        except DiagnosticError as exc:
            result.append(
                {
                    "path": str(path),
                    "filename": path.name,
                    "error": exc.to_dict(),
                }
            )
    return sorted(
        result,
        key=lambda item: (
            int(item.get("plan_version") or -1),
            int(item.get("created_ms") or -1),
            item["filename"],
        ),
    )
