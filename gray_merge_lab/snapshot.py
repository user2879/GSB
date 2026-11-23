from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, canonical_json, first_difference, sha256_value
from .fixedpoint import BatchFixedStats, aggregate_batch_stats, trunc_div, validate_scale_exp
from .errors import LabError
from .ledger import LEDGER_FORMAT_VERSION, Ledger
from .specs import LoadedSpec


SNAPSHOT_FORMAT_VERSION = 1


def build_manifest(
    spec: LoadedSpec,
    backend_summaries: list[dict[str, Any]],
    scale_exp: int,
) -> dict[str, Any]:
    validate_scale_exp(scale_exp)
    summaries = sorted(backend_summaries, key=lambda item: item["backend_id"])
    return {
        "backend_set_version": sha256_value(
            {
                "algorithm": "gray-merge-lab-backend-set-v1",
                "backends": [item.declaration() for item in sorted(spec.backends, key=lambda x: x.backend_id)],
                "parameter_summaries": summaries,
            }
        ),
        "rules_version": spec.rules_version(),
        "samples_version": spec.samples_version(),
        "scale_exp": scale_exp,
        "scale": 10**scale_exp,
        "backend_summaries": summaries,
    }


def _rule_stats(batches: list[dict[str, Any]], scale_exp: int) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for batch in batches:
        grouped.setdefault(batch["rule_id"], []).append(batch)
    result: dict[str, Any] = {}
    for rule_id in sorted(grouped):
        items = sorted(grouped[rule_id], key=lambda item: item["batch_index"])
        stats = aggregate_batch_stats(
            BatchFixedStats(
                batch_id=item["batch_id"],
                raw_duration_ms=int(item["raw_duration_ms"]),
                raw_samples=int(item["raw_sample_count"]),
                scale_exp=scale_exp,
            )
            for item in items
        )
        result[rule_id] = {
            "rule_id": rule_id,
            "batch_count": len(items),
            "stable_samples": sum(
                int(item["raw_sample_count"]) for item in items if item["destination"] == "stable"
            ),
            "canary_samples": sum(
                int(item["raw_sample_count"]) for item in items if item["destination"] == "canary"
            ),
            "stable_batches": sum(1 for item in items if item["destination"] == "stable"),
            "canary_batches": sum(1 for item in items if item["destination"] == "canary"),
            "raw_duration_sum_ms": stats.raw_duration_sum_ms,
            "raw_sample_sum": stats.raw_sample_sum,
            "avg_duration_q": stats.avg_duration_q,
            "avg_samples_q": stats.avg_samples_q,
            "avg_duration_original_approx": stats.avg_duration_q / stats.scale,
            "avg_samples_original_approx": stats.avg_samples_q / stats.scale,
            "sampling_error_affected": False,
            "fixedpoint": stats.to_dict(),
        }
    return result


def build_payload(ledger: Ledger, scale_exp: int) -> dict[str, Any]:
    validate_scale_exp(scale_exp)
    records = ledger.batches()
    mismatched = [item["batch_id"] for item in records if int(item["scale_exp"]) != scale_exp]
    if mismatched:
        raise LabError(
            "E_LEDGER",
            "ledger batches use a different fixed-point scale",
            batch_id=mismatched[0],
            details={"requested_scale_exp": scale_exp, "batch_scale_exp": int(records[0]["scale_exp"])},
        )
    preview_batches: list[dict[str, Any]] = []
    conclusions: list[dict[str, Any]] = []
    for record in records:
        preview = record.get("preview")
        if preview is None:
            preview = {
                "batch_id": record["batch_id"],
                "batch_index": record["batch_index"],
                "rule_id": record["rule_id"],
                "destination": record["destination"],
                "raw_duration_ms": record["raw_duration_ms"],
                "raw_sample_count": record["raw_sample_count"],
                "duration_q": record["duration_q"],
                "samples_q": record["samples_q"],
                "output_digest": record["output_digest"],
                "backend_param_digest": record.get("backend_param_digest"),
                "backend_param_version": record.get("backend_param_version"),
            }
        else:
            preview = dict(preview)
            preview["batch_index"] = record["batch_index"]
        preview_batches.append(preview)
        conclusions.append(record["conclusion"])
    cursor = ledger.cursor()
    return {
        "snapshot_kind": "gray-merge-quick-view",
        "scale_exp": scale_exp,
        "truncation": "toward_zero",
        "single_value_error_bound_original": 1.0 / (10**scale_exp),
        "batch_aggregate_rule": "N batches independently re-aggregated => N/S; canonical raw-sum aggregation => 1/S",
        "preview_batches": preview_batches,
        "rule_stats": _rule_stats(records, scale_exp),
        "conclusions": conclusions,
        "progress": {
            "completed_batches": cursor["completed_batches"],
            "next_batch_index": cursor["next_batch_index"],
            "batch_ids": cursor["batch_ids"],
        },
        "source_ledger": {
            "ledger_format_version": LEDGER_FORMAT_VERSION,
            "content_digest": ledger.content_digest(),
        },
    }


def snapshot_document(manifest: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
        "manifest": manifest,
        "payload": payload,
        "content_digest": sha256_value(
            {
                "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
                "manifest": manifest,
                "payload": payload,
            }
        ),
    }


def verify_snapshot_document(document: Any) -> None:
    if not isinstance(document, dict):
        raise LabError("E_SNAPSHOT", "snapshot must be a JSON object")
    if document.get("snapshot_format_version") != SNAPSHOT_FORMAT_VERSION:
        raise LabError(
            "E_SNAPSHOT",
            "unsupported snapshot format version",
            details={
                "field": "snapshot_format_version",
                "expected": SNAPSHOT_FORMAT_VERSION,
                "actual": document.get("snapshot_format_version"),
            },
        )
    expected = sha256_value(
        {
            "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
            "manifest": document.get("manifest"),
            "payload": document.get("payload"),
        }
    )
    if document.get("content_digest") != expected:
        raise LabError(
            "E_SNAPSHOT",
            "snapshot content digest mismatch",
            details={
                "field": "content_digest",
                "snapshot": document.get("content_digest"),
                "computed": expected,
            },
        )


@dataclass(frozen=True)
class SnapshotLoadResult:
    payload: dict[str, Any]
    replaced: bool
    differences: list[dict[str, Any]]


class SnapshotStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def write(self, manifest: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        document = snapshot_document(manifest, payload)
        atomic_write_json(self.path, document)
        return document

    def _read_document(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise LabError("E_SNAPSHOT", f"cannot read snapshot: {exc}") from exc
        if not isinstance(value, dict):
            raise LabError("E_SNAPSHOT", "snapshot root must be an object")
        return value

    def _preserve_rejected(self, document: dict[str, Any] | None) -> None:
        if document is None:
            return
        rejected_path = self.path.with_suffix(self.path.suffix + ".rejected.json")
        atomic_write_json(rejected_path, document)

    def load_or_replace(
        self,
        manifest: dict[str, Any],
        replacement_payload: dict[str, Any] | None = None,
    ) -> SnapshotLoadResult:
        document = self._read_document()
        if document is None:
            payload = replacement_payload or {}
            self.write(manifest, payload)
            return SnapshotLoadResult(payload=payload, replaced=True, differences=[])

        try:
            verify_snapshot_document(document)
        except LabError as exc:
            self._preserve_rejected(document)
            payload = replacement_payload or {}
            self.write(manifest, payload)
            raise LabError(
                "E_SNAPSHOT",
                "snapshot was rejected and replaced wholesale",
                details={
                    **exc.details,
                    "differences": [exc.details],
                    "replacement": str(self.path),
                },
            ) from exc

        old_manifest = document["manifest"]
        checked_fields = ("backend_set_version", "rules_version", "samples_version", "scale_exp")
        differences: list[dict[str, Any]] = []
        for field in checked_fields:
            if old_manifest.get(field) != manifest.get(field):
                differences.append(
                    {
                        "field": f"manifest.{field}",
                        "snapshot": old_manifest.get(field),
                        "current": manifest.get(field),
                    }
                )
        if differences:
            self._preserve_rejected(document)
            payload = replacement_payload or {}
            self.write(manifest, payload)
            raise LabError(
                "E_SNAPSHOT",
                "snapshot manifest differs from current session; replaced wholesale",
                details={"differences": differences, "replacement": str(self.path)},
            )
        return SnapshotLoadResult(
            payload=document["payload"],
            replaced=False,
            differences=[],
        )

    def read_valid(self) -> dict[str, Any]:
        document = self._read_document()
        if document is None:
            raise LabError("E_SNAPSHOT", "snapshot file does not exist")
        verify_snapshot_document(document)
        return document

    def compare_manifest(self, manifest: dict[str, Any]) -> list[dict[str, Any]]:
        document = self.read_valid()
        differences: list[dict[str, Any]] = []
        for field in ("backend_set_version", "rules_version", "samples_version", "scale_exp"):
            left = document["manifest"].get(field)
            right = manifest.get(field)
            if left != right:
                differences.append({"field": f"manifest.{field}", "snapshot": left, "current": right})
        return differences
