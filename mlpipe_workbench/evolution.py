"""Specification evolution, dependency impact, and shard invalidation."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .errors import E_STATE, WorkbenchError
from .loader import prepare_from_objects
from .models import PreparedSpec, StageSpec


def prepared_from_spec_row(row: dict[str, Any]) -> PreparedSpec:
    dataset_obj = json.loads(row["dataset_json"])
    pipeline_obj = json.loads(row["pipeline_json"])
    prepared = prepare_from_objects(dataset_obj, pipeline_obj)
    if prepared.fingerprint != row["fingerprint"]:
        raise WorkbenchError(
            E_STATE,
            "ledger specification snapshot does not reproduce its fingerprint",
            location="specs.fingerprint",
            details={
                "expected": row["fingerprint"],
                "actual": prepared.fingerprint,
                "spec_version": row["spec_version"],
            },
        )
    return prepared


def _direct_reasons(old: StageSpec, new: StageSpec) -> list[str]:
    reasons: list[str] = []
    if old.type != new.type:
        reasons.append("type_changed")
    if old.version != new.version:
        reasons.append("version_changed")
    if old.params != new.params:
        reasons.append("params_changed")
    if old.upstream != new.upstream:
        reasons.append("upstream_changed")
    return reasons


def compare_specs(
    old: PreparedSpec | None,
    new: PreparedSpec,
    old_spec_version: int | None,
) -> dict[str, Any]:
    if old is None:
        affected = [
            {
                "stage_id": stage.id,
                "ordinal": stage.ordinal,
                "type": stage.type,
                "old_version": None,
                "new_version": stage.version,
                "old_fingerprint": None,
                "new_fingerprint": stage.config_digest,
                "output_digest_changed": True,
                "reasons": ["new_specification"],
            }
            for stage in new.pipeline.stages
        ]
        return {
            "old_spec_version": None,
            "new_spec_version": old_spec_version,
            "old_fingerprint": None,
            "new_fingerprint": new.fingerprint,
            "old_dataset_digest": None,
            "new_dataset_digest": new.dataset_digest,
            "dataset_changed": True,
            "fixed_scale_changed": False,
            "sharding_changed": False,
            "affected_stages": affected,
            "config_changed_stages": affected,
            "removed_stages": [],
            "unaffected_stages": [],
            "view_replaced": True,
        }

    old_by_id = old.pipeline.stage_by_id
    new_by_id = new.pipeline.stage_by_id
    dataset_changed = old.dataset_digest != new.dataset_digest
    fixed_scale_changed = old.pipeline.fixed_scale != new.pipeline.fixed_scale
    sharding_changed = old.pipeline.sharding != new.pipeline.sharding
    global_data_change = dataset_changed or fixed_scale_changed

    direct_changes: dict[str, list[str]] = {}
    for stage_id in sorted(set(old_by_id) | set(new_by_id)):
        old_stage = old_by_id.get(stage_id)
        new_stage = new_by_id.get(stage_id)
        if old_stage is None:
            direct_changes[stage_id] = ["added_stage"]
        elif new_stage is None:
            direct_changes[stage_id] = ["removed_stage"]
        else:
            reasons = _direct_reasons(old_stage, new_stage)
            if sharding_changed and new_stage.type == "shard_pack":
                reasons.append("sharding_changed")
            if reasons:
                direct_changes[stage_id] = reasons

    affected_ids: set[str] = set()
    reasons_by_id: dict[str, list[str]] = {}
    for stage in new.pipeline.stages:
        reasons: list[str] = []
        old_stage = old_by_id.get(stage.id)
        if old_stage is None:
            reasons.append("added_stage")
        elif global_data_change:
            if dataset_changed:
                reasons.append("dataset_changed")
            if fixed_scale_changed:
                reasons.append("fixed_scale_changed")
        if stage.id in direct_changes:
            reasons.extend(direct_changes[stage.id])
        if stage.upstream is not None and stage.upstream in affected_ids:
            reasons.append("upstream_affected")
        if reasons:
            deduped: list[str] = []
            for reason in reasons:
                if reason not in deduped:
                    deduped.append(reason)
            affected_ids.add(stage.id)
            reasons_by_id[stage.id] = deduped

    affected: list[dict[str, Any]] = []
    for stage in new.pipeline.stages:
        if stage.id not in affected_ids:
            continue
        old_stage = old_by_id.get(stage.id)
        affected.append(
            {
                "stage_id": stage.id,
                "ordinal": stage.ordinal,
                "type": stage.type,
                "old_version": old_stage.version if old_stage else None,
                "new_version": stage.version,
                "old_fingerprint": old_stage.config_digest if old_stage else None,
                "new_fingerprint": stage.config_digest,
                "output_digest_changed": None,
                "reasons": reasons_by_id[stage.id],
            }
        )

    config_changed = [
        {
            "stage_id": stage_id,
            "old_version": old_by_id[stage_id].version if stage_id in old_by_id else None,
            "new_version": new_by_id[stage_id].version if stage_id in new_by_id else None,
            "old_fingerprint": (
                old_by_id[stage_id].config_digest if stage_id in old_by_id else None
            ),
            "new_fingerprint": (
                new_by_id[stage_id].config_digest if stage_id in new_by_id else None
            ),
            "reasons": reasons,
        }
        for stage_id, reasons in sorted(
            direct_changes.items(),
            key=lambda item: (
                new_by_id[item[0]].ordinal if item[0] in new_by_id else 10**9,
                item[0],
            ),
        )
    ]
    removed = [
        {
            "stage_id": stage_id,
            "old_version": old_by_id[stage_id].version,
            "old_fingerprint": old_by_id[stage_id].config_digest,
            "reasons": reasons,
        }
        for stage_id, reasons in sorted(direct_changes.items())
        if stage_id not in new_by_id
    ]
    unaffected = [
        {
            "stage_id": stage.id,
            "ordinal": stage.ordinal,
            "version": stage.version,
            "config_fingerprint": stage.config_digest,
        }
        for stage in new.pipeline.stages
        if stage.id not in affected_ids
    ]
    return {
        "old_spec_version": old_spec_version,
        "new_spec_version": old_spec_version,
        "old_fingerprint": old.fingerprint,
        "new_fingerprint": new.fingerprint,
        "old_dataset_digest": old.dataset_digest,
        "new_dataset_digest": new.dataset_digest,
        "dataset_changed": dataset_changed,
        "fixed_scale_changed": fixed_scale_changed,
        "sharding_changed": sharding_changed,
        "affected_stages": affected,
        "config_changed_stages": config_changed,
        "removed_stages": removed,
        "unaffected_stages": unaffected,
        "view_replaced": True,
    }


def enrich_with_shard_invalidation(
    report: dict[str, Any],
    old_shards: list[dict[str, Any]],
    new_shards: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[int, int]]:
    old_counter = Counter(
        (
            shard["stage_id"],
            int(shard["start_row"]),
            int(shard["end_row"]),
            int(shard["row_count"]),
            int(shard["source_stage_version"]),
            shard["shard_fingerprint"],
        )
        for shard in old_shards
    )
    reused_versions: dict[int, int] = {}
    preserved_indices: list[int] = []
    preserved_old_indices: set[int] = set()
    for shard in new_shards:
        key = (
            shard["source_stage_id"],
            int(shard["start_row"]),
            int(shard["end_row"]),
            int(shard["row_count"]),
            int(shard["source_stage_version"]),
            shard["shard_fingerprint"],
        )
        if old_counter[key] > 0:
            old_counter[key] -= 1
            preserved_indices.append(int(shard["shard_index"]))
            if old_shards:
                reused_versions[int(shard["shard_index"])] = int(
                    old_shards[0]["spec_version"]
                )
                for old_shard in old_shards:
                    old_key = (
                        old_shard["stage_id"],
                        int(old_shard["start_row"]),
                        int(old_shard["end_row"]),
                        int(old_shard["row_count"]),
                        int(old_shard["source_stage_version"]),
                        old_shard["shard_fingerprint"],
                    )
                    if old_key == key and int(old_shard["shard_index"]) not in preserved_old_indices:
                        preserved_old_indices.add(int(old_shard["shard_index"]))
                        break
    preserved_count = len(preserved_indices)
    old_count = len(old_shards)
    report = dict(report)
    report.update(
        {
            "old_shard_count": old_count,
            "new_shard_count": len(new_shards),
            "invalidated_shard_count": max(0, old_count - preserved_count),
            "preserved_shard_count": preserved_count,
            "preserved_shard_indices": preserved_indices,
            "polluted_old_shard_indices": sorted(
                {
                    int(shard["shard_index"])
                    for shard in old_shards
                    if int(shard["shard_index"]) not in preserved_old_indices
                }
            ),
        }
    )
    return report, reused_versions


def enrich_stage_outputs(
    report: dict[str, Any],
    old_view: dict[str, Any] | None,
    new_stage_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    old_outputs = {
        stage["stage_id"]: stage.get("output_digest")
        for stage in (old_view or {}).get("stages", [])
    }
    new_outputs = {stage["stage_id"]: stage.get("output_digest") for stage in new_stage_rows}
    output_changed = [
        {
            "stage_id": stage_id,
            "old_output_digest": old_outputs.get(stage_id),
            "new_output_digest": new_outputs.get(stage_id),
        }
        for stage_id in sorted(new_outputs)
        if old_outputs.get(stage_id) != new_outputs.get(stage_id)
    ]
    unaffected_output_mismatches = []
    for stage in report.get("unaffected_stages", []):
        stage_id = stage["stage_id"]
        if old_outputs.get(stage_id) != new_outputs.get(stage_id):
            unaffected_output_mismatches.append(
                {
                    "stage_id": stage_id,
                    "old_output_digest": old_outputs.get(stage_id),
                    "new_output_digest": new_outputs.get(stage_id),
                }
            )
    report = dict(report)
    report["stage_output_changes"] = output_changed
    report["unaffected_output_mismatches"] = unaffected_output_mismatches
    report["unaffected_outputs_stable"] = not unaffected_output_mismatches
    return report


def classify_current_shards(
    shards: list[dict[str, Any]], prepared: PreparedSpec
) -> list[dict[str, Any]]:
    """Mark shard trust against the currently compiled plan.

    A shard is stale when its source stage version or bound spec fingerprint no
    longer matches the current compiled specification.
    """

    stage = prepared.pipeline.shard_stage
    result = []
    for shard in shards:
        stale_reasons: list[str] = []
        if shard.get("source_stage_version") != stage.version:
            stale_reasons.append("stage_version_mismatch")
        if shard.get("fingerprint") != prepared.fingerprint:
            stale_reasons.append("spec_fingerprint_mismatch")
        if shard.get("source_stage_id") != stage.id:
            stale_reasons.append("source_stage_mismatch")
        payload = dict(shard)
        payload["stale"] = bool(stale_reasons)
        payload["stale_reasons"] = stale_reasons
        result.append(payload)
    return result
