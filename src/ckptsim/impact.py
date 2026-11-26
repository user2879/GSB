from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .canonical import canonical_json, sha256_value
from .model import build_batch_samples, compute_loss, generate_parameter_set
from .plan import EventPlan
from .spec import LayerConfig, ParameterVersion, SpecBundle


COMPARED_LAYER_FIELDS = ("dimension", "scale_factor", "derived_from")


def expected_batch_records(
    bundle: SpecBundle,
    plan: EventPlan,
    *,
    layers: LayerConfig | None = None,
    only_keys: set[tuple[str, str]] | None = None,
) -> dict[tuple[str, str], dict[str, Any]]:
    layer_config = layers or bundle.layers
    versions = layer_config.by_version()
    shards = bundle.shards.by_id()
    caches: dict[str, dict[int, Any]] = {shard_id: {} for shard_id in shards}
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for batch in plan.batches:
        if only_keys is not None and batch.key not in only_keys:
            continue
        shard = shards[batch.shard_id]
        version = versions[batch.parameter_version]
        parameters = generate_parameter_set(
            shard, version, cache=caches[batch.shard_id]
        )
        samples = build_batch_samples(
            shard, parameters, batch.batch_id, batch.sample_ids
        )
        loss = compute_loss(
            shard,
            parameters,
            batch.batch_id,
            batch.sample_ids,
            samples=samples,
        )
        records[batch.key] = {
            "shard_id": batch.shard_id,
            "batch_id": batch.batch_id,
            "seq": batch.event.seq,
            "parameter_version": batch.parameter_version,
            "generation": batch.generation,
            "sample_ids": list(batch.sample_ids),
            "samples_digest": loss.samples_digest,
            "parameter_digest": parameters.digest,
            "loss": loss.payload,
            "loss_digest": loss.digest,
            "checkpoint_id": batch.checkpoint_id,
            "status": "committed",
        }
    return records


def _layer_changes(
    original: ParameterVersion, edited: ParameterVersion
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    before = original.by_id()
    after = edited.by_id()
    for layer_id in sorted(set(before) | set(after)):
        if layer_id not in before:
            changes.append(
                {
                    "parameter_version": edited.parameter_version,
                    "layer_id": layer_id,
                    "field": "layer",
                    "before": None,
                    "after": after[layer_id].to_dict(),
                }
            )
            continue
        if layer_id not in after:
            changes.append(
                {
                    "parameter_version": original.parameter_version,
                    "layer_id": layer_id,
                    "field": "layer",
                    "before": before[layer_id].to_dict(),
                    "after": None,
                }
            )
            continue
        old_layer = before[layer_id]
        new_layer = after[layer_id]
        for field in COMPARED_LAYER_FIELDS:
            old_value = getattr(old_layer, field)
            new_value = getattr(new_layer, field)
            if field == "derived_from":
                old_value = sorted(old_value)
                new_value = sorted(new_value)
            if old_value != new_value:
                changes.append(
                    {
                        "parameter_version": edited.parameter_version,
                        "layer_id": layer_id,
                        "field": field,
                        "before": old_value,
                        "after": new_value,
                    }
                )
    return changes


def _affected_closure(
    original: ParameterVersion,
    edited: ParameterVersion,
    changed: set[str],
) -> dict[str, list[str]]:
    closure: dict[str, list[str]] = {}
    combined_descendants: dict[str, set[str]] = {
        layer_id: set() for layer_id in set(original.by_id()) | set(edited.by_id())
    }
    for version in (original, edited):
        for layer_id, descendants in version.descendants().items():
            combined_descendants.setdefault(layer_id, set()).update(descendants)
    for layer_id in sorted(changed):
        closure[layer_id] = sorted(combined_descendants.get(layer_id, set()))
    return closure


def analyze_impact(
    bundle: SpecBundle,
    edited_layers: LayerConfig,
    plan: EventPlan,
    baseline_records: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    original_versions = bundle.layers.by_version()
    edited_versions = edited_layers.by_version()
    all_versions = sorted(set(original_versions) | set(edited_versions))
    changes: list[dict[str, Any]] = []
    affected_layers: dict[int, tuple[str, ...]] = {}
    affected_versions: list[int] = []
    steps: list[dict[str, Any]] = []
    for version_number in all_versions:
        before = original_versions.get(version_number)
        after = edited_versions.get(version_number)
        if before is None and after is None:
            continue
        if before is None:
            version_changes = [
                {
                    "parameter_version": version_number,
                    "layer_id": layer_id,
                    "field": "layer",
                    "before": None,
                    "after": after.by_id()[layer_id].to_dict(),
                }
                for layer_id in sorted(after.by_id())
            ]
        elif after is None:
            version_changes = [
                {
                    "parameter_version": version_number,
                    "layer_id": layer_id,
                    "field": "layer",
                    "before": before.by_id()[layer_id].to_dict(),
                    "after": None,
                }
                for layer_id in sorted(before.by_id())
            ]
        else:
            version_changes = _layer_changes(before, after)
        if not version_changes:
            continue
        affected_versions.append(version_number)
        changes.extend(version_changes)
        changed_layers = {change["layer_id"] for change in version_changes}
        if before is not None and after is not None:
            closure = _affected_closure(before, after, changed_layers)
        else:
            only = before or after
            assert only is not None
            closure = {
                layer_id: list(only.descendants().get(layer_id, []))
                for layer_id in sorted(changed_layers)
            }
        closure_union: set[str] = set(changed_layers)
        for layer_id in sorted(closure):
            closure_union.add(layer_id)
            for descendant in closure[layer_id]:
                closure_union.add(descendant)
                steps.append(
                    {
                        "kind": "dependency",
                        "parameter_version": version_number,
                        "from_layer_id": layer_id,
                        "to_layer_id": descendant,
                        "basis": f"derived_from closure {layer_id} -> {descendant}",
                    }
                )
        affected_layers[version_number] = tuple(sorted(closure_union))
        for change in sorted(
            version_changes,
            key=lambda item: (item["layer_id"], item["field"]),
        ):
            steps.append(
                {
                    "kind": "edit",
                    "parameter_version": version_number,
                    "layer_id": change["layer_id"],
                    "field": change["field"],
                    "basis": f"edited {change['layer_id']}.{change['field']}",
                }
            )
        steps.append(
            {
                "kind": "parameter-version",
                "parameter_version": version_number,
                "affected_layer_ids": list(affected_layers[version_number]),
                "basis": "parameter set and every downstream layer digest change",
            }
        )

    affected_versions_set = set(affected_versions)
    affected_batches: list[dict[str, Any]] = []
    unchanged_batches: list[dict[str, Any]] = []
    affected_keys: set[tuple[str, str]] = set()
    for batch in sorted(
        plan.batches, key=lambda item: (item.shard_id, item.event.seq, item.batch_id)
    ):
        record = baseline_records.get(batch.key)
        if record is None:
            continue
        if batch.parameter_version in affected_versions_set:
            affected_keys.add(batch.key)
            layer_ids = affected_layers[batch.parameter_version]
            affected_batches.append(
                {
                    "shard_id": batch.shard_id,
                    "batch_id": batch.batch_id,
                    "seq": batch.event.seq,
                    "parameter_version": batch.parameter_version,
                    "checkpoint_id": record.get("checkpoint_id"),
                    "loss_digest": record.get("loss_digest"),
                    "affected_layer_ids": list(layer_ids),
                    "basis": (
                        f"batch uses parameter_version {batch.parameter_version}; "
                        f"affected layers {', '.join(layer_ids)}"
                    ),
                }
            )
            steps.append(
                {
                    "kind": "batch",
                    "shard_id": batch.shard_id,
                    "batch_id": batch.batch_id,
                    "parameter_version": batch.parameter_version,
                    "basis": f"parameter_version {batch.parameter_version} -> batch",
                }
            )
            steps.append(
                {
                    "kind": "loss-summary",
                    "shard_id": batch.shard_id,
                    "batch_id": batch.batch_id,
                    "basis": "layer tensor digest -> fixed-point forward -> loss digest",
                }
            )
        else:
            unchanged_batches.append(
                {
                    "shard_id": batch.shard_id,
                    "batch_id": batch.batch_id,
                    "parameter_version": batch.parameter_version,
                    "checkpoint_id": record.get("checkpoint_id"),
                    "loss_digest": record.get("loss_digest"),
                }
            )

    affected_checkpoint_ids: set[str] = set()
    checkpoint_details: list[dict[str, Any]] = []
    for checkpoint_id, checkpoint in plan.checkpoints.items():
        batch_keys = set(checkpoint.batch_keys)
        owned_affected = sorted(batch_keys & affected_keys, key=lambda key: (key[0], key[1]))
        version_affected = checkpoint.spec.parameter_version in affected_versions_set
        if owned_affected or version_affected:
            affected_checkpoint_ids.add(checkpoint_id)
            checkpoint_details.append(
                {
                    "checkpoint_id": checkpoint_id,
                    "parameter_version": checkpoint.spec.parameter_version,
                    "generation": checkpoint.spec.generation,
                    "affected_batches": [
                        {"shard_id": shard_id, "batch_id": batch_id}
                        for shard_id, batch_id in owned_affected
                    ],
                    "basis": (
                        "checkpoint parameter version changed"
                        if version_affected
                        else "checkpoint owns an affected batch"
                    ),
                }
            )
            steps.append(
                {
                    "kind": "checkpoint",
                    "checkpoint_id": checkpoint_id,
                    "basis": "batch/loss dependency chain -> checkpoint baseline",
                }
            )
    unchanged_checkpoint_ids = sorted(
        set(plan.checkpoints) - affected_checkpoint_ids
    )
    unchanged_evidence = canonical_json(unchanged_batches)
    report = {
        "changes": sorted(
            changes,
            key=lambda item: (
                item["parameter_version"],
                item["layer_id"],
                item["field"],
            ),
        ),
        "affected_parameter_versions": affected_versions,
        "affected_layers": {
            str(version): list(layers)
            for version, layers in sorted(affected_layers.items())
        },
        "affected_batches": affected_batches,
        "affected_loss_digests": [
            {
                "shard_id": batch["shard_id"],
                "batch_id": batch["batch_id"],
                "before": batch["loss_digest"],
            }
            for batch in affected_batches
        ],
        "affected_checkpoints": sorted(affected_checkpoint_ids),
        "checkpoint_details": sorted(
            checkpoint_details, key=lambda item: item["checkpoint_id"]
        ),
        "unchanged_batches": unchanged_batches,
        "unchanged_checkpoints": unchanged_checkpoint_ids,
        "unchanged_digest": sha256_value(unchanged_evidence),
        "steps": steps,
    }
    report["report_digest"] = sha256_value(report)
    return report


def selective_recompute(
    bundle: SpecBundle,
    edited_layers: LayerConfig,
    plan: EventPlan,
    baseline_records: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    impact = analyze_impact(bundle, edited_layers, plan, baseline_records)
    affected_keys = {
        (item["shard_id"], item["batch_id"]) for item in impact["affected_batches"]
    }
    recomputed = expected_batch_records(
        bundle, plan, layers=edited_layers, only_keys=affected_keys
    )
    combined_records = dict(baseline_records)
    combined_records.update(recomputed)
    unchanged_before = [
        baseline_records[key]
        for key in sorted(
            (key for key in baseline_records if key not in affected_keys),
            key=lambda item: (item[0], baseline_records[item].get("seq", 0), item[1]),
        )
    ]
    unchanged_after = list(unchanged_before)
    if canonical_json(unchanged_before) != canonical_json(unchanged_after):
        raise AssertionError("unaffected records changed during selective recomputation")
    changes: list[dict[str, Any]] = []
    for key in sorted(affected_keys, key=lambda item: (item[0], baseline_records[item]["seq"], item[1])):
        before = baseline_records[key]
        after = recomputed[key]
        changes.append(
            {
                "shard_id": key[0],
                "batch_id": key[1],
                "parameter_digest_before": before["parameter_digest"],
                "parameter_digest_after": after["parameter_digest"],
                "loss_digest_before": before["loss_digest"],
                "loss_digest_after": after["loss_digest"],
                "loss_before": before["loss"],
                "loss_after": after["loss"],
                "checkpoint_id": before.get("checkpoint_id"),
            }
        )
    checkpoint_changes: list[dict[str, Any]] = []
    for checkpoint_id in impact["affected_checkpoints"]:
        checkpoint = plan.checkpoints[checkpoint_id]
        keys = sorted(checkpoint.batch_keys)

        def baseline_for(records: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
            return [
                {
                    "shard_id": shard_id,
                    "batch_id": batch_id,
                    "parameter_version": records[(shard_id, batch_id)][
                        "parameter_version"
                    ],
                    "generation": records[(shard_id, batch_id)]["generation"],
                    "sample_ids": records[(shard_id, batch_id)]["sample_ids"],
                    "loss_digest": records[(shard_id, batch_id)]["loss_digest"],
                }
                for shard_id, batch_id in keys
            ]

        before = baseline_for(baseline_records)
        after = baseline_for(combined_records)
        checkpoint_changes.append(
            {
                "checkpoint_id": checkpoint_id,
                "parameter_version": checkpoint.spec.parameter_version,
                "generation": checkpoint.spec.generation,
                "batch_keys": [list(key) for key in keys],
                "baseline_digest_before": sha256_value({"loss_baseline": before}),
                "baseline_digest_after": sha256_value({"loss_baseline": after}),
                "loss_baseline_before": before,
                "loss_baseline_after": after,
            }
        )
    unchanged_checkpoints_before = [
        item
        for item in impact["unchanged_checkpoints"]
    ]
    return {
        "impact": impact,
        "recomputed_batch_count": len(recomputed),
        "reused_batch_count": len(unchanged_before),
        "recomputed": changes,
        "recomputed_checkpoint_count": len(checkpoint_changes),
        "recomputed_checkpoints": checkpoint_changes,
        "unchanged_checkpoints": unchanged_checkpoints_before,
        "unchanged_verified": True,
        "unchanged_before_digest": sha256_value(unchanged_before),
        "unchanged_after_digest": sha256_value(unchanged_after),
        "unchanged_digest": impact["unchanged_digest"],
        "recomputed_digest": sha256_value(changes),
    }
