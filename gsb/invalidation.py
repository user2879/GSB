"""Stable invalidation propagation and full-rebuild consistency checks."""

from __future__ import annotations

import json
from typing import Any

from .canonical import canonical_json, sha256_hex
from .checkpoint import CheckpointManager
from .errors import DiagnosticError
from .ledger import Ledger
from .model import BatchSpec, EventSpec, LayerSpec, SessionSpec, ShardSpec
from .params import compute_batch_summary, get_parameter_set


def _layer_map(shard: ShardSpec) -> dict[str, LayerSpec]:
    return {layer.layer_id: layer for layer in shard.layers}


def shard_change_reasons(old: ShardSpec | None, new: ShardSpec | None) -> list[str]:
    if old is None and new is None:
        return []
    if old is None:
        return ["shard_added"]
    if new is None:
        return ["shard_removed"]
    reasons: list[str] = []
    for field in ("seed", "sample_dim", "parameter_version"):
        if getattr(old, field) != getattr(new, field):
            reasons.append(f"shard.{field}")
    old_layers = _layer_map(old)
    new_layers = _layer_map(new)
    for layer_id in sorted(set(old_layers) | set(new_layers)):
        before = old_layers.get(layer_id)
        after = new_layers.get(layer_id)
        if before is None:
            reasons.append(f"layer:{layer_id}:added")
            continue
        if after is None:
            reasons.append(f"layer:{layer_id}:removed")
            continue
        for field in ("input_dim", "output_dim", "scale"):
            if getattr(before, field) != getattr(after, field):
                reasons.append(f"layer:{layer_id}:{field}")
        if before.depends_on != after.depends_on:
            reasons.append(f"layer:{layer_id}:depends_on")
    return reasons


def effective_versions(session: SessionSpec) -> dict[str, int]:
    """Versions observed by each batch in the declared arrival stream, without I/O."""

    versions = {shard.shard_id: shard.parameter_version for shard in session.shards}
    result: dict[str, int] = {}
    items: list[tuple[int, int, str, BatchSpec | EventSpec]] = []
    for batch in session.batches:
        items.append((batch.order, 1, batch.batch_id, batch))
    for event in session.events:
        items.append((event.seq, 0, "", event))
    for _, kind, _, item in sorted(items):
        if kind == 1:
            batch = item
            assert isinstance(batch, BatchSpec)
            result[batch.batch_id] = versions[batch.shard_id]
        else:
            event = item
            assert isinstance(event, EventSpec)
            if event.type == "rollback":
                assert event.checkpoint_id is not None
                checkpoint = session.checkpoint(event.checkpoint_id)
                versions[checkpoint.shard_id] = checkpoint.parameter_version
    return result


def _checkpoint_commit_seqs(session: SessionSpec) -> dict[str, int]:
    result: dict[str, int] = {}
    for event in session.events:
        if event.type == "commit" and event.checkpoint_id is not None:
            result[event.checkpoint_id] = min(
                result.get(event.checkpoint_id, event.seq), event.seq
            )
    return result


def _summaries(session: SessionSpec) -> dict[str, dict[str, Any]]:
    versions = effective_versions(session)
    summaries: dict[str, dict[str, Any]] = {}
    for batch in session.batches:
        summaries[batch.batch_id] = compute_batch_summary(
            session.shard(batch.shard_id),
            batch.batch_id,
            batch.sample_ids,
            versions[batch.batch_id],
        )
    return summaries


def audit_invalidation(old: SessionSpec, new: SessionSpec) -> dict[str, Any]:
    old_shards = {shard.shard_id: shard for shard in old.shards}
    new_shards = {shard.shard_id: shard for shard in new.shards}
    shard_reasons: dict[str, list[str]] = {}
    for shard_id in sorted(set(old_shards) | set(new_shards)):
        reasons = shard_change_reasons(old_shards.get(shard_id), new_shards.get(shard_id))
        if reasons:
            shard_reasons[shard_id] = reasons

    old_batches = {batch.batch_id: batch for batch in old.batches}
    new_batches = {batch.batch_id: batch for batch in new.batches}
    old_versions = effective_versions(old)
    new_versions = effective_versions(new)
    old_summaries = _summaries(old)
    new_summaries = _summaries(new)
    old_checkpoints = {cp.checkpoint_id: cp for cp in old.checkpoints}
    new_checkpoints = {cp.checkpoint_id: cp for cp in new.checkpoints}

    reasons: dict[str, list[dict[str, Any]]] = {}

    def add_reason(batch_id: str, kind: str, reason: str, **details: Any) -> None:
        reasons.setdefault(batch_id, []).append(
            {"kind": kind, "reason": reason, "details": details}
        )

    for batch_id in sorted(set(old_batches) | set(new_batches)):
        before = old_batches.get(batch_id)
        after = new_batches.get(batch_id)
        if before is None:
            add_reason(batch_id, "direct", "batch_added")
            continue
        if after is None:
            add_reason(batch_id, "direct", "batch_removed")
            continue
        if before.shard_id != after.shard_id:
            add_reason(
                batch_id,
                "direct",
                "batch.shard_id",
                before=before.shard_id,
                after=after.shard_id,
            )
        if tuple(before.sample_ids) != tuple(after.sample_ids):
            add_reason(
                batch_id,
                "direct",
                "batch.sample_ids",
                before=list(before.sample_ids),
                after=list(after.sample_ids),
            )
        if before.checkpoint_id != after.checkpoint_id:
            add_reason(
                batch_id,
                "direct",
                "batch.checkpoint_id",
                before=before.checkpoint_id,
                after=after.checkpoint_id,
            )
        if before.depends_on_batches != after.depends_on_batches:
            add_reason(
                batch_id,
                "direct",
                "batch.depends_on_batches",
                before=list(before.depends_on_batches),
                after=list(after.depends_on_batches),
            )
        if old_versions.get(batch_id) != new_versions.get(batch_id):
            add_reason(
                batch_id,
                "direct",
                "effective_parameter_version",
                before=old_versions.get(batch_id),
                after=new_versions.get(batch_id),
            )
        shard_id = after.shard_id
        if shard_id in shard_reasons:
            add_reason(
                batch_id,
                "direct",
                "shard_parameter_changed",
                shard_id=shard_id,
                fields=shard_reasons[shard_id],
            )
        if (
            canonical_json(old_summaries[batch_id])
            != canonical_json(new_summaries[batch_id])
            and batch_id not in reasons
        ):
            add_reason(batch_id, "direct", "loss_summary_changed")

    invalid_batch_ids = set(reasons)
    invalid_checkpoint_ids: set[str] = set()
    propagation: list[dict[str, Any]] = []

    def invalidate_checkpoint(checkpoint_id: str, reason: str, **details: Any) -> None:
        if checkpoint_id in invalid_checkpoint_ids:
            return
        invalid_checkpoint_ids.add(checkpoint_id)
        propagation.append(
            {
                "from": details.pop("source", "checkpoint"),
                "to": checkpoint_id,
                "kind": "checkpoint",
                "reason": reason,
                "details": details,
            }
        )

    for checkpoint_id in sorted(set(old_checkpoints) | set(new_checkpoints)):
        before = old_checkpoints.get(checkpoint_id)
        after = new_checkpoints.get(checkpoint_id)
        if after is None:
            invalidate_checkpoint(checkpoint_id, "checkpoint_removed")
            continue
        if before is None:
            invalidate_checkpoint(checkpoint_id, "checkpoint_added", shard_id=after.shard_id)
            continue
        if before.to_dict() != after.to_dict():
            invalidate_checkpoint(
                checkpoint_id,
                "checkpoint_declaration_changed",
                shard_id=after.shard_id,
            )
        if after.shard_id in shard_reasons:
            invalidate_checkpoint(
                checkpoint_id,
                "shard_parameter_changed",
                shard_id=after.shard_id,
                fields=shard_reasons[after.shard_id],
            )
        commit_seq = _checkpoint_commit_seqs(new).get(checkpoint_id)
        if commit_seq is not None:
            for batch_id in sorted(invalid_batch_ids):
                batch = new_batches.get(batch_id)
                if (
                    batch is not None
                    and batch.shard_id == after.shard_id
                    and batch.order < commit_seq
                ):
                    invalidate_checkpoint(
                        checkpoint_id,
                        "loss_baseline_depends_on_invalid_batch",
                        source=batch_id,
                        shard_id=after.shard_id,
                        batch_order=batch.order,
                    )

    changed = True
    while changed:
        changed = False
        for batch_id in sorted(set(old_batches) | set(new_batches)):
            if batch_id in invalid_batch_ids:
                continue
            batch = new_batches.get(batch_id)
            if batch is None:
                continue
            for dependency in batch.depends_on_batches:
                if dependency in invalid_batch_ids:
                    add_reason(
                        batch_id,
                        "indirect",
                        "depends_on_invalid_batch",
                        source=dependency,
                    )
                    invalid_batch_ids.add(batch_id)
                    propagation.append(
                        {
                            "from": dependency,
                            "to": batch_id,
                            "kind": "batch",
                            "reason": "depends_on_invalid_batch",
                            "details": {"shard_id": batch.shard_id},
                        }
                    )
                    changed = True
                    break
            if batch_id in invalid_batch_ids:
                continue
            if batch.checkpoint_id is not None and batch.checkpoint_id in invalid_checkpoint_ids:
                add_reason(
                    batch_id,
                    "indirect",
                    "references_invalid_checkpoint",
                    checkpoint_id=batch.checkpoint_id,
                )
                invalid_batch_ids.add(batch_id)
                propagation.append(
                    {
                        "from": batch.checkpoint_id,
                        "to": batch_id,
                        "kind": "batch",
                        "reason": "references_invalid_checkpoint",
                        "details": {"shard_id": batch.shard_id},
                    }
                )
                changed = True

    # Indirect batch invalidation can change a checkpoint's frozen baseline;
    # close the fixed point until neither side creates new invalidation.
    commit_seqs = _checkpoint_commit_seqs(new)
    changed = True
    while changed:
        changed = False
        for checkpoint_id in sorted(new_checkpoints):
            if checkpoint_id in invalid_checkpoint_ids:
                continue
            checkpoint = new_checkpoints[checkpoint_id]
            commit_seq = commit_seqs.get(checkpoint_id)
            if commit_seq is None:
                continue
            for batch_id in sorted(invalid_batch_ids):
                batch = new_batches.get(batch_id)
                if (
                    batch is not None
                    and batch.shard_id == checkpoint.shard_id
                    and batch.order < commit_seq
                ):
                    invalidate_checkpoint(
                        checkpoint_id,
                        "loss_baseline_depends_on_invalid_batch",
                        source=batch_id,
                        shard_id=checkpoint.shard_id,
                        batch_order=batch.order,
                    )
                    changed = True
                    break
        for batch_id in sorted(set(old_batches) | set(new_batches)):
            if batch_id in invalid_batch_ids:
                continue
            batch = new_batches.get(batch_id)
            if batch is None:
                continue
            for dependency in batch.depends_on_batches:
                if dependency in invalid_batch_ids:
                    add_reason(
                        batch_id,
                        "indirect",
                        "depends_on_invalid_batch",
                        source=dependency,
                    )
                    invalid_batch_ids.add(batch_id)
                    propagation.append(
                        {
                            "from": dependency,
                            "to": batch_id,
                            "kind": "batch",
                            "reason": "depends_on_invalid_batch",
                            "details": {"shard_id": batch.shard_id},
                        }
                    )
                    changed = True
                    break
            if batch_id in invalid_batch_ids:
                continue
            if batch.checkpoint_id is not None and batch.checkpoint_id in invalid_checkpoint_ids:
                add_reason(
                    batch_id,
                    "indirect",
                    "references_invalid_checkpoint",
                    checkpoint_id=batch.checkpoint_id,
                )
                invalid_batch_ids.add(batch_id)
                propagation.append(
                    {
                        "from": batch.checkpoint_id,
                        "to": batch_id,
                        "kind": "batch",
                        "reason": "references_invalid_checkpoint",
                        "details": {"shard_id": batch.shard_id},
                    }
                )
                changed = True

    invalid_batches: list[dict[str, Any]] = []
    for batch_id in sorted(
        invalid_batch_ids,
        key=lambda key: (
            (new_batches.get(key) or old_batches.get(key)).shard_id,
            (new_batches.get(key) or old_batches.get(key)).order,
            key,
        ),
    ):
        before = old_batches.get(batch_id)
        after = new_batches.get(batch_id)
        invalid_batches.append(
            {
                "batch_id": batch_id,
                "shard_id": (after or before).shard_id,
                "batch_order": (after or before).order,
                "reasons": sorted(
                    reasons.get(batch_id, []), key=lambda item: (item["kind"], item["reason"])
                ),
                "old_loss_summary": old_summaries.get(batch_id),
                "new_loss_summary": new_summaries.get(batch_id),
            }
        )

    unaffected: list[dict[str, Any]] = []
    for batch_id in sorted(set(old_batches) & set(new_batches)):
        if batch_id in invalid_batch_ids:
            continue
        before_summary = old_summaries[batch_id]
        after_summary = new_summaries[batch_id]
        identical = canonical_json(before_summary) == canonical_json(after_summary)
        if not identical:
            raise DiagnosticError(
                "E_REPLAY",
                "an unaffected batch changed loss summary; propagation is incomplete",
                details={"batch_id": batch_id, "field": "loss_digest"},
            )
        unaffected.append(
            {
                "batch_id": batch_id,
                "shard_id": new_batches[batch_id].shard_id,
                "batch_order": new_batches[batch_id].order,
                "loss_digest": after_summary["loss_digest"],
                "canonical_summary": canonical_json(after_summary),
                "byte_identical": True,
            }
        )

    affected_checkpoints: list[dict[str, Any]] = []
    for checkpoint_id in sorted(invalid_checkpoint_ids):
        cp = new_checkpoints.get(checkpoint_id) or old_checkpoints[checkpoint_id]
        affected_checkpoints.append(
            {
                "checkpoint_id": checkpoint_id,
                "shard_id": cp.shard_id,
                "parameter_version": cp.parameter_version,
                "old_declaration": old_checkpoints[checkpoint_id].to_dict()
                if checkpoint_id in old_checkpoints
                else None,
                "new_declaration": new_checkpoints[checkpoint_id].to_dict()
                if checkpoint_id in new_checkpoints
                else None,
            }
        )

    unaffected_checkpoints: list[dict[str, Any]] = []
    for checkpoint_id in sorted(set(old_checkpoints) & set(new_checkpoints)):
        if checkpoint_id in invalid_checkpoint_ids:
            continue
        before = old_checkpoints[checkpoint_id]
        after = new_checkpoints[checkpoint_id]
        if canonical_json(before.to_dict()) != canonical_json(after.to_dict()):
            raise DiagnosticError(
                "E_REPLAY",
                "an unaffected checkpoint declaration changed",
                details={"checkpoint_id": checkpoint_id, "field": "checkpoint"},
            )
        unaffected_checkpoints.append(
            {
                "checkpoint_id": checkpoint_id,
                "shard_id": after.shard_id,
                "parameter_version": after.parameter_version,
                "canonical_declaration": canonical_json(after.to_dict()),
                "byte_identical": True,
            }
        )

    report_payload = {
        "invalid_batches": invalid_batches,
        "invalid_checkpoints": affected_checkpoints,
        "unaffected_batches": unaffected,
        "unaffected_checkpoints": unaffected_checkpoints,
        "propagation": sorted(
            propagation, key=lambda item: (item["kind"], item["from"], item["to"], item["reason"])
        ),
        "shard_change_reasons": shard_reasons,
    }
    return {**report_payload, "report_digest": sha256_hex(report_payload)}


def full_rebuild(session: SessionSpec) -> dict[str, Any]:
    versions = effective_versions(session)
    summaries: dict[str, dict[str, Any]] = {}
    manager = CheckpointManager(
        session,
        lambda shard_id, version: get_parameter_set(
            session.shard(shard_id), version
        ).digest,
    )
    items: list[tuple[int, int, BatchSpec | EventSpec]] = []
    for batch in session.batches:
        items.append((batch.order, 1, batch))
    for event in session.events:
        items.append((event.seq, 0, event))
    for _, kind, item in sorted(items, key=lambda row: (row[0], row[1])):
        if kind == 1:
            batch = item
            assert isinstance(batch, BatchSpec)
            summary = compute_batch_summary(
                session.shard(batch.shard_id),
                batch.batch_id,
                batch.sample_ids,
                versions[batch.batch_id],
            )
            summaries[batch.batch_id] = summary
            manager.record_batch(
                sequence=batch.order,
                shard_id=batch.shard_id,
                batch_id=batch.batch_id,
                loss_digest=summary["loss_digest"],
            )
        else:
            event = item
            assert isinstance(event, EventSpec)
            if event.type in {"commit", "rollback"}:
                manager.apply_event(event)
    ordered_summaries = [
        summaries[batch.batch_id]
        for batch in sorted(session.batches, key=lambda item: (item.shard_id, item.order, item.batch_id))
    ]
    checkpoint_state = manager.snapshot()
    payload = {
        "loss_summaries": ordered_summaries,
        "checkpoint_state": checkpoint_state,
        "batch_count": len(ordered_summaries),
    }
    return {**payload, "rebuild_digest": sha256_hex(payload)}


def compare_ledger_with_rebuild(session: SessionSpec, ledger: Ledger) -> dict[str, Any]:
    ledger.replay()
    rebuilt = full_rebuild(session)
    rows = ledger.connection.execute(
        "SELECT checkpoint_id,payload,state_digest FROM checkpoints ORDER BY checkpoint_id"
    ).fetchall()
    for row in rows:
        checkpoint_id = row["checkpoint_id"]
        expected = rebuilt["checkpoint_state"]["checkpoints"].get(checkpoint_id)
        if expected is None:
            raise DiagnosticError(
                "E_REPLAY",
                "ledger contains a checkpoint absent from full rebuild",
                details={"field": "checkpoint_id", "checkpoint_id": checkpoint_id},
            )
        if row["payload"] != canonical_json(expected):
            raise DiagnosticError(
                "E_REPLAY",
                f"checkpoint {checkpoint_id} differs from full rebuild",
                details={
                    "field": "checkpoint_state",
                    "checkpoint_id": checkpoint_id,
                    "expected": expected,
                    "actual": row["payload"],
                },
            )
        if row["state_digest"] != sha256_hex(expected):
            raise DiagnosticError(
                "E_REPLAY",
                f"checkpoint digest {checkpoint_id} differs from full rebuild",
                details={"field": "checkpoint_digest", "checkpoint_id": checkpoint_id},
            )
    expected_ids = set(rebuilt["checkpoint_state"]["checkpoints"])
    actual_ids = {row["checkpoint_id"] for row in rows}
    if expected_ids != actual_ids:
        raise DiagnosticError(
            "E_REPLAY",
            "checkpoint id set differs from full rebuild",
            details={
                "field": "checkpoints",
                "expected": sorted(expected_ids),
                "actual": sorted(actual_ids),
            },
        )
    runtime_row = ledger.connection.execute(
        "SELECT value FROM runtime_state WHERE key='checkpoint_manager'"
    ).fetchone()
    if runtime_row is not None:
        actual_runtime = json.loads(runtime_row["value"])
        if actual_runtime != rebuilt["checkpoint_state"]:
            for key in sorted(set(actual_runtime) | set(rebuilt["checkpoint_state"])):
                if actual_runtime.get(key) != rebuilt["checkpoint_state"].get(key):
                    raise DiagnosticError(
                        "E_REPLAY",
                        f"runtime checkpoint field {key} differs from full rebuild",
                        details={
                            "field": f"checkpoint_state.{key}",
                            "expected": rebuilt["checkpoint_state"].get(key),
                            "actual": actual_runtime.get(key),
                        },
                    )
    return {
        "consistent": True,
        "batch_count": rebuilt["batch_count"],
        "checkpoint_count": len(expected_ids),
        "rebuild_digest": rebuilt["rebuild_digest"],
    }
