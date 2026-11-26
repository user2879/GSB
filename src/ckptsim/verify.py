from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import canonical_json, sha256_value
from .errors import CkptError
from .impact import analyze_impact, expected_batch_records
from .ledger import FORMAT_VERSION, open_readonly
from .model import generate_parameter_set
from .plan import EventPlan, build_event_plan
from .spec import LayerConfig, SpecBundle, load_layers


@dataclass
class VerifyResult:
    verified: bool
    comparisons: int
    first_mismatch: dict[str, Any] | None
    snapshot: dict[str, Any]
    impact_verified: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "comparisons": self.comparisons,
            "first_mismatch": self.first_mismatch,
            "snapshot": self.snapshot,
            "impact_verified": self.impact_verified,
        }


def _decode_batches(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        "SELECT * FROM batches ORDER BY shard_id, seq, batch_id"
    ).fetchall()
    return [
        {
            "shard_id": row["shard_id"],
            "batch_id": row["batch_id"],
            "seq": row["seq"],
            "parameter_version": row["parameter_version"],
            "generation": row["generation"],
            "sample_ids": json.loads(row["sample_ids"]),
            "samples_digest": row["samples_digest"],
            "parameter_digest": row["parameter_digest"],
            "loss": json.loads(row["loss_json"]) if row["loss_json"] else None,
            "loss_digest": row["loss_digest"],
            "checkpoint_id": row["checkpoint_id"],
            "checkpoint_status": row["checkpoint_status"],
            "lease_id": row["lease_id"],
            "lease_version": row["lease_version"],
            "status": row["status"],
        }
        for row in rows
    ]


def _decode_checkpoints(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        "SELECT * FROM checkpoints ORDER BY generation, checkpoint_id"
    ).fetchall()
    return [
        {
            "checkpoint_id": row["checkpoint_id"],
            "seq": row["seq"],
            "parameter_version": row["parameter_version"],
            "generation": row["generation"],
            "parameter_digests": json.loads(row["parameter_digests"]),
            "batch_keys": [tuple(item) for item in json.loads(row["batch_keys"])],
            "loss_baseline": json.loads(row["loss_baseline"]),
            "baseline_digest": row["baseline_digest"],
            "status": row["status"],
        }
        for row in rows
    ]


def _decode_rollbacks(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = connection.execute("SELECT * FROM rollbacks ORDER BY seq, id").fetchall()
    return [
        {
            "seq": row["seq"],
            "checkpoint_id": row["checkpoint_id"],
            "target_version": row["target_version"],
            "target_generation": row["target_generation"],
            "revoked_versions": json.loads(row["revoked_versions"]),
            "revoked_checkpoints": json.loads(row["revoked_checkpoints"]),
            "affected_committed_batches": json.loads(row["affected_batches"]),
            "generation_before": row["generation_before"],
            "generation_after": row["generation_after"],
        }
        for row in rows
    ]


def _expected_checkpoints(
    bundle: SpecBundle,
    plan: EventPlan,
    records: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    parameter_caches: dict[str, dict[int, Any]] = {
        shard.shard_id: {} for shard in bundle.shards.shards
    }
    rolled_back_checkpoints = {
        rollback.checkpoint_id for rollback in plan.rollbacks
    } | {
        checkpoint_id
        for rollback in plan.rollbacks
        for checkpoint_id in rollback.revoked_checkpoints
    }
    affected_batches = {
        tuple(item)
        for rollback in plan.rollbacks
        for item in rollback.affected_committed_batches
    }
    result: list[dict[str, Any]] = []
    for checkpoint_id in plan.checkpoint_order:
        checkpoint = plan.checkpoints[checkpoint_id]
        baseline: list[dict[str, Any]] = []
        for shard_id, batch_id in sorted(checkpoint.batch_keys):
            record = records[(shard_id, batch_id)]
            baseline.append(
                {
                    "shard_id": shard_id,
                    "batch_id": batch_id,
                    "parameter_version": record["parameter_version"],
                    "generation": record["generation"],
                    "sample_ids": record["sample_ids"],
                    "loss_digest": record["loss_digest"],
                }
            )
        parameter_digests = {
            shard.shard_id: generate_parameter_set(
                shard,
                bundle.layers.by_version()[checkpoint.spec.parameter_version],
                cache=parameter_caches[shard.shard_id],
            ).digest
            for shard in bundle.shards.shards
        }
        status = "rolled_back" if checkpoint.rolled_back else "committed"
        result.append(
            {
                "checkpoint_id": checkpoint_id,
                "seq": checkpoint.event.seq,
                "parameter_version": checkpoint.spec.parameter_version,
                "generation": checkpoint.spec.generation,
                "parameter_digests": parameter_digests,
                "batch_keys": [
                    tuple(key) for key in sorted(checkpoint.batch_keys)
                ],
                "loss_baseline": baseline,
                "baseline_digest": sha256_value({"loss_baseline": baseline}),
                "status": status,
            }
        )
    return result


def _expected_batch_status(
    plan: EventPlan, key: tuple[str, str]
) -> tuple[str | None, str]:
    batch = plan.batch_by_key[key]
    checkpoint_id = batch.checkpoint_id
    if checkpoint_id is None:
        return None, "unassigned"
    checkpoint = plan.checkpoints[checkpoint_id]
    if checkpoint.rolled_back:
        return checkpoint_id, "rolled_back"
    return checkpoint_id, "committed"


def _mismatch(
    *,
    shard_id: str | None,
    batch_id: str | None,
    field: str,
    expected: Any,
    actual: Any,
    scope: str,
) -> dict[str, Any]:
    return {
        "scope": scope,
        "shard_id": shard_id,
        "batch_id": batch_id,
        "field": field,
        "expected": expected,
        "actual": actual,
    }


def verify_ledger(
    bundle: SpecBundle,
    ledger_path: str | Path,
    *,
    edited_layers_path: str | Path | None = None,
    raise_on_error: bool = True,
) -> VerifyResult:
    plan = build_event_plan(bundle)
    connection = open_readonly(ledger_path)
    comparisons = 0
    mismatch: dict[str, Any] | None = None
    impact_verified = False
    snapshot: dict[str, Any] = {}
    try:
        meta_rows = connection.execute("SELECT key,value FROM meta").fetchall()
        meta = {str(row["key"]): str(row["value"]) for row in meta_rows}
        snapshot = {
            "format_version": meta.get("format_version"),
            "shard_set_version": meta.get("shard_set_version"),
            "parameter_version_dialect": meta.get("parameter_version_dialect"),
            "content_digest": meta.get("content_digest"),
        }
        expected_identity = {
            "format_version": str(FORMAT_VERSION),
            "shard_set_version": bundle.shard_set_version,
            "parameter_version_dialect": bundle.parameter_dialect,
            "content_digest": bundle.content_digest,
        }
        for field in expected_identity:
            comparisons += 1
            actual = meta.get(field)
            if actual != expected_identity[field] and mismatch is None:
                mismatch = _mismatch(
                    shard_id=None,
                    batch_id=None,
                    field=f"meta.{field}",
                    expected=expected_identity[field],
                    actual=actual,
                    scope="snapshot",
                )
        if mismatch is not None:
            raise CkptError(
                "E_VERIFY",
                f"first inconsistency at snapshot field {mismatch['field']}",
                field=mismatch["field"],
                extra={"first_mismatch": mismatch},
            )
        actual_batches = _decode_batches(connection)
        actual_batch_map = {
            (item["shard_id"], item["batch_id"]): item for item in actual_batches
        }
        expected_records = expected_batch_records(bundle, plan)
        expected_keys = set(expected_records)
        actual_keys = set(actual_batch_map)
        combined = sorted(
            expected_keys | actual_keys,
            key=lambda key: (
                key[0],
                expected_records.get(key, actual_batch_map.get(key, {})).get("seq", 0),
                key[1],
            ),
        )
        for key in combined:
            if mismatch is not None:
                break
            shard_id, batch_id = key
            expected = expected_records.get(key)
            actual = actual_batch_map.get(key)
            if expected is None:
                mismatch = _mismatch(
                    shard_id=shard_id,
                    batch_id=batch_id,
                    field="record",
                    expected=None,
                    actual="unexpected batch row",
                    scope="batch",
                )
                break
            if actual is None:
                mismatch = _mismatch(
                    shard_id=shard_id,
                    batch_id=batch_id,
                    field="record",
                    expected="batch row",
                    actual=None,
                    scope="batch",
                )
                break
            checkpoint_id, checkpoint_status = _expected_batch_status(plan, key)
            expected_fields = {
                "seq": expected["seq"],
                "parameter_version": expected["parameter_version"],
                "generation": expected["generation"],
                "sample_ids": expected["sample_ids"],
                "samples_digest": expected["samples_digest"],
                "parameter_digest": expected["parameter_digest"],
                "loss_digest": expected["loss_digest"],
                "loss": expected["loss"],
                "checkpoint_id": checkpoint_id,
                "checkpoint_status": checkpoint_status,
                "status": "committed",
                "lease_id": plan.batch_by_key[key].lease_id,
            }
            for field, expected_value in expected_fields.items():
                comparisons += 1
                actual_value = actual.get(field)
                if canonical_json(expected_value) != canonical_json(actual_value):
                    mismatch = _mismatch(
                        shard_id=shard_id,
                        batch_id=batch_id,
                        field=field,
                        expected=expected_value,
                        actual=actual_value,
                        scope="batch",
                    )
                    break
        if mismatch is None:
            expected_checkpoints = _expected_checkpoints(bundle, plan, expected_records)
            actual_checkpoints = _decode_checkpoints(connection)
            actual_checkpoint_map = {
                item["checkpoint_id"]: item for item in actual_checkpoints
            }
            expected_checkpoint_map = {
                item["checkpoint_id"]: item for item in expected_checkpoints
            }
            for checkpoint_id in sorted(
                set(expected_checkpoint_map) | set(actual_checkpoint_map),
                key=lambda item: (
                    expected_checkpoint_map.get(item, actual_checkpoint_map[item])[
                        "generation"
                    ],
                    item,
                ),
            ):
                if mismatch is not None:
                    break
                expected = expected_checkpoint_map.get(checkpoint_id)
                actual = actual_checkpoint_map.get(checkpoint_id)
                if expected is None or actual is None:
                    mismatch = _mismatch(
                        shard_id=None,
                        batch_id=None,
                        field=f"checkpoint.{checkpoint_id}",
                        expected=expected,
                        actual=actual,
                        scope="checkpoint",
                    )
                    break
                for field in (
                    "seq",
                    "parameter_version",
                    "generation",
                    "parameter_digests",
                    "batch_keys",
                    "loss_baseline",
                    "baseline_digest",
                    "status",
                ):
                    comparisons += 1
                    if canonical_json(expected[field]) != canonical_json(actual[field]):
                        mismatch = _mismatch(
                            shard_id=None,
                            batch_id=None,
                            field=f"checkpoint.{checkpoint_id}.{field}",
                            expected=expected[field],
                            actual=actual[field],
                            scope="checkpoint",
                        )
                        break
        if mismatch is None:
            actual_rollbacks = _decode_rollbacks(connection)
            expected_rollbacks = [rollback.to_dict() for rollback in plan.rollbacks]
            if len(actual_rollbacks) != len(expected_rollbacks):
                mismatch = _mismatch(
                    shard_id=None,
                    batch_id=None,
                    field="rollbacks.count",
                    expected=len(expected_rollbacks),
                    actual=len(actual_rollbacks),
                    scope="rollback",
                )
            else:
                for index, (expected, actual) in enumerate(
                    zip(expected_rollbacks, actual_rollbacks)
                ):
                    for field in expected:
                        comparisons += 1
                        if canonical_json(expected[field]) != canonical_json(actual.get(field)):
                            mismatch = _mismatch(
                                shard_id=None,
                                batch_id=None,
                                field=f"rollback[{index}].{field}",
                                expected=expected[field],
                                actual=actual.get(field),
                                scope="rollback",
                            )
                            break
                    if mismatch is not None:
                        break
        if mismatch is None:
            actual_progress = {
                row["shard_id"]: row["cursor_seq"]
                for row in connection.execute("SELECT * FROM progress").fetchall()
            }
            for shard_id in sorted(bundle.shards.by_id()):
                seqs = [
                    batch.event.seq
                    for batch in plan.batches
                    if batch.shard_id == shard_id
                ]
                expected_cursor = max(seqs) if seqs else -1
                comparisons += 1
                if actual_progress.get(shard_id) != expected_cursor:
                    mismatch = _mismatch(
                        shard_id=shard_id,
                        batch_id=None,
                        field="progress.cursor_seq",
                        expected=expected_cursor,
                        actual=actual_progress.get(shard_id),
                        scope="progress",
                    )
                    break
        if edited_layers_path is not None and mismatch is None:
            edited_layers = load_layers(edited_layers_path)
            generated = analyze_impact(bundle, edited_layers, plan, expected_records)
            row = connection.execute(
                "SELECT report_json,report_digest FROM impact_runs ORDER BY id DESC LIMIT 1"
            ).fetchone()
            actual = None if row is None else json.loads(row["report_json"])
            actual_digest = None if row is None else row["report_digest"]
            comparisons += 2
            if actual is None or actual_digest != sha256_value(generated):
                mismatch = _mismatch(
                    shard_id=None,
                    batch_id=None,
                    field="impact.report",
                    expected=generated,
                    actual=actual,
                    scope="impact",
                )
            elif canonical_json(actual) != canonical_json(generated):
                mismatch = _mismatch(
                    shard_id=None,
                    batch_id=None,
                    field="impact.report_bytes",
                    expected=generated,
                    actual=actual,
                    scope="impact",
                )
            else:
                impact_verified = True
    finally:
        connection.close()
    if mismatch is not None:
        if raise_on_error:
            raise CkptError(
                "E_VERIFY",
                f"first inconsistency at {mismatch['field']}",
                shard_id=mismatch.get("shard_id"),
                batch_id=mismatch.get("batch_id"),
                field=mismatch.get("field"),
                extra={"first_mismatch": mismatch},
            )
        return VerifyResult(False, comparisons, mismatch, snapshot, impact_verified)
    return VerifyResult(True, comparisons, None, snapshot, impact_verified)
