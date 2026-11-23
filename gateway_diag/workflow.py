from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .backend_client import BackendClient
from .engine import build_batch_plans, execute_rule
from .errors import E_REPLAY, E_SNAPSHOT, GatewayError
from .graph import compute_invalidation, dependency_graph_digest, initial_invalidation, topological_order
from .ledger import Ledger
from .loaders import spec_declarations_from_payload
from .logging_utils import StructuredLogger
from .models import Spec
from .routing import distribution_from_receipts
from .snapshot import inspect_snapshot, write_snapshot
from .util import canonical_json


def _trace_steps_from_invalidation(invalidation: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "step": step["step"],
            "rule_id": step["rule_id"],
            "code": step.get("code", "E_STALE"),
            "basis": step.get("basis", {}),
        }
        for step in invalidation.get("invalidated_rules", [])
    ]


def _verify_trace(
    ledger: Ledger,
    *,
    before: Spec | None,
    target: Spec,
    generation: int,
) -> None:
    expected = (
        initial_invalidation(target)
        if before is None
        else compute_invalidation(before, target)
    )
    expected_steps = _trace_steps_from_invalidation(expected)
    stored = ledger.invalidation_payload(generation=generation)
    stored_steps = [
        {
            "step": int(step["step"]),
            "rule_id": step["rule_id"],
            "code": step["code"],
            "basis": step["basis"],
        }
        for step in stored
    ]
    if canonical_json(expected_steps) != canonical_json(stored_steps):
        for index, expected_step in enumerate(expected_steps):
            actual = stored_steps[index] if index < len(stored_steps) else None
            if actual != expected_step:
                raise GatewayError(
                    E_REPLAY,
                    "persisted invalidation trace differs from the replayed derivation",
                    rule_id=expected_step.get("rule_id"),
                    field="invalidation_trace",
                    details={"step": index + 1, "expected": expected_step, "actual": actual},
                )
        raise GatewayError(
            E_REPLAY,
            "persisted invalidation trace length differs from the replayed derivation",
            field="invalidation_trace",
            details={"expected": expected_steps, "actual": stored_steps},
        )


def _validate_completed_replay_state(spec: Spec, ledger: Ledger) -> None:
    states = ledger.get_rule_states()
    for rule_id in topological_order(spec):
        state = states.get(rule_id)
        if state is None or state.get("status") != "complete":
            continue
        receipts = ledger.get_rule_receipts(rule_id)
        if len(receipts) != int(state.get("batch_count", -1)):
            raise GatewayError(
                E_REPLAY,
                "completed rule batch count differs during replay recovery",
                rule_id=rule_id,
                field="batch_count",
                details={"ledger": state.get("batch_count"), "receipts": len(receipts)},
            )
        digest = spec.rule_receipt_set_digest(
            receipt["receipt_digest"] for receipt in receipts
        )
        if digest != state.get("receipt_digest"):
            raise GatewayError(
                E_REPLAY,
                "completed rule receipt digest differs during replay recovery",
                rule_id=rule_id,
                field="receipt_digest",
                details={"ledger": state.get("receipt_digest"), "receipts": digest},
            )
        distribution = distribution_from_receipts(receipts)
        if json.loads(state.get("distribution_json", "{}") or "{}") != distribution:
            raise GatewayError(
                E_REPLAY,
                "completed rule destination distribution differs during replay recovery",
                rule_id=rule_id,
                field="distribution",
                details={
                    "ledger": json.loads(state.get("distribution_json", "{}") or "{}"),
                    "receipts": distribution,
                },
            )
        for receipt in receipts:
            if receipt.get("receipt_version") != state.get("receipt_version"):
                raise GatewayError(
                    E_REPLAY,
                    "completed rule receipt version differs during replay recovery",
                    rule_id=rule_id,
                    field="receipt_version",
                    details={
                        "batch_index": receipt.get("batch_index"),
                        "ledger": state.get("receipt_version"),
                        "receipt": receipt.get("receipt_version"),
                    },
                )


def _pending_rule_ids(spec: Spec, states: dict[str, dict[str, Any]]) -> list[str]:
    pending = []
    for rule_id in topological_order(spec):
        state = states.get(rule_id)
        if state is None or state.get("status") != "complete":
            pending.append(rule_id)
    return pending


def _result_rule_summary(
    spec: Spec,
    ledger: Ledger,
    *,
    previous_states: dict[str, dict[str, Any]],
    pending: set[str],
) -> list[dict[str, Any]]:
    result = []
    final_states = ledger.get_rule_states()
    for rule_id in topological_order(spec):
        state = final_states.get(rule_id, {})
        before = previous_states.get(rule_id, {})
        result.append(
            {
                "rule_id": rule_id,
                "status": state.get("status"),
                "rule_version": state.get("rule_version"),
                "batch_count": state.get("batch_count", 0),
                "receipt_version": state.get("receipt_version"),
                "receipt_digest": state.get("receipt_digest"),
                "distribution": json.loads(state.get("distribution_json", "{}") or "{}"),
                "recomputed": rule_id in pending,
                "reused_receipt_bytes": bool(
                    rule_id not in pending
                    and before.get("receipt_digest")
                    and before.get("receipt_digest") == state.get("receipt_digest")
                ),
            }
        )
    return result


def run_session(
    spec: Spec,
    *,
    workdir: str | Path,
    client: BackendClient,
    logger: StructuredLogger,
    crash_after_batches: int | None = None,
) -> dict[str, Any]:
    """Run, update, or recover a session depending only on persisted commits."""
    workdir_path = Path(workdir)
    workdir_path.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(workdir_path)
    try:
        run = ledger.get_run()
        snapshot_info = inspect_snapshot(
            workdir_path, ledger, spec, repair_missing=True
        )
        recovered = False
        update_invalidation: dict[str, Any] | None = None
        generation = 0
        previous_states = ledger.get_rule_states()
        mode = "full"
        reused_existing = False
        replacement_rules: list[str] = []
        trace_before: Spec | None = None

        if run is None:
            invalidation = initial_invalidation(spec)
            run_id = ledger.create_run(spec, invalidation, mode="full")
            generation = 1
            mode = "full"
            update_invalidation = invalidation
            replacement_rules = list(invalidation["replacement_rules"])
            previous_states = {}
            logger.emit(
                "run_created",
                run_id=run_id,
                rule_count=len(spec.rules),
                request_count=spec.request_count,
                graph_digest=dependency_graph_digest(spec),
            )
        else:
            run_id = str(run["run_id"])
            if json.loads(run["target_spec_json"]) != json.loads(spec.canonical_declaration_json()):
                if run["status"] == "running":
                    raise GatewayError(
                        E_SNAPSHOT,
                        "an unfinished target declaration exists; resume it before loading a different declaration",
                        field="target_spec_json",
                        details={"run_id": run_id},
                    )
                before_spec = spec_declarations_from_payload(
                    json.loads(run["target_spec_json"]), requests=spec.requests
                )
                trace_before = before_spec
                invalidation = compute_invalidation(before_spec, spec)
                update_invalidation = invalidation
                replacement_rules = list(invalidation["replacement_rules"])
                generation = ledger.begin_update(spec, invalidation, base_spec=before_spec)
                mode = "update"
                logger.emit(
                    "run_updated",
                    run_id=run_id,
                    generation=generation,
                    replacement_rules=replacement_rules,
                    changed_backends=invalidation["changed_backends"],
                )
            else:
                if run["status"] == "complete":
                    snapshot_repaired = False
                    if snapshot_info.get("replacement_mode") == "regenerate_snapshot":
                        write_snapshot(workdir_path, ledger, spec)
                        snapshot_info = inspect_snapshot(workdir_path, ledger, spec)
                        snapshot_repaired = True
                        logger.emit("snapshot_regenerated", run_id=run_id)
                    if snapshot_info.get("same_declaration") is not True:
                        raise GatewayError(
                            E_SNAPSHOT,
                            "completed run and snapshot declaration state are inconsistent",
                            field="declaration_digest",
                        )
                    ledger.validate_complete_rule_receipts(spec)
                    report = {
                        "ok": True,
                        "run_id": run_id,
                        "mode": "unchanged",
                        "recovered": False,
                        "snapshot_repaired": snapshot_repaired,
                        "replacement": {
                            "mode": "none",
                            "rules": [],
                        },
                        "invalidation": None,
                        "rules": _result_rule_summary(
                            spec,
                            ledger,
                            previous_states=ledger.get_rule_states(),
                            pending=set(),
                        ),
                        "ledger": {
                            "path": str(ledger.path),
                            "content_digest": ledger._meta_value("content_digest"),
                            "generation": ledger.latest_generation(),
                        },
                    }
                    return report
                mode = "recover"
                generation = ledger.latest_generation(run_id)
                base_payload = json.loads(run["base_spec_json"]) if run.get("base_spec_json") else None
                before_spec = (
                    spec_declarations_from_payload(base_payload, requests=spec.requests)
                    if base_payload is not None
                    else None
                )
                trace_before = before_spec
                _verify_trace(
                    ledger,
                    before=before_spec,
                    target=spec,
                    generation=generation,
                )
                recovered = True
                logger.emit(
                    "run_recovery_started",
                    run_id=run_id,
                    generation=generation,
                    unknown_rules=sorted(
                        rule_id
                        for rule_id, state in previous_states.items()
                        if state.get("status") != "complete"
                    ),
                )

        if update_invalidation is None:
            base_payload = json.loads(run["base_spec_json"]) if run.get("base_spec_json") else None
            before_spec = (
                spec_declarations_from_payload(base_payload, requests=spec.requests)
                if base_payload is not None
                else None
            )
            update_invalidation = (
                initial_invalidation(spec) if before_spec is None else compute_invalidation(before_spec, spec)
            )
        if update_invalidation is not None and not recovered:
            # The transaction that created/updated the run already stored this
            # trace.  Verify it before any inference is attempted.
            _verify_trace(
                ledger,
                before=trace_before,
                target=spec,
                generation=generation,
            )

        if recovered:
            _validate_completed_replay_state(spec, ledger)
        states = ledger.get_rule_states()
        pending = _pending_rule_ids(spec, states)
        if not pending and run is not None and run.get("status") == "complete":
            reused_existing = True

        partial_expected: dict[str, dict[int, dict[str, Any]]] = {}
        for rule_id in pending:
            state = states.get(rule_id, {})
            if state.get("status") in {"in_progress", "unknown"}:
                partial_expected[rule_id] = ledger.expected_receipts_for_rule(rule_id)
        marked_unknown = ledger.mark_in_progress_rules_unknown(run_id)
        if marked_unknown:
            logger.emit("rules_marked_unknown", run_id=run_id, rule_ids=marked_unknown)

        # Rules that were already complete are left byte-for-byte untouched.
        ledger.validate_complete_rule_receipts(spec)
        plans = build_batch_plans(spec)
        trace_payload = ledger.invalidation_payload(generation=generation)
        committed_batches = 0
        pending_set = set(pending)

        for rule_id in pending:
            expected = partial_expected.get(rule_id, {})
            ledger.mark_rule_started(rule_id, run_id=run_id)
            logger.emit(
                "rule_recompute_started",
                run_id=run_id,
                rule_id=rule_id,
                expected_committed_batches=sorted(expected),
            )

            def on_batch(receipt: dict[str, Any], cursor: int) -> None:
                nonlocal committed_batches
                ledger.commit_batch(
                    rule_id,
                    receipt,
                    generation=generation,
                    trace_payload=trace_payload,
                    cursor=cursor,
                    run_id=run_id,
                )
                committed_batches += 1
                logger.emit(
                    "batch_committed",
                    run_id=run_id,
                    rule_id=rule_id,
                    batch_index=receipt["batch_index"],
                    cursor=cursor,
                    receipt_digest=receipt["receipt_digest"],
                    backend_param_digest=receipt["backend_param_digest"],
                )
                if crash_after_batches is not None and committed_batches >= crash_after_batches:
                    # Deliberately terminate without an exit report or snapshot.
                    os._exit(137)

            execution = execute_rule(
                spec,
                rule_id,
                plans[rule_id],
                client,
                on_batch=on_batch,
                expected_partial=expected,
            )
            ledger.complete_rule(
                execution,
                generation=generation,
                trace_payload=trace_payload,
                run_id=run_id,
            )
            logger.emit(
                "rule_recompute_completed",
                run_id=run_id,
                rule_id=rule_id,
                receipt_digest=execution.receipt_digest,
                batch_count=len(execution.receipts),
                distribution=execution.distribution,
            )

        ledger.finish_run(run_id)
        ledger.validate_complete_rule_receipts(spec)
        snapshot = write_snapshot(workdir_path, ledger, spec)
        final_states = ledger.get_rule_states()
        if any(state.get("status") != "complete" for state in final_states.values()):
            raise GatewayError(
                E_SNAPSHOT,
                "run finished with rules that are not complete",
                field="rule_state",
            )

        report = {
            "ok": True,
            "run_id": run_id,
            "mode": mode if not reused_existing else "recover",
            "recovered": recovered,
            "recovery": {
                "marked_unknown_rules": marked_unknown,
                "committed_partial_batches": {
                    rule_id: sorted(batches)
                    for rule_id, batches in sorted(partial_expected.items())
                },
                "requires_rerun": sorted(marked_unknown),
            },
            "replacement": {
                "mode": "none" if not replacement_rules else "rule_subset",
                "rules": sorted(replacement_rules),
            },
            "invalidation": {
                "changed_backends": update_invalidation.get("changed_backends", []),
                "affected_backends": update_invalidation.get("affected_backends", []),
                "affected_rules": [
                    step["rule_id"] for step in update_invalidation.get("invalidated_rules", [])
                ],
                "steps": _trace_steps_from_invalidation(update_invalidation),
                "unaffected_rules": update_invalidation.get("unaffected_rules", []),
                "removed_rules": update_invalidation.get("removed_rules", []),
                "added_rules": update_invalidation.get("added_rules", []),
            },
            "rules": _result_rule_summary(
                spec,
                ledger,
                previous_states=previous_states,
                pending=pending_set,
            ),
            "ledger": {
                "path": str(ledger.path),
                "content_digest": ledger._meta_value("content_digest"),
                "generation": ledger.latest_generation(),
                "snapshot_path": str(workdir_path / "snapshot.json"),
                "snapshot_full_receipt_digest": snapshot["full_receipt_digest"],
            },
        }
        logger.emit(
            "run_completed",
            run_id=run_id,
            ledger_digest=report["ledger"]["content_digest"],
            recomputed_rules=sorted(pending_set),
        )
        return report
    finally:
        ledger.close()
