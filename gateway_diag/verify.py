from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .backend_client import BackendClient
from .errors import E_VERIFY, GatewayError
from .graph import (
    compute_invalidation,
    dependency_graph_digest,
    initial_invalidation,
    topological_order,
)
from .ledger import Ledger
from .loaders import spec_declarations_from_payload
from .logging_utils import StructuredLogger
from .models import Request, Rule, Spec
from .routing import Router, distribution_from_receipts
from .snapshot import inspect_snapshot
from .util import canonical_json, digest_value

VERIFY_FIELDS = [
    "schema",
    "rule_id",
    "rule_version",
    "receipt_version",
    "batch_index",
    "destination",
    "backend_id",
    "backend_param_digest",
    "request_ids",
    "output_digest",
    "receipt_digest",
]


def _independent_expected_receipts(
    spec: Spec,
    rule: Rule,
    client: BackendClient,
) -> list[dict[str, Any]]:
    router = Router(spec)
    matching: list[Request] = []
    for request in sorted(spec.requests, key=lambda item: (item.arrival_seq, item.request_id)):
        matched_rule = router.matching_rule(request)
        if matched_rule.id == rule.id:
            matching.append(request)
    groups: dict[tuple[str, str], list[Request]] = defaultdict(list)
    for request in matching:
        decision = router.decide(rule.id, request.client_key)
        groups[(decision.destination, decision.backend_id)].append(request)
    batch_index = 0
    result: list[dict[str, Any]] = []
    for destination, backend_id in sorted(
        groups, key=lambda item: (0 if item[0] == "stable" else 1, item[1])
    ):
        backend = spec.backends[backend_id]
        requests = sorted(groups[(destination, backend_id)], key=lambda item: (item.arrival_seq, item.request_id))
        for start in range(0, len(requests), rule.batch_size):
            chunk = requests[start : start + rule.batch_size]
            response = client.infer_batch(backend, chunk)
            body: dict[str, Any] = {
                "schema": "route-receipt.v1",
                "rule_id": rule.id,
                "rule_version": rule.version,
                "receipt_version": rule.version,
                "batch_index": batch_index,
                "destination": destination,
                "backend_id": backend_id,
                "backend_param_digest": backend.param_digest,
                "request_ids": sorted(request.request_id for request in chunk),
                "output_digest": response["output_digest"],
            }
            body["receipt_digest"] = digest_value(body)
            result.append(body)
            batch_index += 1
    return result


def _verify_receipt_internal(receipt: dict[str, Any], rule_id: str, batch_index: int) -> None:
    if not isinstance(receipt, dict):
        raise GatewayError(E_VERIFY, "persisted receipt is not an object", rule_id=rule_id, field="receipt_json")
    claimed = receipt.get("receipt_digest")
    material = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    actual = digest_value(material)
    if claimed != actual:
        raise GatewayError(
            E_VERIFY,
            "persisted receipt self-digest is invalid",
            rule_id=rule_id,
            field="receipt_digest",
            details={"batch_index": batch_index, "claimed": claimed, "actual": actual},
        )


def _compare_rule_receipts(
    rule_id: str,
    persisted: list[dict[str, Any]],
    expected: list[dict[str, Any]],
) -> int:
    persisted_by_index = {int(item["batch_index"]): item for item in persisted}
    expected_by_index = {int(item["batch_index"]): item for item in expected}
    if sorted(persisted_by_index) != sorted(expected_by_index):
        first = min(set(persisted_by_index) ^ set(expected_by_index), default=0)
        raise GatewayError(
            E_VERIFY,
            "persisted receipt batch indexes differ from independent recomputation",
            rule_id=rule_id,
            field="batch_index",
            details={
                "batch_index": first,
                "persisted": sorted(persisted_by_index),
                "expected": sorted(expected_by_index),
            },
        )
    for batch_index in sorted(expected_by_index):
        actual = persisted_by_index[batch_index]
        wanted = expected_by_index[batch_index]
        for field in VERIFY_FIELDS:
            if actual.get(field) != wanted.get(field):
                raise GatewayError(
                    E_VERIFY,
                    "persisted receipt differs from independent recomputation",
                    rule_id=rule_id,
                    field=field,
                    details={
                        "batch_index": batch_index,
                        "persisted": actual.get(field),
                        "expected": wanted.get(field),
                    },
                )
    return len(expected_by_index)


def _verify_trace(ledger: Ledger, spec: Spec, run: dict[str, Any]) -> dict[str, Any]:
    base_payload = json.loads(run["base_spec_json"]) if run.get("base_spec_json") else None
    before = (
        spec_declarations_from_payload(base_payload, requests=spec.requests)
        if base_payload is not None
        else None
    )
    expected = initial_invalidation(spec) if before is None else compute_invalidation(before, spec)
    expected_steps = [
        {
            "step": int(step["step"]),
            "rule_id": step["rule_id"],
            "code": step.get("code", "E_STALE"),
            "basis": step.get("basis", {}),
        }
        for step in expected["invalidated_rules"]
    ]
    stored = ledger.invalidation_payload()
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
        for index in range(max(len(expected_steps), len(stored_steps))):
            actual = stored_steps[index] if index < len(stored_steps) else None
            wanted = expected_steps[index] if index < len(expected_steps) else None
            if actual != wanted:
                raise GatewayError(
                    E_VERIFY,
                    "persisted invalidation trace differs from independent derivation",
                    rule_id=(wanted or actual or {}).get("rule_id"),
                    field="invalidation_trace",
                    details={"step": index + 1, "persisted": actual, "expected": wanted},
                )
    return {
        "steps_checked": len(stored_steps),
        "generation": ledger.latest_generation(),
        "result": "identical",
    }


def verify_ledger(
    spec: Spec,
    *,
    workdir: str | Path,
    client: BackendClient,
    logger: StructuredLogger,
) -> dict[str, Any]:
    root = Path(workdir)
    ledger = Ledger(root, read_only=True)
    try:
        snapshot_info = inspect_snapshot(root, ledger, spec)
        run = ledger.get_run()
        if run is None:
            raise GatewayError(E_VERIFY, "ledger has no active run", field="run_id")
        if run["status"] != "complete":
            raise GatewayError(
                E_VERIFY,
                "ledger run is not complete",
                field="run_status",
                details={"actual": run["status"]},
            )
        if run["declaration_digest"] != spec.declaration_digest:
            raise GatewayError(
                E_VERIFY,
                "ledger declaration digest differs from the supplied declarations",
                field="declaration_digest",
                details={"ledger": run["declaration_digest"], "supplied": spec.declaration_digest},
            )
        if run["request_digest"] != spec.request_digest:
            raise GatewayError(
                E_VERIFY,
                "ledger request digest differs from the supplied request stream",
                field="request_digest",
            )
        graph_digest = dependency_graph_digest(spec)
        if run["graph_digest"] != graph_digest:
            raise GatewayError(
                E_VERIFY,
                "ledger dependency graph digest differs from independent derivation",
                field="graph_digest",
                details={"ledger": run["graph_digest"], "expected": graph_digest},
            )
        states = ledger.get_rule_states()
        verified_rules = []
        checked_batches = 0
        for rule_id in topological_order(spec):
            state = states.get(rule_id)
            if state is None or state.get("status") != "complete":
                raise GatewayError(
                    E_VERIFY,
                    "rule has no complete state in the ledger",
                    rule_id=rule_id,
                    field="status",
                    details={"actual": None if state is None else state.get("status")},
                )
            persisted = ledger.get_rule_receipts(rule_id)
            for receipt in persisted:
                _verify_receipt_internal(receipt, rule_id, int(receipt.get("batch_index", -1)))
            expected = _independent_expected_receipts(spec, spec.rules[rule_id], client)
            checked_batches += _compare_rule_receipts(rule_id, persisted, expected)
            persisted_distribution = distribution_from_receipts(persisted)
            expected_distribution = distribution_from_receipts(expected)
            if persisted_distribution != expected_distribution:
                raise GatewayError(
                    E_VERIFY,
                    "persisted destination distribution differs from independent recomputation",
                    rule_id=rule_id,
                    field="distribution",
                    details={"persisted": persisted_distribution, "expected": expected_distribution},
                )
            if json.loads(state.get("distribution_json", "{}") or "{}") != expected_distribution:
                raise GatewayError(
                    E_VERIFY,
                    "persisted rule-state distribution differs from independent recomputation",
                    rule_id=rule_id,
                    field="rule_state.distribution",
                )
            if int(state.get("rule_version", -1)) != spec.rules[rule_id].version:
                raise GatewayError(
                    E_VERIFY,
                    "persisted rule version differs from the declaration",
                    rule_id=rule_id,
                    field="rule_version",
                )
            expected_set_digest = digest_value(
                [receipt["receipt_digest"] for receipt in expected]
            )
            if state.get("receipt_digest") != expected_set_digest:
                raise GatewayError(
                    E_VERIFY,
                    "persisted rule receipt-set digest differs from independent recomputation",
                    rule_id=rule_id,
                    field="receipt_digest",
                )
            verified_rules.append(
                {
                    "rule_id": rule_id,
                    "batches": len(expected),
                    "distribution": expected_distribution,
                }
            )
        expected_snapshot = ledger.snapshot_payload(spec)
        if snapshot_info.get("snapshot_full_receipt_digest") != expected_snapshot["full_receipt_digest"]:
            raise GatewayError(
                E_VERIFY,
                "snapshot full receipt digest differs from independently verified receipts",
                field="full_receipt_digest",
            )
        trace = _verify_trace(ledger, spec, run)
        report = {
            "ok": True,
            "command": "verify",
            "workdir": str(root),
            "run_id": run["run_id"],
            "declaration_digest": spec.declaration_digest,
            "graph_digest": graph_digest,
            "ledger_content_digest": ledger._meta_value("content_digest"),
            "snapshot": snapshot_info,
            "trace": trace,
            "rules_verified": verified_rules,
            "batches_checked": checked_batches,
            "result": "verified",
        }
        logger.emit(
            "verification_completed",
            run_id=run["run_id"],
            rules=len(verified_rules),
            batches=checked_batches,
            ledger_digest=report["ledger_content_digest"],
        )
        return report
    finally:
        ledger.close()
