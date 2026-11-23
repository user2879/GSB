from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import E_SPEC, E_SNAPSHOT, GatewayError
from .ledger import Ledger
from .loaders import spec_declarations_from_payload
from .models import Spec
from .util import atomic_write_json, canonical_json

SNAPSHOT_FILENAME = "snapshot.json"


def snapshot_path(workdir: str | Path) -> Path:
    return Path(workdir) / SNAPSHOT_FILENAME


def write_snapshot(workdir: str | Path, ledger: Ledger, spec: Spec) -> dict[str, Any]:
    payload = ledger.snapshot_payload(spec)
    atomic_write_json(snapshot_path(workdir), payload)
    return payload


def load_snapshot(workdir: str | Path) -> dict[str, Any] | None:
    path = snapshot_path(workdir)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise GatewayError(
            E_SNAPSHOT,
            "session snapshot is missing or malformed",
            field="snapshot",
            entry=str(path),
            details={"reason": str(exc)},
        )
    if not isinstance(payload, dict) or payload.get("schema") != "session-snapshot.v1":
        raise GatewayError(
            E_SNAPSHOT,
            "session snapshot schema is unsupported",
            field="snapshot.schema",
            entry=str(path),
        )
    return payload


def _all_rules_complete(ledger: Ledger, spec: Spec) -> bool:
    states = ledger.get_rule_states()
    return set(states) == set(spec.rules) and all(
        state.get("status") == "complete" for state in states.values()
    )


def inspect_snapshot(
    workdir: str | Path,
    ledger: Ledger,
    current_spec: Spec,
    *,
    repair_missing: bool = False,
) -> dict[str, Any]:
    snapshot = load_snapshot(workdir)
    run = ledger.get_run()
    if run is None:
        if snapshot is not None:
            raise GatewayError(
                E_SNAPSHOT,
                "snapshot exists without an active ledger run",
                field="active_run_id",
            )
        return {"present": False, "stale": False, "replacement_mode": "none", "replacement_rules": []}

    run_status = str(run["status"])
    if snapshot is None:
        if run_status == "complete":
            if repair_missing and _all_rules_complete(ledger, current_spec):
                return {
                    "present": False,
                    "stale": True,
                    "same_declaration": None,
                    "replacement_mode": "regenerate_snapshot",
                    "replacement_rules": [],
                }
            raise GatewayError(
                E_SNAPSHOT,
                "completed ledger has no session snapshot",
                field="snapshot",
                entry=str(snapshot_path(workdir)),
            )
        return {
            "present": False,
            "stale": True,
            "replacement_mode": "recover_ledger",
            "replacement_rules": [],
        }

    if snapshot.get("run_id") != run["run_id"]:
        raise GatewayError(
            E_SNAPSHOT,
            "snapshot run id does not match the ledger",
            field="snapshot.run_id",
            details={"snapshot": snapshot.get("run_id"), "ledger": run["run_id"]},
        )
    if snapshot.get("request_digest") != current_spec.request_digest:
        raise GatewayError(
            E_SPEC,
            "request stream digest differs from the persisted session",
            field="request_digest",
            details={
                "snapshot": snapshot.get("request_digest"),
                "current": current_spec.request_digest,
            },
        )

    if run_status == "complete":
        stored_digest = ledger._meta_value("content_digest")
        if snapshot.get("ledger_content_digest") != stored_digest:
            if repair_missing and _all_rules_complete(ledger, current_spec):
                # The ledger committed completion, but the process stopped
                # between the final transaction and atomic snapshot replace.
                return {
                    "present": True,
                    "stale": True,
                    "same_declaration": None,
                    "replacement_mode": "regenerate_snapshot",
                    "replacement_rules": [],
                    "snapshot_declaration_digest": snapshot.get("declaration_digest"),
                    "snapshot_full_receipt_digest": snapshot.get("full_receipt_digest"),
                    "snapshot_ledger_content_digest": snapshot.get("ledger_content_digest"),
                    "ledger_content_digest": stored_digest,
                }
            raise GatewayError(
                E_SNAPSHOT,
                "snapshot ledger digest does not match the ledger",
                field="snapshot.ledger_content_digest",
                details={
                    "snapshot": snapshot.get("ledger_content_digest"),
                    "ledger": stored_digest,
                },
            )
        snapshot_spec = spec_declarations_from_payload(snapshot.get("spec"))
        expected = ledger.snapshot_payload(snapshot_spec)
        if snapshot.get("full_receipt_digest") != expected["full_receipt_digest"]:
            raise GatewayError(
                E_SNAPSHOT,
                "snapshot full receipt digest does not match persisted receipts",
                field="snapshot.full_receipt_digest",
                details={
                    "snapshot": snapshot.get("full_receipt_digest"),
                    "ledger": expected["full_receipt_digest"],
                },
            )
        if canonical_json(snapshot.get("completed_rules")) != canonical_json(expected["completed_rules"]):
            raise GatewayError(
                E_SNAPSHOT,
                "snapshot completed rule set does not match persisted receipts",
                field="snapshot.completed_rules",
            )

    snapshot_versions = snapshot.get("rule_versions", {})
    current_versions = current_spec.rule_versions
    changed_rules = sorted(
        rule_id
        for rule_id in set(snapshot_versions) | set(current_versions)
        if snapshot_versions.get(rule_id) != current_versions.get(rule_id)
    )
    snapshot_backends = snapshot.get("backend_versions", {})
    current_backends = current_spec.backend_versions
    changed_backends = sorted(
        backend_id
        for backend_id in set(snapshot_backends) | set(current_backends)
        if snapshot_backends.get(backend_id) != current_backends.get(backend_id)
    )
    replacements = sorted(set(changed_rules) | set(changed_backends))
    same_declaration = snapshot.get("declaration_digest") == current_spec.declaration_digest
    return {
        "present": True,
        "stale": run_status != "complete",
        "same_declaration": same_declaration,
        "changed_rule_versions": changed_rules,
        "changed_backend_versions": changed_backends,
        "declaration_change_candidates": replacements,
        "replacement_mode": "none" if same_declaration else "rule_subset",
        "replacement_rules": [] if same_declaration else replacements,
        "snapshot_declaration_digest": snapshot.get("declaration_digest"),
        "snapshot_full_receipt_digest": snapshot.get("full_receipt_digest"),
    }
