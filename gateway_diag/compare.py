from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from .backend_client import BackendClient
from .errors import E_DIVERGE, GatewayError
from .graph import topological_order
from .ledger import Ledger
from .logging_utils import StructuredLogger
from .models import Spec
from .routing import distribution_from_receipts
from .snapshot import inspect_snapshot
from .util import atomic_write_json
from .workflow import run_session

COMPARISON_FIELDS = [
    "destination",
    "backend_id",
    "backend_param_digest",
    "request_ids",
    "output_digest",
    "receipt_version",
    "receipt_digest",
]


def _load_ledger(directory: Path) -> tuple[Ledger, dict[str, Any]]:
    ledger = Ledger(directory, read_only=True)
    run = ledger.get_run()
    if run is None:
        raise GatewayError(E_DIVERGE, "comparison ledger has no active run", field="run_id")
    return ledger, run


def _compare_unaffected(
    baseline: Ledger,
    incremental: Ledger,
    unaffected_rules: list[str],
) -> dict[str, Any]:
    checked = 0
    for rule_id in sorted(unaffected_rules):
        before = baseline.get_rule_receipt_records(rule_id)
        after = incremental.get_rule_receipt_records(rule_id)
        if len(before) != len(after):
            raise GatewayError(
                E_DIVERGE,
                "an unaffected rule changed its batch count",
                rule_id=rule_id,
                field="batch_count",
                details={"baseline": len(before), "incremental": len(after)},
            )
        for index, (old, new) in enumerate(zip(before, after)):
            if old["receipt_json"] != new["receipt_json"]:
                raise GatewayError(
                    E_DIVERGE,
                    "an unaffected rule receipt changed byte-for-byte",
                    rule_id=rule_id,
                    field="receipt_json",
                    details={"batch_index": index},
                )
            if old["receipt_version"] != new["receipt_version"]:
                raise GatewayError(
                    E_DIVERGE,
                    "an unaffected rule receipt version changed",
                    rule_id=rule_id,
                    field="receipt_version",
                    details={
                        "batch_index": index,
                        "baseline": old["receipt_version"],
                        "incremental": new["receipt_version"],
                    },
                )
            checked += 1
    return {"rules": sorted(unaffected_rules), "receipts_checked": checked, "identical": True}


def _compare_rule_results(
    spec: Spec,
    incremental: dict[str, list[dict[str, Any]]],
    full: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    allowed: list[dict[str, Any]] = []
    checked_batches = 0
    for rule_id in topological_order(spec):
        inc_receipts = incremental.get(rule_id, [])
        full_receipts = full.get(rule_id, [])
        inc_distribution = distribution_from_receipts(inc_receipts)
        full_distribution = distribution_from_receipts(full_receipts)
        if inc_distribution != full_distribution:
            raise GatewayError(
                E_DIVERGE,
                "incremental and full destination distributions differ",
                rule_id=rule_id,
                field="distribution",
                details={
                    "batch_index": 0,
                    "incremental": inc_distribution,
                    "full": full_distribution,
                },
            )
        by_index_inc = {int(receipt["batch_index"]): receipt for receipt in inc_receipts}
        by_index_full = {int(receipt["batch_index"]): receipt for receipt in full_receipts}
        if sorted(by_index_inc) != sorted(by_index_full):
            first = min(set(by_index_inc) ^ set(by_index_full), default=0)
            raise GatewayError(
                E_DIVERGE,
                "incremental and full receipts have different batch indexes",
                rule_id=rule_id,
                field="batch_index",
                details={
                    "batch_index": first,
                    "incremental": sorted(by_index_inc),
                    "full": sorted(by_index_full),
                },
            )
        for batch_index in sorted(by_index_inc):
            inc_receipt = by_index_inc[batch_index]
            full_receipt = by_index_full[batch_index]
            backend_digest_differs = (
                inc_receipt.get("backend_param_digest")
                != full_receipt.get("backend_param_digest")
            )
            fields = COMPARISON_FIELDS
            for field in fields:
                old = inc_receipt.get(field)
                new = full_receipt.get(field)
                if old == new:
                    continue
                if field == "backend_param_digest" and backend_digest_differs:
                    allowed.append(
                        {
                            "source": "backend_parameter_digest_change",
                            "rule_id": rule_id,
                            "batch_index": batch_index,
                            "field": field,
                            "incremental": old,
                            "full": new,
                        }
                    )
                    continue
                if field in {"output_digest", "receipt_digest"} and backend_digest_differs:
                    allowed.append(
                        {
                            "source": "backend_parameter_digest_change",
                            "rule_id": rule_id,
                            "batch_index": batch_index,
                            "field": field,
                            "incremental": old,
                            "full": new,
                        }
                    )
                    continue
                raise GatewayError(
                    E_DIVERGE,
                    "incremental and full recomputation differ",
                    rule_id=rule_id,
                    field=field,
                    details={
                        "batch_index": batch_index,
                        "incremental": old,
                        "full": new,
                    },
                )
            checked_batches += 1
    return (
        {
            "rules_checked": len(spec.rules),
            "batches_checked": checked_batches,
            "result": "identical",
        },
        allowed,
    )


def _backend_change_sources(before: Spec, after: Spec, affected_rules: list[str]) -> list[dict[str, Any]]:
    sources = []
    for backend_id in sorted(set(before.backends) | set(after.backends)):
        old = before.backends.get(backend_id)
        new = after.backends.get(backend_id)
        old_digest = old.param_digest if old else None
        new_digest = new.param_digest if new else None
        if old_digest == new_digest:
            continue
        sources.append(
            {
                "source": "backend_parameter_change",
                "backend_id": backend_id,
                "old_param_digest": old_digest,
                "new_param_digest": new_digest,
                "rules": sorted(affected_rules),
                "classification": "declared_parameter_change",
            }
        )
    return sources


def validate_incremental_vs_full(
    *,
    before: Spec,
    after: Spec,
    workdir: str | Path,
    client: BackendClient,
    logger: StructuredLogger,
) -> dict[str, Any]:
    """Run baseline -> incremental update and a separate after-config rebuild."""
    root = Path(workdir)
    stamp = time.monotonic_ns()
    session_root = root / f"validation-{stamp}"
    baseline_dir = session_root / "baseline"
    incremental_dir = session_root / "incremental"
    full_dir = session_root / "full"
    session_root.mkdir(parents=True, exist_ok=False)

    logger.emit("validation_started", session_dir=str(session_root))
    baseline_report = run_session(
        before,
        workdir=baseline_dir,
        client=client,
        logger=logger,
    )
    shutil.copytree(baseline_dir, incremental_dir)
    incremental_report = run_session(
        after,
        workdir=incremental_dir,
        client=client,
        logger=logger,
    )
    full_report = run_session(
        after,
        workdir=full_dir,
        client=client,
        logger=logger,
    )

    baseline, before_run = _load_ledger(baseline_dir)
    incremental, incr_run = _load_ledger(incremental_dir)
    full, full_run = _load_ledger(full_dir)
    try:
        inspect_snapshot(baseline_dir, baseline, before)
        inspect_snapshot(incremental_dir, incremental, after)
        inspect_snapshot(full_dir, full, after)
        if incr_run["declaration_digest"] != after.declaration_digest:
            raise GatewayError(
                E_DIVERGE,
                "incremental ledger target declaration differs from the requested update",
                field="declaration_digest",
            )
        if full_run["declaration_digest"] != after.declaration_digest:
            raise GatewayError(
                E_DIVERGE,
                "full ledger declaration differs from the requested update",
                field="declaration_digest",
            )

        invalidation = incremental_report.get("invalidation") or {}
        unaffected = list(invalidation.get("unaffected_rules", []))
        unaffected_result = _compare_unaffected(baseline, incremental, unaffected)
        incremental_receipts = incremental.all_rule_receipts()
        full_receipts = full.all_rule_receipts()
        comparison_result, allowed = _compare_rule_results(
            after, incremental_receipts, full_receipts
        )
        affected_rules = list(invalidation.get("affected_rules", []))
        allowed.extend(_backend_change_sources(before, after, affected_rules))
        # Stable de-duplication of identical legal difference classifications.
        dedup: dict[str, dict[str, Any]] = {}
        for item in allowed:
            key = json.dumps(item, sort_keys=True, separators=(",", ":"))
            dedup[key] = item
        allowed = [dedup[key] for key in sorted(dedup)]

        baseline_snap = json.loads((baseline_dir / "snapshot.json").read_text(encoding="utf-8"))
        incremental_snap = json.loads((incremental_dir / "snapshot.json").read_text(encoding="utf-8"))
        full_snap = json.loads((full_dir / "snapshot.json").read_text(encoding="utf-8"))
        if incremental_snap["full_receipt_digest"] != full_snap["full_receipt_digest"]:
            raise GatewayError(
                E_DIVERGE,
                "incremental and full full-receipt digests differ",
                field="full_receipt_digest",
                details={
                    "incremental": incremental_snap["full_receipt_digest"],
                    "full": full_snap["full_receipt_digest"],
                },
            )

        report = {
            "ok": True,
            "command": "validate",
            "session_dir": str(session_root),
            "decision_rule": {
                "comparison_order": "sorted(rule_id), then ascending batch_index",
                "fields": COMPARISON_FIELDS,
                "backend_parameter_change_policy": (
                    "A changed backend parameter digest attributable to the declared update is "
                    "reported as an allowed difference source; request sets, destinations and "
                    "batch indexes must still match. No comparison depends on execution order."
                ),
            },
            "before_declaration_digest": before.declaration_digest,
            "after_declaration_digest": after.declaration_digest,
            "invalidation": invalidation,
            "unaffected_receipts": unaffected_result,
            "comparison": comparison_result,
            "allowed_differences": allowed,
            "baseline_report": baseline_report,
            "incremental_report": incremental_report,
            "full_report": full_report,
            "snapshots": {
                "baseline_full_receipt_digest": baseline_snap["full_receipt_digest"],
                "incremental_full_receipt_digest": incremental_snap["full_receipt_digest"],
                "full_full_receipt_digest": full_snap["full_receipt_digest"],
            },
            "ledgers": {
                "baseline": str(baseline_dir),
                "incremental": str(incremental_dir),
                "full": str(full_dir),
            },
        }
        atomic_write_json(session_root / "validation-summary.json", report)
        logger.emit(
            "validation_completed",
            session_dir=str(session_root),
            unaffected_rules=unaffected,
            allowed_differences=len(allowed),
        )
        return report
    finally:
        baseline.close()
        incremental.close()
        full.close()
