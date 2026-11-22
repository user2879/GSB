"""High-level aggregate, resume, replace, and evolve orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import canonical_json, digest_value
from .catalog import load_catalog
from .engine import Execution, build_execution
from .errors import EBudget, EReplay
from .ledger import Ledger
from .planner import plan_slots


def _check_budget(budget: int) -> None:
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise EBudget(budget if isinstance(budget, int) else -1, 1)


@dataclass
class Prepared:
    catalog: Any
    execution: Execution
    effective_batch_size: int

    @property
    def config(self) -> dict[str, Any]:
        return {
            "catalog_digest": self.catalog.digest,
            "rows_per_page": self.catalog.rows_per_page,
            "group_key": self.catalog.group_key,
            "budget": self.execution.budget,
            "route": self.execution.route.selected_route,
            "batch_size": self.effective_batch_size,
        }


def prepare_execution(
    catalog_path: str | Path,
    budget: int,
    *,
    force_route: str | None = None,
    group_key_override: str | None = None,
    batch_size_override: int | None = None,
) -> Prepared:
    _check_budget(budget)
    catalog = load_catalog(catalog_path, group_key_override=group_key_override)
    execution = build_execution(
        catalog,
        budget,
        force_route=force_route,
        batch_size=batch_size_override,
    )
    effective_batch_size = execution.batches[0].cursor_end - execution.batches[0].cursor_start
    return Prepared(
        catalog=catalog,
        execution=execution,
        effective_batch_size=effective_batch_size,
    )


def _config_reasons(old_config: dict[str, Any], new_config: dict[str, Any]) -> list[str]:
    labels = {
        "catalog_digest": "page_catalog_content_digest_changed",
        "rows_per_page": "rows_per_page_changed",
        "group_key": "group_key_changed",
        "budget": "slot_budget_changed",
        "route": "selected_route_changed",
        "batch_size": "batch_size_changed",
    }
    return [labels[key] for key in sorted(old_config) if old_config.get(key) != new_config.get(key)]


def _old_catalog_raw(snapshot: dict[str, Any]) -> dict[str, Any]:
    return json.loads(snapshot["catalog_json"])


def _page_entries(raw_catalog: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(item["page_id"]): item for item in raw_catalog.get("pages", [])}


def _all_page_ids(old_raw: dict[str, Any], new_catalog: Any) -> set[int]:
    return set(_page_entries(old_raw)) | set(new_catalog.pages)


def _new_group_refs(catalog: Any) -> dict[int, set[int]]:
    refs: dict[int, set[int]] = {page_id: set() for page_id in catalog.pages}
    group_key = catalog.group_key
    for page_id, page in catalog.pages.items():
        refs[page_id] = {int(value) for value in page.columns[group_key].tolist()}
    return refs


def _plan_signature(scan_sequence: list[int], budget: int) -> list[tuple[int, bool, int, int | None, int | None]]:
    plan = plan_slots(scan_sequence, budget)
    return [
        (
            access.page_id,
            access.hit,
            access.slot,
            access.evicted_page_id,
            access.eviction_seq,
        )
        for access in plan.accesses
    ]


def compute_propagation(
    old_snapshot: dict[str, Any] | None,
    prepared: Prepared,
    *,
    old_page_refs: list[dict[str, Any]] | None,
    old_result: dict[str, Any] | None,
) -> dict[str, Any]:
    new_catalog = prepared.catalog
    new_result = prepared.execution.result
    new_refs = _new_group_refs(new_catalog)
    old_group_map = {
        int(group["group_key"]): group for group in (old_result or {}).get("groups", [])
    }
    new_group_map = {int(group["group_key"]): group for group in new_result.get("groups", [])}

    if old_snapshot is None:
        propagation_steps = [
            {
                "step": 1,
                "source": "initial_build",
                "changed": True,
                "affected_pages": sorted(new_catalog.pages),
                "affected_groups": sorted(new_group_map),
                "basis": "no prior ledger snapshot; build complete current view",
            }
        ]
        return {
            "replacement_required": True,
            "reasons": ["initial_build"],
            "propagation_steps": propagation_steps,
            "affected_pages": sorted(new_catalog.pages),
            "value_affected_pages": sorted(new_catalog.pages),
            "route_affected_pages": [],
            "affected_groups": sorted(new_group_map),
            "route_affected_groups": [],
            "added_groups": sorted(new_group_map),
            "removed_groups": [],
            "changed_groups": [],
            "unaffected_groups": [],
            "unaffected_group_count": 0,
            "unaffected_group_digests": [],
            "byte_unchanged_groups": [],
            "unchanged_groups": [],
            "unchanged_group_count": 0,
            "unaffected_groups_byte_identical": True,
            "old_group_digest": None,
            "new_group_digest": digest_value(new_group_map),
        }

    old_raw = _old_catalog_raw(old_snapshot)
    old_pages = _page_entries(old_raw)
    new_pages = _page_entries(new_catalog.raw)
    old_columns = old_raw.get("columns")
    new_columns = new_catalog.raw.get("columns")
    old_scan = list(old_raw.get("scan_sequence", []))
    new_scan = list(new_catalog.scan_sequence)
    old_rows = int(old_raw.get("rows_per_page", old_snapshot["rows_per_page"]))
    new_rows = new_catalog.rows_per_page
    old_group_key = old_snapshot["group_key"]
    new_group_key = new_catalog.group_key
    old_budget = int(old_snapshot["budget"])
    new_budget = prepared.execution.budget
    all_ids = _all_page_ids(old_raw, new_catalog)

    catalog_changed_pages: set[int] = set()
    for page_id in all_ids:
        old_entry = old_pages.get(page_id)
        new_entry = new_pages.get(page_id)
        if old_entry is None or new_entry is None:
            catalog_changed_pages.add(page_id)
            continue
        if old_entry.get("content_sha256") != new_entry.get("content_sha256"):
            catalog_changed_pages.add(page_id)
        if old_entry.get("rows") != new_entry.get("rows"):
            catalog_changed_pages.add(page_id)

    scan_changed_pages: set[int] = set()
    if old_scan != new_scan:
        max_length = max(len(old_scan), len(new_scan))
        for index in range(max_length):
            old_id = old_scan[index] if index < len(old_scan) else None
            new_id = new_scan[index] if index < len(new_scan) else None
            if old_id != new_id:
                if old_id is not None:
                    scan_changed_pages.add(int(old_id))
                if new_id is not None:
                    scan_changed_pages.add(int(new_id))

    route_affected_pages: set[int] = set()
    route_changed = old_budget != new_budget or old_snapshot["route"] != prepared.execution.route.selected_route
    if old_scan == new_scan:
        try:
            old_sig = _plan_signature(old_scan, old_budget)
            new_sig = _plan_signature(new_scan, new_budget)
        except Exception:
            old_sig = []
            new_sig = []
        for index in range(min(len(old_sig), len(new_sig))):
            if old_sig[index] != new_sig[index]:
                route_affected_pages.add(int(old_sig[index][0]))
                route_affected_pages.add(int(new_sig[index][0]))
        if len(old_sig) != len(new_sig):
            route_changed = True
            route_affected_pages |= {int(entry[0]) for entry in old_sig + new_sig}
    if old_snapshot["route"] != prepared.execution.route.selected_route:
        route_affected_pages |= {int(page_id) for page_id in new_scan}

    schema_changed = old_columns != new_columns
    rows_changed = old_rows != new_rows
    group_key_changed = old_group_key != new_group_key

    value_affected_pages: set[int] = set(catalog_changed_pages) | scan_changed_pages
    affected_pages: set[int] = set(value_affected_pages) | route_affected_pages
    if rows_changed or schema_changed or group_key_changed:
        affected_pages |= all_ids
        value_affected_pages |= all_ids

    old_refs_by_page: dict[int, set[int]] = {}
    for row in old_page_refs or []:
        old_refs_by_page.setdefault(int(row["page_id"]), set()).add(int(row["group_key"]))
    affected_groups: set[int] = set()
    if group_key_changed or rows_changed or schema_changed:
        affected_groups |= set(old_group_map) | set(new_group_map)
        affected_groups |= set().union(*old_refs_by_page.values()) if old_refs_by_page else set()
        affected_groups |= set().union(*new_refs.values()) if new_refs else set()
    else:
        for page_id in value_affected_pages:
            affected_groups |= old_refs_by_page.get(page_id, set())
            affected_groups |= new_refs.get(page_id, set())
    route_affected_groups: set[int] = set()
    for page_id in route_affected_pages:
        route_affected_groups |= old_refs_by_page.get(page_id, set())
        route_affected_groups |= new_refs.get(page_id, set())

    steps: list[dict[str, Any]] = []
    step = 1
    steps.append(
        {
            "step": step,
            "source": "page_catalog",
            "changed": bool(catalog_changed_pages),
            "affected_pages": sorted(catalog_changed_pages),
            "affected_groups": sorted(
                set().union(*(new_refs.get(page_id, set()) for page_id in catalog_changed_pages))
                if catalog_changed_pages
                else set()
            ),
            "basis": "page id add/remove, page row count, or page content digest changed",
        }
    )
    step += 1
    steps.append(
        {
            "step": step,
            "source": "rows_per_page",
            "changed": rows_changed,
            "affected_pages": sorted(all_ids) if rows_changed else [],
            "affected_groups": sorted(set(new_group_map)) if rows_changed else [],
            "basis": "fixed page row count changed; every page layout and group aggregate is invalidated",
        }
    )
    step += 1
    steps.append(
        {
            "step": step,
            "source": "group_key",
            "changed": group_key_changed,
            "affected_pages": sorted(all_ids) if group_key_changed else [],
            "affected_groups": sorted(set(old_group_map) | set(new_group_map)) if group_key_changed else [],
            "basis": "group key reference changed; group membership is recomputed for all pages",
        }
    )
    step += 1
    steps.append(
        {
            "step": step,
            "source": "slot_budget_route",
            "changed": route_changed,
            "affected_pages": sorted(route_affected_pages),
            "affected_groups": sorted(route_affected_groups),
            "affected_pages_role": "route_replan",
            "basis": "resident hit/miss/slot/victim plan or selected route differs under the new budget",
        }
    )
    step += 1
    steps.append(
        {
            "step": step,
            "source": "scan_sequence",
            "changed": bool(scan_changed_pages),
            "affected_pages": sorted(scan_changed_pages),
            "affected_groups": sorted(
                set().union(*(new_refs.get(page_id, set()) for page_id in scan_changed_pages))
                if scan_changed_pages
                else set()
            ),
            "basis": "scan access position or multiplicity changed",
        }
    )
    step += 1
    steps.append(
        {
            "step": step,
            "source": "columns",
            "changed": schema_changed,
            "affected_pages": sorted(all_ids) if schema_changed else [],
            "affected_groups": sorted(set(old_group_map) | set(new_group_map)) if schema_changed else [],
            "basis": "column name/type layout changed",
        }
    )

    common = set(old_group_map) & set(new_group_map)
    added = sorted(set(new_group_map) - set(old_group_map))
    removed = sorted(set(old_group_map) - set(new_group_map))
    changed = sorted(
        key
        for key in common
        if canonical_json(old_group_map[key]) != canonical_json(new_group_map[key])
    )
    unchanged = sorted(key for key in common if key not in set(changed))
    unaffected_groups = sorted(key for key in (set(old_group_map) | set(new_group_map)) if key not in affected_groups)
    unaffected_group_digests = [
        {
            "group_key": key,
            "old_digest": digest_value(old_group_map[key]) if key in old_group_map else None,
            "new_digest": digest_value(new_group_map[key]) if key in new_group_map else None,
        }
        for key in unaffected_groups
    ]
    for key in unaffected_groups:
        if key in old_group_map and key in new_group_map and canonical_json(old_group_map[key]) != canonical_json(new_group_map[key]):
            raise EReplay(
                "an unaffected group changed bytes",
                field=f"propagation.unaffected_group[{key}]",
                details={"group_key": key},
            )
    reasons: list[str] = []
    if old_snapshot["catalog_digest"] != new_catalog.digest:
        reasons.append("page_catalog_content_digest_changed")
    if rows_changed:
        reasons.append("rows_per_page_changed")
    if group_key_changed:
        reasons.append("group_key_changed")
    if old_budget != new_budget:
        reasons.append("slot_budget_changed")
    if old_snapshot["route"] != prepared.execution.route.selected_route:
        reasons.append("selected_route_changed")
    if int(old_snapshot["batch_size"]) != prepared.effective_batch_size:
        reasons.append("batch_size_changed")
    if not reasons and (catalog_changed_pages or scan_changed_pages or schema_changed):
        reasons.append("page_catalog_changed")

    return {
        "replacement_required": bool(reasons),
        "reasons": reasons,
        "propagation_steps": steps,
        "affected_pages": sorted(affected_pages),
        "value_affected_pages": sorted(value_affected_pages),
        "route_affected_pages": sorted(route_affected_pages),
        "affected_groups": sorted(affected_groups),
        "route_affected_groups": sorted(route_affected_groups),
        "added_groups": added,
        "removed_groups": removed,
        "changed_groups": changed,
        "unaffected_groups": unaffected_groups,
        "unaffected_group_count": len(unaffected_groups),
        "unaffected_group_digests": unaffected_group_digests,
        "byte_unchanged_groups": unchanged,
        "unchanged_group_keys": unchanged,
        "unchanged_group_count": len(unchanged),
        "unaffected_groups_byte_identical": True,
        "old_group_digest": digest_value(old_group_map),
        "new_group_digest": digest_value(new_group_map),
        "old_result_digest": old_snapshot.get("result_digest"),
        "new_result_digest": prepared.execution.result_digest,
    }


def _existing_snapshot_context(
    ledger: Ledger,
    snapshot: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    if snapshot is None:
        return [], None
    refs = ledger.page_group_refs(int(snapshot["snapshot_id"]))
    result = json.loads(snapshot["final_result_json"]) if snapshot.get("final_result_json") else None
    return refs, result


def _apply_new_snapshot(
    ledger: Ledger,
    prepared: Prepared,
    *,
    replacement_reason: str | None,
    replacement_diff: dict[str, Any] | None,
    crash_during_batch: int | None,
) -> dict[str, Any]:
    old = ledger.get_active_snapshot()
    old_version = int(old["plan_version"]) if old else None
    old_catalog_digest = old["catalog_digest"] if old else None
    old_result_digest = old.get("result_digest") if old else None
    snapshot_id, plan_version, _ = ledger.create_snapshot(
        prepared.execution,
        catalog_path=str(prepared.catalog.path),
        replacement_reason=replacement_reason,
    )
    if replacement_diff is not None:
        ledger.record_replacement(
            from_plan_version=old_version,
            to_plan_version=plan_version,
            reason=replacement_reason or "snapshot_replaced",
            old_catalog_digest=old_catalog_digest,
            new_catalog_digest=prepared.catalog.digest,
            old_result_digest=old_result_digest,
            new_result_digest=prepared.execution.result_digest,
            diff=replacement_diff,
        )
    committed = ledger.persist_batches(
        prepared.execution,
        snapshot_id,
        start_batch_index=1,
        crash_during_batch=crash_during_batch,
    )
    return {
        "action": "replaced" if old is not None else "created",
        "replaced": old is not None,
        "previous_plan_version": old_version,
        "plan_version": plan_version,
        "snapshot_id": snapshot_id,
        "committed_batches": committed,
    }


def aggregate_database(
    catalog_path: str | Path,
    db_path: str | Path,
    budget: int,
    *,
    force_route: str | None = None,
    group_key_override: str | None = None,
    batch_size_override: int | None = None,
    crash_during_batch: int | None = None,
) -> dict[str, Any]:
    prepared = prepare_execution(
        catalog_path,
        budget,
        force_route=force_route,
        group_key_override=group_key_override,
        batch_size_override=batch_size_override,
    )
    with Ledger(db_path) as ledger:
        old = ledger.get_active_snapshot()
        unknown = ledger.unknown_attempts()
        replacement_diff = None
        if old is not None and ledger.snapshot_config(old) != prepared.config:
            refs, result = _existing_snapshot_context(ledger, old)
            replacement_diff = compute_propagation(
                old,
                prepared,
                old_page_refs=refs,
                old_result=result,
            )
            reasons = _config_reasons(ledger.snapshot_config(old), prepared.config)
            replacement_reason = ",".join(reasons) or "configuration_changed"
            applied = _apply_new_snapshot(
                ledger,
                prepared,
                replacement_reason=replacement_reason,
                replacement_diff=replacement_diff,
                crash_during_batch=crash_during_batch,
            )
        elif old is None:
            replacement_diff = compute_propagation(
                None,
                prepared,
                old_page_refs=[],
                old_result=None,
            )
            applied = _apply_new_snapshot(
                ledger,
                prepared,
                replacement_reason="initial_build",
                replacement_diff=None,
                crash_during_batch=crash_during_batch,
            )
        else:
            # Recompute and verify all committed batches before resuming.
            from .replay import compare_execution_with_ledger

            refreshed = ledger.get_active_snapshot()
            assert refreshed is not None
            committed_count = ledger.committed_batch_count(int(refreshed["snapshot_id"]))
            if int(refreshed["cursor"]) > 0:
                comparison = compare_execution_with_ledger(ledger, refreshed, prepared.execution)
            else:
                comparison = {"resume_cursor": 0, "complete": False}
            if refreshed["status"] == "COMPLETE":
                applied = {
                    "action": "resumed_complete",
                    "replaced": False,
                    "previous_plan_version": None,
                    "plan_version": int(refreshed["plan_version"]),
                    "snapshot_id": int(refreshed["snapshot_id"]),
                    "committed_batches": [],
                    "verified": comparison,
                }
            else:
                committed = ledger.persist_batches(
                    prepared.execution,
                    int(refreshed["snapshot_id"]),
                    start_batch_index=committed_count + 1,
                    crash_during_batch=crash_during_batch,
                )
                applied = {
                    "action": "resumed",
                    "replaced": False,
                    "previous_plan_version": None,
                    "plan_version": int(refreshed["plan_version"]),
                    "snapshot_id": int(refreshed["snapshot_id"]),
                    "committed_batches": committed,
                    "verified": comparison,
                }
        final_snapshot = ledger.get_active_snapshot()
        assert final_snapshot is not None
        ledger_summary = ledger.ledger_summary()

    response = {
        "ok": True,
        "command": "aggregate",
        "catalog": {
            "path": str(prepared.catalog.path),
            "digest": prepared.catalog.digest,
            "dataset_id": prepared.catalog.dataset_id,
        },
        "configuration": prepared.config,
        "route_decision": prepared.execution.route.payload(),
        "snapshot_action": applied,
        "unknown_attempts": unknown,
        "execution": prepared.execution.summary_payload(),
        "replacement_summary": replacement_diff,
        "result": prepared.execution.result,
        "ledger": ledger_summary,
    }
    return response


def recover_database(db_path: str | Path) -> dict[str, Any]:
    with Ledger(db_path) as ledger:
        snapshot = ledger.get_active_snapshot()
        unknown = ledger.unknown_attempts()
        return {
            "ok": True,
            "command": "recover",
            "forced_exit_report": None,
            "forced_exit_report_available": False,
            "recovery_rule": "only completely committed batches are trusted",
            "unknown_batches_require_rerun": bool(unknown),
            "unknown_attempts": unknown,
            "snapshot": None
            if snapshot is None
            else {
                "snapshot_id": snapshot["snapshot_id"],
                "plan_version": snapshot["plan_version"],
                "status": snapshot["status"],
                "cursor": snapshot["cursor"],
                "route": snapshot["route"],
            },
            "ledger": ledger.ledger_summary(),
        }


def evolve_database(
    db_path: str | Path,
    catalog_path: str | Path,
    budget: int,
    *,
    force_route: str | None = None,
    group_key_override: str | None = None,
    batch_size_override: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    prepared = prepare_execution(
        catalog_path,
        budget,
        force_route=force_route,
        group_key_override=group_key_override,
        batch_size_override=batch_size_override,
    )
    with Ledger(db_path) as ledger:
        old = ledger.get_active_snapshot()
        refs, old_result = _existing_snapshot_context(ledger, old)
        propagation = compute_propagation(
            old,
            prepared,
            old_page_refs=refs,
            old_result=old_result,
        )
        if not propagation["replacement_required"]:
            return {
                "ok": True,
                "command": "evolve",
                "dry_run": dry_run,
                "snapshot_action": {
                    "action": "no_op",
                    "replaced": False,
                    "package": "configuration is unchanged",
                    "plan_version": int(old["plan_version"]) if old else None,
                },
                "propagation": propagation,
                "ledger": ledger.ledger_summary(),
            }
        if dry_run:
            return {
                "ok": True,
                "command": "evolve",
                "dry_run": True,
                "snapshot_action": {
                    "action": "would_replace",
                    "replaced": False,
                    "previous_plan_version": int(old["plan_version"]) if old else None,
                    "predicted_plan_version": (
                        int(old["plan_version"]) + 1
                        if old
                        else ledger.ledger_summary()["plan_version_counter"] + 1
                    ),
                },
                "propagation": propagation,
                "ledger": ledger.ledger_summary(),
            }
        reason = ",".join(propagation["reasons"]) or "evolved_configuration"
        applied = _apply_new_snapshot(
            ledger,
            prepared,
            replacement_reason=reason,
            replacement_diff=propagation,
            crash_during_batch=None,
        )
        return {
            "ok": True,
            "command": "evolve",
            "dry_run": False,
            "snapshot_action": applied,
            "propagation": propagation,
            "route_decision": prepared.execution.route.payload(),
            "execution": prepared.execution.summary_payload(),
            "ledger": ledger.ledger_summary(),
        }
