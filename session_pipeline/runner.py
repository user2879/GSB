from __future__ import annotations

import copy
import os
import shutil
from pathlib import Path
from typing import Any, Callable

from .edits import apply_edit_batch, materialize_final_edits
from .engine import compare_state_projection
from .injections import apply_batch_injections
from .loading import (
    config_fingerprint,
    load_scenario_base,
    machine_fingerprint,
    make_failure,
    normalize_machine,
    validate_events,
)
from .scheduler import run_sessions
from .snapshot import SnapshotStore
from .util import atomic_write_json, stable_hash


def _merge_pollution(
    pollution: dict[str, Any] | None,
    change: dict[str, Any],
) -> dict[str, Any]:
    merged = copy.deepcopy(pollution) if pollution else {
        "affected_sessions": [],
        "session_turns": {},
        "dirty_slots": {},
        "reasons": [],
    }
    affected = set(merged.get("affected_sessions", []))
    affected.update(change.get("affected_sessions", []))
    merged["affected_sessions"] = sorted(affected)
    merged["session_turns"] = dict(merged.get("session_turns", {}))
    for session_id, turn in change.get("session_turns", {}).items():
        previous = merged["session_turns"].get(session_id)
        merged["session_turns"][session_id] = turn if previous is None else min(previous, turn)
    merged["dirty_slots"] = copy.deepcopy(merged.get("dirty_slots", {}))
    for session_id, slots in change.get("dirty_slots", {}).items():
        current = set(merged["dirty_slots"].get(session_id, []))
        current.update(slots)
        merged["dirty_slots"][session_id] = sorted(current)
    merged["reasons"] = list(dict.fromkeys(list(merged.get("reasons", [])) + list(change.get("reasons", []))))
    return merged


def _add_injection_pollution(
    pollution: dict[str, Any],
    hits: list[dict[str, Any]],
    quota_overrides: dict[str, int],
) -> dict[str, Any]:
    updated = copy.deepcopy(pollution)
    affected = set(updated.get("affected_sessions", []))
    for hit in hits:
        session_id = hit.get("session_id")
        if not isinstance(session_id, str):
            continue
        affected.add(session_id)
        turn = hit.get("turn") if isinstance(hit.get("turn"), int) else 1
        if hit.get("type") == "quota_overflow" or session_id in quota_overrides:
            turn = 1
        previous = updated["session_turns"].get(session_id)
        updated["session_turns"][session_id] = turn if previous is None else min(previous, turn)
        applied = hit.get("applied", {})
        slot = applied.get("slot") if isinstance(applied, dict) else None
        if isinstance(slot, str):
            slots = set(updated["dirty_slots"].get(session_id, []))
            slots.add(slot)
            updated["dirty_slots"][session_id] = sorted(slots)
        updated["reasons"].append(f"fault_injection:{hit.get('injection_id')}")
    updated["affected_sessions"] = sorted(affected)
    updated["reasons"] = list(dict.fromkeys(updated["reasons"]))
    return updated


def _prepare_validated(
    *,
    events: list[dict[str, Any]],
    machine: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], str]:
    normalized_machine, machine_failures = normalize_machine(
        machine, config, apply_config_overrides=False
    )
    valid_events, event_failures = validate_events(events, normalized_machine, config)
    return (
        normalized_machine,
        valid_events,
        machine_failures + event_failures,
        machine_fingerprint(normalized_machine),
    )


def _session_report(state_result: dict[str, Any]) -> dict[str, Any]:
    sessions: dict[str, Any] = {}
    transition_count = 0
    rejected_count = 0
    for session_id, session in sorted(state_result["sessions"].items()):
        transitions = session.get("transition_trace", [])
        rejected = session.get("rejected_transitions", [])
        transition_count += len(transitions)
        rejected_count += len(rejected)
        sessions[session_id] = {
            "session_id": session_id,
            "status": session.get("status"),
            "committed": session.get("committed", True),
            "current_state": session.get("current_state"),
            "slots": session.get("slots", {}),
            "transition_trace": transitions,
            "rejected_transitions": rejected,
            "slot_changes": session.get("slot_changes", []),
            "pending_cursor": session.get("pending_cursor"),
            "state_entries": session.get("state_entries"),
            "computed_state_entries": session.get("computed_state_entries"),
            "quota": session.get("quota"),
            "state_cache_key": session.get("state_cache_key"),
            "transition_trace_cache_key": session.get("transition_trace_cache_key"),
            "failures": session.get("failures", []),
            "reuse": session.get("reuse", {}),
        }
    contamination = state_result.get("contamination", {})
    dirty_slot_count = sum(len(values) for values in contamination.get("dirty_slots", {}).values())
    return {
        "sessions": sessions,
        "global_state_table": state_result.get("global_state_table", {}),
        "summary": {
            "session_count": len(sessions),
            "committed_session_count": len(state_result.get("global_state_table", {})),
            "transition_count": transition_count,
            "rejected_transition_count": rejected_count,
            "dirty_slot_count": dirty_slot_count,
            "affected_session_count": len(contamination.get("affected_sessions", [])),
            "reused_session_count": state_result.get("execution", {}).get("reused_session_count", 0),
            "replayed_event_count": state_result.get("execution", {}).get("replayed_events", 0),
        },
    }


def build_diagnostic_report(
    *,
    mode: str,
    batch_number: int,
    state_result: dict[str, Any],
    pollution: dict[str, Any] | None,
    load_failures: list[dict[str, Any]],
    edit_failures: list[dict[str, Any]],
    injection_hits: list[dict[str, Any]],
    injection_misses: list[dict[str, Any]],
    recovery: dict[str, Any],
    snapshot: dict[str, Any] | None,
    config: dict[str, Any],
    machine: dict[str, Any],
    run_status: str = "complete",
) -> dict[str, Any]:
    session_report = _session_report(state_result)
    failures = list(load_failures) + list(edit_failures) + list(state_result.get("failures", []))
    contamination = copy.deepcopy(pollution or {
        "affected_sessions": [],
        "session_turns": {},
        "dirty_slots": {},
        "reasons": [],
    })
    contamination["dirty_slot_count"] = sum(
        len(values) for values in contamination.get("dirty_slots", {}).values()
    )
    report = {
        "schema_version": "1.0",
        "mode": mode,
        "run_status": run_status,
        "batch_number": batch_number,
        "pipeline": {
            "config_hash": config_fingerprint(config),
            "machine_hash": machine_fingerprint(machine),
            "machine_id": machine.get("machine_id"),
            "confidence_decimals": config.get("confidence_decimals"),
            "worker_count_configured": config.get("worker_count"),
            "max_events_per_batch": config.get("max_events_per_batch"),
            "snapshot_interval_batches": config.get("snapshot_interval_batches"),
            "global_state_budget": config.get("global_state_budget"),
            "session_state_quota": config.get("session_state_quota"),
        },
        "sessions": session_report["sessions"],
        "global_state_table": session_report["global_state_table"],
        "summary": session_report["summary"],
        "contamination": contamination,
        "reuse": {
            "reused_sessions": state_result.get("execution", {}).get("reused_sessions", []),
            "reused_session_count": state_result.get("execution", {}).get(
                "reused_session_count", 0
            ),
            "recomputed_session_count": state_result.get("execution", {}).get(
                "recomputed_session_count", 0
            ),
            "replayed_events": state_result.get("execution", {}).get("replayed_events", 0),
            "dirty_slots": contamination.get("dirty_slots", {}),
            "reused_slot_count": sum(
                len(session.get("reuse", {}).get("reused_slots", []))
                for session in session_report["sessions"].values()
            ),
        },
        "execution": state_result.get("execution", {}),
        "budget": state_result.get("budget", {}),
        "recovery": recovery,
        "snapshot": snapshot or {},
        "injection_hits": injection_hits,
        "injection_misses": injection_misses,
        "failures": failures,
        "failure_count": len(failures),
        "state_projection_hash": stable_hash(
            compare_state_projection({"sessions": session_report["sessions"]})
        ),
    }
    return report


def run_full_build(
    *,
    scenario_dir: Path,
    report_path: Path,
    serial: bool,
    enable_injections: bool = False,
    max_edit_batches: int | None = None,
    snapshot_workdir: Path | None = None,
) -> dict[str, Any]:
    base = load_scenario_base(scenario_dir)
    config = base["config"]
    effective_edit_plan = base["edit_plan"]
    if max_edit_batches is not None:
        effective_edit_plan = copy.deepcopy(base["edit_plan"])
        effective_edit_plan["batches"] = effective_edit_plan.get("batches", [])[:max_edit_batches]
    raw_events, edited_machine, edit_batches, edit_failures = materialize_final_edits(
        events=base["raw_events"],
        machine=base["machine"],
        edit_plan=effective_edit_plan,
        config=config,
    )
    machine, valid_events, load_failures, machine_hash = _prepare_validated(
        events=raw_events,
        machine=edited_machine,
        config=config,
    )
    injection_hits: list[dict[str, Any]] = []
    injection_misses: list[dict[str, Any]] = []
    quota_overrides: dict[str, int] = {}
    if enable_injections:
        # Full mode applies all injections as a deterministic single preparation pass.
        for batch in effective_edit_plan.get("batches", []):
            batch_number = int(batch.get("batch_id", 1))
            valid_events, batch_quota, hits, misses = apply_batch_injections(
                batch_number=batch_number,
                events=valid_events,
                injection_config=base["injection_config"],
                enabled=True,
            )
            quota_overrides.update(batch_quota)
            injection_hits.extend(hits)
            injection_misses.extend(misses)
        valid_events, injected_event_failures = validate_events(valid_events, machine, config)
        load_failures.extend(injected_event_failures)

    state_result = run_sessions(
        events=valid_events,
        machine=machine,
        config=config,
        machine_hash=machine_hash,
        previous_sessions=None,
        pollution={
            "affected_sessions": sorted({event["session_id"] for event in valid_events}),
            "session_turns": {
                session_id: min(
                    int(event["turn"]) for event in valid_events if event["session_id"] == session_id
                )
                for session_id in sorted({event["session_id"] for event in valid_events})
            },
            "dirty_slots": {},
            "reasons": ["full_build"],
        },
        quota_overrides=quota_overrides,
        serial=serial,
    )
    state_result["contamination"] = {
        "affected_sessions": sorted({event["session_id"] for event in valid_events}),
        "session_turns": {},
        "dirty_slots": {},
        "reasons": ["full_build"],
    }
    preliminary_report = build_diagnostic_report(
        mode="full_serial" if serial else "full_parallel",
        batch_number=len(edit_batches),
        state_result=state_result,
        pollution=state_result["contamination"],
        load_failures=load_failures + list(base["failures"]),
        edit_failures=edit_failures,
        injection_hits=injection_hits,
        injection_misses=injection_misses,
        recovery={"resumed": False, "recovery_point": None},
        snapshot={"status": "pending_atomic_commit", "completed_batch": len(edit_batches)},
        config=config,
        machine=machine,
    )
    snapshot_store = SnapshotStore(snapshot_workdir or (report_path.parent / "full-snapshot-work"))
    snapshot_info = snapshot_store.commit(
        events=raw_events,
        machine=machine,
        config=config,
        sessions=state_result["sessions"],
        global_state_table=state_result["global_state_table"],
        budget=state_result["budget"],
        execution=state_result["execution"],
        failures=state_result["failures"],
        run_metadata={
            "injection_hits": injection_hits,
            "injection_misses": injection_misses,
            "injection_failures": [],
            "applied_batch_ids": [int(batch.get("batch_id", index)) for index, batch in enumerate(effective_edit_plan.get("batches", []), start=1)],
            "batch_history": edit_batches,
            "load_failures": load_failures + list(base["failures"]),
            "edit_failures": edit_failures,
        },
        diagnostic=preliminary_report,
        completed_batch=len(edit_batches),
        config_hash=config_fingerprint(config),
        machine_hash=machine_fingerprint(machine),
    )
    report = build_diagnostic_report(
        mode="full_serial" if serial else "full_parallel",
        batch_number=len(edit_batches),
        state_result=state_result,
        pollution=state_result["contamination"],
        load_failures=load_failures + list(base["failures"]),
        edit_failures=edit_failures,
        injection_hits=injection_hits,
        injection_misses=injection_misses,
        recovery={"resumed": False, "recovery_point": None},
        snapshot=snapshot_info,
        config=config,
        machine=machine,
    )
    report["edit_batches"] = edit_batches
    atomic_write_json(report_path, report)
    return report


def _snapshot_summary(report: dict[str, Any]) -> dict[str, Any]:
    return report


def run_incremental(
    *,
    scenario_dir: Path,
    workdir: Path,
    report_path: Path,
    reset: bool = False,
    enable_injections: bool = False,
    stop_requested: Callable[[], bool] | None = None,
    crash_after_batch: int | None = None,
    max_batch: int | None = None,
) -> tuple[dict[str, Any], int]:
    if reset and workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    store = SnapshotStore(workdir)
    latest = store.latest()
    base = load_scenario_base(scenario_dir)
    scenario_config_hash = base["config_hash"]

    if latest is not None:
        raw_events = latest["events"]
        machine = latest["machine"]
        config = latest["config"]
        previous_sessions = latest["sessions"]
        completed_batch = int(latest["manifest"].get("completed_batch", 0))
        run_metadata = latest["run_metadata"]
        recovery = {
            "resumed": True,
            "recovery_point": {
                "global_version": latest["version"],
                "snapshot_version": latest["manifest"].get("snapshot_version"),
                "completed_batch": completed_batch,
                "path": latest["path"],
            },
        }
        last_snapshot = {
            "global_version": latest["version"],
            "snapshot_version": latest["manifest"].get("snapshot_version"),
            "path": latest["path"],
            "completed_batch": completed_batch,
        }
    else:
        raw_events = copy.deepcopy(base["raw_events"])
        machine = copy.deepcopy(base["machine"])
        config = base["config"]
        previous_sessions = {}
        completed_batch = 0
        run_metadata = {
            "injection_hits": [],
            "injection_misses": [],
            "injection_failures": [],
            "applied_batch_ids": [],
            "batch_history": [],
            "load_failures": list(base["failures"]),
            "edit_failures": [],
        }
        recovery = {"resumed": False, "recovery_point": None}
        last_snapshot = None

    if config_fingerprint(config) != scenario_config_hash:
        mismatch_message = (
            "scenario configuration differs from the resumed snapshot; snapshot configuration retained"
            if latest is not None
            else "configuration hash mismatch during initialization"
        )
        run_metadata.setdefault("load_failures", []).append(
            make_failure(config, "input_invalid", "loading", mismatch_message)
        )

    batches = base["edit_plan"].get("batches", [])
    interrupted = False
    state_result: dict[str, Any] | None = None
    pollution: dict[str, Any] | None = None
    load_failures: list[dict[str, Any]] = []
    edit_failures: list[dict[str, Any]] = list(run_metadata.get("edit_failures", []))

    def evaluate_current(
        *,
        pollution_value: dict[str, Any] | None,
        batch_number: int,
        load_failure_values: list[dict[str, Any]],
        edit_failure_values: list[dict[str, Any]],
        batch_hits: list[dict[str, Any]],
        batch_misses: list[dict[str, Any]],
        quota_overrides: dict[str, int],
        mode: str,
        run_status: str,
    ) -> dict[str, Any]:
        nonlocal machine
        normalized_machine, valid_events, validation_failures, machine_hash = _prepare_validated(
            events=raw_events,
            machine=machine,
            config=config,
        )
        machine = normalized_machine
        result = run_sessions(
            events=valid_events,
            machine=machine,
            config=config,
            machine_hash=machine_hash,
            previous_sessions=previous_sessions,
            pollution=pollution_value,
            quota_overrides=quota_overrides,
            serial=False,
        )
        result["contamination"] = copy.deepcopy(pollution_value or {})
        all_load_failures = (
            list(run_metadata.get("load_failures", []))
            + list(load_failure_values)
            + validation_failures
        )
        report = build_diagnostic_report(
            mode=mode,
            batch_number=batch_number,
            state_result=result,
            pollution=pollution_value,
            load_failures=all_load_failures,
            edit_failures=edit_failure_values,
            injection_hits=run_metadata.get("injection_hits", []) + batch_hits,
            injection_misses=run_metadata.get("injection_misses", []) + batch_misses,
            recovery=recovery,
            snapshot=last_snapshot,
            config=config,
            machine=machine,
            run_status=run_status,
        )
        report["source_config_hash"] = scenario_config_hash
        return report, result

    if latest is None:
        initial_report, initial_result = evaluate_current(
            pollution_value=None,
            batch_number=0,
            load_failure_values=[],
            edit_failure_values=[],
            batch_hits=[],
            batch_misses=[],
            quota_overrides={},
            mode="incremental_initial",
            run_status="complete",
        )
        state_result = initial_result
        initial_snapshot = store.commit(
            events=raw_events,
            machine=machine,
            config=config,
            sessions=initial_result["sessions"],
            global_state_table=initial_result["global_state_table"],
            budget=initial_result["budget"],
            execution=initial_result["execution"],
            failures=initial_result["failures"],
            run_metadata=run_metadata,
            diagnostic=_snapshot_summary(initial_report),
            completed_batch=0,
            config_hash=config_fingerprint(config),
            machine_hash=machine_fingerprint(machine),
        )
        last_snapshot = initial_snapshot
        previous_sessions = initial_result["sessions"]
        if crash_after_batch == 0:
            os._exit(137)

    for batch_index, batch in enumerate(batches, start=1):
        if not isinstance(batch, dict):
            edit_failures.append(
                make_failure(config, "input_invalid", "editing", f"batch #{batch_index} is not an object")
            )
            continue
        batch_number = int(batch.get("batch_id", batch_index))
        if max_batch is not None and batch_number > max_batch:
            break
        if batch_number <= completed_batch:
            continue
        if stop_requested is not None and stop_requested():
            interrupted = True
            break

        next_events, next_machine, change, batch_edit_failures = apply_edit_batch(
            events=raw_events,
            machine=machine,
            batch=batch,
            config=config,
            batch_number=batch_number,
        )
        raw_events = next_events
        machine = next_machine
        pollution = _merge_pollution(None, change)
        batch_hits: list[dict[str, Any]] = []
        batch_misses: list[dict[str, Any]] = []
        quota_overrides: dict[str, int] = {}
        if enable_injections:
            injected_events, quota_overrides, batch_hits, batch_misses = apply_batch_injections(
                batch_number=batch_number,
                events=raw_events,
                injection_config=base["injection_config"],
                enabled=True,
            )
            raw_events_for_validation = injected_events
            pollution = _add_injection_pollution(pollution, batch_hits, quota_overrides)
        else:
            raw_events_for_validation = raw_events

        normalized_machine, valid_events, validation_failures, machine_hash = _prepare_validated(
            events=raw_events_for_validation,
            machine=machine,
            config=config,
        )
        machine = normalized_machine
        edit_failures.extend(batch_edit_failures)
        result = run_sessions(
            events=valid_events,
            machine=machine,
            config=config,
            machine_hash=machine_hash,
            previous_sessions=previous_sessions,
            pollution=pollution,
            quota_overrides=quota_overrides,
            serial=False,
        )
        result["contamination"] = copy.deepcopy(pollution)
        state_result = result

        cumulative_hits = list(run_metadata.get("injection_hits", [])) + batch_hits
        cumulative_misses = list(run_metadata.get("injection_misses", [])) + batch_misses
        cumulative_load_failures = list(run_metadata.get("load_failures", [])) + validation_failures
        report = build_diagnostic_report(
            mode="incremental",
            batch_number=batch_number,
            state_result=result,
            pollution=pollution,
            load_failures=cumulative_load_failures,
            edit_failures=edit_failures,
            injection_hits=cumulative_hits,
            injection_misses=cumulative_misses,
            recovery=recovery,
            snapshot=last_snapshot,
            config=config,
            machine=machine,
            run_status="complete",
        )
        report["source_config_hash"] = scenario_config_hash
        report["edit_batches"] = list(run_metadata.get("batch_history", [])) + [change]

        should_stop = stop_requested is not None and stop_requested()
        interval = int(config["snapshot_interval_batches"])
        is_last = batch_index == len(batches)
        is_boundary = max_batch is not None and batch_number >= max_batch
        run_metadata = {
            "injection_hits": cumulative_hits,
            "injection_misses": cumulative_misses,
            "injection_failures": list(run_metadata.get("injection_failures", [])),
            "applied_batch_ids": list(run_metadata.get("applied_batch_ids", [])) + [batch_number],
            "batch_history": list(run_metadata.get("batch_history", [])) + [change],
            "load_failures": cumulative_load_failures,
            "edit_failures": edit_failures,
        }
        if batch_number % interval == 0 or should_stop or is_last or is_boundary:
            last_snapshot = store.commit(
                events=raw_events,
                machine=machine,
                config=config,
                sessions=result["sessions"],
                global_state_table=result["global_state_table"],
                budget=result["budget"],
                execution=result["execution"],
                failures=result["failures"],
                run_metadata=run_metadata,
                diagnostic=_snapshot_summary(report),
                completed_batch=batch_number,
                config_hash=config_fingerprint(config),
                machine_hash=machine_fingerprint(machine),
            )
            report["snapshot"] = last_snapshot
            previous_sessions = result["sessions"]
            completed_batch = batch_number
            if crash_after_batch == batch_number:
                os._exit(137)
        else:
            previous_sessions = result["sessions"]
        if should_stop:
            interrupted = True
            break

    if state_result is None:
        if latest is not None:
            state_result = {
                "sessions": latest["sessions"],
                "global_state_table": latest["global_state_table"],
                "budget": latest["budget"],
                "execution": latest["execution"],
                "failures": latest["failures"],
            }
        else:
            state_result, _ = evaluate_current(
                pollution_value=None,
                batch_number=0,
                load_failure_values=[],
                edit_failure_values=[],
                batch_hits=[],
                batch_misses=[],
                quota_overrides={},
                mode="incremental_initial",
                run_status="complete",
            )

    final_pollution = copy.deepcopy(pollution or state_result.get("contamination") or {
        "affected_sessions": [],
        "session_turns": {},
        "dirty_slots": {},
        "reasons": [],
    })
    state_result["contamination"] = copy.deepcopy(final_pollution)
    final_report = build_diagnostic_report(
        mode="incremental",
        batch_number=completed_batch,
        state_result=state_result,
        pollution=final_pollution,
        load_failures=list(run_metadata.get("load_failures", [])) + load_failures,
        edit_failures=edit_failures,
        injection_hits=run_metadata.get("injection_hits", []),
        injection_misses=run_metadata.get("injection_misses", []),
        recovery=recovery,
        snapshot=last_snapshot,
        config=config,
        machine=machine,
        run_status="interrupted" if interrupted else "complete",
    )
    final_report["source_config_hash"] = scenario_config_hash
    final_report["edit_batches"] = run_metadata.get("batch_history", [])
    atomic_write_json(report_path, final_report)
    return final_report, 0 if interrupted else 0


def projection_from_report(report: dict[str, Any]) -> dict[str, Any]:
    return compare_state_projection({"sessions": report.get("sessions", {})})
