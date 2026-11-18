from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .engine import evaluate_session, state_entry_count
from .loading import make_failure


def group_events(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        grouped.setdefault(event["session_id"], []).append(event)
    for session_events in grouped.values():
        session_events.sort(key=lambda event: int(event["turn"]))
    return grouped


def _error_session(
    session_id: str,
    machine: dict[str, Any],
    config: dict[str, Any],
    error: Exception,
) -> dict[str, Any]:
    return {
        "session_id": session_id,
        "status": "error",
        "current_state": machine["initial_state"],
        "slots": {},
        "transition_trace": [],
        "rejected_transitions": [],
        "slot_changes": [],
        "failures": [
            make_failure(
                config,
                "internal_error",
                "session",
                str(error),
                session_id=session_id,
                exception_type=type(error).__name__,
            )
        ],
        "pending_cursor": 1,
        "state_entries": 1,
        "state_cache_key": "",
        "transition_trace_cache_key": "",
        "checkpoints": [],
        "committed": False,
        "reuse": {
            "checkpoint_turn": None,
            "replayed_events": 0,
            "dirty_slots": [],
            "pollution_reasons": ["session_exception"],
        },
    }


def run_sessions(
    *,
    events: list[dict[str, Any]],
    machine: dict[str, Any],
    config: dict[str, Any],
    machine_hash: str,
    previous_sessions: dict[str, dict[str, Any]] | None = None,
    pollution: dict[str, Any] | None = None,
    quota_overrides: dict[str, int] | None = None,
    serial: bool = False,
) -> dict[str, Any]:
    previous_sessions = previous_sessions or {}
    quota_overrides = quota_overrides or {}
    grouped = group_events(events)
    all_session_ids = sorted(set(grouped) | set(previous_sessions))
    pollution = pollution or {}
    affected_sessions = set(pollution.get("affected_sessions", []))
    session_turns = pollution.get("session_turns", {})
    dirty_slots_by_session = pollution.get("dirty_slots", {})
    pollution_reasons = pollution.get("reasons", [])

    tasks: list[tuple[str, int | None, list[str], list[str]]] = []
    results: dict[str, dict[str, Any]] = {}
    reused_sessions: list[str] = []
    for session_id in all_session_ids:
        session_events = grouped.get(session_id, [])
        previous = previous_sessions.get(session_id)
        reusable_previous = previous if previous is not None and previous.get("committed", True) else None
        if reusable_previous is not None and session_id not in affected_sessions:
            reused = copy.deepcopy(reusable_previous)
            previous_slot_names = set(reused.get("slots", {}))
            reused["reuse"] = {
                "checkpoint_turn": reused["checkpoints"][-1]["turn"] if reused.get("checkpoints") else None,
                "replayed_events": 0,
                "dirty_slots": [],
                "reused_slots": sorted(previous_slot_names),
                "pollution_reasons": ["unaffected_session_reused"],
                "session_reused": True,
            }
            reused["committed"] = True
            results[session_id] = reused
            reused_sessions.append(session_id)
            continue
        start_turn = session_turns.get(session_id)
        dirty_slots = list(dirty_slots_by_session.get(session_id, []))
        reasons = list(pollution_reasons)
        if reusable_previous is None:
            start_turn = None
            reasons = reasons + ["new_session_or_full_build"]
        previous = reusable_previous
        tasks.append((session_id, start_turn, dirty_slots, reasons))

    configured_workers = max(1, int(config.get("worker_count", 1)))
    effective_workers = 1 if serial or not tasks else min(configured_workers, len(tasks))

    def invoke(task: tuple[str, int | None, list[str], list[str]]) -> dict[str, Any]:
        session_id, start_turn, dirty_slots, reasons = task
        session_events = grouped.get(session_id, [])
        previous = previous_sessions.get(session_id)
        try:
            return evaluate_session(
                session_id=session_id,
                events=session_events,
                machine=machine,
                config=config,
                machine_hash=machine_hash,
                start_turn=start_turn,
                previous_session=previous,
                dirty_slots=dirty_slots,
                pollution_reasons=reasons,
                quota_override=quota_overrides.get(session_id),
            )
        except Exception as exc:
            return _error_session(session_id, machine, config, exc)

    if effective_workers == 1:
        for task in tasks:
            result = invoke(task)
            results[result["session_id"]] = result
    else:
        with ThreadPoolExecutor(max_workers=effective_workers, thread_name_prefix="session-state") as pool:
            futures = {pool.submit(invoke, task): task[0] for task in tasks}
            for future in as_completed(futures):
                result = future.result()
                results[result["session_id"]] = result

    budget_limit = int(config["global_state_budget"])
    remaining = budget_limit
    committed_sessions: dict[str, dict[str, Any]] = {}
    budget_exhausted: list[str] = []
    budget_allocations: dict[str, dict[str, Any]] = {}
    for session_id in sorted(results):
        session = results[session_id]
        previous = previous_sessions.get(session_id)
        if previous is not None and session.get("reuse", {}).get("replayed_events", 0) > 0:
            dirty = set(session.get("reuse", {}).get("dirty_slots", []))
            reused_slots = sorted(set(previous.get("slots", {})) - dirty)
            session.setdefault("reuse", {})["reused_slots"] = reused_slots
            session["reuse"]["reused_slot_count"] = len(reused_slots)
        entries = state_entry_count(session.get("slots", {}))
        session["computed_state_entries"] = entries
        if entries <= remaining:
            session["state_entries"] = entries
            session["committed"] = True
            results[session_id] = session
            committed_sessions[session_id] = session
            remaining -= entries
            budget_allocations[session_id] = {
                "requested_entries": entries,
                "granted_entries": entries,
                "committed": True,
            }
        else:
            previous_cursor = previous.get("pending_cursor") if previous else None
            first_turn = 1
            if grouped.get(session_id):
                first_turn = int(grouped[session_id][0]["turn"])
            session["status"] = "budget_exhausted"
            session["committed"] = False
            session["state_entries"] = 0
            session["pending_cursor"] = previous_cursor or first_turn
            session["failures"] = list(session.get("failures", [])) + [
                make_failure(
                    config,
                    "state_budget_exhausted",
                    "scheduling",
                    f"global state budget exhausted: need {entries}, remaining {remaining}",
                    session_id=session_id,
                    turn=session["pending_cursor"],
                    exception_type="StateBudgetExhausted",
                )
            ]
            budget_exhausted.append(session_id)
            budget_allocations[session_id] = {
                "requested_entries": entries,
                "granted_entries": 0,
                "committed": False,
                "pending_cursor": session["pending_cursor"],
            }
            results[session_id] = session

    global_state_table = {
        session_id: {
            "current_state": session["current_state"],
            "slots": session["slots"],
            "state_cache_key": session["state_cache_key"],
            "state_entries": session["state_entries"],
            "status": session["status"],
        }
        for session_id, session in sorted(committed_sessions.items())
    }
    consumed = budget_limit - remaining
    all_failures: list[dict[str, Any]] = []
    for session_id in sorted(results):
        all_failures.extend(results[session_id].get("failures", []))

    return {
        "sessions": {session_id: results[session_id] for session_id in sorted(results)},
        "global_state_table": global_state_table,
        "budget": {
            "limit": budget_limit,
            "consumed": consumed,
            "remaining": remaining,
            "exhausted_sessions": budget_exhausted,
            "committed_sessions": sorted(committed_sessions),
            "allocations": {session_id: budget_allocations[session_id] for session_id in sorted(budget_allocations)},
        },
        "execution": {
            "worker_count_configured": configured_workers,
            "worker_count_effective": effective_workers,
            "session_count": len(results),
            "reused_sessions": reused_sessions,
            "reused_session_count": len(reused_sessions),
            "recomputed_session_count": len(tasks),
            "session_order": all_session_ids,
            "replayed_events": sum(
                int(results[session_id].get("reuse", {}).get("replayed_events", 0)) for session_id in results
            ),
        },
        "failures": all_failures,
    }
