"""Diagnostic report assembly and failure-list ordering.

Reports are built from the same deterministic primitives as the pipeline, so a
full replay and an incremental run over identical final inputs produce
identical JSON (the snapshot/ledger bookkeeping fields aside, which are kept
under separate keys).
"""

import json

import numpy as np

from .hashing import digest

FAILURE_ORDER = (
    "E_STREAM_INVALID",
    "E_FSM_INVALID",
    "E_CONFIG_INVALID",
    "E_SLOT_INVALID",
    "E_TRANSITION_REJECTED",
    "E_QUOTA_EXCEEDED",
    "E_BUDGET_PENDING",
    "E_SNAPSHOT_STALE",
)
_CODE_RANK = {code: idx for idx, code in enumerate(FAILURE_ORDER)}


def sort_failures(failures):
    """Stable, business-code-defined failure list ordering."""
    return sorted(
        failures,
        key=lambda f: (
            _CODE_RANK.get(f["code"], len(FAILURE_ORDER)),
            str(f.get("session_id", "")),
            f.get("turn") is None,
            f.get("turn") if f.get("turn") is not None else -1,
            str(f.get("stage", "")),
            str(f.get("message", "")),
        ),
    )


def build_report(results, load_failures, stale_failure, scheduler, fsm, config,
                 materialised, injection_config, applied_ids, completed_batches,
                 mode):
    sessions = {}
    all_failures = list(load_failures)
    if stale_failure is not None:
        all_failures.append(stale_failure)

    for sid in sorted(results):
        result = results[sid]
        slot_view = {
            name: record.to_dict()
            for name, record in sorted(result.slots.items())
        }
        sessions[sid] = {
            "status": result.status,
            "current_state": result.final_state,
            "slots": slot_view,
            "transition_count": len(result.transition_trace),
            "transition_trace": result.transition_trace,
            "rejected": result.rejected,
            "reused_turns": list(result.reused_turns),
            "replayed_turns": list(result.processed_turns),
            "consumed_entries": result.consumed_entries,
        }
        all_failures.extend(result.failures)

    requested_ids = [item["injection_id"]
                     for item in injection_config.get("injections", [])]
    report = {
        "report_version": 1,
        "mode": mode,
        "fsm_version": fsm.version,
        "fsm_fingerprint": fsm.fingerprint,
        "config_fingerprint": config.fingerprint,
        "completed_batches": completed_batches,
        "sessions": sessions,
        "budget": {
            "global_budget": config.global_budget,
            "consumed": scheduler.budget_consumed,
            "remaining": scheduler.budget_remaining,
            "pending_sessions": list(scheduler.pending),
            "deduction_sequence": list(scheduler.ledger),
        },
        "injections": {
            "requested": requested_ids,
            "applied": list(dict.fromkeys(applied_ids)),
            "hits": _injection_hits(all_failures, requested_ids),
        },
        "failures": sort_failures(_normalise(all_failures)),
    }
    report["statistics"] = _statistics(sessions)
    if stale_failure is not None:
        report["snapshot_event"] = stale_failure
        # Staleness is a run-level recovery event, not a session business
        # failure, so it stays out of the field-level comparison projection.
        report["failures"] = [f for f in report["failures"]
                              if f["code"] != "E_SNAPSHOT_STALE"]
    report["report_fingerprint"] = digest(_comparison_view(report))
    return report


def write_report(report, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True, ensure_ascii=False, indent=2)
        handle.write("\n")


def _normalise(failures):
    out = []
    for item in failures:
        if hasattr(item, "to_dict"):
            out.append(item.to_dict())
        else:
            out.append(dict(item))
    return out


def _injection_hits(failures, requested_ids):
    hit = {}
    for failure in failures:
        inj_id = failure.get("injection_id")
        if inj_id and inj_id in requested_ids:
            hit.setdefault(inj_id, []).append({
                "code": failure["code"],
                "stage": failure["stage"],
                "session_id": failure.get("session_id"),
                "turn": failure.get("turn"),
            })
    return {inj_id: hit[inj_id] for inj_id in requested_ids if inj_id in hit}


def comparison_view(report):
    """Project the report onto fields that must match between runs."""
    return _comparison_view(report)


def _comparison_view(report):
    sessions = {}
    for sid, session in report["sessions"].items():
        sessions[sid] = {
            key: value for key, value in session.items()
            # reused/replayed turn lists describe *how* the run obtained its
            # result and differ by construction between incremental and full.
            if key not in ("reused_turns", "replayed_turns")
        }
    return {
        "sessions": sessions,
        "failures": report["failures"],
        "injections": report["injections"],
    }


def _statistics(sessions):
    """Deterministic NumPy summary over per-session processed volume.

    Values are rounded to a fixed number of decimals so the JSON output is
    stable across platforms; NumPy is used for the aggregate calculations.
    """
    consumed = np.array(
        [session["consumed_entries"] for session in sessions.values()],
        dtype=np.int64)
    transition_counts = np.array(
        [session["transition_count"] for session in sessions.values()],
        dtype=np.int64)
    if consumed.size == 0:
        return {"session_count": 0}
    return {
        "session_count": int(consumed.size),
        "consumed_entries": {
            "total": int(consumed.sum()),
            "mean": round(float(consumed.mean()), 6),
            "stddev_population": round(float(consumed.std()), 6),
            "max": int(consumed.max()),
        },
        "transition_counts": {
            "total": int(transition_counts.sum()),
            "mean": round(float(transition_counts.mean()), 6),
            "stddev_population": round(float(transition_counts.std()), 6),
        },
    }
