"""Repeatable end-to-end demonstration entry point.

Runs the fixed sequence required by the specification for every deterministic
sample: generation, offline synthesis, online frame-driven replay (with jump
and speed changes), reconciliation and budget reporting. All machine-measured
timings are replaced by a scripted clock whose per-tick costs are fixed, so two
runs with the same inputs and budget produce byte-identical outputs and hashes.
"""
from __future__ import annotations

import os

from .demo_clock import build_scripted_costs
from .engine import TickEngine
from .errors import ErrorCode
from .events import load_events
from .generator import PROFILES, generate_all, load_rules
from .hashing import sha256_file
from .offline import run_offline, write_offline_reports
from .player import ReplayPlayer, ScriptedClock
from .store import write_json, write_text
from .verifier import error_code_audit, reconcile, verify_jump_equivalence

BUDGET_MS = 4.0
BASE_TICKS_PER_FRAME = 4
SPIKE_FACTOR = 6.0


def _run_online(workspace_dir: str, scenario: str, rules, events, last_tick: int):
    costs = build_scripted_costs(last_tick, BUDGET_MS, SPIKE_FACTOR)
    clock = ScriptedClock(costs)
    snapshot_dir = os.path.join(workspace_dir, "snapshots", scenario)
    player = ReplayPlayer(
        rules,
        events,
        budget_ms=BUDGET_MS,
        base_ticks_per_frame=BASE_TICKS_PER_FRAME,
        snapshot_dir=snapshot_dir,
        clock=clock,
    )
    player.resume()
    midpoint = max(1, last_tick // 2)

    while player.current_tick < min(midpoint, last_tick):
        player.run_frame()

    jump_target = min(last_tick, midpoint + 5)
    player.pause()
    state_at_jump = player.jump_to(jump_target)
    jump_hash = state_at_jump.state_hash()

    player.set_speed(4)
    player.resume()
    while not player.finished:
        player.run_frame()

    return player, {
        "budget_ms": BUDGET_MS,
        "base_ticks_per_frame": BASE_TICKS_PER_FRAME,
        "jump_tick": jump_target,
        "jump_state_hash": jump_hash,
        "speed_sequence": [1, 4],
        "final_state_hash": player.engine.state.state_hash(),
        **player.budget_report.to_dict(),
    }


def run_demo(workspace_dir: str) -> dict:
    manifests = generate_all(workspace_dir)
    reports_root = os.path.join(workspace_dir, "reports")
    scenario_reports = []

    for manifest in manifests:
        scenario = manifest["scenario"]
        sample_dir = os.path.join(workspace_dir, "samples", scenario)
        rules = load_rules(os.path.join(sample_dir, manifest["rules_file"]))
        events = load_events(os.path.join(sample_dir, manifest["events_file"]))
        report_dir = os.path.join(reports_root, scenario)
        snapshot_dir = os.path.join(workspace_dir, "snapshots", scenario)

        offline = run_offline(
            rules, events, snapshot_dir=snapshot_dir
        )
        write_offline_reports(report_dir, offline)

        player, online_report = _run_online(
            workspace_dir, scenario, rules, events, offline.engine.last_tick
        )
        write_json(os.path.join(report_dir, "online_budget.json"), online_report)

        reconciliation = reconcile(
            rules,
            events,
            offline,
            budget_ms=BUDGET_MS,
            base_ticks_per_frame=BASE_TICKS_PER_FRAME,
        )
        jump_ticks = sorted(
            {0, 1, offline.engine.last_tick // 2, offline.engine.last_tick - 1}
        )
        jumps = verify_jump_equivalence(rules, events, offline, jump_ticks)
        audit = error_code_audit(offline)

        verify_payload = {
            "reconciliation": reconciliation.to_dict(),
            "jump_equivalence": jumps,
            "error_audit": audit,
        }
        write_json(os.path.join(report_dir, "verification.json"), verify_payload)
        write_text(
            os.path.join(report_dir, "budget_report.txt"),
            _format_budget_report(scenario, online_report, verify_payload),
        )
        scenario_reports.append(
            {
                "scenario": scenario,
                "kind": manifest["kind"],
                "event_count": manifest["event_count"],
                "tick_count": manifest["tick_count"],
                "final_state_hash": offline.engine.state.state_hash(),
                "online_final_state_hash": online_report["final_state_hash"],
                "reconcile_ok": reconciliation.ok,
                "ledger_match": reconciliation.ledger_match,
                "jump_all_match": jumps["all_match"],
                "rng_drift_tick": reconciliation.rng_drift_tick,
                "over_budget_count": online_report["over_budget_count"],
                "skipped_refresh_count": online_report["skipped_refresh_count"],
                "error_codes": sorted(audit["error_counts"]),
            }
        )

    master = {
        "budget_ms": BUDGET_MS,
        "base_ticks_per_frame": BASE_TICKS_PER_FRAME,
        "spike_factor": SPIKE_FACTOR,
        "scenarios": scenario_reports,
    }
    write_json(os.path.join(reports_root, "master_report.json"), master)
    write_text(
        os.path.join(reports_root, "master_report.txt"),
        _format_master(master),
    )
    return master


def _format_budget_report(scenario: str, online_report: dict, verify_payload: dict) -> str:
    lines = [
        f"ONLINE REPLAY BUDGET REPORT - {scenario}",
        f"budget_ms={online_report['budget_ms']}",
        f"base_ticks_per_frame={online_report['base_ticks_per_frame']}",
        f"frames={len(online_report['frames'])}",
        f"over_budget={online_report['over_budget_count']}",
        f"skipped_refresh={online_report['skipped_refresh_count']}",
        f"min_allowance={online_report['min_allowance']}",
        f"max_elapsed_ms={online_report['max_elapsed_ms']}",
        f"jump_tick={online_report['jump_tick']} jump_state_hash={online_report['jump_state_hash']}",
        "reconcile_ok=" + str(verify_payload["reconciliation"]["ok"]),
        "ledger_match=" + str(verify_payload["reconciliation"]["ledger_match"]),
        "rng_drift_tick=" + str(verify_payload["reconciliation"]["rng_drift_tick"]),
        "jump_all_match=" + str(verify_payload["jump_equivalence"]["all_match"]),
        "error_counts=" + str(verify_payload["error_audit"]["error_counts"]),
    ]
    return "\n".join(lines) + "\n"


def _format_master(master: dict) -> str:
    lines = [
        "RTS REPLAY WORKBENCH MASTER REPORT",
        f"budget_ms={master['budget_ms']} base_ticks_per_frame={master['base_ticks_per_frame']}",
        "scenarios:",
    ]
    for item in master["scenarios"]:
        lines.append(
            "  {scenario} ({kind}): events={event_count} ticks={tick_count} "
            "reconcile_ok={reconcile_ok} ledger_match={ledger_match} "
            "jump_all_match={jump_all_match} rng_drift_tick={rng_drift_tick} "
            "over_budget={over_budget_count} errors={error_codes}".format(**item)
        )
    return "\n".join(lines) + "\n"
