"""Offline full-scan synthesis path.

The offline path reads the whole event stream once and synthesizes every tick
straight through. Its outputs (per-tick state hashes, ledger entries, tick
errors) are the authoritative reference that the online replay is verified
against.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .engine import TickEngine
from .events import Event
from .rules import RuleConfig
from .state import Snapshot
from .store import save_snapshot, write_json, write_text
from .hashing import canonical_dumps


@dataclass
class OfflineResult:
    engine: TickEngine
    tick_hashes: list[dict]
    ledger: list[dict]
    errors: list[dict]
    final_state: dict


def run_offline(
    config: RuleConfig,
    events: list[Event],
    *,
    snapshot_dir: str | None = None,
    snapshot_interval: int = 10,
) -> OfflineResult:
    engine = TickEngine(
        config, events, snapshot_interval=snapshot_interval
    )
    engine.advance_to(engine.last_tick)
    if snapshot_dir is not None:
        for tick, snapshot in sorted(engine.snapshots.items()):
            save_snapshot(snapshot_dir, snapshot)
    return OfflineResult(
        engine=engine,
        tick_hashes=[record.to_dict() for record in engine.records],
        ledger=[entry.to_dict() for entry in engine.ledger],
        errors=[error.to_dict() for error in engine.all_errors()],
        final_state=engine.state.to_dict(),
    )


def write_offline_reports(report_dir: str, result: OfflineResult) -> dict[str, str]:
    os.makedirs(report_dir, exist_ok=True)
    hashes_path = os.path.join(report_dir, "tick_hashes.jsonl")
    with open(hashes_path, "wb") as handle:
        for record in result.tick_hashes:
            handle.write(canonical_dumps(record) + b"\n")

    ledger_path = os.path.join(report_dir, "ledger.jsonl")
    with open(ledger_path, "wb") as handle:
        for entry in result.ledger:
            handle.write(canonical_dumps(entry) + b"\n")

    errors_path = os.path.join(report_dir, "tick_errors.jsonl")
    with open(errors_path, "wb") as handle:
        for error in result.errors:
            handle.write(canonical_dumps(error) + b"\n")

    summary = {
        "tick_count": len(result.tick_hashes),
        "ledger_entry_count": len(result.ledger),
        "error_count": len(result.errors),
        "final_state_hash": result.engine.state.state_hash(),
        "final_state": result.final_state,
    }
    summary_path = os.path.join(report_dir, "offline_summary.json")
    write_json(summary_path, summary)
    write_text(
        os.path.join(report_dir, "final_state.txt"),
        _format_final_state(summary),
    )
    return {
        "hashes": hashes_path,
        "ledger": ledger_path,
        "errors": errors_path,
        "summary": summary_path,
    }


def _format_final_state(summary: dict) -> str:
    lines = [
        "OFFLINE SYNTHESIS SUMMARY",
        f"ticks={summary['tick_count']}",
        f"ledger_entries={summary['ledger_entry_count']}",
        f"errors={summary['error_count']}",
        f"final_state_hash={summary['final_state_hash']}",
        "factions:",
    ]
    for name, faction in sorted(summary["final_state"]["factions"].items()):
        lines.append(
            "  {name}: resources={resources} collected={total_collected} "
            "spent={total_spent} idle={idle_units} deployed={deployed_units} "
            "queued={queued}".format(
                name=name, queued=len(faction["queue"]), **faction
            )
        )
    return "\n".join(lines) + "\n"
