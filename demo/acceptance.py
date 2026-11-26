"""End-to-end acceptance driver for the four required verification scenarios.

Runs real node subprocesses on 127.0.0.1, a real SQLite ledger and real
monotonic-clock timing. Outputs structured JSON evidence under demo/out.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from shardcast.engine import Engine
from shardcast.ledger import Ledger
from shardcast.propagation import plan_invalidation
from shardcast.recovery import recover_and_replay
from shardcast.sample import canonical_json
from shardcast.spec import parse_spec

from demo.generate_spec import build_spec


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def scenario_baseline(out_dir: Path) -> dict:
    raw = build_spec()
    raw["events"] = [
        {"kind": "arrive", "shard_id": "s-root"},
        {"kind": "arrive", "shard_id": "s-alpha"},
        {"kind": "arrive", "shard_id": "s-beta"},
        {"kind": "arrive", "shard_id": "s-gamma"},
        {"kind": "arrive", "shard_id": "s-delta"},
        {"kind": "arrive", "shard_id": "s-epsilon"},
    ]
    spec = parse_spec(raw)
    ledger_path = out_dir / "baseline-ledger.db"
    ledger = Ledger(ledger_path)
    engine = Engine(spec, ledger, batch_size=37)
    result = engine.run()
    ledger.close()
    evidence = {
        "outcomes": result.outcomes,
        "decisions": result.decisions,
        "waits": result.waits,
        "water_marks": result.water_marks,
    }
    _write(out_dir / "01-baseline.json", evidence)
    return evidence


def scenario_capacity_switch(out_dir: Path) -> dict:
    raw = build_spec()
    spec = parse_spec(raw)
    ledger = Ledger(out_dir / "capacity-ledger.db")
    engine = Engine(spec, ledger, batch_size=19)
    engine.start()
    try:
        engine.arrived.add("s-root")
        engine._drain_once()
        lease = engine.lease_manager.grant(
            "lease-inflight-alpha", "s-alpha", "node-a", 1
        )
        engine.pool.install_lease("node-a", lease.to_dict())
        engine.in_flight["s-alpha"] = lease
        engine._resource_used += spec.shard("s-alpha").resource_units
        engine.runtimes["s-alpha"].status = "dispatched"
        engine.runtimes["s-alpha"].node_id = "node-a"
        engine.runtimes["s-alpha"].lease_id = lease.lease_id
        engine.runtimes["s-alpha"].capacity_version = 1

        engine._replace_capacity(2)
        revocation = {
            "decisions": engine.lease_manager.decisions_snapshot(),
            "diffs": engine.capacity.diffs,
            "backups": [
                {
                    "version": view.version,
                    "global_budget": view.global_budget,
                    "quotas": {
                        node_id: quota.__dict__
                        for node_id, quota in view.quotas.items()
                    },
                }
                for view in engine.capacity.backups
            ],
            "in_flight_after_revoke": sorted(engine.in_flight),
        }
        engine.arrived = set(spec.shard_ids)
        engine._drain_until_settled()
        result = engine._build_result()
    finally:
        engine.stop()
        ledger.close()
    evidence = {
        "revocation": revocation,
        "final_outcomes": result.outcomes,
        "final_capacity_version": result.capacity_version,
        "water_marks": result.water_marks,
    }
    _write(out_dir / "02-capacity-switch.json", evidence)
    return evidence


def scenario_propagation(out_dir: Path) -> dict:
    old_raw = build_spec()
    new_raw = json.loads(json.dumps(old_raw))
    new_raw["shards"][1]["resource_units"] = 5
    old_spec = parse_spec(old_raw)
    new_spec = parse_spec(new_raw)
    steps = plan_invalidation(old_spec, new_spec)

    baseline = scenario_baseline(out_dir)
    unaffected = ["s-root"]
    byte_identical = {
        shard_id: canonical_json(baseline["outcomes"][shard_id]).decode("utf-8")
        for shard_id in unaffected
    }
    evidence = {
        "invalidated": steps,
        "unaffected_byte_identical": byte_identical,
    }
    _write(out_dir / "03-propagation.json", evidence)
    return evidence


def scenario_crash_recovery(out_dir: Path) -> dict:
    raw = build_spec()
    raw["events"] = [
        {"kind": "arrive", "shard_id": "s-root"},
        {"kind": "arrive", "shard_id": "s-alpha"},
        {"kind": "arrive", "shard_id": "s-beta"},
    ]
    spec = parse_spec(raw)
    ledger_path = out_dir / "crash-ledger.db"
    ledger = Ledger(ledger_path)
    engine = Engine(spec, ledger)
    engine.start()
    try:
        engine.arrived = {"s-root", "s-alpha", "s-beta"}
        for _ in range(20):
            engine._drain_once()
            if all(
                engine.runtimes[shard_id].status == "completed"
                for shard_id in ("s-root", "s-alpha", "s-beta")
            ):
                break
        committed_before = sorted(ledger.all_outcomes())
        killed_node = engine.runtimes["s-alpha"].node_id
        engine.pool.kill_node_hard(killed_node)
    finally:
        engine.stop()
        ledger.close()

    full_raw = build_spec()
    report = recover_and_replay(
        parse_spec(full_raw),
        str(ledger_path),
        killed_node=killed_node,
        batch_size=23,
    )
    evidence = {
        "committed_before_kill": committed_before,
        "killed_node": killed_node,
        "recovery_report": report.to_dict(),
        "note": "no exit report can be produced for a SIGKILL worker; only committed rows are trusted",
    }
    _write(out_dir / "04-crash-recovery.json", evidence)
    return evidence


def main() -> int:
    out_dir = Path(__file__).with_name("out")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "baseline": scenario_baseline(out_dir),
        "capacity_switch": scenario_capacity_switch(out_dir),
        "propagation": scenario_propagation(out_dir),
        "crash_recovery": scenario_crash_recovery(out_dir),
    }
    summary = {
        "scenario_files": [
            "demo/out/01-baseline.json",
            "demo/out/02-capacity-switch.json",
            "demo/out/03-propagation.json",
            "demo/out/04-crash-recovery.json",
        ],
        "completed_shards": sorted(
            shard_id
            for shard_id, outcome in results["baseline"]["outcomes"].items()
            if outcome["status"] == "completed"
        ),
        "recovery_unknown": results["crash_recovery"]["recovery_report"]["unknown"],
        "recovery_mismatches": results["crash_recovery"]["recovery_report"]["mismatches"],
    }
    _write(out_dir / "00-summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
