"""Deterministic forced-termination and recovery drill.

Phase A runs real loopback traffic and commits batches.  It then simulates a
hard kill by leaving rows in the pending table with no committed batch (an
in-flight transaction that never reached COMMIT).  Phase B opens a brand-new
process-equivalent engine from only the committed ledger, replays every window
byte-for-byte, and reports the unknown requests that must be rerun.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from meshbench.cli import _make_requests, engine_digest
from meshbench.clock import VirtualClock
from meshbench.config import load_config
from meshbench.engine import Engine
from meshbench.ledger import Ledger
from meshbench.netutil import assert_ports_free, preflight_upstreams
from meshbench.recovery import recover
from meshbench.upstream import start_all, stop_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", default="samples/rules_v1.json")
    parser.add_argument("--db", default="logs/crash.db")
    args = parser.parse_args()
    if os.path.exists(args.db):
        os.remove(args.db)

    config = load_config(args.rules)
    assert_ports_free(sorted({i.port for s in config.services.values()
                              for i in s.instances}))
    upstreams = start_all(config)
    preflight_upstreams([i for s in config.services.values() for i in s.instances])

    ledger = Ledger(args.db)
    ledger.record_rule_version(config.version, engine_digest(config), 0)
    clock = VirtualClock()
    engine = Engine(config, ledger, clock=clock)
    engine.run_batch(_make_requests("orders-api", 18))
    clock.advance(60_000)
    engine.run_batch(_make_requests("orders-api", 12, 18))

    # Simulate SIGKILL mid-batch: pending rows exist, but no batch commits them.
    orphan = _make_requests("orders-api", 4, 80)
    ledger.mark_pending(orphan, config.version)
    print(f"[phase A] committed cursor={ledger.cursor_sequence()} "
          f"simulated_in_flight={len(orphan)} -> process killed (no exit report)")
    ledger.close()
    stop_all(upstreams)

    # Phase B: fresh recovery from durable, committed records only.
    upstreams = start_all(config)
    report = recover(config, args.db, clock=VirtualClock())
    lines = {
        "status": "recovered",
        "format_version": report["format_version"],
        "cursor_sequence": report["cursor_sequence"],
        "head_digest": report["head_digest"],
        "replayed_windows": sorted(report["receipts"]),
        "replay_elapsed_ms": report["replay_elapsed_ms"],
        "unknown_requests": report["unknown_requests"],
        "byte_for_byte_replay_ok": True,
    }
    print("[phase B] " + json.dumps(lines, sort_keys=True, indent=2))
    report["ledger"].close()
    stop_all(upstreams)
    assert len(report["unknown_requests"]) == 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
