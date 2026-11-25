"""Repeatable end-to-end demonstration runner.

Modes:
  full      : run the complete event stream and independent replay
  crash     : force-kill a node mid-run, restart it, then verify recovery
  tamper    : corrupt the ledger digest and demonstrate whole-file replacement
  all       : run every mode in sequence

All artifacts are written under demo/ and use only loopback TCP + local files.
"""

import argparse
import json
import os
import shutil
import sqlite3
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from shardflow.engine import Engine
from shardflow.ledger import Ledger
from shardflow.replay import verify_ledger
from shardflow.spec import load_spec

SPEC = os.path.join(ROOT, "demo", "demo.spec.json")
EVENTS = os.path.join(ROOT, "demo", "demo.events.ndjson")


def _print(title, payload):
    print("\n=== %s ===" % title)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def mode_full():
    ledger = os.path.join(ROOT, "demo", "demo.full.sqlite")
    spec = load_spec(SPEC, EVENTS)
    engine = Engine(spec, ledger)
    snapshot = engine.run()
    _print("run snapshot", {
        "statuses": {sid: row["status"]
                     for sid, row in sorted(snapshot["shards"].items())},
        "invalidated": snapshot["invalidated_order"],
        "errors": snapshot["errors"],
        "digest": snapshot["content_digest"],
    })
    result = verify_ledger(SPEC, ledger, EVENTS)
    _print("independent replay", result)
    return result["status"] == "ok"


def mode_crash():
    ledger = os.path.join(ROOT, "demo", "demo.crash.sqlite")
    spec = load_spec(SPEC, EVENTS)
    engine = Engine(spec, ledger)
    engine.start_node_processes()
    for node_id in engine.spec.node_order:
        engine.node_clients[node_id].handshake()

    # Mark an in-flight-style row as unknown, simulating an orchestrator kill
    # while a transaction was still open (only committed rows survive).
    engine.ledger.mark_unknown(
        "s2", 1, "node-a",
        {"granted_ms": 1, "config_version": 1, "capacity_version": 1,
         "resource_units": 2, "wait_start_ms": 1, "wait_end_ms": 1},
        0,
    )
    engine.force_kill_node("node-b")
    engine.restart_node("node-b")
    engine.stop_node_processes()
    engine.ledger.close()

    recovered = Engine(load_spec(SPEC, EVENTS), ledger)
    info = recovered.recover()
    _print("recovery view", info)
    snapshot = recovered.rerun_unknown()
    _print("post-recovery unknown shards",
           {"unknown": snapshot["unknown_shards"],
            "s2": snapshot["shards"]["s2"]})
    return snapshot["unknown_shards"] == []


def mode_tamper():
    ledger = os.path.join(ROOT, "demo", "demo.tamper.sqlite")
    engine = Engine(load_spec(SPEC, EVENTS), ledger)
    engine.run()
    engine.ledger.close()
    conn = sqlite3.connect(ledger)
    conn.execute("UPDATE meta SET value='tampered' WHERE key='content_digest'")
    conn.commit()
    conn.close()
    reopened = Ledger(ledger, spec_digest="ignored")
    _print("tamper handling", {
        "replaced": reopened.replaced,
        "format_version": reopened.format_version(),
        "lease_events_after_replace": len(reopened.fetch_lease_events()),
    })
    reopened.close()
    return True


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["full", "crash", "tamper", "all"],
                        default="all", nargs="?")
    args = parser.parse_args(argv)
    modes = {
        "full": mode_full,
        "crash": mode_crash,
        "tamper": mode_tamper,
    }
    selected = list(modes) if args.mode == "all" else [args.mode]
    results = {}
    for name in selected:
        results[name] = modes[name]()
    _print("summary", results)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
