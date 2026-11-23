"""Command line interface: run / resume / verify / summary."""

import argparse
import json
import sys

from . import LEDGER_FORMAT_VERSION
from .aggregate import verify_aggregates
from .errors import ConvError
from .ledger import Ledger, reconcile_snapshot
from .orchestrator import Orchestrator
from .replay import check_meta, replay_verify, unknown_rounds
from .scene import load_executor_set, load_scenario


def _parse_drops(items):
    drop_map = {}
    for item in items or []:
        executor_id, _, rounds = item.partition(":")
        drop_map[executor_id] = [int(r) for r in rounds.split(",") if r]
    return drop_map


def _conclusions(ledger):
    out = {}
    for round_no, executor_id, entry in ledger.round_entries():
        out.setdefault(str(round_no), {})[executor_id] = entry["conclusion"]
    return out


def _init_ledger(ledger, scenario):
    ledger.set_meta(format_version=LEDGER_FORMAT_VERSION,
                    scenario_id=scenario.scenario_id,
                    scenario_version=scenario.digest,
                    budget_version=scenario.budget_version,
                    executor_set_version=scenario.executor_set_version)


def cmd_run(args):
    scenario = load_scenario(args.scenario)
    executor_set = load_executor_set(args.executors)
    ledger = Ledger(args.db)
    _init_ledger(ledger, scenario)
    drop_map = _parse_drops(args.drop)
    with Orchestrator(scenario, executor_set, ledger,
                      base_ms=args.base_ms, jitter_ms=args.jitter_ms,
                      drop_map=drop_map) as orch:
        digest = orch.run()
    reconcile_snapshot(args.db, scenario, _conclusions(ledger), digest)
    result = {"status": "ok", "cursor": ledger.cursor(),
              "ledger_digest": digest,
              "aggregate": verify_aggregates(ledger)}
    print(json.dumps(result, indent=2, sort_keys=True))
    ledger.close()
    return 0


def cmd_resume(args):
    scenario = load_scenario(args.scenario)
    executor_set = load_executor_set(args.executors)
    ledger = Ledger(args.db)
    check_meta(ledger, scenario, executor_set)
    pending = unknown_rounds(ledger, scenario)
    if not pending:
        print(json.dumps({"status": "ok", "resumed_rounds": [],
                          "note": "nothing to resume"}))
        ledger.close()
        return 0
    # Rounds past the cursor were never committed: unknown, re-run them.
    drop_map = _parse_drops(args.drop)
    with Orchestrator(scenario, executor_set, ledger,
                      base_ms=args.base_ms, jitter_ms=args.jitter_ms,
                      drop_map=drop_map) as orch:
        orch.restore_tracker()
        digest = orch.run(start_round=pending[0])
    reconcile_snapshot(args.db, scenario, _conclusions(ledger), digest)
    print(json.dumps({"status": "ok", "resumed_rounds": pending,
                      "ledger_digest": digest,
                      "aggregate": verify_aggregates(ledger)},
                     indent=2, sort_keys=True))
    ledger.close()
    return 0


def cmd_verify(args):
    scenario = load_scenario(args.scenario)
    executor_set = load_executor_set(args.executors)
    ledger = Ledger(args.db)
    drop_map = _parse_drops(args.drop)
    replay = replay_verify(scenario, executor_set, ledger,
                           base_ms=args.base_ms, jitter_ms=args.jitter_ms,
                           drop_map=drop_map)
    aggregate = verify_aggregates(ledger)
    print(json.dumps({"status": "ok", "replay": replay,
                      "aggregate": aggregate}, indent=2, sort_keys=True))
    ledger.close()
    return 0 if aggregate["ok"] else 1


def cmd_summary(args):
    ledger = Ledger(args.db)
    print(json.dumps({"cursor": ledger.cursor(),
                      "ledger_digest": ledger.content_digest(),
                      "summary": ledger.summary(),
                      "anomalies": [
                          {"round": r, "executor_id": e, "kind": k,
                           "payload": p}
                          for r, e, k, p in ledger.anomaly_entries()],
                      "retries": [
                          {"round": r, "executor_id": e, "attempt": a,
                           "payload": p}
                          for r, e, a, p in ledger.retry_entries()]},
                     indent=2, sort_keys=True))
    ledger.close()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="convd")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--scenario", required=True)
        p.add_argument("--executors", required=True)
        p.add_argument("--db", required=True)
        p.add_argument("--base-ms", type=int, default=1)
        p.add_argument("--jitter-ms", type=int, default=4)
        p.add_argument("--drop", action="append",
                       help="executor_id:round1,round2 drop first attempt")

    common(sub.add_parser("run"))
    common(sub.add_parser("resume"))
    common(sub.add_parser("verify"))
    summary_p = sub.add_parser("summary")
    summary_p.add_argument("--db", required=True)
    args = parser.parse_args(argv)
    handler = {"run": cmd_run, "resume": cmd_resume,
               "verify": cmd_verify, "summary": cmd_summary}[args.command]
    try:
        return handler(args)
    except ConvError as exc:
        print(json.dumps(exc.to_dict(), indent=2, sort_keys=True),
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
