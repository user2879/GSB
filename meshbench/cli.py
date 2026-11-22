"""Command-line entry point with deterministic demos and fault injection."""

import argparse
import json
import os
import sys

from .clock import VirtualClock
from .config import load_config
from .engine import Engine
from .errors import BenchError, RuleError
from .ledger import Ledger
from .logging_util import StructuredLogger
from .netutil import assert_ports_free, preflight_upstreams
from .protocol import Request
from .snapshot import (assert_ledger_matches_snapshot, load_snapshot,
                       save_snapshot)
from .upstream import start_all, stop_all


def _all_ports(config):
    return sorted({inst.port for service in config.services.values()
                   for inst in service.instances})


def _all_instances(config):
    return [inst for service in config.services.values()
            for inst in service.instances]


def _make_requests(route_id, count, start_sequence=0):
    return [
        Request(
            request_id=f"req-{route_id}-{start_sequence + index:04d}",
            route_id=route_id,
            client_key=f"client-{index % 7}",
            sequence=start_sequence + index,
        )
        for index in range(count)
    ]


def cmd_validate(args, logger):
    config = load_config(args.rules)
    print(json.dumps({"status": "ok", "version": config.version,
                      "services": sorted(config.services),
                      "routes": sorted(config.routes)}, sort_keys=True))
    return 0


def _open_engine(rules_path, db_path, logger, virtual=True):
    config = load_config(rules_path)
    ports = _all_ports(config)
    assert_ports_free(ports)
    ledger = Ledger(db_path)
    ledger.verify_integrity()
    clock = VirtualClock() if virtual else None
    return config, ledger, clock


def cmd_demo(args, logger):
    import time
    lines = []
    db_path = args.db
    snapshot_path = args.snapshot
    if os.path.exists(db_path):
        os.remove(db_path)

    config = load_config(args.rules)
    ports = _all_ports(config)
    assert_ports_free(ports)
    upstreams = start_all(config)
    preflight_upstreams(_all_instances(config))
    ledger = Ledger(db_path)
    ledger.record_rule_version(config.version, engine_digest(config), 0)
    clock = VirtualClock()
    engine = Engine(config, ledger, clock=clock, logger=logger)

    # Scenario 1: normal deterministic dispatch across two windows.
    batch_one = _make_requests("orders-api", 24, 0)
    r1 = engine.run_batch(batch_one)
    clock.advance(60_000)
    batch_two = _make_requests("orders-api", 18, 24) + _make_requests(
        "payments-api", 12, 0)
    r2 = engine.run_batch(batch_two)
    lines.append(f"[1] normal dispatch: batches={r1['batch_id']},{r2['batch_id']} "
                 f"requests={len(batch_one) + len(batch_two)} windows="
                 f"{len(r1['receipts']) + len(r2['receipts'])}")
    view_text, view_ms = engine.render_view()
    lines.append(f"[1] cached view rendered in {view_ms}ms (monotonic)")

    # Scenario 2: isolated breaker scenario on a fresh ledger/registry so the
    # consecutive-failure counters start at zero.  Real monotonic cooldown is
    # kept short by the rule (400ms); traffic concentrates on both instances.
    breaker_db = os.path.join(os.path.dirname(db_path), "breaker.db")
    if os.path.exists(breaker_db):
        os.remove(breaker_db)
    breaker_ledger = Ledger(breaker_db)
    breaker_clock = VirtualClock()
    breaker_engine = Engine(config, breaker_ledger, clock=breaker_clock)
    flaky = _make_requests("payments-api", 16, 100)
    r3 = breaker_engine.run_batch(flaky)
    open_events = [m for m in r3["migrations"] if m["to_state"] == "OPEN"]
    lines.append(f"[2] breaker migrations={len(r3['migrations'])} "
                 f"open_events={len(open_events)} "
                 f"instances={sorted({m['instance_id'] for m in open_events})}")
    # Real monotonic wait beyond cooldown -> OPEN -> HALF_OPEN probe path.
    time.sleep(0.6)
    probe = breaker_engine.run_batch(_make_requests("payments-api", 4, 300))
    half_open_seen = any(
        attempt.error_code == "E_CIRCUIT"
        for result in probe["results"] for attempt in result.attempts
    ) or any(m["from_state"] == "HALF_OPEN" for m in probe["migrations"])
    lines.append(f"[2] post-cooldown probe migrations={len(probe['migrations'])} "
                 f"half_open_exercised={half_open_seen}")
    breaker_ledger.close()
    breaker_clock.advance(60_000)

    # Scenario 3: rule evolution -> wholesale receipt replacement.
    new_config = load_config(args.rules_v2)
    report = engine.evolve_rules(new_config)
    ledger.record_rule_version(new_config.version, engine_digest(new_config),
                               clock.now_ms())
    evolved = _make_requests("orders-api", 10, 200)
    r4 = engine.run_batch(evolved)
    lines.append(
        f"[3] evolved v{report['old_version']}->v{report['new_version']} "
        f"replaced_windows={report['replaced_windows']} "
        f"affected_instances={len(report['affected_instances'])} "
        f"new_receipts={len(r4['receipts'])}"
    )

    # Authority separation: cached view must match byte-for-byte recomputation.
    receipts, recompute_ms = engine.authoritative_recompute()
    lines.append(f"[3] authoritative recompute matched cache in {recompute_ms}ms "
                 f"windows={len(receipts)}")
    final_view, final_view_ms = engine.render_view()

    snapshot = engine.snapshot()
    save_snapshot(snapshot_path, snapshot)
    lines.append(f"[snapshot] saved cursor={snapshot['cursor_sequence']} "
                 f"digest={snapshot['content_digest'][:12]}")

    stop_all(upstreams)
    ledger.close()
    summary = "\n".join(lines)
    print(summary)
    print("\n--- headless view (final) ---")
    print(final_view)
    logger.log("info", "demo_complete",
               batches=[r1["batch_id"], r2["batch_id"], r3["batch_id"], r4["batch_id"]])
    return 0

def engine_digest(config):
    from .canonical import content_digest
    from dataclasses import asdict
    return content_digest({
        "version": config.version,
        "global_concurrency": config.global_concurrency,
        "services": {sid: asdict(svc) for sid, svc in config.services.items()},
        "routes": {rid: asdict(route) for rid, route in config.routes.items()},
    })


def cmd_inject(args, logger):
    """Inject one of rule-corrupt / upstream-down / snapshot-corrupt."""
    fault = args.fault
    try:
        if fault == "rule-corrupt":
            import shutil
            import tempfile
            directory = tempfile.mkdtemp(prefix="meshbench-inject-")
            path = os.path.join(directory, "corrupt-rules.json")
            shutil.copyfile(args.rules, path)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write("\n{ this is not valid json")
            load_config(path)  # must raise E_RULE
        elif fault == "upstream-down":
            config = load_config(args.rules)
            # Preflight against ports with nothing listening -> E_UPSTREAM.
            assert_ports_free(_all_ports(config))
            preflight_upstreams(_all_instances(config))
        elif fault == "snapshot-corrupt":
            config = load_config(args.rules)
            assert_ports_free(_all_ports(config))
            ledger = Ledger(args.db)
            ledger.record_rule_version(config.version, engine_digest(config), 0)
            original_head = ledger.head_digest()
            # Persist a valid snapshot, then corrupt the ledger out-of-band so
            # reopening detects the content-digest mismatch (E_SNAPSHOT).
            snapshot = _with_digest({
                "snapshot_format": 1,
                "rule_version": config.version,
                "params_version": 1,
                "breaker_snapshots": [],
                "receipt_digests": {},
                "cursor_sequence": 0,
                "ledger_head_digest": original_head,
            })
            save_snapshot(args.snapshot, snapshot)
            ledger.tamper_head_digest()
            loaded = load_snapshot(args.snapshot)
            assert_ledger_matches_snapshot(loaded, ledger)
            ledger.close()
    except BenchError as exc:
        logger.log("error", "injected_fault", fault=fault, **exc.as_dict())
        print(json.dumps({"injected": fault, "error": exc.as_dict()},
                         sort_keys=True, ensure_ascii=False))
        return 4
    logger.log("error", "injection_did_not_fire", fault=fault)
    return 5


def _with_digest(snapshot):
    from .canonical import content_digest
    snapshot = dict(snapshot)
    snapshot["content_digest"] = content_digest(
        {k: v for k, v in snapshot.items()}
    )
    return snapshot




def cmd_recover(args, logger):
    from .recovery import recover
    config = load_config(args.rules)
    if args.rules_version and config.version != args.rules_version:
        raise RuleError(
            "recover rule version does not match --rules-version",
            config_version=config.version, requested=args.rules_version,
        )
    assert_ports_free(_all_ports(config))
    upstreams = start_all(config)
    try:
        report = recover(config, args.db, clock=VirtualClock(), logger=logger)
    finally:
        stop_all(upstreams)
    unknown = report["unknown_requests"]
    print(json.dumps({
        "status": "recovered",
        "format_version": report["format_version"],
        "cursor_sequence": report["cursor_sequence"],
        "head_digest": report["head_digest"],
        "replay_elapsed_ms": report["replay_elapsed_ms"],
        "replayed_windows": sorted(report["receipts"]),
        "unknown_requests": unknown,
        "note": "unknown requests were never committed and must be rerun; "
                "no forced-kill exit report is fabricated",
    }, sort_keys=True, indent=2))
    report["ledger"].close()
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description="MeshBench workbench CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="strongly validate a rule file")
    p_validate.add_argument("--rules", default="samples/rules_v1.json")
    p_validate.set_defaults(func=cmd_validate)

    p_demo = sub.add_parser("demo", help="run the deterministic three-scenario demo")
    p_demo.add_argument("--rules", default="samples/rules_v1.json")
    p_demo.add_argument("--rules-v2", default="samples/rules_v2.json")
    p_demo.add_argument("--db", default="logs/meshbench.db")
    p_demo.add_argument("--snapshot", default="logs/session.snapshot.json")
    p_demo.set_defaults(func=cmd_demo)

    p_inject = sub.add_parser("inject", help="inject a structured fault")
    p_inject.add_argument("fault",
                          choices=["rule-corrupt", "upstream-down", "snapshot-corrupt"])
    p_inject.add_argument("--rules", default="samples/rules_v1.json")
    p_inject.add_argument("--db", default="logs/inject.db")
    p_inject.add_argument("--snapshot", default="logs/inject.snapshot.json")
    p_inject.set_defaults(func=cmd_inject)

    p_recover = sub.add_parser("recover", help="recover after a forced termination")
    p_recover.add_argument("--rules", default="samples/rules_v2.json")
    p_recover.add_argument("--rules-version", type=int, default=2)
    p_recover.add_argument("--db", default="logs/meshbench.db")
    p_recover.set_defaults(func=cmd_recover)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = StructuredLogger("logs/meshbench.log")
    try:
        return args.func(args, logger)
    except BenchError as exc:
        logger.log("error", "cli_error", **exc.as_dict())
        print(json.dumps({"error": exc.as_dict()}, sort_keys=True, ensure_ascii=False),
              file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
