"""命令行入口：validate / run / recover / replay。

所有路径、槽位参数覆盖与异常注入均通过显式命令行参数提供。
"""

import argparse
import json
import os
import sys

from .digest import dump_canonical, content_digest
from .clock import MonotonicClock, FakeClock
from .engine import Engine
from .errors import EngineError
from .logging_util import StructuredLogger
from .replay import replay_and_compare, verify_snapshot, verify_digests
from .samplegen import generate
from .spec import load_spec
from .ledger import Ledger


def _apply_overrides(spec_raw, args):
    topo = spec_raw["topology"]
    if args.max_connections is not None:
        topo["global_max_connections"] = args.max_connections
    if args.key_cap is not None:
        for part in topo["partitions"]:
            part["key_cap_per_connection"] = args.key_cap
    if args.idle_ms is not None:
        for part in topo["partitions"]:
            part["idle_recycle_ms"] = args.idle_ms
    if args.partition_count is not None:
        from .hashing import validate_partition_count
        validate_partition_count(args.partition_count)
        old = topo["partition_count"]
        topo["partition_count"] = args.partition_count
        if len(topo["partitions"]) == old:
            template = topo["partitions"][0]
            topo["partitions"] = [
                dict(template, partition_id=i) for i in range(args.partition_count)
            ]
        topo["borrows"] = [
            b for b in topo.get("borrows", [])
            if b["lender"] < args.partition_count and b["borrower"] < args.partition_count
        ]
    if args.batch_size is not None:
        spec_raw["batch_size"] = args.batch_size
    if args.hotspot_threshold is not None:
        spec_raw.setdefault("hotspot", {})["row_threshold"] = args.hotspot_threshold


def _load_with_overrides(args, allow_cycle=False):
    with open(args.sample, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    _apply_overrides(raw, args)
    if args.inject_borrow_cycle:
        borrows = raw["topology"].setdefault("borrows", [])
        if borrows:
            first = borrows[0]
            reverse = {"lender": first["borrower"], "borrower": first["lender"],
                       "borrowed_key_cap": first["borrowed_key_cap"],
                       "deadline_ms": first["deadline_ms"]}
            borrows.append(reverse)
    overridden = os.path.join(args.out_dir, "effective_spec.json")
    os.makedirs(args.out_dir, exist_ok=True)
    dump_canonical(overridden, raw)
    spec = load_spec(
        overridden,
        validate_borrow_acyclic=not (allow_cycle and args.inject_borrow_cycle))
    return spec


def cmd_validate(args):
    try:
        spec = _load_with_overrides(args, allow_cycle=False)
    except EngineError as exc:
        print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 2
    dump_canonical(os.path.join(args.out_dir, "validate_ok.json"),
                   {"status": "ok", "tables": [t.name for t in spec.tables],
                    "partition_count": spec.topology.partition_count})
    print("E_SPEC check passed")
    return 0


def _make_clock(args):
    if args.virtual_clock:
        return FakeClock(start_ms=0)
    return MonotonicClock()


def cmd_run(args):
    logger = StructuredLogger(args.out_dir)
    try:
        spec = _load_with_overrides(args, allow_cycle=True)
        engine = Engine(
            spec, args.out_dir, _make_clock(args), workers=args.workers,
            inject_cycle=args.inject_borrow_cycle,
            inject_deadline=args.inject_early_deadline,
            inject_digest=args.inject_digest_mismatch,
            crash_after_batch=args.crash_after_batch,
        )
        logger.log("INFO", "run_start", sample=args.sample,
                   partition_count=spec.topology.partition_count,
                   workers=args.workers)
        plan, replaced = engine.run()
        report = engine.finalize(plan)
        report["replaced_snapshot_items"] = replaced
        report["slot_params_version"] = spec.slot_params_version
        dump_canonical(os.path.join(args.out_dir, "report.json"), report)
        dump_canonical(os.path.join(args.out_dir, "mapping_summary.json"),
                       report["mapping_summary"])
        dump_canonical(os.path.join(args.out_dir, "watermark.json"),
                       report["watermark"])
        logger.log("INFO", "run_complete",
                   result_digest=report["result_digest"],
                   hot_partitions=report["mapping_summary"]["hot_partitions"])
        engine.close()
        print(json.dumps({"status": "ok",
                          "result_digest": report["result_digest"]},
                         ensure_ascii=False, sort_keys=True))
        return 0
    except EngineError as exc:
        logger.log("ERROR", "engine_error", **exc.to_dict())
        print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 3
    finally:
        logger.close()


def cmd_recover(args):
    logger = StructuredLogger(args.out_dir)
    try:
        spec = _load_with_overrides(args, allow_cycle=True)
        ledger = Ledger(os.path.join(args.out_dir, "ledger.sqlite"))
        snap = verify_snapshot(ledger, spec)
        verify_digests(ledger, args.inject_digest_mismatch)
        crash_events = ledger.crash_events()
        last_batch = ledger.last_batch_id()
        ledger.close()
        replay = replay_and_compare(spec, args.out_dir, _make_clock(args),
                                    args.inject_digest_mismatch)
        engine = Engine(spec, args.out_dir, _make_clock(args), workers=args.workers)
        engine.seed_slot_sequence(last_batch)
        engine.hydrate_aggregators_from_ledger(last_batch)
        plan, replaced = engine.run(task_start=last_batch + 1)
        report = engine.finalize(plan)
        report["recovery"] = {
            "last_committed_batch": last_batch,
            "crash_events": crash_events,
            "replaced_snapshot_items": replaced,
            "snapshot": snap,
            "replay": replay,
        }
        dump_canonical(os.path.join(args.out_dir, "recovery_report.json"), report)
        dump_canonical(os.path.join(args.out_dir, "report.json"), report)
        logger.log("INFO", "recover_complete", last_batch=last_batch,
                   unknown=[c for e in crash_events for c in e["connections"]])
        engine.close()
        dump_canonical(os.path.join(args.out_dir, "replay_report.json"), replay)
        print(json.dumps({"status": "recovered", "replay": replay},
                         ensure_ascii=False, sort_keys=True))
        return 0
    except EngineError as exc:
        logger.log("ERROR", "engine_error", **exc.to_dict())
        print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 3
    finally:
        logger.close()


def cmd_replay(args):
    logger = StructuredLogger(args.out_dir)
    try:
        spec = _load_with_overrides(args, allow_cycle=True)
        replay = replay_and_compare(spec, args.out_dir, _make_clock(args),
                                    args.inject_digest_mismatch)
        dump_canonical(os.path.join(args.out_dir, "replay_report.json"), replay)
        print(json.dumps({"status": "replay_ok", **replay},
                         ensure_ascii=False, sort_keys=True))
        return 0
    except EngineError as exc:
        logger.log("ERROR", "engine_error", **exc.to_dict())
        print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 3
    finally:
        logger.close()


def build_parser():
    parser = argparse.ArgumentParser(description="哈希分组聚合连接水位诊断引擎")
    parser.add_argument("--sample", default="samples/skew_sample.json")
    parser.add_argument("--out-dir", dest="out_dir", default="out")
    parser.add_argument("--max-connections", type=int)
    parser.add_argument("--key-cap", type=int)
    parser.add_argument("--idle-ms", type=int)
    parser.add_argument("--partition-count", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--hotspot-threshold", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--virtual-clock", action="store_true")
    parser.add_argument("--inject-borrow-cycle", action="store_true")
    parser.add_argument("--inject-early-deadline", action="store_true")
    parser.add_argument("--inject-digest-mismatch", action="store_true")
    parser.add_argument("--crash-after-batch", type=int)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate").set_defaults(func=cmd_validate)
    sub.add_parser("run").set_defaults(func=cmd_run)
    sub.add_parser("recover").set_defaults(func=cmd_recover)
    sub.add_parser("replay").set_defaults(func=cmd_replay)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
