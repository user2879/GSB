"""Command line interface for the kvdiag diagnostic component."""

import argparse
import json
import sys
from pathlib import Path

from .errors import KvDiagError
from .log import StructuredLogger


def _logger_for(args):
    path = getattr(args, "log", None) or "artifacts/kvdiag.log.jsonl"
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return StructuredLogger(path)


def _emit_error(logger, err):
    logger.emit_error(err)
    print(json.dumps(err.to_dict(), sort_keys=True, ensure_ascii=False))
    return 1


def cmd_gen_samples(args):
    from .samples import write_example_tree
    decl = write_example_tree(args.out, seed=args.seed)
    print(json.dumps({"written": args.out, "seed": args.seed,
                      "shards": len(decl["shards"])}, sort_keys=True))
    return 0


def cmd_run(args):
    from .pipeline import run_pipeline
    logger = _logger_for(args)
    try:
        report = run_pipeline(
            args.spec, args.db, renewals_path=args.renewals,
            session_out=args.session_out, batch_size=args.batch_size,
            max_workers=args.max_workers, per_shard_quota=args.per_shard_quota,
            crash_before_batch=args.crash_before_batch,
            only_shards=args.shards, logger=logger)
    except KvDiagError as err:
        return _emit_error(logger, err)
    print(json.dumps({"status": "ok", "replay_ok": report["replay_ok"],
                      "batches": report["batches"]}, sort_keys=True))
    return 0


def cmd_replay(args):
    from .ledger import Ledger
    from .pipeline import replay_verify
    from .spec import load_spec
    logger = _logger_for(args)
    try:
        spec = load_spec(args.spec)
        ledger = Ledger(args.db)
        ok = replay_verify(ledger, spec)
        ledger.close()
    except KvDiagError as err:
        return _emit_error(logger, err)
    print(json.dumps({"replay_ok": ok}, sort_keys=True))
    return 0 if ok else 1


def cmd_recover(args):
    from .ledger import Ledger
    from .pipeline import run_pipeline
    from .recovery import recover
    from .spec import load_spec
    logger = _logger_for(args)
    try:
        spec = load_spec(args.spec)
        ledger = Ledger(args.db)
        report = recover(spec, ledger, args.session, logger=logger)
        ledger.close()
        if report["unknown_shards"] and args.rerun_unknown:
            run_pipeline(args.spec, args.db, renewals_path=args.renewals,
                         session_out=args.session,
                         only_shards=report["unknown_shards"], logger=logger)
            report["rerun"] = report["unknown_shards"]
    except KvDiagError as err:
        return _emit_error(logger, err)
    print(json.dumps(report, sort_keys=True))
    return 0


def cmd_inject(args):
    from . import inject
    logger = _logger_for(args)
    if args.kind == "sample-corrupt":
        result = inject.inject_sample_corrupt(args.spec, args.shard)
        code = "E_SPEC"
    elif args.kind == "lease-version-regression":
        result = inject.inject_lease_version_regression(args.renewals)
        code = "E_LEASE"
    else:
        result = inject.inject_ledger_digest_mismatch(args.session)
        code = "E_SNAPSHOT"
    logger.emit("fault_injected", code=code, shard_id=getattr(args, "shard", None),
                kind=args.kind, **result)
    print(json.dumps({"injected": args.kind, "expects": code, **result},
                     sort_keys=True))
    return 0


def cmd_ui(args):
    try:
        from .ui import run_ui
        run_ui(spec_path=args.spec, db_path=args.db,
               log_path=getattr(args, "log", None) or "artifacts/kvdiag.log.jsonl")
    except ImportError as exc:
        print(json.dumps({"code": "E_UI",
                          "message": "tkinter unavailable: %s" % exc}))
        return 1
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="kvdiag",
        description="Key-value index shard and lease-expiry diagnostics")
    parser.add_argument("--log", default=None,
                        help="structured log path (JSON lines)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("gen-samples", help="write deterministic example tree")
    p.add_argument("--out", default="examples")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_gen_samples)

    p = sub.add_parser("run", help="run the batch pipeline")
    p.add_argument("--spec", required=True)
    p.add_argument("--db", required=True)
    p.add_argument("--renewals", default=None)
    p.add_argument("--session-out", default=None)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-workers", type=int, default=2)
    p.add_argument("--per-shard-quota", type=int, default=1)
    p.add_argument("--crash-before-batch", type=int, default=None)
    p.add_argument("--shards", nargs="*", default=None)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("replay", help="verify snapshot replay vs direct rebuild")
    p.add_argument("--spec", required=True)
    p.add_argument("--db", required=True)
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("recover", help="session-snapshot recovery")
    p.add_argument("--spec", required=True)
    p.add_argument("--db", required=True)
    p.add_argument("--session", required=True)
    p.add_argument("--renewals", default=None)
    p.add_argument("--rerun-unknown", action="store_true")
    p.set_defaults(func=cmd_recover)

    p = sub.add_parser("inject", help="inject a fault into input files")
    p.add_argument("kind", choices=["sample-corrupt", "lease-version-regression",
                                    "ledger-digest-mismatch"])
    p.add_argument("--spec", default="examples/keyspace.json")
    p.add_argument("--shard", default="shard-1")
    p.add_argument("--renewals", default="examples/renewals.json")
    p.add_argument("--session", default="artifacts/session_snapshot.json")
    p.set_defaults(func=cmd_inject)

    p = sub.add_parser("ui", help="launch the Tkinter diagnostic UI")
    p.add_argument("--spec", default="examples/keyspace.json")
    p.add_argument("--db", default="artifacts/kvdiag.sqlite")
    p.set_defaults(func=cmd_ui)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
