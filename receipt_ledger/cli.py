"""命令行入口：run / resume / preview / recompute / ui / verify。"""

import argparse
import json
import sys

from .errors import LedgerError
from .ledger import Ledger
from .merge import build_batch_plan
from .preview import preview, recompute, rederive_conclusions
from .scheduler import Scheduler, WorkerPool
from .spec import load_spec


def _load(args):
    return load_spec(
        args.workers, args.requests, args.rules,
        spec_version=args.spec_version,
        global_concurrency=args.global_concurrency,
        per_worker_concurrency=args.per_worker_concurrency,
    )


def cmd_run(args):
    spec = _load(args)
    with Ledger(args.db) as ledger:
        diff = ledger.register_spec(spec)
        plan = build_batch_plan(spec)
        state = ledger.recovery_state(spec, plan)
        if not state["unknown"]:
            print(json.dumps({"status": "already-complete",
                              "batches": len(plan)}, ensure_ascii=False))
            return
        with WorkerPool(spec) as pool:
            scheduler = Scheduler(
                spec, ledger, pool,
                crash_after_commits=args.crash_after_commits,
            )
            scheduler.run(plan, start_seq=state["resume_from_seq"])
        ledger.verify_chain(spec.spec_version)
        result = {"status": "committed", "batches": len(plan), "diff": diff}
        print(json.dumps(result, ensure_ascii=False))


def cmd_resume(args):
    spec = _load(args)
    with Ledger(args.db) as ledger:
        ledger.register_spec(spec)
        plan = build_batch_plan(spec)
        state = ledger.recovery_state(spec, plan)
        if state["unknown"]:
            with WorkerPool(spec) as pool:
                scheduler = Scheduler(spec, ledger, pool)
                scheduler.run(plan, start_seq=state["resume_from_seq"])
        ledger.verify_chain(spec.spec_version)
        rederived = rederive_conclusions(spec, plan)
        ledger.verify_replay(spec.spec_version, rederived)
        print(json.dumps(
            {"status": "recovered", "state": state,
             "replay": "identical"}, ensure_ascii=False))


def cmd_preview(args):
    spec = _load(args)
    with Ledger(args.db) as ledger:
        print(json.dumps(preview(ledger, spec), ensure_ascii=False,
                         indent=2 if args.pretty else None))


def cmd_recompute(args):
    spec = _load(args)
    with Ledger(args.db) as ledger:
        print(json.dumps(recompute(ledger, spec), ensure_ascii=False, indent=2))


def cmd_verify(args):
    spec = _load(args)
    with Ledger(args.db) as ledger:
        plan = build_batch_plan(spec)
        state = ledger.recovery_state(spec, plan)
        ledger.verify_chain(spec.spec_version)
        rederived = rederive_conclusions(spec, plan)
        committed = {bid: rederived[bid] for bid in state["committed"]}
        ledger.verify_replay(spec.spec_version, committed)
        print(json.dumps(
            {"chain": "ok", "recovery": state,
             "latency_preview": ledger.latency_stats("preview"),
             "latency_recompute": ledger.latency_stats("recompute")},
            ensure_ascii=False, indent=2))


def cmd_ui(args):
    spec = _load(args)
    with Ledger(args.db) as ledger:
        from .ui import launch_ui
        launch_ui(ledger, spec)


def build_parser():
    parser = argparse.ArgumentParser(prog="receipt-ledger")
    parser.add_argument("--workers", required=True)
    parser.add_argument("--requests", required=True)
    parser.add_argument("--rules", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--spec-version", type=int, default=1)
    parser.add_argument("--global-concurrency", type=int, default=2)
    parser.add_argument("--per-worker-concurrency", type=int, default=1)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--crash-after-commits", type=int, default=None)
    run.set_defaults(func=cmd_run)
    resume = sub.add_parser("resume")
    resume.set_defaults(func=cmd_resume)
    pv = sub.add_parser("preview")
    pv.add_argument("--pretty", action="store_true")
    pv.set_defaults(func=cmd_preview)
    rc = sub.add_parser("recompute")
    rc.set_defaults(func=cmd_recompute)
    vf = sub.add_parser("verify")
    vf.set_defaults(func=cmd_verify)
    ui = sub.add_parser("ui")
    ui.set_defaults(func=cmd_ui)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except LedgerError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (RuntimeError, ConnectionError) as exc:
        print(f"[E_RUNTIME] {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
