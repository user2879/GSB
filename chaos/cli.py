"""Command line entry point.

Subcommands:
    run     execute an experiment (optionally simulate forced termination)
    resume  recover from the ledger, mark unknowns, redo and continue
    verify  replay-check the ledger against the declaration + seed
    fault   inject one of the three anomaly classes
    ui      launch the Tkinter diagnostic interface

Exit codes: 0 ok, 2 E_SPEC, 3 E_SEED, 4 E_SNAPSHOT, 5 E_REVERT, 6 E_REPLAY.
"""

import argparse
import json
import os
import sys

from . import faults, recover, replay, timeline
from .errors import EXIT_CODES, E_SEED, ChaosError
from .orchestrator import Orchestrator, check_observation_isolation
from .ports import resolve_ports
from .spec import load_spec
from .structlog import StructLog


def _log_path(args):
    os.makedirs(args.log_dir, exist_ok=True)
    return os.path.join(args.log_dir, "run.jsonl")


def _load_resolved_spec(args, log):
    spec = load_spec(args.spec)
    needs_ports = any(s.port == 0 or s.control_port == 0 for s in spec.services)
    if needs_ports:
        out = args.resolved_spec or (str(args.ledger) + ".resolved.json")
        resolve_ports(spec.to_dict(), existing_path=out)
        log.emit("ports_resolved", resolved_spec=out)
        spec = load_spec(out)
    return spec


def _prepare(args):
    log = StructLog(_log_path(args))
    spec = _load_resolved_spec(args, log)
    entries = timeline.generate(spec)
    tdigest = timeline.timeline_digest(entries)
    return log, spec, entries, tdigest


def cmd_run(args):
    log, spec, entries, tdigest = _prepare(args)
    ledger, report = recover.reconcile(spec, tdigest, args.ledger,
                                       args.snapshot_dir, log)
    # uncommitted rows (including recovery-marked unknowns) are (re)done;
    # committed seqs are skipped automatically
    orch = Orchestrator(spec, ledger, entries, log,
                        kill_after=args.kill_after)
    result = orch.run()
    violations = check_observation_isolation(ledger)
    summary = {"recovery": report.to_dict(), "run": result,
               "isolation_violations": violations}
    log.emit("run_summary", **summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    ledger.close()
    return 0 if not result["errors"] and not violations else 1


def cmd_resume(args):
    log, spec, entries, tdigest = _prepare(args)
    ledger, report = recover.reconcile(spec, tdigest, args.ledger,
                                       args.snapshot_dir, log)
    orch = Orchestrator(spec, ledger, entries, log)
    result = orch.run()
    count = replay.replay(spec, entries, ledger)
    violations = check_observation_isolation(ledger)
    summary = {"recovery": report.to_dict(), "run": result,
               "replayed_injections": count,
               "isolation_violations": violations}
    log.emit("resume_summary", **summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    ledger.close()
    return 0 if not result["errors"] and not violations else 1


def cmd_verify(args):
    log, spec, entries, tdigest = _prepare(args)
    from .ledger import Ledger
    ledger = Ledger(args.ledger)
    try:
        meta = ledger.meta_dict()
        if meta.get("timeline_digest") != tdigest or \
                meta.get("seed") != str(spec.seed):
            raise ChaosError(
                E_SEED, "seed/timeline digest does not match ledger",
                field="meta.timeline_digest",
                evidence={"ledger": meta.get("timeline_digest"),
                          "session": tdigest})
        if not ledger.verify_digest():
            from .errors import E_SNAPSHOT
            raise ChaosError(E_SNAPSHOT, "ledger content digest mismatch",
                             field="meta.content_digest")
        count = replay.replay(spec, entries, ledger)
        violations = check_observation_isolation(ledger)
        summary = {"replayed_injections": count,
                   "isolation_violations": violations, "ok": not violations}
        log.emit("verify_summary", **summary)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if not violations else 1
    finally:
        ledger.close()


def cmd_fault(args):
    log = StructLog(_log_path(args))
    if args.fault == "corrupt-spec":
        info = faults.corrupt_spec(args.spec)
        code = "E_SPEC"
    elif args.fault == "seed-mismatch":
        info = faults.seed_mismatch(args.spec)
        code = "E_SEED"
    elif args.fault == "ledger-digest":
        info = faults.ledger_digest_mismatch(args.ledger)
        code = "E_SNAPSHOT"
    else:
        raise SystemExit("unknown fault " + args.fault)
    log.emit("fault_injected", expected_code=code, **info)
    print(json.dumps({"injected": info, "expected_code": code}, indent=2))
    return 0


def cmd_ui(args):
    from .ui import launch
    launch(args.ledger, args.snapshot_dir)
    return 0


def build_parser():
    ap = argparse.ArgumentParser(prog="chaos")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--spec", default="samples/experiment.json")
        p.add_argument("--ledger", default="ledger.sqlite")
        p.add_argument("--snapshot-dir", default="snapshots")
        p.add_argument("--log-dir", default="logs")
        p.add_argument("--resolved-spec", default=None)

    run = sub.add_parser("run")
    common(run)
    run.add_argument("--kill-after", type=int, default=None,
                     help="simulate forced termination after N commits")
    run.set_defaults(func=cmd_run)

    resume = sub.add_parser("resume")
    common(resume)
    resume.set_defaults(func=cmd_resume)

    verify = sub.add_parser("verify")
    common(verify)
    verify.set_defaults(func=cmd_verify)

    fault = sub.add_parser("fault")
    common(fault)
    fault.add_argument("fault",
                       choices=["corrupt-spec", "seed-mismatch", "ledger-digest"])
    fault.set_defaults(func=cmd_fault)

    ui = sub.add_parser("ui")
    common(ui)
    ui.set_defaults(func=cmd_ui)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ChaosError as exc:
        log = StructLog(_log_path(args))
        log.error(exc.code, exc.message, field=exc.field,
                  evidence=exc.evidence)
        print(json.dumps(exc.to_dict(), indent=2, sort_keys=True))
        return EXIT_CODES.get(exc.code, 1)


if __name__ == "__main__":
    sys.exit(main())
