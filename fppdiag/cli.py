"""Command line interface for the fixed-point pipeline diagnostics tool."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .errors import DiagError
from .orchestrator import Orchestrator
from .samplegen import write_sample
from .spec import load_spec_file
from .util import StructuredLogger, canonical_json


def _print(obj) -> None:
    print(canonical_json(obj))


def _logger(workdir: str) -> StructuredLogger:
    return StructuredLogger(os.path.join(workdir, "events.log.jsonl"))


def cmd_init(args) -> int:
    path = write_sample(args.outdir, rows=args.rows, version=args.version)
    _print({"written": path, "rows": args.rows, "version": args.version})
    return 0


def cmd_validate(args) -> int:
    spec = load_spec_file(args.spec)
    _print({"ok": True, "spec_version": spec.version,
            "stages": [s.stage_id for s in spec.stages],
            "sources": sorted(spec.sources), "digest": spec.digest()})
    return 0


def _load_orchestrator(args) -> Orchestrator:
    spec = load_spec_file(args.spec)
    return Orchestrator(spec, args.workdir, schedule=args.schedule)


def cmd_run(args) -> int:
    log = _logger(args.workdir)
    try:
        orch = _load_orchestrator(args)
        report = orch.run_full()
        log.event("INFO", "run_full", spec_version=orch.spec.version,
                  features=len(report["features"]))
        _print(report)
        orch.close()
        return 0
    except DiagError as exc:
        log.event("ERROR", exc.code, **exc.to_dict())
        _print(exc.to_dict())
        return _exit_for(exc.code)


def cmd_recover(args) -> int:
    log = _logger(args.workdir)
    try:
        orch = _load_orchestrator(args)
        report = orch.recover()
        log.event("INFO", "recover", spec_version=orch.spec.version)
        _print(report)
        orch.close()
        return 0
    except DiagError as exc:
        log.event("ERROR", exc.code, **exc.to_dict())
        _print(exc.to_dict())
        return _exit_for(exc.code)


def cmd_crash(args) -> int:
    """Run until a stage, leave its batch uncommitted, then recover."""
    log = _logger(args.workdir)
    try:
        orch = _load_orchestrator(args)
        crashed = orch.run_full(crash_after_stage=args.stage)
        log.event("WARN", "forced_termination", **crashed)
        # New process: reopen and recover from committed batches only.
        orch.close()
        orch2 = _load_orchestrator(args)
        report = orch2.recover()
        log.event("INFO", "post_crash_recover", crashed=crashed["crashed"])
        _print({"crashed": crashed, "recovered": report})
        orch2.close()
        return 0
    except DiagError as exc:
        log.event("ERROR", exc.code, **exc.to_dict())
        _print(exc.to_dict())
        return _exit_for(exc.code)


def cmd_verify(args) -> int:
    """Compare an incremental recompute against a from-scratch full recompute."""
    log = _logger(args.workdir)
    try:
        import shutil
        spec = load_spec_file(args.spec)
        # Fresh full recompute in an isolated workdir.
        full_dir = os.path.join(args.workdir, "_full_shadow")
        shutil.rmtree(full_dir, ignore_errors=True)
        full_orch = Orchestrator(spec, full_dir, schedule=args.schedule)
        full_report = full_orch.run_full()
        # Incremental run against the persistent ledger.
        inc_orch = Orchestrator(spec, args.workdir, schedule=args.schedule)
        inc_report = inc_orch.run_full()
        inc_orch.verify_incremental(full_report, inc_report)
        log.event("INFO", "verify_incremental", match=True)
        _print({"match": True, "features": len(full_report["features"])})
        full_orch.close()
        inc_orch.close()
        return 0
    except DiagError as exc:
        log.event("ERROR", exc.code, **exc.to_dict())
        _print(exc.to_dict())
        return _exit_for(exc.code)


def cmd_inject(args) -> int:
    """Inject one fault by mutating an input file/param only (source untouched)."""
    log = _logger(args.workdir)
    try:
        if args.fault == "data_corrupt":
            _inject_data_corrupt(args, log)
        elif args.fault == "param_overflow":
            _inject_param_overflow(args, log)
        elif args.fault == "snapshot_digest":
            _inject_snapshot_digest(args, log)
        else:  # pragma: no cover
            raise SystemExit(2)
        return 0
    except DiagError as exc:
        log.event("ERROR", exc.code, **exc.to_dict())
        _print(exc.to_dict())
        return _exit_for(exc.code)


def _inject_data_corrupt(args, log) -> None:
    from .sources import column_path
    orch = _load_orchestrator(args)
    orch.engine.load_sources()  # ensure cache files exist
    sid = args.source or next(iter(orch.spec.sources))
    path = column_path(args.workdir, sid)
    with open(path, "r+b") as fh:
        fh.seek(len(b"FPPCOL01") + 2 + 5)
        fh.write(b"\xff\xff\xff\xff")  # flip payload bytes, digest now invalid
    log.event("WARN", "inject", fault="data_corrupt", source=sid)
    orch.close()
    # A fresh process reading the corrupted cache must raise E_DATA.
    orch2 = _load_orchestrator(args)
    orch2.engine.load_sources()


def _inject_param_overflow(args, log) -> None:
    import dataclasses
    spec = load_spec_file(args.spec)
    stage = spec.stage(args.stage or "std_price")
    log.event("WARN", "inject", fault="param_overflow", stage=stage.stage_id)
    # Mutate an input parameter only (source code untouched). The JSON loader
    # rejects this structurally; here the tampered parameter reaches the engine
    # runtime envelope check, which must raise E_PARAM.
    tampered = dataclasses.replace(stage, params={**stage.params, "sd": 0})
    stages = [tampered if s.stage_id == stage.stage_id else s
              for s in spec.stages]
    spec = dataclasses.replace(spec, stages=stages)
    orch = Orchestrator(spec, args.workdir)
    orch.run_full()


def _inject_snapshot_digest(args, log) -> None:
    orch = _load_orchestrator(args)
    orch.run_full()
    orch.close()
    snap_path = os.path.join(args.workdir, "snapshot.json")
    with open(snap_path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    doc["digest"] = "0" * 64  # tamper the stored content digest
    with open(snap_path, "w", encoding="utf-8") as fh:
        fh.write(canonical_json(doc))
    log.event("WARN", "inject", fault="snapshot_digest")
    # Reopen: digest mismatch must raise E_SNAPSHOT before any merge.
    _load_orchestrator(args)


def _exit_for(code: str) -> int:
    return {"E_SPEC": 10, "E_DATA": 11, "E_PARAM": 12,
            "E_DIVERGE": 13, "E_REPLAY": 14, "E_SNAPSHOT": 15}.get(code, 1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fppdiag",
                                description="Fixed-point feature pipeline diagnostics")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("init-sample")
    sp.add_argument("--outdir", default="samples")
    sp.add_argument("--rows", type=int, default=240)
    sp.add_argument("--version", type=int, default=1)
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("validate")
    sp.add_argument("--spec", required=True)
    sp.set_defaults(func=cmd_validate)

    def add_common(sp):
        sp.add_argument("--spec", required=True)
        sp.add_argument("--workdir", required=True)
        sp.add_argument("--schedule", choices=["stage_major", "pipeline_major"],
                        default="stage_major")

    sp = sub.add_parser("run"); add_common(sp); sp.set_defaults(func=cmd_run)
    sp = sub.add_parser("recover"); add_common(sp); sp.set_defaults(func=cmd_recover)
    sp = sub.add_parser("verify"); add_common(sp); sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("crash")
    add_common(sp)
    sp.add_argument("--stage", required=True)
    sp.set_defaults(func=cmd_crash)

    sp = sub.add_parser("inject")
    add_common(sp)
    sp.add_argument("--fault", required=True,
                    choices=["data_corrupt", "param_overflow", "snapshot_digest"])
    sp.add_argument("--source")
    sp.add_argument("--stage")
    sp.set_defaults(func=cmd_inject)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except DiagError as exc:
        _print(exc.to_dict())
        return _exit_for(exc.code)


if __name__ == "__main__":
    sys.exit(main())
