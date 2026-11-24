"""Command line interface and fault-injection entry point."""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, replace
from typing import Any

from .codec import resolve_stream_path
from .config import load_engine_config, load_source_set
from .diagnostics.viewmodel import build_window_rows, render_text_table
from .engine import ObsFlowEngine
from .errors import ObsFlowError
from .injection import (
    inject_order_regression,
    inject_snapshot_digest,
    inject_truncation,
)
from .logging_utils import StructuredLogger
from .sample_data import generate_samples


def _build_engine(args: argparse.Namespace,
                  ledger_path: str | None = None) -> ObsFlowEngine:
    source_set = load_source_set(args.declarations)
    engine_config = load_engine_config(args.params)
    return ObsFlowEngine(engine_config, source_set,
                         ledger_path=ledger_path)


def _window_dicts(engine: ObsFlowEngine) -> list[dict[str, Any]]:
    return [asdict(w) for w in engine.query_windows()]


def cmd_generate(args: argparse.Namespace) -> int:
    result = generate_samples(args.output_dir, seed=args.seed,
                              records_per_source=args.records)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    logger = StructuredLogger(args.log)
    ledger_path = os.path.join(args.workdir, "obsflow.db")
    os.makedirs(args.workdir, exist_ok=True)
    engine = _build_engine(args, ledger_path)
    engine.read_records()
    report = engine.advance()
    rows = build_window_rows(engine.query_windows(), engine.lease_manager)
    print(render_text_table(rows))
    print(f"batch={report.batch_id} recomputed={len(report.recomputed)} "
          f"reused={len(report.reused)} unknown={len(report.unknown)}")
    logger.info("run_complete",
                windows=len(report.windows),
                recomputed=len(report.recomputed),
                reused=len(report.reused))
    engine.close()
    return 0


def cmd_recover(args: argparse.Namespace) -> int:
    logger = StructuredLogger(args.log)
    ledger_path = os.path.join(args.workdir, "obsflow.db")
    engine = _build_engine(args, ledger_path)
    windows, unknown = engine.recover()
    rows = build_window_rows(engine.query_windows(), engine.lease_manager)
    print(render_text_table(rows))
    print(f"recovered windows={len(windows)} unknown={len(unknown)}")
    logger.info("recover_complete", windows=len(windows),
                unknown=list(unknown))
    if args.rerun:
        engine.read_records()
        engine.advance()
        engine.replay_verify()
        print("rerun + replay verification OK")
    engine.close()
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    logger = StructuredLogger(args.log)
    ledger_path = os.path.join(args.workdir, "obsflow.db")
    engine = _build_engine(args, ledger_path)
    engine.recover()
    engine.read_records()
    replayed = engine.replay_verify()
    print(f"replay verified {len(replayed)} windows field by field")
    logger.info("replay_verified", windows=len(replayed))
    engine.close()
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    ledger_path = os.path.join(args.workdir, "obsflow.db")
    engine = _build_engine(args, ledger_path)
    if os.path.exists(ledger_path):
        engine.recover()
    engine.read_records()
    engine.advance()
    try:
        from .diagnostics.tk_ui import launch_tk_ui

        launch_tk_ui(engine)
    except RuntimeError:
        rows = build_window_rows(engine.query_windows(),
                                 engine.lease_manager)
        print("Tk unavailable in this runtime - headless table:")
        print(render_text_table(rows))
    engine.close()
    return 0


def cmd_inject(args: argparse.Namespace) -> int:
    logger = StructuredLogger(args.log)
    os.makedirs(args.workdir, exist_ok=True)
    ledger_path = os.path.join(args.workdir, "obsflow.db")
    try:
        if args.fault == "truncate":
            source_set = load_source_set(args.declarations)
            target = source_set.sources[0]
            original = resolve_stream_path(source_set.base_dir,
                                           target.file_path)
            work = os.path.join(args.workdir, "truncated.jsonl")
            inject_truncation(original, work)
            target = replace(target, file_path=os.path.abspath(work))
            source_set = source_set.__class__(
                source_set_version=source_set.source_set_version,
                sources=(target,),
                base_dir=source_set.base_dir)
            config = load_engine_config(args.params)
            engine = ObsFlowEngine(config, source_set,
                                   ledger_path=ledger_path)
            engine.read_records()
        elif args.fault == "order":
            source_set = load_source_set(args.declarations)
            target = source_set.sources[0]
            original = resolve_stream_path(source_set.base_dir,
                                           target.file_path)
            work = os.path.join(args.workdir, "regressed.jsonl")
            inject_order_regression(original, work)
            target = replace(target, file_path=os.path.abspath(work))
            source_set = source_set.__class__(
                source_set_version=source_set.source_set_version,
                sources=(target,),
                base_dir=source_set.base_dir)
            config = load_engine_config(args.params)
            engine = ObsFlowEngine(config, source_set,
                                   ledger_path=ledger_path)
            engine.read_records()
        elif args.fault == "snapshot":
            engine = _build_engine(args, ledger_path)
            engine.read_records()
            engine.advance()
            engine.close()
            inject_snapshot_digest(ledger_path)
            engine = _build_engine(args, ledger_path)
            identity = engine.current_identity([])
            engine.ledger.verify_content_digest(identity)
        else:  # pragma: no cover - argparse enforces choices
            raise ValueError(f"unknown fault {args.fault}")
    except ObsFlowError as exc:
        entry = logger.error("injected_fault", fault=args.fault,
                            **exc.as_dict())
        print(json.dumps(entry, indent=2, sort_keys=True))
        return 2
    print("injection did not raise (unexpected)")
    return 1


def cmd_crash(args: argparse.Namespace) -> int:
    from .process_runner import launch_crashed_worker

    logger = StructuredLogger(args.log)
    os.makedirs(args.workdir, exist_ok=True)
    ledger_path = os.path.join(args.workdir, "obsflow.db")
    engine = _build_engine(args, ledger_path)
    engine.read_records()
    engine.advance()
    result = launch_crashed_worker(ledger_path)
    print(f"worker killed returncode={result.returncode}")
    windows, unknown = engine.recover()
    print(f"after recover: windows={len(windows)} unknown={len(unknown)}")
    logger.info("crash_recover", killed_returncode=result.returncode,
                unknown=list(unknown))
    engine.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="obsflow")
    parser.add_argument("--log", default=os.path.join("logs", "obsflow.log"))
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate deterministic samples")
    gen.add_argument("--output-dir", default=os.path.join("samples", "local"))
    gen.add_argument("--seed", type=int, default=20240517)
    gen.add_argument("--records", type=int, default=24)
    gen.set_defaults(func=cmd_generate)

    def add_common(child: argparse.ArgumentParser) -> None:
        child.add_argument("--declarations",
                           default=os.path.join("samples", "local",
                                                "sources.json"))
        child.add_argument("--params",
                           default=os.path.join("samples", "local",
                                                "window_params.json"))
        child.add_argument("--workdir", default=os.path.join("artifacts",
                                                              "run"))

    run = sub.add_parser("run", help="load, aggregate and persist a batch")
    add_common(run)
    run.set_defaults(func=cmd_run)

    rec = sub.add_parser("recover", help="recover ledger after a crash")
    add_common(rec)
    rec.add_argument("--rerun", action="store_true")
    rec.set_defaults(func=cmd_recover)

    rep = sub.add_parser("replay", help="field-by-field replay verification")
    add_common(rep)
    rep.set_defaults(func=cmd_replay)

    ui = sub.add_parser("ui", help="diagnostics UI (Tk or headless table)")
    add_common(ui)
    ui.set_defaults(func=cmd_ui)

    inj = sub.add_parser("inject", help="fault injection chain")
    add_common(inj)
    inj.add_argument("fault", choices=("truncate", "order", "snapshot"))
    inj.set_defaults(func=cmd_inject)

    crash = sub.add_parser("crash", help="subprocess force-kill + recovery")
    add_common(crash)
    crash.set_defaults(func=cmd_crash)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
