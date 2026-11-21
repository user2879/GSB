"""Command-line interface for the slow-request profiler.

Subcommands:
  generate   deterministic NDJSON generation (delegates to slowprof.generator)
  profile    run a full profiling session with batched ledger commits
  recover    inspect durable batches after a forced kill
  resume     rerun the unknown tail after a kill and finalize
  replay     reload samples and verify persisted conclusions (E_REPLAY)
  diagnose   render diagnostics (Tkinter window when available, else text)
  inject-sample  produce a corrupted NDJSON copy (input-only fault injection)
  cross-check    prove byte-identical full vs streaming folding
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Sequence

from . import diagnostics, generator
from .errors import ProfilerError
from .folding import cross_check, fold_full, fold_streaming
from .samples import load_samples, make_interval
from .session import ForcedKill, ProfileSession, SessionConfig


def _config_from_args(args: argparse.Namespace) -> SessionConfig:
    return SessionConfig(
        sample_path=args.samples,
        ledger_path=args.ledger,
        snapshot_dir=args.snapshot_dir,
        start_ms=args.start_ms,
        end_ms=args.end_ms,
        scale=args.scale,
        top_k=args.top_k,
        batch_size=args.batch_size,
        max_unique_stacks=args.max_unique_stacks,
        max_total_weight=args.max_total_weight,
        force_path=args.force_path,
        interval_version=args.interval_version,
    )


def _print_error(exc: ProfilerError) -> int:
    payload = {"code": exc.code, "message": exc.message}
    if exc.field is not None:
        payload["field"] = exc.field
    if exc.line_number is not None:
        payload["line_number"] = exc.line_number
    print(json.dumps(payload, sort_keys=True), file=sys.stderr)
    return 2


def cmd_generate(args: argparse.Namespace) -> int:
    return generator.main([
        "--rows", str(args.rows),
        "--services", str(args.services),
        "--min-depth", str(args.min_depth),
        "--max-depth", str(args.max_depth),
        "--weight-mode", args.weight_mode,
        "--seed", str(args.seed),
        "--output", args.output,
    ])


def cmd_profile(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    session = ProfileSession(config)
    try:
        summary = session.run(
            crash_after_batches=args.crash_after_batches)
    except ForcedKill as exc:
        info = session.recover()
        print(json.dumps({
            "status": "killed",
            "reason": str(exc),
            "recoverable_batches": info.recoverable_batches,
            "last_committed_line": info.last_committed_line,
            "unknown_rerun_from_line": info.needs_rerun_from_line,
        }, indent=2))
        session.close()
        return 0
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.print_report:
        print()
        print(diagnostics.render_from_ledger(config, session))
    session.close()
    return 0


def cmd_recover(args: argparse.Namespace) -> int:
    session = ProfileSession(_config_from_args(args))
    info = session.recover()
    print(json.dumps(info.__dict__, indent=2, sort_keys=True))
    session.close()
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    session = ProfileSession(_config_from_args(args))
    session.recover()
    summary = session.resume()
    print(json.dumps(summary, indent=2, sort_keys=True))
    session.close()
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    session = ProfileSession(_config_from_args(args))
    try:
        outcome = session.replay()
    except ProfilerError as exc:
        return _print_error(exc)
    print(json.dumps(outcome, indent=2, sort_keys=True))
    session.close()
    return 0


def cmd_diagnose(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    # Strictly read-only: render from the persisted snapshot + ledger SELECTs.
    report = diagnostics.render_from_snapshot(
        config.snapshot_dir, config.ledger_path,
        make_interval(config.start_ms, config.end_ms))
    if args.text or not diagnostics.tkinter_available():
        if not args.text and not diagnostics.tkinter_available():
            print("[info] tkinter unavailable in this interpreter; "
                  "using text fallback\n", file=sys.stderr)
        print(report)
    else:
        opened = diagnostics.launch_gui(
            title="slowprof diagnostics", text_report=report)
        if not opened:
            print(report)
    return 0


def cmd_cross_check(args: argparse.Namespace) -> int:
    interval = make_interval(args.start_ms, args.end_ms)
    samples = load_samples(args.samples, interval)
    same, message = cross_check(
        samples, args.max_unique_stacks, args.max_total_weight)
    full_bytes = fold_full(
        samples, args.max_unique_stacks,
        args.max_total_weight).canonical_bytes()
    stream_bytes = fold_streaming(
        samples, args.max_unique_stacks,
        args.max_total_weight).canonical_bytes()
    print(json.dumps({
        "samples": len(samples),
        "byte_identical": same,
        "canonical_bytes_sha": __import__("hashlib").sha256(
            full_bytes).hexdigest(),
        "full_len": len(full_bytes),
        "streaming_len": len(stream_bytes),
        "detail": message,
    }, indent=2, sort_keys=True))
    return 0 if same else 1


def cmd_inject_sample(args: argparse.Namespace) -> int:
    """Fault injection that only touches the input file, never source code."""

    src = Path(args.samples)
    dst = Path(args.output)
    shutil.copyfile(src, dst)
    lines = dst.read_text(encoding="utf-8").splitlines(keepends=True)
    target = min(args.target_line, len(lines)) - 1
    original = json.loads(lines[target])
    if args.kind == "bad_json":
        lines[target] = '{"request_id": "broken", ,}\n'
        field = "<json>"
    elif args.kind == "bad_weight":
        original["weight"] = 0
        field = "weight"
        lines[target] = json.dumps(original) + "\n"
    elif args.kind == "bad_duration":
        original["duration_ms"] = -3
        field = "duration_ms"
        lines[target] = json.dumps(original) + "\n"
    elif args.kind == "bad_frame":
        original["frames"][-1] = "billing."
        field = "frames[" + str(len(original["frames"]) - 1) + "]"
        lines[target] = json.dumps(original) + "\n"
    elif args.kind == "missing_field":
        del original["service_id"]
        field = "service_id"
        lines[target] = json.dumps(original) + "\n"
    else:  # timestamp regression: mutate current row below previous row
        original["timestamp_ms"] = 1
        field = "timestamp_ms"
        lines[target] = json.dumps(original) + "\n"
    dst.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({
        "injected": args.kind,
        "output": str(dst),
        "line_number": target + 1,
        "expected_field": field,
        "expected_code": "E_SAMPLE",
    }, indent=2, sort_keys=True))
    return 0


def _strip(namespace: argparse.Namespace, names: Sequence[str]):
    data = dict(vars(namespace))
    for name in names:
        data.pop(name, None)
    return [f"--{k.replace('_', '-')}={v}" for k, v in data.items()
            if v is not None]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="slowprof")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="deterministic sample generation")
    gen.add_argument("--rows", type=int, default=5_000)
    gen.add_argument("--services", type=int, default=3)
    gen.add_argument("--min-depth", type=int, default=3)
    gen.add_argument("--max-depth", type=int, default=8)
    gen.add_argument(
        "--weight-mode",
        choices=("uniform", "skewed", "unit"), default="skewed")
    gen.add_argument("--seed", type=int, default=20261122)
    gen.add_argument("--output", type=str, default="data/samples.ndjson")
    gen.set_defaults(func=cmd_generate)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--samples", required=True)
        p.add_argument("--ledger", required=True)
        p.add_argument("--snapshot-dir", required=True)
        p.add_argument("--start-ms", type=int, default=None)
        p.add_argument("--end-ms", type=int, default=None)
        p.add_argument("--scale", type=int, default=10_000)
        p.add_argument("--top-k", type=int, default=10)
        p.add_argument("--batch-size", type=int, default=1_000)
        p.add_argument("--max-unique-stacks", type=int, default=5_000)
        p.add_argument("--max-total-weight", type=int, default=200_000)
        p.add_argument(
            "--force-path", choices=("full", "streaming"), default=None)
        p.add_argument("--interval-version", type=int, default=1)

    prof = sub.add_parser("profile")
    add_common(prof)
    prof.add_argument("--crash-after-batches", type=int, default=None)
    prof.add_argument("--print-report", action="store_true")
    prof.set_defaults(func=cmd_profile)

    rec = sub.add_parser("recover")
    add_common(rec)
    rec.set_defaults(func=cmd_recover)

    res = sub.add_parser("resume")
    add_common(res)
    res.set_defaults(func=cmd_resume)

    rep = sub.add_parser("replay")
    add_common(rep)
    rep.set_defaults(func=cmd_replay)

    diag = sub.add_parser("diagnose")
    add_common(diag)
    diag.add_argument("--text", action="store_true")
    diag.set_defaults(func=cmd_diagnose)

    cc = sub.add_parser("cross-check")
    cc.add_argument("--samples", required=True)
    cc.add_argument("--start-ms", type=int, default=None)
    cc.add_argument("--end-ms", type=int, default=None)
    cc.add_argument("--max-unique-stacks", type=int, default=5_000)
    cc.add_argument("--max-total-weight", type=int, default=200_000)
    cc.set_defaults(func=cmd_cross_check)

    inj = sub.add_parser("inject-sample")
    inj.add_argument("--samples", required=True)
    inj.add_argument("--output", required=True)
    inj.add_argument(
        "--kind",
        choices=("bad_json", "bad_weight", "bad_duration", "bad_frame",
                 "missing_field", "timestamp_regression"),
        required=True)
    inj.add_argument("--target-line", type=int, default=2)
    inj.set_defaults(func=cmd_inject_sample)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ProfilerError as exc:
        return _print_error(exc)


if __name__ == "__main__":
    raise SystemExit(main())
