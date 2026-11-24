"""CLI for the deterministic quota-ledger demo and verification commands."""
from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from typing import Any

from .diagnostics import launch_dashboard
from .engine import Engine
from .errors import BusinessError, StructuredLogger
from .injection import inject_snapshot_corruption, prepare_injected_sample
from .ledger import Ledger, compare_replay
from .samplegen import write_sample
from .spec import load_spec
from .util import monotonic_ms


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _db_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_file() else candidate / "ledger.sqlite3"


def _handle_business_error(error: BusinessError, logger: StructuredLogger | None, *, expected: bool = False) -> int:
    if logger is not None:
        logger.write_error(error)
    _print({"ok": False, "injected": expected, **error.to_dict()})
    return 0 if expected else 2


def _run_injection(kind: str, sample: str, workdir: str) -> int:
    logger = StructuredLogger(Path(workdir) / "structured.jsonl")
    expected_code = {
        "corrupt-nodes": "E_SPEC",
        "cycle": "E_GRAPH",
        "snapshot": "E_SNAPSHOT",
    }[kind]
    try:
        if kind == "snapshot":
            inject_snapshot_corruption(workdir)
            ledger = Ledger(_db_path(workdir))
            ledger.close()
            raise BusinessError("E_SNAPSHOT", "snapshot injection did not produce a mismatch", field="content_digest")
        injected_sample = prepare_injected_sample(sample, workdir, kind)
        load_spec(injected_sample)
        raise BusinessError(expected_code, "injection did not trigger validation failure", field="inject")
    except BusinessError as exc:
        return _handle_business_error(exc, logger, expected=exc.code == expected_code)


def _run_command(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    logger = StructuredLogger(workdir / "structured.jsonl")
    if args.inject in {"corrupt-nodes", "cycle"}:
        return _run_injection(args.inject, args.sample, args.workdir)
    if args.inject == "snapshot":
        return _run_injection("snapshot", args.sample, args.workdir)
    try:
        spec = load_spec(args.sample)
        engine = Engine(
            spec,
            workdir,
            crash_after_commits=args.crash_after_commits,
            graceful_after_commits=args.graceful_after_commits,
            max_attempts=args.max_attempts,
        )
        previous: dict[int, Any] = {}

        def stop_handler(signum: int, frame: Any) -> None:
            _ = frame
            signal_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
            engine.request_stop(signal_name)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                previous[sig] = signal.signal(sig, stop_handler)
            except (ValueError, OSError, AttributeError):
                pass
        try:
            result = engine.run()
        finally:
            for sig, handler in previous.items():
                try:
                    signal.signal(sig, handler)
                except (ValueError, OSError):
                    pass
        verification = None
        if args.verify_baseline:
            verification = compare_replay(_db_path(args.verify_baseline), workdir / "ledger.sqlite3")
        _print(
            {
                "ok": result.success,
                "exit_code": result.exit_code,
                "interrupted": result.interrupted,
                "ledger_digest": result.ledger_digest,
                "completed_count": result.report["completed_count"],
                "unfinished_shards": result.report["unfinished_shards"],
                "unknown_shards_on_recovery": result.report["unknown_shards_on_recovery"],
                "run_report": str(workdir / "run_report.json"),
                "verification": verification,
            }
        )
        if args.gui:
            launch_dashboard(result.report)
        return result.exit_code
    except BusinessError as exc:
        return _handle_business_error(exc, logger)


def _verify_command(args: argparse.Namespace) -> int:
    try:
        result = compare_replay(_db_path(args.baseline), _db_path(args.candidate))
        _print({"ok": True, "comparison": result})
        return 0
    except BusinessError as exc:
        _print({"ok": False, **exc.to_dict()})
        return 4


def _snapshot_command(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    logger = StructuredLogger(workdir / "structured.jsonl")
    ledger = None
    try:
        if args.inject:
            inject_snapshot_corruption(workdir)
        ledger = Ledger(_db_path(workdir))
        result = ledger.verify_snapshots()
        _print({"ok": True, **result})
        return 0
    except BusinessError as exc:
        if exc.code == "E_SNAPSHOT" and args.inject:
            return _handle_business_error(exc, logger, expected=True)
        return _handle_business_error(exc, logger)
    finally:
        if ledger is not None:
            ledger.close()


def _generate_command(args: argparse.Namespace) -> int:
    _print(write_sample(args.out, args.seed, args.base_port))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quota-ledger", description="cross-node shard quota ledger demo")
    sub = parser.add_subparsers(dest="command")
    run = sub.add_parser("run", help="run or resume the deterministic demo")
    run.add_argument("--sample", required=True, help="sample directory or combined spec.json")
    run.add_argument("--workdir", required=True, help="ledger/report working directory")
    run.add_argument(
        "--gui",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="open the read-only Tkinter diagnostics dashboard (--gui/--no-gui)",
    )
    run.add_argument("--inject", choices=("none", "corrupt-nodes", "cycle", "snapshot"), default="none")
    run.add_argument("--crash-after-commits", type=int, default=0, help="simulate forced termination (no exit report)")
    run.add_argument("--graceful-after-commits", type=int, default=0, help="handled stop with complete exit report")
    run.add_argument("--max-attempts", type=int, default=2)
    run.add_argument("--verify-baseline", default=None, help="compare with an uninterrupted full-build workdir")
    run.set_defaults(handler=_run_command)

    verify = sub.add_parser("verify", help="compare incremental/recovered ledger with a full baseline")
    verify.add_argument("--baseline", required=True)
    verify.add_argument("--candidate", required=True)
    verify.set_defaults(handler=_verify_command)

    snapshot = sub.add_parser("snapshot-check", help="verify ledger content digest and snapshots")
    snapshot.add_argument("--workdir", required=True)
    snapshot.add_argument("--inject", action="store_true", help="tamper the content digest before checking")
    snapshot.set_defaults(handler=_snapshot_command)

    inject = sub.add_parser("inject", help="show a CLI-driven structured fault injection")
    inject.add_argument("--kind", choices=("corrupt-nodes", "cycle", "snapshot"), required=True)
    inject.add_argument("--sample", required=True)
    inject.add_argument("--workdir", required=True)
    inject.set_defaults(handler=lambda a: _run_injection(a.kind, a.sample, a.workdir))

    generate = sub.add_parser("generate", help="generate deterministic sample JSON files")
    generate.add_argument("--out", required=True)
    generate.add_argument("--seed", type=int, default=20261125)
    generate.add_argument("--base-port", type=int, default=None)
    generate.set_defaults(handler=_generate_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    commands = {"run", "verify", "snapshot-check", "inject", "generate"}
    if raw and raw[0] not in commands and raw[0] not in {"-h", "--help"}:
        raw.insert(0, "run")
    parser = build_parser()
    args = parser.parse_args(raw)
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
