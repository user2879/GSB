"""Command line entry point for the lease diagnostic application."""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from .clock import RealClock
from .diagnostics import DEFAULT_SCALE, rank_pools, pool_watermark
from .errors import E_GAP, E_ROLLBACK, E_SNAPSHOT, E_REPLAY, LeaseError
from .ledger import Ledger
from .loader import load_workspace
from .logging_utils import StructuredLogger
from .runtime import compute_watermarks, recover_session, run_session
from .samples import write_samples
from .server import is_port_free
from .ui import render


def _logger(workdir: Path) -> StructuredLogger:
    return StructuredLogger(workdir / "structured.log")


def _state_for_ui(workdir: Path, workspace: dict, scale: int) -> dict:
    ledger = Ledger(workdir / "lease_ledger.sqlite")
    leases = ledger.latest_leases()
    watermarks = ledger.latest_watermarks()
    queue = [event for event in ledger.events()
             if event["event_type"] == "QUEUE"]
    pool_marks = [pool_watermark(workspace["pools"][mark["pool_id"]],
                                 mark["effective_leases"],
                                 mark["queue_length"], scale)
                  for mark in watermarks if mark["scope"].startswith("pool:")]
    ledger.close()
    return {
        "versions": {"pool_set_version": workspace["pool_set_version"],
                     "rollback_version": workspace["rollback_version"]},
        "scale": scale,
        "watermarks": watermarks,
        "ranking": rank_pools(pool_marks, scale),
        "leases": leases,
        "queue": queue,
        "batches": [dict(row) for row in
                    Ledger(workdir / "lease_ledger.sqlite").list_batches()],
    }


def cmd_generate(args) -> int:
    info = write_samples(args.workdir, args.inject)
    print(json.dumps({k: str(v) for k, v in info.items()}, indent=2))
    return 0


def _load_or_log(logger: StructuredLogger, pools_path: Path,
                 requests_path: Path):
    try:
        return load_workspace(pools_path, requests_path)
    except LeaseError as exc:
        logger.error("load_rejected", **exc.to_dict())
        print(json.dumps(exc.to_dict(), indent=2))
        return None


def cmd_run(args) -> int:
    workdir = Path(args.workdir)
    logger = _logger(workdir)
    if args.port and not is_port_free(port=args.port):
        print(json.dumps({"code": "E_PORT",
                          "message": f"port {args.port} is not free"}))
        return 2
    workspace = _load_or_log(logger, workdir / "pools.json",
                             workdir / "requests.ndjson")
    if workspace is None:
        return 2
    logger.info("run_start", workdir=str(workdir), inject=args.inject,
                hard_exit_batch=args.hard_exit_batch)
    result = run_session(
        workspace, workdir, scale=args.scale, global_cap=args.global_cap,
        time_scale=args.time_scale, socket_timeout_ms=args.socket_timeout_ms,
        max_retries=args.max_retries,
        hard_exit_batch=args.hard_exit_batch, resume=args.resume)
    logger.info("run_complete",
                batches=len(result["batches"]), port=result["port"])
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_recover(args) -> int:
    workdir = Path(args.workdir)
    logger = _logger(workdir)
    request_file = ("requests_gap.ndjson" if args.inject == "missing_renewal"
                    else "requests.ndjson")
    try:
        workspace = load_workspace(workdir / "pools.json",
                                   workdir / request_file)
        result = recover_session(
            workspace, workdir, scale=args.scale,
            snapshot_tamper=args.inject == "snapshot_digest")
        logger.info("recover_complete", **result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except LeaseError as exc:
        logger.error("recover_rejected", **exc.to_dict())
        print(json.dumps(exc.to_dict(), indent=2, sort_keys=True))
        return 2


def cmd_demo(args) -> int:
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    logger = _logger(workdir)
    # Build the durable ledger from the complete stream first; gap injection
    # swaps in a stream missing one renewal before recovery.
    write_samples(workdir, None if args.inject == "missing_renewal"
                  else args.inject)
    if args.inject == "rollback_cycle":
        workspace = _load_or_log(logger, workdir / "pools.json",
                                 workdir / "requests.ndjson")
        return 2 if workspace is None else 0
    if args.port and not is_port_free(port=args.port):
        print(json.dumps({"code": "E_PORT",
                          "message": f"port {args.port} is not free"}))
        return 2
    workspace = load_workspace(workdir / "pools.json",
                               workdir / "requests.ndjson")
    result = run_session(
        workspace, workdir, scale=args.scale, global_cap=args.global_cap,
        time_scale=args.time_scale, socket_timeout_ms=args.socket_timeout_ms,
        max_retries=args.max_retries,
        hard_exit_batch=args.hard_exit_batch)
    logger.info("demo_run_complete", batches=len(result["batches"]))
    if args.inject == "snapshot_digest":
        return cmd_recover(argparse.Namespace(
            workdir=args.workdir, inject="snapshot_digest",
            scale=args.scale))
    if args.inject == "missing_renewal":
        gap_doc = write_samples(workdir, "missing_renewal")
        (workdir / "requests.ndjson").write_text(
            gap_doc["gap_path"].read_text(encoding="utf-8"),
            encoding="utf-8")
        return cmd_recover(argparse.Namespace(
            workdir=args.workdir, inject="missing_renewal",
            scale=args.scale))
    recovery = recover_session(workspace, workdir, scale=args.scale)
    use_tk = args.ui
    if use_tk:
        try:
            import tkinter  # noqa: F401
        except ImportError:
            logger.error("tk_unavailable_headless_fallback",
                         code="E_UI",
                         message="tkinter missing; using text dashboard")
            use_tk = False
    dashboard = render(_state_for_ui(workdir, workspace, args.scale),
                       use_tk=use_tk)
    summary = {"run": {"batches": result["batches"],
                       "ranking": result["ranking"],
                       "final_watermarks": result["final_watermarks"]},
               "recovery": recovery}
    if dashboard is not None:
        print(dashboard)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lease-diag",
        description="Database config lease & concurrency diagnostic demo")
    parser.add_argument("--workdir", default="workspace/demo")
    parser.add_argument("--inject", choices=[
        "rollback_cycle", "missing_renewal", "snapshot_digest"],
        default=None)
    parser.add_argument("--scale", type=int, default=DEFAULT_SCALE)
    parser.add_argument("--global-cap", type=int, default=2)
    parser.add_argument("--time-scale", type=int, default=20)
    parser.add_argument("--socket-timeout-ms", type=int, default=2000)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--hard-exit-batch", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("generate").set_defaults(func=cmd_generate)
    run_p = sub.add_parser("run")
    run_p.set_defaults(func=cmd_run)
    sub.add_parser("recover").set_defaults(func=cmd_recover)
    demo_p = sub.add_parser("demo")
    demo_p.add_argument("--ui", action="store_true")
    demo_p.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except LeaseError as exc:
        print(json.dumps(exc.to_dict(), indent=2, sort_keys=True))
        return 2
    except Exception:  # pragma: no cover - defensive top-level guard
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
