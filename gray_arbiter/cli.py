"""Command-line entry point for the local arbiter."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import load_config, load_request_stream, validate_config
from .engine import ArbiterEngine
from .errors import BusinessError, StructuredLogger
from .ledger import Ledger


def _free_loopback_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def _inject_borrow_cycle(config: Any) -> Any:
    mirror_pools = {rule.mirror_pool for rule in config.rules.values()}
    candidates = sorted(pool_id for pool_id in config.pools if pool_id not in mirror_pools)
    if len(candidates) < 2:
        raise BusinessError(
            "E_BORROW",
            "cannot inject a borrow cycle with fewer than two non-mirror pools",
            field="pools.lendable_to",
            request_id="*",
            pool_id=candidates[0] if candidates else None,
            phase="fault_injection",
            details={"capacity_override": 1},
        )
    first, second = candidates[0], candidates[1]
    pools = dict(config.pools)
    pools[first] = replace(
        pools[first],
        max_connections=1,
        lendable_to=tuple(sorted(set(pools[first].lendable_to) | {second})),
    )
    pools[second] = replace(
        pools[second],
        max_connections=1,
        lendable_to=tuple(sorted(set(pools[second].lendable_to) | {first})),
    )
    injected = replace(config, pools=pools)
    validate_config(injected, detect_borrow_cycles=True, cycle_code="E_BORROW")
    return injected


def _load_for_injection(path: str, rules_path: str | None, injection: str) -> Any:
    detect_cycles = injection != "borrow-cycle"
    config = load_config(
        path,
        rules_path=rules_path,
        detect_borrow_cycles=detect_cycles,
        cycle_code="E_BORROW" if injection == "borrow-cycle" else "E_POOL",
    )
    if injection == "borrow-cycle":
        return _inject_borrow_cycle(config)
    return config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="execute request batches")
    run.add_argument("--config", required=True)
    run.add_argument("--rules")
    run.add_argument("--requests", required=True)
    run.add_argument("--ledger", default="state/arbiter.sqlite")
    run.add_argument("--snapshots", default="snapshots")
    run.add_argument("--log", default="logs/structured.ndjson")
    run.add_argument("--batch-size", type=int, default=100)
    run.add_argument("--workers", type=int, default=16)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--reset-ledger", action="store_true")
    run.add_argument("--borrow-policy", choices=["force_return", "recompute", "invalidate"], default="force_return")
    run.add_argument(
        "--inject",
        choices=["none", "borrow-cycle", "target-unreachable", "snapshot-digest"],
        default="none",
    )
    run.add_argument(
        "--inject-pool-cycle",
        action="store_true",
        help="alias for --inject borrow-cycle",
    )
    run.add_argument(
        "--inject-target-unreachable",
        nargs="?",
        const="auto",
        metavar="POOL",
        help="override a pool endpoint with a closed loopback port",
    )
    run.add_argument(
        "--inject-snapshot-digest",
        action="store_true",
        help="force an E_SNAPSHOT content-digest verification failure",
    )

    replay = sub.add_parser("replay", help="verify committed batches and replay fields")
    replay.add_argument("--ledger", default="state/arbiter.sqlite")
    replay.add_argument("--log", default="logs/structured.ndjson")
    replay.add_argument("--inject-snapshot-digest", action="store_true")

    recover = sub.add_parser("recover", help="recover only complete batches")
    recover.add_argument("--ledger", default="state/arbiter.sqlite")
    recover.add_argument("--log", default="logs/structured.ndjson")
    recover.add_argument("--replay", action="store_true", default=True)

    ui = sub.add_parser("ui", help="open the Tkinter diagnostics window")
    ui.add_argument("--ledger", default="state/arbiter.sqlite")
    ui.add_argument("--snapshots", default="snapshots")
    return parser


def _write_json(payload: Any) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _prepare_logger(path: str, *, append: bool = False) -> StructuredLogger:
    log_path = Path(path)
    if log_path.exists() and not append:
        log_path.unlink()
    return StructuredLogger(log_path)


def _reset_ledger(path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            candidate.unlink()


def _run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    injection = args.inject
    if args.inject_pool_cycle:
        injection = "borrow-cycle"
    if args.inject_target_unreachable is not None:
        injection = "target-unreachable"
    if args.inject_snapshot_digest:
        injection = "snapshot-digest"
    logger = _prepare_logger(args.log)
    ledger_path = Path(args.ledger)
    if args.reset_ledger:
        _reset_ledger(ledger_path)
    config = _load_for_injection(args.config, args.rules, injection)
    requests = load_request_stream(args.requests)
    endpoint_overrides: dict[str, tuple[str, int]] = {}
    injected_pool: str | None = None
    if injection == "target-unreachable":
        requested_pool = args.inject_target_unreachable
        mirror_pools = {rule.mirror_pool for rule in config.rules.values()}
        if requested_pool in (None, "auto"):
            requested_pool = sorted(pool_id for pool_id in config.pools if pool_id not in mirror_pools)[0]
        if requested_pool not in config.pools:
            raise BusinessError(
                "E_POOL",
                "injected unreachable pool does not exist",
                field="pool_id",
                request_id="*",
                pool_id=requested_pool,
                phase="fault_injection",
            )
        injected_pool = requested_pool
        endpoint_overrides[injected_pool] = ("127.0.0.1", _free_loopback_port())
        logger.emit(
            "E_TARGET",
            "*",
            injected_pool,
            "fault_injection",
            "loopback target endpoint forced to a closed port",
            injected_endpoint=endpoint_overrides[injected_pool],
        )

    ledger = Ledger(ledger_path, logger)
    engine = ArbiterEngine(
        config,
        ledger=ledger,
        logger=logger,
        snapshot_dir=args.snapshots,
        endpoint_overrides=endpoint_overrides,
        worker_count=args.workers,
    )
    try:
        result = engine.run(requests, batch_size=args.batch_size, resume=args.resume)
        if injection == "snapshot-digest":
            batches = ledger.list_batches()
            if not batches:
                raise BusinessError(
                    "E_REPLAY",
                    "no committed batch is available for digest injection",
                    field="batch_id",
                    request_id="*",
                    pool_id=None,
                    phase="fault_injection",
                )
            ledger.verify_batch(batches[-1]["batch_id"], force_digest_mismatch=True)
        result["structured_log"] = str(Path(args.log))
        result["injection"] = {"kind": injection, "pool_id": injected_pool}
        logger.write_summary(Path(args.log).with_name(Path(args.log).stem + ".summary.json"))
        codes = {
            outcome.get("main_error_code")
            for batch in result.get("batches", [])
            for outcome in batch.get("requests", [])
        } | {
            outcome.get("mirror_error_code")
            for batch in result.get("batches", [])
            for outcome in batch.get("requests", [])
        }
        if "E_SNAPSHOT" in codes or "E_REPLAY" in codes:
            _write_json(result)
            return 4, result
        if "E_BORROW" in codes or "E_TARGET" in codes:
            _write_json(result)
            return 3, result
        _write_json(result)
        return 0, result
    finally:
        engine.close()


def _replay(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    logger = _prepare_logger(args.log, append=True)
    ledger = Ledger(args.ledger, logger)
    try:
        recovery = ledger.recover()
        replay = ledger.replay_self_check()
        batches = ledger.list_batches()
        if args.inject_snapshot_digest:
            if not batches:
                raise BusinessError(
                    "E_REPLAY",
                    "no committed batch exists for digest verification",
                    field="batch_id",
                    request_id="*",
                    pool_id=None,
                    phase="replay",
                )
            ledger.verify_batch(batches[-1]["batch_id"], force_digest_mismatch=True)
        result = {"recovery": recovery, "replay": replay, "exit_report_available": recovery["exit_report_available"]}
        logger.write_summary(Path(args.log).with_name(Path(args.log).stem + ".summary.json"))
        _write_json(result)
        return 0, result
    finally:
        ledger.close()


def _recover(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    logger = _prepare_logger(args.log, append=True)
    ledger = Ledger(args.ledger, logger)
    try:
        recovery = ledger.recover()
        replay = ledger.replay_self_check()
        result = {"recovery": recovery, "replay": replay}
        logger.write_summary(Path(args.log).with_name(Path(args.log).stem + ".summary.json"))
        _write_json(result)
        return 0, result
    finally:
        ledger.close()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "ui":
        from .ui import launch

        return launch(args.ledger, args.snapshots)
    try:
        if args.command == "run":
            code, _ = _run(args)
            return code
        if args.command == "replay":
            code, _ = _replay(args)
            return code
        if args.command == "recover":
            code, _ = _recover(args)
            return code
    except BusinessError as error:
        if error.request_id is None:
            error.request_id = "*"
        if error.pool_id is None and error.code == "E_BORROW":
            cycle = error.details.get("cycle") if error.details else None
            if cycle:
                error.pool_id = cycle[0]
        logger = StructuredLogger(args.log)
        logger.emit_error(error)
        logger.write_summary(Path(args.log).with_name(Path(args.log).stem + ".summary.json"))
        _write_json({"status": "error", "error": error.to_dict(), "structured_log": args.log})
        if error.code in {"E_SNAPSHOT", "E_REPLAY"}:
            return 4
        if error.code in {"E_BORROW", "E_TARGET"}:
            return 3
        return 2
    except Exception as exc:
        _write_json(
            {
                "status": "error",
                "error": {
                    "code": "E_CONFIG",
                    "message": str(exc),
                    "exception": type(exc).__name__,
                },
            }
        )
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
