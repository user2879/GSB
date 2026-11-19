"""Command line interface for the configuration distribution platform."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config.grid import Grid
from .config.manifest import load_manifest, load_registry
from .errors import PlatformError
from .injection import prepare_manifest_injection, validate_grid_q
from .persistence.database import Database
from .persistence.snapshot import SnapshotStore
from .runtime.replay import verify_snapshot_replay
from .runtime.session import Session, new_session_id
from .util.logging_util import StructuredLogger
from . import gui as gui_module

_EXIT_CODES = {
    "E_MANIFEST": 2,
    "E_CYCLE": 3,
    "E_GRID": 4,
    "E_REF": 5,
    "E_NODE": 6,
    "E_HANDSHAKE": 7,
    "E_PORT": 8,
    "E_CONVERGE": 9,
    "E_VIEW": 10,
    "E_REPLAY": 11,
    "E_STATE": 12,
    "E_IO": 13,
}


def _emit_error(error: PlatformError, logger: StructuredLogger | None) -> int:
    payload = error.to_dict()
    if logger is not None:
        for detail in error.details:
            logger.log(
                "business_error",
                code=error.code,
                key=detail.key,
                source=detail.source,
                reason=detail.reason,
            )
    sys.stderr.write(json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n")
    return _EXIT_CODES.get(error.code, 1)


def _base_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    samples = Path(args.samples_dir)
    data = Path(args.data_dir)
    db_path = data / "cfgdist.sqlite"
    snapshot_dir = data / "snapshots"
    log_dir = Path(args.log_dir)
    return samples, data, db_path, snapshot_dir, log_dir


def _load_grid(args: argparse.Namespace) -> Grid:
    return Grid.from_q(validate_grid_q(args.grid_q))


def cmd_validate(args: argparse.Namespace) -> int:
    samples, _, _, _, _ = _base_paths(args)
    registry = load_registry(samples / "registry.json")
    manifest = load_manifest(samples / "manifest.json", registry)
    out = {
        "status": "ok",
        "versions": [v.version for v in manifest.versions],
        "nodes": sorted(registry.nodes),
        "items_per_version": {
            str(v.version): len(v.items) for v in manifest.versions
        },
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    samples, data, db_path, snapshot_dir, log_dir = _base_paths(args)
    data.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = StructuredLogger(log_dir / f"session-{args.session_id}.log")
    try:
        grid = _load_grid(args)
        registry = load_registry(samples / "registry.json")
        manifest = load_manifest(samples / "manifest.json", registry)
        with Database(db_path, logger) as db:
            store = SnapshotStore(db, snapshot_dir, logger)
            session = Session(
                args.session_id,
                manifest,
                registry,
                grid,
                db,
                store,
                logger,
                concurrency=args.concurrency,
                max_attempts=args.max_attempts,
                log_dir=str(log_dir),
                inject_by_node=_node_injections(args),
                operation_timeout_ms=args.timeout_ms,
            )
            results = []
            target_versions = set(args.versions) if args.versions else None
            for manifest_version in manifest.versions:
                if target_versions and manifest_version.version not in target_versions:
                    continue
                result = session.run_version(manifest_version)
                results.append(_summarize_run(result))
            print(json.dumps({"session_id": args.session_id, "runs": results}, indent=2, ensure_ascii=False))
        return 0
    except PlatformError as error:
        return _emit_error(error, logger)
    finally:
        logger.close()


def _node_injections(args: argparse.Namespace) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for token in args.inject_node or []:
        node_id, _, flag = token.partition("=")
        mapping.setdefault(node_id, []).append(flag)
    return mapping


def _summarize_run(result: Any) -> dict[str, Any]:
    return {
        "version": result.version,
        "state": result.state,
        "invalidated": sorted(result.invalidated),
        "invalidation_steps": result.invalidation_steps,
        "byte_identical_untouched": result.byte_check,
        "plan_digest": result.plan.content_digest,
        "observed_max_inflight": result.scheduler.observed_max_inflight,
        "concurrency_max": result.scheduler.concurrency_max,
        "converged": result.convergence.converged if result.convergence else None,
        "view_digest": result.view.get("view_digest"),
        "snapshot_replaced": result.identity.replaced,
    }


def cmd_replay(args: argparse.Namespace) -> int:
    samples, _, db_path, _, log_dir = _base_paths(args)
    logger = StructuredLogger(log_dir / f"replay-{args.session_id}.log")
    try:
        grid = _load_grid(args)
        registry = load_registry(samples / "registry.json")
        manifest = load_manifest(samples / "manifest.json", registry)
        with Database(db_path, logger) as db:
            store = SnapshotStore(db, Path(args.data_dir) / "snapshots", logger)
            versions = store.db.list_versions(args.session_id)
            if args.version is not None:
                versions = [v for v in versions if v == args.version]
            reports = []
            previous = None
            for number in versions:
                snapshot = store.load(args.session_id, number)
                current_manifest = next(v for v in manifest.versions if v.version == number)
                report = verify_snapshot_replay(
                    manifest, registry, grid, snapshot, previous
                )
                reports.append(report)
                previous = current_manifest
            print(json.dumps({"session_id": args.session_id, "replay": reports}, indent=2, ensure_ascii=False))
        return 0
    except PlatformError as error:
        return _emit_error(error, logger)
    finally:
        logger.close()


def cmd_view(args: argparse.Namespace) -> int:
    snapshot_path = Path(args.snapshot)
    if args.gui:  # pragma: no cover - requires a display
        gui_module.launch(snapshot_path)
        return 0
    print(gui_module.render_text(snapshot_path))
    return 0


def cmd_inject(args: argparse.Namespace) -> int:
    samples, data, _, _, log_dir = _base_paths(args)
    work_dir = data / "injected"
    logger = StructuredLogger(log_dir / "injection.log")
    try:
        if args.kind in ("corrupt", "cycle"):
            target = prepare_manifest_injection(args.kind, samples, work_dir)
            registry = load_registry(samples / "registry.json")
            # Loading the poisoned file must raise the documented code.
            load_manifest(target, registry)
            raise AssertionError("injected manifest unexpectedly accepted")
        if args.kind == "bad_grid":
            # Use the explicitly poisoned exponent, bypassing the global
            # --grid-q validation which runs for the normal pipeline.
            validate_grid_q(args.bad_grid_q)
            raise AssertionError("injected grid unexpectedly accepted")
        raise ValueError(f"unknown injection kind {args.kind}")
    except PlatformError as error:
        return _emit_error(error, logger)
    finally:
        logger.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cfgdist", description="cross-component config distribution platform")
    parser.add_argument("--samples-dir", default="samples")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--grid-q", type=int, default=2)
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="hard-validate manifest and registry")
    p_validate.set_defaults(func=cmd_validate)

    p_run = sub.add_parser("run", help="run one or more version evolutions")
    p_run.add_argument("--session-id", default=new_session_id())
    p_run.add_argument("--concurrency", type=int, default=2)
    p_run.add_argument("--max-attempts", type=int, default=3)
    p_run.add_argument("--timeout-ms", type=int, default=10000)
    p_run.add_argument("--versions", type=int, nargs="*")
    p_run.add_argument(
        "--inject-node",
        action="append",
        help="node_id=flag, e.g. node-a=handshake_fail / node-b=node_fail",
    )
    p_run.set_defaults(func=cmd_run)

    p_replay = sub.add_parser("replay", help="authoritative ledger replay")
    p_replay.add_argument("--session-id", required=True)
    p_replay.add_argument("--version", type=int)
    p_replay.set_defaults(func=cmd_replay)

    p_view = sub.add_parser("view", help="render a snapshot view")
    p_view.add_argument("--snapshot", required=True)
    p_view.add_argument("--gui", action="store_true")
    p_view.set_defaults(func=cmd_view)

    p_inject = sub.add_parser("inject", help="fault injection (corrupt|cycle|bad_grid)")
    p_inject.add_argument("kind", choices=["corrupt", "cycle", "bad_grid"])
    p_inject.add_argument(
        "--bad-grid-q",
        type=int,
        default=99,
        help="illegal grid exponent used only by the bad_grid injection",
    )
    p_inject.set_defaults(func=cmd_inject)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
