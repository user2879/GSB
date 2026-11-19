from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .content import content_digest_from_versions, initial_versions
from .demo import run_demo
from .errors import PipelineError
from .fixed import FixedGrid, parse_grid_power
from .ledger import Ledger
from .orchestrator import run_session
from .replay import replay_snapshot
from .spec import load_spec
from .verify import verify_ledger


def _success(payload: dict[str, Any]) -> int:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def _failure(error: PipelineError) -> int:
    print(json.dumps(error.to_dict(), ensure_ascii=False, sort_keys=True, indent=2), file=sys.stderr)
    return error.exit_code


def _grid_from_args(args: argparse.Namespace, *, required: bool = False) -> FixedGrid | None:
    text = getattr(args, "grid", None)
    if text is None:
        if required:
            raise PipelineError("E_GRID", "grid exponent is required", path="grid", field="power")
        return None
    return FixedGrid(parse_grid_power(str(text)))


def _summary_batch(batch: dict[str, Any]) -> dict[str, Any]:
    metrics = batch["metrics"]
    return {
        "batch_id": batch["batch_id"],
        "snapshot_id": batch["snapshot_id"],
        "start_seq": batch["start_seq"],
        "end_seq": batch["end_seq"],
        "cursor": batch["cursor"],
        "content_digest": batch["content_digest"],
        "request_count": metrics["request_count"],
        "source_counts": metrics["source_counts"],
        "aggregate": {
            name: {
                "value_units": metrics["aggregate"]["metrics"][name]["value_units"],
                "error_bound_units": metrics["aggregate"]["metrics"][name]["error_bound_units"],
                "dominant_error_source": metrics["aggregate"]["metrics"][name]["dominant_error_source"],
            }
            for name in sorted(metrics["aggregate"]["metrics"])
        },
        "nodes": {
            node_id: {
                name: {
                    "value_units": metrics["nodes"][node_id]["metrics"][name]["value_units"],
                    "error_bound_units": metrics["nodes"][node_id]["metrics"][name]["error_bound_units"],
                    "dominant_error_source": metrics["nodes"][node_id]["metrics"][name]["dominant_error_source"],
                }
                for name in sorted(metrics["nodes"][node_id]["metrics"])
            }
            for node_id in sorted(metrics["nodes"])
        },
        "requests": [
            {
                "seq": record["seq"],
                "request_id": record["request_id"],
                "node_id": record["node_id"],
                "path": record["path"],
                "hit_source": record["hit_source"],
                "etag": record["returned_etag"],
                "version_consistent": record["version_consistent"],
                "remaining_ttl_ms": record["remaining_ttl_ms"],
                "entry_version": record["entry_version"],
                "stale": record["stale"],
                "error_code": record["error_code"],
            }
            for record in batch["requests"]
        ],
    }


def _spec_check(args: argparse.Namespace) -> int:
    grid = _grid_from_args(args)
    spec = load_spec(args.spec, grid_power=None if grid is None else grid.power)
    return _success(
        {
            "ok": True,
            "spec": str(Path(args.spec).resolve()),
            "session_id": spec.session.session_id,
            "grid": spec.grid.as_dict(),
            "content_paths": sorted(spec.content),
            "derived_paths": sorted(path for path, entry in spec.content.items() if entry.derived),
            "node_ids": sorted(spec.nodes),
            "request_count": len(spec.requests),
            "batch_ids": spec.batch_ids(),
            "source_update_count": len(spec.source_updates),
            "validation": "ok",
        }
    )


def _run(args: argparse.Namespace) -> int:
    grid = _grid_from_args(args)
    spec = load_spec(args.spec, grid_power=None if grid is None else grid.power)
    result = run_session(
        spec,
        args.ledger,
        work_dir=args.work_dir,
        session_id=args.session_id,
    )
    verification = verify_ledger(args.ledger, session_id=result["session_id"])
    return _success(
        {
            "ok": True,
            "session_id": result["session_id"],
            "grid": result["grid"],
            "ledger": str(Path(args.ledger).resolve()),
            "batch_count": result["batch_count"],
            "batches": [_summary_batch(batch) for batch in result["batches"]],
            "verification": verification,
        }
    )


def _verify(args: argparse.Namespace) -> int:
    return _success(verify_ledger(args.ledger, session_id=args.session))


def _snapshots(args: argparse.Namespace) -> int:
    ledger = Ledger(args.ledger, create=False)
    try:
        snapshots = ledger.list_snapshots(args.session)
        active = ledger.active_state()
        return _success(
            {
                "ok": True,
                "ledger": str(Path(args.ledger).resolve()),
                "sessions": ledger.list_sessions(),
                "snapshots": [
                    {
                        **snapshot,
                        "cursor": json.loads(snapshot["cursor_json"]),
                    }
                    for snapshot in snapshots
                ],
                "active_state": None
                if active is None
                else {
                    "snapshot_id": active["snapshot_id"],
                    "session_id": active["session_id"],
                    "batch_id": active["batch_id"],
                    "content_digest": active["content_digest"],
                    "grid": {
                        "power": active["grid_power"],
                        "scale": active["grid_scale"],
                    },
                },
            }
        )
    finally:
        ledger.close()


def _restore(args: argparse.Namespace) -> int:
    ledger = Ledger(args.ledger, create=True)
    try:
        snapshot = ledger.get_snapshot(args.snapshot_id)
        active = ledger.active_state()
        current_digest = args.current_content_digest
        current_grid = _grid_from_args(args)
        if args.spec is not None:
            spec = load_spec(args.spec, grid_power=None if current_grid is None else current_grid.power)
            if current_grid is None:
                current_grid = spec.grid
            snapshot_at = int(snapshot["payload"].get("content_snapshot", {}).get("at_ms", 0))
            current_digest = initial_versions(spec, spec.source_updates, snapshot_at)["content_digest"]
        elif current_digest is None and active is not None:
            current_digest = str(active["content_digest"])
            if current_grid is None:
                current_grid = FixedGrid(int(active["grid_power"]))
        report = ledger.restore_snapshot(
            args.snapshot_id,
            current_content_digest=current_digest,
            current_grid=current_grid,
            expected_content_digest=args.expected_content_digest,
            mode=args.mode,
        )
        restored_active = ledger.active_state()
        if restored_active is None:
            raise PipelineError("E_STATE", "restore did not establish active state", path=args.snapshot_id, field="active_state")
        report["active_state"] = {
            "snapshot_id": restored_active["snapshot_id"],
            "batch_id": restored_active["batch_id"],
            "active_cache_entries": ledger.active_cache_count(),
        }
        return _success(report)
    finally:
        ledger.close()


def _replay(args: argparse.Namespace) -> int:
    grid = _grid_from_args(args)
    return _success(
        replay_snapshot(
            args.ledger,
            args.snapshot_id,
            grid_override=grid,
            current_spec_path=args.spec,
            work_dir=args.work_dir,
        )
    )


def _select_demo_ledger(requested: str | None) -> Path:
    if requested:
        return Path(requested)
    base = Path("artifacts") / "demo-ledger.sqlite3"
    if not base.exists():
        return base
    index = 1
    while True:
        candidate = Path("artifacts") / f"demo-ledger-{index}.sqlite3"
        if not candidate.exists():
            return candidate
        index += 1


def _demo(args: argparse.Namespace) -> int:
    ledger = _select_demo_ledger(args.ledger)
    inject = args.inject
    if args.inject_cycle:
        inject = "cycle"
    if args.inject_grid:
        inject = "grid"
    if args.inject_snapshot:
        inject = "snapshot"
    result = run_demo(ledger, port_base=args.port_base, inject=inject)
    return _success(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cdn_edge",
        description="Deterministic local CDN edge-cache precision and error-propagation diagnostics",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    spec_check = subparsers.add_parser("spec-check", help="validate content and cache-policy JSON")
    spec_check.add_argument("--spec", required=True)
    spec_check.add_argument("--grid", help="precision exponent k where scale=10**k")
    spec_check.set_defaults(func=_spec_check)

    run = subparsers.add_parser("run", help="run a deterministic origin/edge request session")
    run.add_argument("--spec", required=True)
    run.add_argument("--ledger", required=True)
    run.add_argument("--grid")
    run.add_argument("--session-id")
    run.add_argument("--work-dir")
    run.set_defaults(func=_run)

    verify = subparsers.add_parser("verify", help="independently validate all ledger records")
    verify.add_argument("--ledger", required=True)
    verify.add_argument("--session")
    verify.set_defaults(func=_verify)

    snapshots = subparsers.add_parser("snapshots", help="list immutable sessions and snapshots")
    snapshots.add_argument("--ledger", required=True)
    snapshots.add_argument("--session")
    snapshots.set_defaults(func=_snapshots)

    restore = subparsers.add_parser("restore", help="restore a snapshot by whole-state replacement")
    restore.add_argument("--ledger", required=True)
    restore.add_argument("--snapshot-id", required=True)
    restore.add_argument("--spec", help="current spec used to detect content-version changes")
    restore.add_argument("--grid")
    restore.add_argument("--current-content-digest")
    restore.add_argument("--expected-content-digest")
    restore.add_argument("--mode", choices=("replace", "strict"), default="replace")
    restore.set_defaults(func=_restore)

    replay = subparsers.add_parser("replay", help="re-drive recorded requests from a snapshot")
    replay.add_argument("--ledger", required=True)
    replay.add_argument("--snapshot-id", required=True)
    replay.add_argument("--grid")
    replay.add_argument("--spec", help="current spec used to prove content-version reproducibility")
    replay.add_argument("--work-dir")
    replay.set_defaults(func=_replay)

    gui = subparsers.add_parser("gui", help="open the optional tkinter ledger viewer")
    gui.add_argument("--ledger", required=True)
    gui.set_defaults(func=lambda args: __import__("cdn_edge.gui", fromlist=["main"]).main(["--ledger", args.ledger]))

    demo = subparsers.add_parser("demo", help="run cold fill, version update, and grid perturbation sessions")
    demo.add_argument("--ledger")
    demo.add_argument("--port-base", type=int, default=28400)
    demo.add_argument("--inject", choices=("cycle", "grid", "snapshot", "replay", "port"))
    demo.add_argument("--inject-cycle", action="store_true", help="alias for --inject cycle")
    demo.add_argument("--inject-grid", action="store_true", help="alias for --inject grid")
    demo.add_argument("--inject-snapshot", action="store_true", help="alias for --inject snapshot")
    demo.set_defaults(func=_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except PipelineError as exc:
        return _failure(exc)
    except KeyboardInterrupt:
        return _failure(PipelineError("E_PROTOCOL", "interrupted", path="$", field="signal"))
    except Exception as exc:
        return _failure(
            PipelineError(
                "E_PROTOCOL",
                "unexpected command failure",
                path="$",
                field="cli",
                details={"exception_type": type(exc).__name__, "reason": str(exc)},
            )
        )


if __name__ == "__main__":
    raise SystemExit(main())
