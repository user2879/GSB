from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .errors import GSBError
from .fixedpoint import (
    budget_boundary_view,
    global_water_error_bound,
    single_shard_error_attribution,
)
from .ledger import Ledger
from .protocol import check_port_available
from .reconcile import run_evolution_check
from .runner import run_spec
from .snapshot import SessionSnapshot
from .spec import load_spec
from .verify import verify_ledger


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _error_exit(exc: GSBError) -> int:
    _print_json(exc.to_dict())
    return 2 if exc.code == "E_SPEC" else 3


def cmd_validate(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    _print_json(
        {
            "ok": True,
            "session_id": spec["session_id"],
            "spec_digest": spec["spec_digest"],
            "nodes": len(spec["nodes"]),
            "shards": len(spec["shards"]),
            "topo_order": spec["topo_order"],
        }
    )
    return 0


def cmd_ports(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    for node in spec["nodes"]:
        check_port_available(node["host"], int(node["port"]))
    _print_json(
        {
            "ok": True,
            "available": [
                {"node_id": node["node_id"], "host": node["host"], "port": node["port"]}
                for node in spec["nodes"]
            ],
        }
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    outcome = run_spec(
        args.spec,
        args.db,
        log_dir=args.log_dir,
        snapshot_path=args.snapshot,
        run_id=args.run_id,
    )
    result = outcome["result"]
    _print_json(
        {
            "ok": True,
            "status": result["status"],
            "completed": result["completed"],
            "decision_count": len(result["decisions"]),
            "wait_records": result["wait_records"],
            "recovery": outcome["recovery"],
            "db_path": outcome["db_path"],
            "snapshot": args.snapshot,
        }
    )
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    _print_json(verify_ledger(args.spec, args.db))
    return 0


def cmd_recover(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    with Ledger(args.db) as ledger:
        report = ledger.recover(spec)
    _print_json({"ok": True, **report})
    return 0


def cmd_evolve(args: argparse.Namespace) -> int:
    _print_json(run_evolution_check(old_spec_path=args.old, new_spec_path=args.new, work_dir=args.workdir))
    return 0


def cmd_fixedpoint_report(args: argparse.Namespace) -> int:
    scales = [10, 100, 1000]
    views = []
    for scale in scales:
        views.append(
            {
                "scale": scale,
                "single_shard": single_shard_error_attribution(12, 40, scale),
                "boundary_over_by_one": budget_boundary_view(
                    used_units=29,
                    demand_units=12,
                    budget_units=40,
                    scale=scale,
                    active_term_count=2,
                ),
                "global_bound_six_shards": global_water_error_bound(6, scale).to_dict(40),
            }
        )
    _print_json({"ok": True, "rule": "floor downward; int64 saturation", "views": views})
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    from .tk_view import launch

    try:
        return launch(args.snapshot)
    except RuntimeError as exc:
        _print_json({"ok": False, "code": "E_TK", "error": str(exc)})
        return 5


def cmd_snapshot_show(args: argparse.Namespace) -> int:
    snapshot = SessionSnapshot.load(args.snapshot)
    _print_json({"ok": True, "snapshot": snapshot.to_dict()})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gsb", description="local multi-node shard delivery lab")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate a JSON specification")
    validate.add_argument("spec")
    validate.set_defaults(func=cmd_validate)

    ports = sub.add_parser("ports", help="check loopback port conflicts")
    ports.add_argument("spec")
    ports.set_defaults(func=cmd_ports)

    run = sub.add_parser("run", help="run all shards over real node subprocesses/TCP")
    run.add_argument("spec")
    run.add_argument("--db", required=True)
    run.add_argument("--log-dir")
    run.add_argument("--snapshot")
    run.add_argument("--run-id", default="cli-run")
    run.set_defaults(func=cmd_run)

    verify = sub.add_parser("verify", help="read-only independent ledger replay")
    verify.add_argument("--spec", required=True)
    verify.add_argument("--db", required=True)
    verify.set_defaults(func=cmd_verify)

    recover = sub.add_parser("recover", help="reclaim stale active leases and list unknown shards")
    recover.add_argument("--spec", required=True)
    recover.add_argument("--db", required=True)
    recover.set_defaults(func=cmd_recover)

    evolve = sub.add_parser("evolve", help="run incremental and blank-full consistency comparison")
    evolve.add_argument("--old", required=True)
    evolve.add_argument("--new", required=True)
    evolve.add_argument("--workdir", required=True)
    evolve.set_defaults(func=cmd_evolve)

    fp = sub.add_parser("fixedpoint-report", help="print error attribution and boundary views")
    fp.set_defaults(func=cmd_fixedpoint_report)

    gui = sub.add_parser("gui", help="open an optional Tkinter snapshot viewer")
    gui.add_argument("--snapshot", required=True)
    gui.set_defaults(func=cmd_gui)

    snapshot_show = sub.add_parser("snapshot-show", help="identify and print an atomic session snapshot")
    snapshot_show.add_argument("--snapshot", required=True)
    snapshot_show.set_defaults(func=cmd_snapshot_show)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except GSBError as exc:
        return _error_exit(exc)
    except FileNotFoundError as exc:
        error = GSBError(str(exc), field="path")
        print(json.dumps(error.to_dict(), ensure_ascii=False, indent=2, sort_keys=True), file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
