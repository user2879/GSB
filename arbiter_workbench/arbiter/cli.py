"""Command line entry points for the arbitration diagnostic workbench."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

from .engine import ArbitrationEngine
from .errors import WorkbenchError
from .ledger import Ledger
from .propagation import ChangeSet
from .recovery import inspect_db, recover
from .sample import DEFAULT_KEYS
from .service_manager import ServiceManager
from .spec import load_declarations

DECL_DIR = Path(__file__).resolve().parents[1] / "decls"
RUN_DIR = Path(__file__).resolve().parents[1] / "run"


def _json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def _dump(payload) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=_json_default)


def _build_engine(args) -> tuple[ArbitrationEngine, Ledger, ServiceManager]:
    decls = load_declarations(args.decls, DEFAULT_KEYS)
    RUN_DIR.mkdir(exist_ok=True)
    ledger = Ledger(RUN_DIR / "ledger.sqlite", session_scale=args.scale)
    engine = ArbitrationEngine(decls, scale_exponent=args.scale, ledger=ledger,
                               batch_size=args.batch_size)
    manager = ServiceManager(list(decls.environments.values()))
    return engine, ledger, manager


def cmd_run(args) -> int:
    engine, ledger, manager = _build_engine(args)
    try:
        manager.start_all()
        engine.start_services_and_ingest(manager)
        outputs = engine.run_full()
        report = {
            "mode": "full",
            "scale_exponent": args.scale,
            "views": {
                view_id: {
                    "path": item["path"],
                    "digest": item["digest"],
                    "version": item["invalidation"].new_version,
                    "decision": asdict(item["decision"]),
                    "key_count": len(item["records"]),
                }
                for view_id, item in sorted(outputs.items())
            },
            "meta_digest": ledger.meta_digest(),
        }
        print(_dump(report))
    finally:
        manager.shutdown_all()
        ledger.close()
    return 0


def cmd_incremental(args) -> int:
    engine, ledger, manager = _build_engine(args)
    try:
        manager.start_all()
        engine.start_services_and_ingest(manager)
        engine.run_full()
        change = ChangeSet(
            replaced_obs_envs=tuple(args.changed_envs.split(",")) if args.changed_envs else (),
            changed_policies=json.loads(args.policy_changes) if args.policy_changes else {},
        )
        invalidations = engine.propagation.propagate(change)
        outputs = engine.run_incremental(invalidations)
        print(_dump({
            "mode": "incremental",
            "invalidations": [asdict(i) for i in invalidations],
            "views": {
                view_id: {
                    "decision": asdict(item["decision"]),
                    "version": item["invalidation"].new_version,
                    "dirty_keys": list(item["invalidation"].dirty_keys),
                    "digest": item["digest"],
                }
                for view_id, item in sorted(outputs.items())
            },
        }))
    finally:
        manager.shutdown_all()
        ledger.close()
    return 0


def cmd_snapshot(args) -> int:
    decls = load_declarations(args.decls, DEFAULT_KEYS)
    ledger = Ledger(RUN_DIR / "ledger.sqlite", session_scale=args.scale)
    try:
        out = Path(args.out)
        digest = ledger.export_snapshot(out)
        print(_dump({"snapshot": str(out), "content_digest": digest}))
    finally:
        ledger.close()
    return 0


def cmd_restore(args) -> int:
    ledger = Ledger(RUN_DIR / "ledger.sqlite", session_scale=args.scale)
    try:
        digest = ledger.replace_with_snapshot(args.snapshot)
        print(_dump({"mode": "whole_snapshot_replacement", "content_digest": digest}))
    finally:
        ledger.close()
    return 0


def cmd_recover(args) -> int:
    latest = inspect_db(RUN_DIR / "ledger.sqlite")

    def replay_fn(view_id: str, version: int):
        # Deterministic rebuild using a fresh engine on the same declarations.
        decls = load_declarations(args.decls, DEFAULT_KEYS)
        ledger_ro = Ledger(RUN_DIR / "replay.sqlite", session_scale=args.scale)
        manager = ServiceManager(list(decls.environments.values()))
        engine = ArbitrationEngine(decls, args.scale, ledger_ro, args.batch_size)
        try:
            manager.start_all()
            engine.start_services_and_ingest(manager)
            committed_version = latest[view_id]["version"]
            # Replay exactly the key set that was committed in that version;
            # incremental commits only contain their dirty keys.
            with Ledger(RUN_DIR / "ledger.sqlite", session_scale=args.scale) as probe:
                committed_keys = tuple(
                    r.key for r in probe.load_records(view_id, committed_version)
                )
            records = engine.arbitrate_view(
                view_id, committed_version, only_keys=committed_keys
            )
        finally:
            manager.shutdown_all()
            ledger_ro.close()
        return records

    report = recover(RUN_DIR / "ledger.sqlite", args.scale, replay_fn)
    print(_dump(asdict(report)))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cross-environment config arbitration workbench")
    parser.add_argument("--decls", default=str(DECL_DIR))
    parser.add_argument("--scale", type=int, default=2, choices=(0, 1, 2, 3))
    parser.add_argument("--batch-size", type=int, default=7)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="full arbitration across all views")
    p_run.set_defaults(func=cmd_run)

    p_inc = sub.add_parser("incremental", help="change-set driven incremental arbitration")
    p_inc.add_argument("--changed-envs", default="")
    p_inc.add_argument("--policy-changes", default="")
    p_inc.set_defaults(func=cmd_incremental)

    p_snap = sub.add_parser("snapshot", help="export session snapshot")
    p_snap.add_argument("--out", required=True)
    p_snap.set_defaults(func=cmd_snapshot)

    p_restore = sub.add_parser("restore", help="whole snapshot replacement recovery")
    p_restore.add_argument("--snapshot", required=True)
    p_restore.set_defaults(func=cmd_restore)

    p_rec = sub.add_parser("recover", help="crash recovery + replay validation")
    p_rec.set_defaults(func=cmd_recover)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except WorkbenchError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
