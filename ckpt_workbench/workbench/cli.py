"""Command-line interface for the checkpoint workbench."""

import argparse
import json
import os
import sys

from . import app
from .changeset import commit, load_change_set
from .errors import WorkbenchError
from .gui import collect_view, launch_gui, render_text
from .inject import (
    inject_dangling_parent,
    inject_shard_corruption,
    inject_snapshot_mismatch,
    run_validation_expect_failure,
)
from .model import load_declaration
from .recovery import (
    detect_inflight,
    replay_ledger,
    rerun_pending,
    rollback_to_parent,
)
from .changeset import full_recompute


def print_json(payload):
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def build_parser():
    parser = argparse.ArgumentParser(
        prog="ckptwb",
        description="Deterministic training checkpoint & recovery diagnostic workbench",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init", help="build deterministic baseline ledger")
    p.add_argument("--workdir", required=True)
    p.add_argument("--decl", required=True)

    p = sub.add_parser("submit", help="incrementally commit a change set")
    p.add_argument("--workdir", required=True)
    p.add_argument("--decl", required=True)
    p.add_argument("--change", required=True)
    p.add_argument("--crash-point", choices=["before_commit"], default=None)

    p = sub.add_parser("verify", help="incremental vs full recompute check")
    p.add_argument("--workdir", required=True)
    p.add_argument("--decl", required=True)
    p.add_argument("--changes", nargs="+", required=True)

    p = sub.add_parser("snapshot-save")
    p.add_argument("--workdir", required=True)
    p.add_argument("--snapshot", required=True)

    p = sub.add_parser("snapshot-status")
    p.add_argument("--workdir", required=True)
    p.add_argument("--snapshot", required=True)

    p = sub.add_parser("snapshot-restore")
    p.add_argument("--workdir", required=True)
    p.add_argument("--snapshot", required=True)

    p = sub.add_parser("status", help="validate and print a diagnostic view")
    p.add_argument("--workdir", required=True)
    p.add_argument("--snapshots", nargs="*", default=[])
    p.add_argument("--gui", action="store_true")

    p = sub.add_parser("recover", help="detect forced-termination residue")
    p.add_argument("--workdir", required=True)
    p.add_argument("--mode", choices=["rollback", "rerun"], required=True)
    p.add_argument("--decl")
    p.add_argument("--change")

    p = sub.add_parser("replay", help="replay the change trace and compare")
    p.add_argument("--workdir", required=True)
    p.add_argument("--decl", required=True)

    p = sub.add_parser("inject", help="inject one of the three anomalies")
    p.add_argument("--workdir")
    p.add_argument("--kind", required=True,
                   choices=["shard_corruption", "dangling_parent", "snapshot_mismatch"])
    p.add_argument("--checkpoint")
    p.add_argument("--shard")
    p.add_argument("--snapshot")

    p = sub.add_parser("demo", help="run the full repeatable demo")
    p.add_argument("--root", required=True)
    p.add_argument("--decl", required=True)
    p.add_argument("--changes", nargs="+", required=True)

    return parser


def _open(workdir):
    return app.open_ledger(workdir)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            print_json(app.initialize(args.workdir, args.decl))

        elif args.command == "submit":
            print_json(app.submit_change(args.workdir, args.decl, args.change,
                                         crash_point=args.crash_point))

        elif args.command == "verify":
            print_json(app.verify_all(args.workdir, args.decl, args.changes))

        elif args.command == "snapshot-save":
            print_json(app.snapshot_save(args.workdir, args.snapshot))

        elif args.command == "snapshot-status":
            print_json(app.snapshot_status(args.workdir, args.snapshot))

        elif args.command == "snapshot-restore":
            print_json(app.snapshot_restore(args.workdir, args.snapshot))

        elif args.command == "status":
            ledger = _open(args.workdir)
            ledger.validate()
            view = collect_view(ledger, args.snapshots)
            if args.gui:
                launch_gui(ledger, args.snapshots)
            else:
                print(render_text(view))
            ledger.close()

        elif args.command == "recover":
            ledger = _open(args.workdir)
            inflight = detect_inflight(ledger)
            if inflight is None:
                print_json({"inflight": None, "message": "no interrupted commit"})
                ledger.close()
            elif args.mode == "rollback":
                print_json(rollback_to_parent(ledger))
                ledger.close()
            else:
                if not (args.decl and args.change):
                    raise WorkbenchError("rerun requires --decl and --change")
                decl = load_declaration(args.decl)
                change = load_change_set(args.change)
                result = rerun_pending(ledger, change, decl, commit)
                ledger.close()
                print_json(result)

        elif args.command == "replay":
            ledger = _open(args.workdir)
            decl = load_declaration(args.decl)
            report = replay_ledger(ledger, decl, full_recompute)
            ledger.close()
            print_json(report)

        elif args.command == "inject":
            if args.kind == "shard_corruption":
                print_json(inject_shard_corruption(
                    os.path.join(args.workdir, "ledger.db"),
                    os.path.join(args.workdir, "shards"),
                    args.checkpoint, args.shard))
                print_json(run_validation_expect_failure(
                    os.path.join(args.workdir, "ledger.db"),
                    os.path.join(args.workdir, "shards")))
            elif args.kind == "dangling_parent":
                print_json(inject_dangling_parent(
                    os.path.join(args.workdir, "ledger.db"),
                    os.path.join(args.workdir, "shards"), args.checkpoint))
                print_json(run_validation_expect_failure(
                    os.path.join(args.workdir, "ledger.db"),
                    os.path.join(args.workdir, "shards")))
            else:
                print_json(inject_snapshot_mismatch(args.snapshot))

        elif args.command == "demo":
            from .demo import run_demo
            print_json(run_demo(args.root, args.decl, args.changes))

        return 0
    except WorkbenchError as error:
        print(str(error), file=sys.stderr)
        return 2
    except (ValueError, KeyError, FileNotFoundError) as error:
        print(f"E_INPUT: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
