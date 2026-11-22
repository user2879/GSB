"""Command line entry point for local Windows / desktop use.

Examples
--------
python -m snapshot_migration.cli validate --chain samples/snapshot_chain.json
python -m snapshot_migration.cli demo --chain samples/snapshot_chain.json \
    --db work.sqlite3 --out artifacts/demo_output.txt
python -m snapshot_migration.cli migrate --chain samples/snapshot_chain.json \
    --db work.sqlite3 --snapshot snapshot.json
python -m snapshot_migration.cli scan --db work.sqlite3 --generation G3 \
    --range R4 --limit 5
python -m snapshot_migration.cli inject --db work.sqlite3 --fault E_BOUND \
    --generation G0 --fragment frag:G0:R2:000
python -m snapshot_migration.cli ui --chain samples/snapshot_chain.json --db work.sqlite3
"""

import argparse
import json
import os
import sys

from .chain import load_chain
from .demo import run_demo
from .engine import MigrationEngine
from .errors import MigrationError
from .readview import Cursor


def _print_error(exc: MigrationError) -> int:
    sys.stderr.write(json.dumps(exc.to_dict(), ensure_ascii=False,
                                sort_keys=True) + "\n")
    return 2


def build_parser():
    parser = argparse.ArgumentParser(
        prog="snapshot_migration",
        description="Snapshot generation migration and read-view diagnostics")
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="validate a chain JSON (E_CHAIN)")
    p_validate.add_argument("--chain", required=True)

    p_migrate = sub.add_parser("migrate", help="migrate all generations")
    p_migrate.add_argument("--chain", required=True)
    p_migrate.add_argument("--db", required=True)
    p_migrate.add_argument("--snapshot")
    p_migrate.add_argument("--no-cleanup", action="store_true",
                           help="leave discarded fragments pending (for injection)")

    p_scan = sub.add_parser("scan", help="two-dimensional cursor scan")
    p_scan.add_argument("--chain", required=True)
    p_scan.add_argument("--db", required=True)
    p_scan.add_argument("--generation", required=True)
    p_scan.add_argument("--range")
    p_scan.add_argument("--key-low")
    p_scan.add_argument("--key-high")
    p_scan.add_argument("--limit", type=int, default=10)
    p_scan.add_argument("--cursor-range")
    p_scan.add_argument("--cursor-key")
    p_scan.add_argument("--cursor-fragment")
    p_scan.add_argument("--cursor-version", type=int)

    p_demo = sub.add_parser("demo", help="run the six deterministic scenarios")
    p_demo.add_argument("--chain", required=True)
    p_demo.add_argument("--out")

    p_inject = sub.add_parser("inject", help="inject E_BOUND/E_INHERIT/E_CLEAN")
    p_inject.add_argument("--chain", required=True)
    p_inject.add_argument("--db", required=True)
    p_inject.add_argument("--fault", required=True,
                          choices=["E_BOUND", "E_INHERIT", "E_CLEAN"])
    p_inject.add_argument("--generation", required=True)
    p_inject.add_argument("--fragment")
    p_inject.add_argument("--check", action="store_true",
                          help="run integrity verification after injection")

    p_replay = sub.add_parser("replay", help="rebuild and field-level verify")
    p_replay.add_argument("--chain", required=True)
    p_replay.add_argument("--db", required=True)

    p_ui = sub.add_parser("ui", help="launch the Tkinter diagnostic UI")
    p_ui.add_argument("--chain", required=True)
    p_ui.add_argument("--db", required=True)

    return parser


def _open_engine(args):
    chain = load_chain(args.chain)
    return chain, MigrationEngine(args.db, chain)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            chain = load_chain(args.chain)
            print(json.dumps({
                "ok": True, "title": chain.title,
                "generations": [g.generation_id for g in chain.generations],
                "limits": [chain.limits.lower, chain.limits.upper]},
                ensure_ascii=False, indent=2))
            return 0

        if args.command == "demo":
            output = run_demo(args.chain)
            if args.out:
                os.makedirs(os.path.dirname(os.path.abspath(args.out)),
                            exist_ok=True)
                with open(args.out, "w", encoding="utf-8") as handle:
                    handle.write(output + "\n")
            print(output)
            return 0

        if args.command == "migrate":
            chain, engine = _open_engine(args)
            for gen in chain.generations:
                engine.migrate_generation(gen.generation_id)
            if not args.no_cleanup:
                engine.run_cleanup()
            if args.snapshot:
                manifest = engine.save_snapshot(args.snapshot)
                print(json.dumps({"ok": True,
                                  "content_digest": manifest["content_digest"],
                                  "fragments": len(manifest["fragments"])},
                                 indent=2))
            else:
                print(json.dumps({"ok": True,
                                  "head": engine.repo.get_meta("head_generation")},
                                 indent=2))
            engine.close()
            return 0

        if args.command == "scan":
            chain, engine = _open_engine(args)
            cursor = None
            if args.cursor_range and args.cursor_key:
                cursor = Cursor(args.cursor_range, args.cursor_key,
                                args.cursor_fragment or "",
                                args.cursor_version or 0)
            with engine.open_read_view("cli-view", args.generation) as view:
                result = view.scan(limit=args.limit, cursor=cursor,
                                   range_id=args.range, key_low=args.key_low,
                                   key_high=args.key_high)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            engine.close()
            return 0

        if args.command == "inject":
            from .recovery import (inject_bound_fault, inject_clean_fault,
                                   inject_inherit_fault, verify_integrity)
            chain, engine = _open_engine(args)
            if args.fault == "E_BOUND":
                report = inject_bound_fault(engine, args.generation,
                                            args.fragment)
            elif args.fault == "E_INHERIT":
                report = inject_inherit_fault(engine, args.generation,
                                              args.fragment)
            else:
                report = inject_clean_fault(engine, args.generation)
            if args.check:
                verify_integrity(engine)
            print(json.dumps(report, indent=2))
            engine.close()
            return 0

        if args.command == "replay":
            from .recovery import replay
            chain, engine = _open_engine(args)
            result = replay(engine)
            print(json.dumps({"ok": result.ok,
                              "first_mismatch": result.first_mismatch,
                              "fragment_differences": result.fragment_differences,
                              "cleanup_differences": result.cleanup_differences},
                             ensure_ascii=False, indent=2))
            engine.close()
            return 0

        if args.command == "ui":
            from .ui import launch_ui
            launch_ui(args.chain, args.db)
            return 0

    except MigrationError as exc:
        return _print_error(exc)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
