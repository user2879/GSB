"""Command line entry point for the cache ledger workbench."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pcl.engine import Engine, EngineConfig  # noqa: E402
from pcl.report import render_diagnostic_report  # noqa: E402
from pcl.samples import write_sample_workspace  # noqa: E402
from pcl.workbench import TK_AVAILABLE, launch_workbench  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Pipeline artifact cache and reuse diagnostic workbench"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="verify python, tkinter and sqlite3")

    init_p = sub.add_parser("init", help="create deterministic sample workspace")
    init_p.add_argument("root", help="workspace directory")

    run_p = sub.add_parser("run", help="load manifest, schedule, judge hits")
    run_p.add_argument("root")
    run_p.add_argument("--manifest", default=None)
    run_p.add_argument("--global-limit", type=int, default=2)
    run_p.add_argument("--stage-quota", type=int, default=1)

    view_p = sub.add_parser("view", help="render the diagnostic report")
    view_p.add_argument("root")

    snap_p = sub.add_parser("snapshot", help="save a named session snapshot")
    snap_p.add_argument("root")
    snap_p.add_argument("name")

    restore_p = sub.add_parser("restore", help="full-replacement snapshot restore")
    restore_p.add_argument("root")
    restore_p.add_argument("name")

    args = parser.parse_args(argv)

    if args.cmd == "check":
        import sqlite3

        print(f"python  : {sys.version.split()[0]}")
        print(f"sqlite3 : {sqlite3.sqlite_version}")
        print(f"tkinter : {'available' if TK_AVAILABLE else 'MISSING (text fallback active)'}")
        return 0 if TK_AVAILABLE else 2

    if args.cmd == "init":
        root = Path(args.root)
        root.mkdir(parents=True, exist_ok=True)
        manifest_path, _ = write_sample_workspace(root)
        print(f"sample workspace ready at {root}")
        print(f"manifest: {manifest_path}")
        return 0

    engine = Engine(
        Path(args.root),
        EngineConfig(
            global_limit=getattr(args, "global_limit", 2),
            default_stage_quota=getattr(args, "stage_quota", 1),
        ),
    )

    if args.cmd == "run":
        manifest = Path(args.manifest) if args.manifest else (
            Path(args.root) / "pipeline_manifest.json"
        )
        engine.load(manifest)
        records, _ = engine.run_plan()
        for rec in records:
            print(
                f"{rec.key:<14} t{rec.queued_tick}->t{rec.start_tick}->"
                f"t{rec.end_tick} slot=G{rec.slot} {rec.outcome}/{rec.reason}"
            )
        return 0

    if args.cmd == "view":
        manifest = Path(args.root) / "pipeline_manifest.json"
        if manifest.exists():
            engine.load(manifest)
        result = launch_workbench(engine.diagnostic_view())
        if isinstance(result, str):
            print(result)
        return 0

    if args.cmd == "snapshot":
        manifest = Path(args.root) / "pipeline_manifest.json"
        if manifest.exists():
            engine.load(manifest)
        meta = engine.save_snapshot(args.name)
        print(f"snapshot saved: {args.name} digest={meta['payload_digest']}")
        return 0

    if args.cmd == "restore":
        result = engine.restore_snapshot(args.name)
        print(f"restored {args.name} via full replacement")
        print(render_diagnostic_report(engine.diagnostic_view()))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
