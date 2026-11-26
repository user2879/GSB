from __future__ import annotations

import argparse
import json
from pathlib import Path

from .commands import apply_commands, initialize_workspace, load_command_stream
from .config import load_config
from .diagnostics import run_diagnostics
from .ledger import ContentLedger, SnapshotManager
from .pipeline import Pipeline, canonical_fingerprint
from .samplegen import generate_sample_set
from .utils import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mdr", description="Deterministic multi-view reconstruction diagnostics")
    parser.add_argument("--root", default=".", help="workspace root")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=argparse.SUPPRESS, help="workspace root")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init-sample", parents=[common], help="create config and deterministic PNG samples")
    run = sub.add_parser("run", parents=[common], help="run incremental pipeline")
    run.add_argument("--serial", action="store_true")
    run.add_argument("--no-snapshot", action="store_true")
    cmd = sub.add_parser("apply", parents=[common], help="apply a JSON command stream and run")
    cmd.add_argument("commands")
    cmd.add_argument("--serial", action="store_true")
    sub.add_parser("diagnostics", parents=[common], help="generate error attribution report")
    sub.add_parser("recover", parents=[common], help="inspect committed snapshot recovery decision")
    sub.add_parser("fingerprint", parents=[common], help="print canonical product fingerprint")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).resolve()

    if args.command == "init-sample":
        config_path = initialize_workspace(root)
        config = load_config(config_path)
        info = generate_sample_set(root / "samples", config)
        print(json.dumps({"ok": True, "views": len(info["dataset"]["views"]), "root": str(root)}, sort_keys=True))
        return 0

    if args.command == "apply":
        commands = load_command_stream(args.commands)
        result = apply_commands(root, commands)
        pipeline = Pipeline(root, parallel=not args.serial, command_errors=result["errors"])
        state = pipeline.run()
        print(json.dumps({
            "commands": result["business_code_counts"],
            "business_codes": state["business_code_counts"],
            "fingerprint": canonical_fingerprint(state),
            "pending": len(state["pending"]),
        }, sort_keys=True))
        return 0

    if args.command == "run":
        state = Pipeline(root, parallel=not args.serial, use_snapshot=not args.no_snapshot).run()
        print(json.dumps({
            "business_codes": state["business_code_counts"],
            "fingerprint": canonical_fingerprint(state),
            "pending": len(state["pending"]),
            "voxel_counts": state["voxels"].get("counts", {}),
        }, sort_keys=True))
        return 0

    if args.command == "diagnostics":
        report = run_diagnostics(root)
        print(json.dumps(report["default_comparison"]["counts"], sort_keys=True))
        return 0

    if args.command == "recover":
        manager = SnapshotManager(ContentLedger(root))
        print(json.dumps(manager.recover(), sort_keys=True, default=str))
        return 0

    if args.command == "fingerprint":
        state = Pipeline(root, parallel=False).run()
        print(canonical_fingerprint(state))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
