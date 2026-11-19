"""Command line entry point for the RTS replay workbench."""
from __future__ import annotations

import argparse
import json
import os
import sys

from .demo import run_demo
from .errors import RtsError
from .events import load_events
from .generator import generate_all, load_rules
from .hashing import sha256_file
from .offline import run_offline, write_offline_reports


def _default_workspace() -> str:
    return os.path.join(os.getcwd(), "workspace")


def cmd_generate(args: argparse.Namespace) -> int:
    manifests = generate_all(args.workspace)
    for manifest in manifests:
        print(
            f"{manifest['scenario']}: events={manifest['event_count']} "
            f"ticks={manifest['tick_count']} sha256={manifest['events_sha256'][:16]}"
        )
    return 0


def cmd_offline(args: argparse.Namespace) -> int:
    sample_dir = os.path.join(args.workspace, "samples", args.scenario)
    rules = load_rules(os.path.join(sample_dir, "rules.json"))
    events = load_events(os.path.join(sample_dir, "events.jsonl"))
    snapshot_dir = os.path.join(args.workspace, "snapshots", args.scenario)
    report_dir = os.path.join(args.workspace, "reports", args.scenario)
    result = run_offline(rules, events, snapshot_dir=snapshot_dir)
    paths = write_offline_reports(report_dir, result)
    for name, path in paths.items():
        print(f"{name}: {path}")
    print(f"final_state_hash={result.engine.state.state_hash()}")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    master = run_demo(args.workspace)
    print(json.dumps(master, indent=2, sort_keys=True))
    return 0


def cmd_gui(args: argparse.Namespace) -> int:
    from .gui.main_window import run_gui

    return run_gui(args.workspace)


def cmd_hashcheck(args: argparse.Namespace) -> int:
    root = os.path.join(args.workspace, "samples")
    ok = True
    for name in sorted(os.listdir(root)):
        manifest_path = os.path.join(root, name, "manifest.json")
        if not os.path.isfile(manifest_path):
            continue
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.loads(handle.read())
        events_path = os.path.join(root, name, manifest["events_file"])
        actual = sha256_file(events_path)
        match = actual == manifest["events_sha256"]
        ok = ok and match
        print(f"{name}: hash_match={match}")
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rts-replay",
        description="Deterministic RTS resource/unit replay workbench",
    )
    parser.add_argument(
        "--workspace", default=_default_workspace(), help="writable workspace directory"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("generate", help="generate deterministic sample set").set_defaults(
        func=cmd_generate
    )
    offline_parser = sub.add_parser("offline", help="run offline full synthesis")
    offline_parser.add_argument("scenario")
    offline_parser.set_defaults(func=cmd_offline)
    sub.add_parser("demo", help="run the fixed end-to-end demo sequence").set_defaults(
        func=cmd_demo
    )
    sub.add_parser("gui", help="launch the interactive timeline window").set_defaults(
        func=cmd_gui
    )
    sub.add_parser(
        "hashcheck", help="verify sample files against manifest hashes"
    ).set_defaults(func=cmd_hashcheck)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except RtsError as exc:
        print(json.dumps(exc.to_dict(), indent=2, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
