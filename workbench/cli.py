from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .engine import Workbench
from .hashing import canonical_dumps
from .sample import write_sample_files
from .snapshot import SnapshotStore
from .validation import load_and_validate


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def cmd_init_sample(args: argparse.Namespace) -> int:
    paths = write_sample_files(args.dir)
    _print_json({"created": {key: str(path) for key, path in paths.items()}})
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    result = load_and_validate(args.config, events_path=args.events)
    summary = {
        "valid": result["valid"],
        "config_hash": result["config_hash"],
        "loaded_nodes": sorted(result["nodes"]),
        "valid_events": len(result["events"]),
        "failure_count": len(result["failures"]),
        "failures": result["failures"],
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    _print_json(summary)
    return 0 if result["valid"] else 2


def cmd_run(args: argparse.Namespace) -> int:
    validation = load_and_validate(args.config, events_path=args.events)
    out_dir = Path(args.out)
    store = SnapshotStore(out_dir, config_hash=validation["config_hash"], fresh=args.fresh)
    recovery = store.recovery_plan(validation["events"])
    recovery["parallel"] = bool(args.parallel)
    recovery["max_events"] = args.max_events
    (out_dir).mkdir(parents=True, exist_ok=True)
    (out_dir / "recovery.json").write_text(
        json.dumps(recovery, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    workbench = Workbench(
        validation,
        parallel=bool(args.parallel),
        log_dir=out_dir / "logs",
        snapshot_store=store,
        protocol_self_check=bool(args.protocol_self_check),
    )
    try:
        state = workbench.run(max_events=args.max_events, close_cluster=False)
        if args.protocol_self_check:
            workbench.protocol_self_check()
            store.commit(workbench, final=True)
    finally:
        workbench.close()
    pointer = store._read_pointer()
    report = store.read_current_report() if pointer else None
    summary = {
        "status": "completed",
        "version": __version__,
        "out_dir": str(out_dir),
        "current_pointer": pointer,
        "recovery": recovery,
        "cursor": state["cursor"],
        "processed_event_count": state["processed_event_count"],
        "node_count": len(state["node_versions"]),
        "authority_keys": sorted(state["authority"]),
        "eviction_count": len(state["evictions"]),
        "code_counts": state["code_counts"],
        "failure_count": len(state["failure_list"]),
        "report_available": report is not None,
    }
    _print_json(summary)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="workbench", description="Lease and watermark diagnostic workbench")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    init_parser = sub.add_parser("init-sample", help="write a deterministic local sample")
    init_parser.add_argument("--dir", default="examples/generated")
    init_parser.set_defaults(func=cmd_init_sample)

    validate_parser = sub.add_parser("validate", help="validate config, manifest, and event stream")
    validate_parser.add_argument("--config", required=True)
    validate_parser.add_argument("--events")
    validate_parser.add_argument("--out")
    validate_parser.set_defaults(func=cmd_validate)

    run_parser = sub.add_parser("run", help="start node subprocesses and run the stream")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--events")
    run_parser.add_argument("--out", default="artifacts/run")
    run_parser.add_argument("--parallel", action="store_true")
    run_parser.add_argument("--fresh", action="store_true")
    run_parser.add_argument("--max-events", type=int, default=None)
    run_parser.add_argument("--protocol-self-check", action="store_true")
    run_parser.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
