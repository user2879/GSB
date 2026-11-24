"""Command line interface for the vibration anomaly pipeline."""

import argparse
import json
import signal
import sys
import threading
import time
from pathlib import Path

from .config import load_config
from .edits import delete_segment, edit_config, replace_window
from .pipeline import run_pipeline
from .scheduler import ShutdownRequested
from .synth import load_descriptions


def _write_diagnostics(workspace, payload):
    path = Path(workspace) / "output" / "diagnostics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True, indent=2)
    return path


def cmd_init(args):
    from tools.generate_sample import build_descriptions
    from vibdet.config import DEFAULT_CONFIG
    import yaml

    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    with (workspace / "config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(DEFAULT_CONFIG, fh, sort_keys=True, allow_unicode=True)
    with (workspace / "descriptions.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(build_descriptions(), fh, sort_keys=True,
                       allow_unicode=True)
    print(f"initialized workspace at {workspace}")


def cmd_run(args):
    shutdown_event = threading.Event()

    def _handle(signum, frame):
        shutdown_event.set()

    # signal handlers can only be installed in the main thread; failures are
    # recorded rather than silently leaving the default terminating behavior
    signal_installed = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle)
            signal_installed.append(sig.name)
        except (ValueError, OSError):
            continue

    started = time.time()
    cfg = load_config(Path(args.workspace) / "config.yaml", args.override)
    descriptions = load_descriptions(
        Path(args.workspace) / "descriptions.yaml")
    try:
        result = run_pipeline(args.workspace, cfg, descriptions,
                              shutdown_event=shutdown_event,
                              point_cost=args.point_cost)
        status = "shutdown_after_stage" if shutdown_event.is_set() else "ok"
    except ShutdownRequested:
        result = run_pipeline(args.workspace, cfg, descriptions,
                              shutdown_event=shutdown_event,
                              point_cost=args.point_cost)
        status = "shutdown_after_stage"
    diagnostics = {
        "status": status,
        "elapsed_seconds": round(time.time() - started, 6),
        "shutdown_requested": shutdown_event.is_set(),
        "signal_handlers": signal_installed,
        "failures": result["failures"],
        "pending_queue_size": len(result["pending_queue"]),
        "snapshots": result["snapshots"],
        "cache_stats": result["cache_stats"],
        "device_count": len(result["devices"]),
    }
    diag_path = _write_diagnostics(args.workspace, diagnostics)
    print(json.dumps({"status": status,
                      "diagnostics": str(diag_path),
                      "output": str(Path(args.workspace) / 'output' / 'results.json')},
                     ensure_ascii=False, indent=2))


def cmd_edit(args):
    workspace = Path(args.workspace)
    if args.kind == "config":
        record = edit_config(workspace, args.path, _parse(args.value))
    elif args.kind == "replace-window":
        record = replace_window(workspace, args.device, args.index,
                                float(args.value) if args.value else None)
    elif args.kind == "delete-segment":
        record = delete_segment(workspace, args.device, args.segment_id)
    else:
        raise SystemExit(f"unknown edit kind {args.kind}")
    log_path = workspace / "edits.log.json"
    history = []
    if log_path.exists():
        history = json.loads(log_path.read_text(encoding="utf-8"))
    history.append(record)
    log_path.write_text(json.dumps(history, sort_keys=True, indent=2),
                        encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False))


def cmd_verify(args):
    from tools.acceptance import main as acceptance_main

    sys.argv = ["acceptance", "--workspace", args.workspace]
    acceptance_main()


def _parse(text):
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        low = text.lower()
        if low in ("true", "false"):
            return low == "true"
        try:
            return int(text)
        except ValueError:
            return float(text) if "." in text else text


def build_parser():
    parser = argparse.ArgumentParser(prog="vibdet")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init")
    init.add_argument("--workspace", required=True)
    init.set_defaults(func=cmd_init)

    run = sub.add_parser("run")
    run.add_argument("--workspace", required=True)
    run.add_argument("--override", action="append", default=[])
    run.add_argument("--point-cost", type=int, default=1)
    run.set_defaults(func=cmd_run)

    edit = sub.add_parser("edit")
    edit.add_argument("--workspace", required=True)
    edit.add_argument("--kind", required=True,
                      choices=["config", "replace-window", "delete-segment"])
    edit.add_argument("--path")
    edit.add_argument("--device")
    edit.add_argument("--index", type=int)
    edit.add_argument("--segment-id")
    edit.add_argument("--value", default="")
    edit.set_defaults(func=cmd_edit)

    verify = sub.add_parser("verify")
    verify.add_argument("--workspace", required=True)
    verify.set_defaults(func=cmd_verify)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
