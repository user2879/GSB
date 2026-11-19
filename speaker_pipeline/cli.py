"""Command line entry points for the speaker pipeline."""

import argparse
import json
import os
import shutil
import signal
import sys

from .commands import load_command_stream
from .config import Config
from .engine import PipelineEngine
from .injections import apply_file_injections, load_injections
from .report import build_report, canonical_result, write_report


def _prepare_work(dataset_dir, work_dir, reset=False):
    os.makedirs(work_dir, exist_ok=True)
    if reset:
        for name in os.listdir(work_dir):
            path = os.path.join(work_dir, name)
            if os.path.isfile(path) or os.path.islink(path):
                os.unlink(path)
            else:
                shutil.rmtree(path)
    copied = []
    for name in sorted(os.listdir(dataset_dir)):
        if name.lower().endswith(".wav"):
            shutil.copy2(
                os.path.join(dataset_dir, name), os.path.join(work_dir, name)
            )
            copied.append(name)
    return copied


def _make_engine(args, work_dir, store_dir, injections=None):
    config = Config.load(args.config)
    return PipelineEngine(
        work_dir, config, store_dir, injections=injections
    )


def cmd_full(args):
    work_dir = os.path.join(args.workspace, "work_audio")
    store_dir = os.path.join(args.workspace, "store_full")
    _prepare_work(args.dataset, work_dir, reset=True)
    engine = _make_engine(args, work_dir, store_dir)
    result = engine.run_pipeline()
    manifest = engine.commit_snapshot(result)
    report = build_report(result, engine.config, recovery_version=None)
    report["snapshot_version"] = manifest["global_version"]
    write_report(report, os.path.join(args.workspace, "report_full.json"))
    with open(os.path.join(args.workspace, "canonical_full.json"), "w",
              encoding="utf-8") as handle:
        json.dump(canonical_result(result), handle, sort_keys=True, indent=2)
    print(
        json.dumps(
            {
                "status": report["status"],
                "clusters": report["n_clusters"],
                "intervals": report["n_intervals"],
                "snapshot_version": manifest["global_version"],
                "report": os.path.join(args.workspace, "report_full.json"),
            },
            indent=2,
        )
    )
    return 0


def cmd_incremental(args):
    work_dir = os.path.join(args.workspace, "work_audio")
    store_dir = os.path.join(args.workspace, "store_incr")
    injections = load_injections(args.injections)
    _prepare_work(args.dataset, work_dir, reset=not args.resume)
    engine = _make_engine(args, work_dir, store_dir, injections=injections)
    recovery_version = None
    if args.resume:
        manifest = engine.restore()
        recovery_version = manifest["global_version"] if manifest else None
        print(json.dumps({"recovered_version": recovery_version}))
    else:
        apply_file_injections(work_dir, injections)
    commands = load_command_stream(args.commands)
    stop_after = args.stop_after
    stop = {"flag": False}

    def handle_signal(signum, frame):
        stop["flag"] = True

    if sys.platform != "win32":
        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)

    result = engine.run_pipeline()
    engine.commit_snapshot(result)
    command_failures = []
    processed = 0
    for command in commands:
        if stop_after is not None and processed >= stop_after:
            break
        try:
            engine.apply_command(command, current_result=result)
        except Exception as exc:  # reported through failures below
            record = {
                "file": command.get("file"),
                "interval_index": None,
                "stage": "command",
                "code": getattr(exc, "code", "E_COMMAND"),
                "error_type": type(exc).__name__,
                "message": str(exc),
                "injection_id": getattr(exc, "injection_id", None),
            }
            if record["injection_id"] is None and command.get("injection_id"):
                record["injection_id"] = command["injection_id"]
            command_failures.append(record)
        result = engine.run_pipeline()
        processed += 1
        if (
            engine.command_cursor
            % max(1, engine.config.data["snapshot_interval"])
            == 0
        ):
            engine.commit_snapshot(result)
        if stop["flag"]:
            break
    manifest = engine.commit_snapshot(result)
    result["failures"] = result.get("failures", []) + command_failures
    injection_hits = [
        {
            "id": item["id"],
            "fault": item["fault"],
            "code": item["fault"],
        }
        for item in injections
    ]
    report = build_report(
        result, engine.config, injection_hits, recovery_version
    )
    report["snapshot_version"] = manifest["global_version"]
    report["commands_processed"] = processed
    write_report(
        report, os.path.join(args.workspace, "report_incremental.json")
    )
    with open(
        os.path.join(args.workspace, "canonical_incremental.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(canonical_result(result), handle, sort_keys=True, indent=2)
    print(
        json.dumps(
            {
                "status": report["status"],
                "commands_processed": processed,
                "snapshot_version": manifest["global_version"],
                "recovered_version": recovery_version,
                "failures": len(report["failures"]),
            },
            indent=2,
        )
    )
    return 0


def cmd_verify(args):
    with open(args.full, "r", encoding="utf-8") as handle:
        full = json.load(handle)
    with open(args.incremental, "r", encoding="utf-8") as handle:
        incr = json.load(handle)
    if full == incr:
        print(json.dumps({"parity": True}))
        return 0
    print(json.dumps({"parity": False}, indent=2))
    return 2



def cmd_rebuild_final(args):
    """Full rebuild on the already materialized final audio + final state."""
    work_dir = os.path.join(args.workspace, "work_audio")
    incr_store = os.path.join(args.workspace, "store_incr")
    full_store = os.path.join(args.workspace, "store_full_rebuild")
    config = Config.load(args.config)

    # Recover the final mutable state produced by the incremental run.
    state_engine = PipelineEngine(work_dir, config, incr_store)
    manifest = state_engine.restore()
    if manifest is None:
        raise SystemExit("no committed snapshot found; run incremental first")

    engine = PipelineEngine(work_dir, config, full_store)
    engine.load_state(
        {
            "threshold": manifest["threshold"],
            "override_ledger": manifest["override_ledger"],
            "command_cursor": manifest["command_cursor"],
        }
    )
    result = engine.run_pipeline()
    manifest_out = engine.commit_snapshot(result)
    report = build_report(result, engine.config)
    report["snapshot_version"] = manifest_out["global_version"]
    write_report(
        report, os.path.join(args.workspace, "report_full_rebuild.json")
    )
    with open(
        os.path.join(args.workspace, "canonical_full.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(canonical_result(result), handle, sort_keys=True, indent=2)
    print(
        json.dumps(
            {
                "status": report["status"],
                "clusters": report["n_clusters"],
                "intervals": report["n_intervals"],
                "snapshot_version": manifest_out["global_version"],
            },
            indent=2,
        )
    )
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description="Speaker pipeline CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--dataset", required=True)
        p.add_argument("--config", required=True)
        p.add_argument("--workspace", required=True)

    p_full = sub.add_parser("full", help="one-shot full rebuild")
    add_common(p_full)
    p_full.set_defaults(func=cmd_full)

    p_incr = sub.add_parser("incremental", help="command-driven incremental run")
    add_common(p_incr)
    p_incr.add_argument("--commands", required=True)
    p_incr.add_argument("--injections")
    p_incr.add_argument("--resume", action="store_true")
    p_incr.add_argument("--stop-after", type=int, default=None)
    p_incr.set_defaults(func=cmd_incremental)

    p_rebuild = sub.add_parser(
        "rebuild-final", help="full rebuild on final edited audio"
    )
    add_common(p_rebuild)
    p_rebuild.set_defaults(func=cmd_rebuild_final)

    p_verify = sub.add_parser("verify", help="compare canonical results")
    p_verify.add_argument("--full", required=True)
    p_verify.add_argument("--incremental", required=True)
    p_verify.set_defaults(func=cmd_verify)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
