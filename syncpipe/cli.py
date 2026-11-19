"""Command line entry points.

One-shot end-to-end demo::

    python -m syncpipe.cli demo --workspace workspace --seed 20261119

Independent validation::

    python -m syncpipe.cli validate --root workspace/runs/normal_game --seed 20261119
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List

from .engine import Engine, rebuild_from_anchor
from .generator import SCENARIOS, generate_all, read_stream, write_streams
from .parser import parse_stream
from .recovery import recover
from .state import canonical_bytes, state_digest
from .storage import Storage
from .validator import validate_directory
from .worker import run_worker

DEFAULT_SEED = 20261119


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(canonical_bytes(payload) + b"\n")


def _run_scenario(root: str, events: List[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    storage = Storage(root)
    lines = [json.dumps(event, sort_keys=True, separators=(",", ":")) for event in events]
    tick_groups, parser_errors, quarantined = parse_stream(lines)
    engine = Engine(storage, seed)
    summary = engine.run(tick_groups, parser_errors)
    engine.write_shutdown()
    summary["quarantined_events"] = quarantined
    return summary


def cmd_generate(args: argparse.Namespace) -> int:
    paths = write_streams(args.samples, args.seed)
    for name, path in paths.items():
        print(f"generated {name}: {path}")
    return 0


def cmd_synthesize(args: argparse.Namespace) -> int:
    events = read_stream(args.stream)
    summary = _run_scenario(args.root, events, args.seed)
    _write_json(os.path.join(args.root, "summary.json"), summary)
    print(json.dumps({k: summary[k] for k in ("head_version", "head_state_hash", "applied", "rejected")}, indent=2))
    return 0


def cmd_compact(args: argparse.Namespace) -> int:
    storage = Storage(args.root)
    manifest = storage.load_manifest()
    before = len(storage.available_snapshot_versions())
    engine = Engine(storage, manifest["seed"])
    engine.manifest = manifest
    compaction = engine.compact()
    after = len(storage.available_snapshot_versions())
    report = {"before_snapshots": before, "after_snapshots": after, **compaction}
    _write_json(os.path.join(args.root, "compaction.json"), report)
    validation = validate_directory(args.root, args.seed)
    _write_json(os.path.join(args.root, "post_compaction_validation.json"), validation)
    print(json.dumps(report, indent=2))
    print("post-compaction validation ok:", validation["ok"])
    return 0 if validation["ok"] else 2


def cmd_catchup(args: argparse.Namespace) -> int:
    storage = Storage(args.root)
    manifest = storage.load_manifest()
    target = manifest["head_version"]
    anchors = [v for v in storage.available_snapshot_versions() if v < target]
    if not anchors:
        print("no ancestor anchor available", file=sys.stderr)
        return 2
    anchor_version = min(anchors)
    rebuilt_snapshot, rebuilt_state, replayed = rebuild_from_anchor(storage, target)
    direct = storage.read_snapshot_file(target)
    report = {
        "anchor_version": anchor_version,
        "target_version": target,
        "replayed_log_versions": [entry["version"] for entry in replayed],
        "field_equal": rebuilt_state == direct["state"],
        "hash_equal": state_digest(rebuilt_state) == direct["state_hash"],
        "rebuilt_snapshot_hash": rebuilt_snapshot["snapshot_hash"],
        "direct_snapshot_hash": direct["snapshot_hash"],
        "snapshot_byte_equal": canonical_bytes(rebuilt_snapshot["state"])
        == canonical_bytes(direct["state"]),
    }
    _write_json(os.path.join(args.root, "catchup.json"), report)
    print(json.dumps(report, indent=2))
    return 0 if report["field_equal"] and report["hash_equal"] else 2


def cmd_validate(args: argparse.Namespace) -> int:
    report = validate_directory(args.root, args.seed)
    _write_json(os.path.join(args.root, "validation_report.json"), report)
    print("validation ok:", report["ok"])
    if report["first_mismatch"]:
        print(json.dumps(report["first_mismatch"], indent=2))
    return 0 if report["ok"] else 3


def cmd_worker(args: argparse.Namespace) -> int:
    result = run_worker(
        args.root,
        args.stream,
        args.seed,
        stop_after_version=args.stop_after_version,
        clean_shutdown=not args.no_shutdown,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def _kill_subprocess(root: str, stream_path: str, seed: int, kill_after_version: int) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "syncpipe.cli",
        "worker",
        "--root",
        root,
        "--stream",
        stream_path,
        "--seed",
        str(seed),
        "--no-shutdown",
        "--stop-after-version",
        str(kill_after_version),
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    progress_path = os.path.join(root, "progress", "worker_progress.json")
    deadline = time.time() + 60
    while time.time() < deadline:
        if os.path.exists(progress_path):
            with open(progress_path, "rb") as handle:
                progress = json.loads(handle.read().decode("utf-8"))
            if progress["version"] == kill_after_version:
                break
        if process.poll() is not None:
            break
        time.sleep(0.02)
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
        killed = True
    else:
        killed = False
    stdout, stderr = process.communicate()
    return {
        "killed": killed,
        "returncode": process.returncode,
        "worker_stdout": stdout.decode("utf-8", "replace").strip(),
        "worker_stderr": stderr.decode("utf-8", "replace").strip(),
    }


# Fixed logical name for the uncommitted tail used by the kill drill. By
# pinning the name (and never rotating the segment the worker is killed in),
# the on-disk layout and all error messages are identical across runs.
TAIL_LOG_NAME = "log_uncommitted_tail.jsonl"


def _corrupt_tail(root: str) -> str:
    storage = Storage(root)
    target_dir = os.path.join(root, "logs")
    # Normalize whatever the killed worker left behind to the fixed tail name.
    pending = storage.list_pending_logs()
    segments = storage.list_log_segments()
    source = pending[0] if pending else segments[-1]
    target = os.path.join(target_dir, TAIL_LOG_NAME)
    if os.path.abspath(source) != os.path.abspath(target):
        os.replace(source, target)
    with open(target, "ab") as handle:
        handle.write(b'{"version": 999999, "tick": 1, "broken')
        handle.flush()
        os.fsync(handle.fileno())
    return TAIL_LOG_NAME


def cmd_demo(args: argparse.Namespace) -> int:
    workspace = os.path.abspath(args.workspace)
    samples_dir = os.path.join(workspace, "samples")
    runs_dir = os.path.join(workspace, "runs")
    reports_dir = os.path.join(workspace, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    if os.path.exists(runs_dir):
        shutil.rmtree(runs_dir)
    os.makedirs(runs_dir)

    streams = generate_all(args.seed)
    write_streams(samples_dir, args.seed)
    report: Dict[str, Any] = {"seed": args.seed, "scenarios": {}}

    # 1. normal game: full synthesis + shutdown marker
    normal_root = os.path.join(runs_dir, "normal_game")
    normal_summary = _run_scenario(normal_root, streams["normal_game"], args.seed)
    report["scenarios"]["normal_game"] = {
        "head_version": normal_summary["head_version"],
        "head_state_hash": normal_summary["head_state_hash"],
        "errors": normal_summary["errors"],
    }

    # 2. conflict game: deterministic LWW adjudication recorded
    conflict_root = os.path.join(runs_dir, "conflict_game")
    conflict_summary = _run_scenario(conflict_root, streams["conflict_game"], args.seed)
    report["scenarios"]["conflict_game"] = {
        "head_version": conflict_summary["head_version"],
        "head_state_hash": conflict_summary["head_state_hash"],
        "conflict_count": conflict_summary["conflict_count"],
        "sample_conflicts": conflict_summary["hash_chain"][:0],
        "conflicts": conflict_summary["conflicts"],
    }

    # 3. catch-up game: E_BASE_VERSION suspensions then anchor rebuild
    catchup_root = os.path.join(runs_dir, "catchup_game")
    catchup_summary = _run_scenario(catchup_root, streams["catchup_game"], args.seed)
    Storage(catchup_root)  # ensure layout exists (already created by engine)
    storage = Storage(catchup_root)
    rebuilt_snapshot, rebuilt_state, replayed = rebuild_from_anchor(
        storage, catchup_summary["head_version"]
    )
    direct = storage.read_snapshot_file(catchup_summary["head_version"])
    report["scenarios"]["catchup_game"] = {
        "head_version": catchup_summary["head_version"],
        "head_state_hash": catchup_summary["head_state_hash"],
        "base_version_errors": [e for e in catchup_summary["errors"] if e["code"] == "E_BASE_VERSION"],
        "catchup_field_equal": rebuilt_state == direct["state"],
        "catchup_hash_equal": state_digest(rebuilt_state) == direct["state_hash"],
        "replayed_versions": [entry["version"] for entry in replayed],
    }

    # 4. compaction on the normal run, then re-validate reconstructibility
    normal_storage = Storage(normal_root)
    manifest = normal_storage.load_manifest()
    compactor = Engine(normal_storage, args.seed)
    compactor.manifest = manifest
    compaction = compactor.compact()
    post_compaction = validate_directory(normal_root, args.seed)
    report["scenarios"]["normal_game"]["compaction"] = compaction
    report["scenarios"]["normal_game"]["post_compaction_ok"] = post_compaction["ok"]

    # 5. forced termination: kill a live worker subprocess mid-stream
    kill_root = os.path.join(runs_dir, "kill_game")
    os.makedirs(kill_root, exist_ok=True)
    stream_path = os.path.join(samples_dir, "kill_game.jsonl")
    kill_after = 12
    kill_info = _kill_subprocess(kill_root, stream_path, args.seed, kill_after)
    truncated_file = _corrupt_tail(kill_root)
    recovery_report = recover(Storage(kill_root))
    recovery_report["kill"] = kill_info
    recovery_report["corrupted_log_file"] = truncated_file
    kill_validation = validate_directory(kill_root, args.seed)
    truncation_recognized = (
        not kill_validation["ok"]
        and kill_validation["first_mismatch"] is not None
        and kill_validation["first_mismatch"].get("kind") == "log_truncation"
    )
    # Cross-check: an uninterrupted reference run must reach the same state
    # hash at the version the crashed worker recovered to.
    reference_root = os.path.join(runs_dir, "kill_reference")
    reference_summary = _run_scenario(reference_root, streams["kill_game"], args.seed)
    ref_manifest = Storage(reference_root).load_manifest()
    ref_at_recovered = next(
        item
        for item in ref_manifest["versions"]
        if item["version"] == recovery_report["recovered_version"]
    )
    recovery_byte_identical = (
        ref_at_recovered["state_hash"] == recovery_report["recovered_state_hash"]
    )
    report["scenarios"]["kill_game"] = {
        "kill": kill_info,
        "corrupted_log_file": truncated_file,
        "recovery": recovery_report,
        "truncation_recognized_by_validator": truncation_recognized,
        "validator_truncation": kill_validation["first_mismatch"],
        "reference_state_hash_at_recovered_version": ref_at_recovered["state_hash"],
        "recovery_byte_identical_to_reference": recovery_byte_identical,
    }

    # 6. per-version hash manifest across all runs (for byte-diff comparison)
    hash_manifest = {}
    validation_summaries = {}
    for name, root in (
        ("normal_game", normal_root),
        ("conflict_game", conflict_root),
        ("catchup_game", catchup_root),
        ("kill_reference", reference_root),
    ):
        storage = Storage(root)
        manifest_local = storage.load_manifest()
        hash_manifest[name] = manifest_local["versions"]
        validation_summaries[name] = validate_directory(root, args.seed)
    _write_json(os.path.join(reports_dir, "hash_manifest.json"), hash_manifest)
    _write_json(os.path.join(reports_dir, "validation_summaries.json"), validation_summaries)
    _write_json(os.path.join(reports_dir, "demo_report.json"), report)

    reference_validation_ok = all(item["ok"] for item in validation_summaries.values())
    console = {
        "seed": args.seed,
        "normal_head": normal_summary["head_state_hash"],
        "conflict_head": conflict_summary["head_state_hash"],
        "catchup_head": catchup_summary["head_state_hash"],
        "kill_recovered_version": recovery_report["recovered_version"],
        "kill_recovered_hash": recovery_report["recovered_state_hash"],
        "kill_truncation_recognized": truncation_recognized,
        "kill_recovery_byte_identical": recovery_byte_identical,
        "all_validations_ok": (
            post_compaction["ok"]
            and reference_validation_ok
            and report["scenarios"]["catchup_game"]["catchup_hash_equal"]
            and truncation_recognized
            and recovery_byte_identical
        ),
    }
    print(json.dumps(console, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="syncpipe")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate")
    p.add_argument("--samples", required=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("synthesize")
    p.add_argument("--stream", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.set_defaults(func=cmd_synthesize)

    p = sub.add_parser("compact")
    p.add_argument("--root", required=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.set_defaults(func=cmd_compact)

    p = sub.add_parser("catchup")
    p.add_argument("--root", required=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.set_defaults(func=cmd_catchup)

    p = sub.add_parser("validate")
    p.add_argument("--root", required=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("worker")
    p.add_argument("--root", required=True)
    p.add_argument("--stream", required=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--stop-after-version", type=int, default=None)
    p.add_argument("--no-shutdown", action="store_true")
    p.set_defaults(func=cmd_worker)

    p = sub.add_parser("demo")
    p.add_argument("--workspace", required=True)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.set_defaults(func=cmd_demo)
    return parser


def main(argv: List[str] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
