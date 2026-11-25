"""Command-line entry point and the repeatable end-to-end demo.

Subcommands:

* ``generate`` - write deterministic local samples + manifest,
* ``run``      - load manifest + command stream and execute the engine,
* ``rebuild``  - run the same final segment set from scratch (full rebuild),
* ``recover``  - demonstrate forced-termination recovery from the snapshot.

All reports are written as deterministic JSON; nothing touches the network and
no container runtime is required.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from typing import Any, Dict, List, Optional

from .cache import ProductCache
from .config import Config, load_config, load_injection_manifest
from .engine import VADEngine
from .injection import InjectionHarness
from .ledger import Ledger
from .loader import Command, load_command_stream, load_segment_manifest
from .samplegen import write_dataset
from .segments import Segment
from .snapshot import SnapshotStore, config_hash


def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, sort_keys=True, indent=2, ensure_ascii=False)


def _json_safe(obj: Any) -> Any:
    """Strip non-serializable runtime objects (e.g. Segment) from responses."""
    if isinstance(obj, Segment):
        return {"segment_id": obj.segment_id,
                "num_samples": obj.num_samples,
                "capture_ms": obj.capture_ms,
                "arrival_ms": obj.arrival_ms}
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def build_engine(config: Config, harness: Optional[InjectionHarness] = None,
                 recompute_budget: Optional[int] = None) -> VADEngine:
    os.makedirs(config.snapshot_dir, exist_ok=True)
    os.makedirs(config.ledger_dir, exist_ok=True)
    ledger = Ledger(os.path.join(config.ledger_dir, "ledger.jsonl")).load()
    cache = ProductCache(os.path.join(config.ledger_dir, "cache.json")).load()
    store = SnapshotStore(os.path.join(config.snapshot_dir, "store"))
    store.ensure_dirs()
    return VADEngine(
        config, ledger, cache, store, harness=harness or InjectionHarness([]),
        recompute_budget=recompute_budget)


def _segments_from_manifest(manifest_path: str, engine: VADEngine):
    result = load_segment_manifest(manifest_path)
    for failure in result.failures:
        engine.failures.append(failure)
        engine.ledger.record_failure(failure)
    return result


def cmd_generate(args) -> int:
    info = write_dataset(
        args.out_dir, seed=args.seed, segment_count=args.segments,
        segment_samples=args.segment_samples)
    print(json.dumps({
        "manifest": info["manifest_path"],
        "segment_count": len(info["segment_ids"]),
        "segment_ids": info["segment_ids"],
    }, indent=2))
    return 0


def _run_commands(engine: VADEngine, commands: List[Command],
                  segment_pool: Dict[str, Segment]):
    responses = []
    for command in commands:
        if command.op in ("append", "replace"):
            seg_id = command.payload.get("segment_id")
            if seg_id not in segment_pool:
                raise SystemExit(
                    f"command references unknown segment {seg_id}")
            payload = dict(command.payload)
            payload["segment"] = segment_pool[seg_id]
            command = Command(command.seq, command.op, payload)
        responses.append(engine.handle_command(command))
    return responses


def cmd_run(args) -> int:
    config = load_config(args.config)
    rules = load_injection_manifest(config.injection_manifest)
    harness = InjectionHarness(rules)
    engine = build_engine(
        config, harness, recompute_budget=args.recompute_budget)
    load_result = _segments_from_manifest(args.manifest, engine)
    commands = load_command_stream(args.commands)
    responses = _run_commands(engine, commands, load_result.segments)
    responses = _json_safe(responses)
    engine.ledger.flush()
    engine.cache.flush()
    report = {
        "config_hash": config_hash(config.to_dict()),
        "responses": responses,
        "exit_report": engine.exit_report(),
        "injection_report": harness.report.to_list(),
        "ledger_tail": engine.ledger.events[-20:],
    }
    _write_json(args.report, report)
    print(json.dumps({"report": args.report,
                      "responses": len(responses),
                      "injections_fired": harness.report.ids()}, indent=2))
    return 0


def cmd_rebuild(args) -> int:
    """Full rebuild: feed every accepted segment once, compare determinism."""
    config = load_config(args.config)
    engine = build_engine(config)
    load_result = _segments_from_manifest(args.manifest, engine)
    commands = [
        Command(i, "append", {"segment_id": sid})
        for i, sid in enumerate(sorted(load_result.segments))
    ]
    responses = _run_commands(engine, commands, load_result.segments)
    final = engine._last_result
    report = {
        "window_count": len(final.windows),
        "posteriors": {p.window_id: p.to_dict()
                       for p in final.posteriors.values()},
        "endpoints": [e.to_dict() for e in final.endpoints],
        "boundary_windows": sorted(p.window_id for p in
                                   final.posteriors.values()
                                   if p.boundary_window),
        "boundary_frames": sorted(d.frame_index for d in final.decisions
                                  if d.boundary_frame),
        "cache_keys": final.keys,
        "failures": [f.to_dict() for f in final.failures],
    }
    _write_json(args.report, report)
    print(json.dumps({"report": args.report,
                      "windows": report["window_count"],
                      "endpoints": len(report["endpoints"])}, indent=2))
    return 0


def cmd_recover(args) -> int:
    config = load_config(args.config)
    engine = build_engine(config)
    state = engine.restore_from_snapshot()
    if state is None:
        print(json.dumps({"recovered": False,
                          "reason": "no matching snapshot"}, indent=2))
        return 0
    print(json.dumps({"recovered": True,
                      "version": state.version,
                      "quant_level": state.quant_level,
                      "completed_segments": state.completed_segments,
                      "pending_windows": state.pending_windows,
                      "recompute_budget_used":
                      state.recompute_budget_used}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Streaming VAD diagnostics")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate deterministic samples")
    gen.add_argument("--out-dir", default="samples/demo")
    gen.add_argument("--seed", type=int, default=20261126)
    gen.add_argument("--segments", type=int, default=20)
    gen.add_argument("--segment-samples", type=int, default=3200)
    gen.set_defaults(func=cmd_generate)

    run = sub.add_parser("run", help="execute a command stream")
    run.add_argument("--config", required=True)
    run.add_argument("--manifest", required=True)
    run.add_argument("--commands", required=True)
    run.add_argument("--report", default="artifacts/run_report.json")
    run.add_argument("--recompute-budget", type=int, default=None)
    run.set_defaults(func=cmd_run)

    rebuild = sub.add_parser("rebuild", help="full from-scratch rebuild")
    rebuild.add_argument("--config", required=True)
    rebuild.add_argument("--manifest", required=True)
    rebuild.add_argument("--report", default="artifacts/rebuild_report.json")
    rebuild.set_defaults(func=cmd_rebuild)

    recover = sub.add_parser("recover", help="recover from snapshot")
    recover.add_argument("--config", required=True)
    recover.set_defaults(func=cmd_recover)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
