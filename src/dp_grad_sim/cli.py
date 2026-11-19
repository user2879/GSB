from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
from typing import Any, Sequence

from .config import load_config
from .data import load_shards
from .engine import Simulator
from .errors import SimError
from .injection import Injection
from .samplegen import generate_sample
from .snapshot import SnapshotManager
from .topology import load_topology
from .verify import verify_snapshot


def _json_print(value: Any, *, stream=None) -> None:
    print(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False), file=stream or sys.stdout)


def _error_payload(error: SimError) -> dict[str, Any]:
    return {
        "status": "error",
        "code": error.code,
        "message": error.message,
        "details": error.details,
        "exception_type": type(error).__name__,
    }


def _cmd_sample(args: argparse.Namespace) -> int:
    paths = generate_sample(
        args.out,
        seed=args.seed,
        workers=args.workers,
        steps=args.steps,
        layers=args.layers,
        hidden_dim=args.hidden_dim,
        topology_type=args.topology,
        link_bandwidth_bytes=args.link_bandwidth_bytes,
        window_budget_bytes=args.window_budget_bytes,
    )
    _json_print({"status": "ok", "files": {name: str(path) for name, path in paths.items()}})
    return 0


def _load_inputs(args: argparse.Namespace):
    config = load_config(args.config)
    shards = load_shards(args.shards, config.workers)
    topology = load_topology(args.topology, config.workers)
    return config, shards, topology


def _cmd_run(args: argparse.Namespace) -> int:
    config, shards, topology = _load_inputs(args)
    injection = Injection.load(args.inject) if args.inject else None
    stop_event = threading.Event()

    def request_stop(signum, frame) -> None:  # noqa: ANN001 - signal handler signature
        stop_event.set()
        print(
            json.dumps(
                {
                    "event": "graceful_stop_requested",
                    "signal": int(signum),
                    "message": "the current window will finish and its snapshot will be committed",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )

    previous_handlers: dict[int, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, request_stop)
    try:
        simulator = Simulator(
            config=config,
            shards=shards,
            topology=topology,
            snapshot_dir=args.snapshot_dir,
            injection=injection,
            stop_event=stop_event,
            max_windows=args.max_windows,
            verbose=args.verbose,
        )
        report = simulator.run()
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)

    summary = {
        "status": report["status"],
        "stopped_reason": report["stopped_reason"],
        "exit_code": report["exit_code"],
        "completed_steps": report["completed_steps"],
        "total_steps": report["total_steps"],
        "snapshot_dir": report["snapshot_dir"],
        "snapshot_version": report["snapshot_version"],
        "snapshot_phase": report["snapshot_phase"],
        "error_count": report["error_count"],
        "parameters_equal_across_workers": report["parameters_equal_across_workers"],
        "retained_fidelity_ids": report["retained_fidelity_ids"],
        "report_path": str(simulator.manager.root / "reports" / "latest.json"),
    }
    _json_print(summary)
    return int(report["exit_code"])


def _cmd_inspect(args: argparse.Namespace) -> int:
    manager = SnapshotManager(
        args.snapshot_dir,
        config_hash="",
        topology_hash="",
        data_hash="",
        snapshot_interval_steps=1,
        fidelity_generation_limit=0,
    )
    state = manager.load_latest()
    if state is None:
        _json_print({"status": "empty", "snapshot_dir": args.snapshot_dir})
        return 1
    summary = {
        "status": "ok",
        "snapshot_dir": str(manager.root),
        "version": state.version,
        "phase": state.phase,
        "completed_steps": state.completed_steps,
        "window_index": state.window_index,
        "window_start": state.window_start,
        "window_end": state.window_end,
        "manifest": state.manifest,
        "warnings": state.warnings,
        "retained_fidelity_ids": [entry.gradient_id for entry in state.fidelity_entries],
        "window_record_count": len(state.window_records),
    }
    _json_print(summary)
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    config, shards, topology = _load_inputs(args)
    snapshot_dir = args.snapshot_dir or str(config.snapshot_dir)
    result = verify_snapshot(
        config=config,
        shards=shards,
        topology=topology,
        snapshot_dir=snapshot_dir,
    )
    _json_print(result)
    return 0 if result["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dp-grad-sim",
        description="Deterministic local-loopback data-parallel gradient communication simulator",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample", help="generate a deterministic local sample directory")
    sample.add_argument("--out", required=True, help="output directory")
    sample.add_argument("--seed", type=int, default=20261119)
    sample.add_argument("--workers", type=int, default=3)
    sample.add_argument("--steps", type=int, default=4)
    sample.add_argument("--layers", type=int, default=2)
    sample.add_argument("--hidden-dim", type=int, default=8)
    sample.add_argument("--topology", choices=("ring", "full"), default="ring")
    sample.add_argument("--link-bandwidth-bytes", type=int, default=4096)
    sample.add_argument("--window-budget-bytes", type=int, default=65536)
    sample.set_defaults(func=_cmd_sample)

    run = subparsers.add_parser("run", help="run or resume a simulation")
    run.add_argument("--config", required=True)
    run.add_argument("--shards", required=True)
    run.add_argument("--topology", required=True)
    run.add_argument("--snapshot-dir", default=None)
    run.add_argument("--inject", default=None, help="optional injection YAML; inactive unless supplied")
    run.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="gracefully stop after processing this many windows in the current invocation",
    )
    run.add_argument("--verbose", action="store_true")
    run.set_defaults(func=_cmd_run)

    inspect = subparsers.add_parser("inspect", help="inspect the latest committed snapshot")
    inspect.add_argument("--snapshot-dir", required=True)
    inspect.set_defaults(func=_cmd_inspect)

    verify = subparsers.add_parser("verify", help="validate snapshot, budgets, ordering, and fidelity state")
    verify.add_argument("--config", required=True)
    verify.add_argument("--shards", required=True)
    verify.add_argument("--topology", required=True)
    verify.add_argument("--snapshot-dir", default=None)
    verify.set_defaults(func=_cmd_verify)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except SimError as error:
        _json_print(_error_payload(error), stream=sys.stderr)
        if error.code == "E_CONFIG_MISMATCH":
            return 2
        return 1
    except KeyboardInterrupt:
        return 130
    except Exception as error:  # last-resort CLI isolation; per-task errors are handled in the engine
        _json_print(
            {
                "status": "error",
                "code": "E_INTERNAL",
                "message": str(error),
                "exception_type": type(error).__name__,
            },
            stream=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
