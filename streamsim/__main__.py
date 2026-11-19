"""Command-line entry point for the deterministic streaming simulator."""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from pathlib import Path

from .canon import canonical_json
from .engine import EXIT_INTERRUPTED, SimulationInterrupted, simulate
from .errors import BusinessError, error_response
from .model import load_session
from .prefetch import run_prefetch_validation
from .samples import generate_samples, clean_sample_runs


def _install_signal_handlers(stop_event: threading.Event) -> dict:
    previous = {}

    def handler(signum: int, frame: object) -> None:
        stop_event.set()

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, handler)
    return previous


def _restore_signal_handlers(previous: dict) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _exit_code_for(error: BusinessError) -> int:
    if error.code.startswith("E_DIGEST") or error.code == "E_SNAPSHOT":
        return 4
    if error.code == "E_PREFETCH":
        return 5
    if error.code == "E_IO":
        return 3
    return 2


def command_run(args: argparse.Namespace) -> int:
    session = load_session(args.session)
    stop_event = threading.Event()
    previous = _install_signal_handlers(stop_event)
    try:
        simulator, _, recovery = simulate(
            session,
            stop_event=stop_event,
            event_delay_ms=args.event_delay_ms,
        )
        if not args.skip_prefetch:
            prefetch = run_prefetch_validation(
                [job.to_dict() for job in simulator.state.planned_jobs],
                concurrency=session.client.concurrency,
                memory_budget_bytes=session.client.memory_budget_bytes,
            )
        else:
            prefetch = None
        simulator.manager.write_log(simulator.state.journal)
        report = simulator.build_report(
            prefetch=prefetch,
            log_path=str(simulator.manager.log_path),
        )
        output_path = (
            Path(args.report).resolve()
            if args.report
            else simulator.manager.report_path
        )
        simulator.manager.write_report(report, output_path)
        report["report_path"] = str(output_path)
        print(canonical_json(report))
        return 0
    except SimulationInterrupted as exc:
        print(
            canonical_json(
                {
                    "ok": True,
                    "interrupted": True,
                    "exit_code": EXIT_INTERRUPTED,
                    "marker": exc.marker,
                }
            )
        )
        return EXIT_INTERRUPTED
    except BusinessError as exc:
        print(canonical_json(error_response(exc)))
        return _exit_code_for(exc)
    finally:
        _restore_signal_handlers(previous)


def command_generate(args: argparse.Namespace) -> int:
    if args.clean:
        clean_sample_runs(args.destination)
    manifest = generate_samples(args.destination, seed=args.seed)
    print(canonical_json({"ok": True, "samples": manifest}))
    return 0


def command_verify(args: argparse.Namespace) -> int:
    session = load_session(args.session)
    from .snapshot import SnapshotManager

    manager = SnapshotManager(
        session.state_dir,
        session.snapshot_dir,
        session.version_id,
        session.events.digest,
    )
    integrity = manager.inspect_integrity()
    payload = {
        "ok": integrity["ok"],
        "version_id": session.version_id,
        "integrity": integrity,
    }
    if not integrity["ok"]:
        payload["error"] = {
            "code": "E_DIGEST",
            "message": "one or more snapshots failed integrity validation",
        }
        print(canonical_json(payload))
        return 4
    print(canonical_json(payload))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m streamsim")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run or resume a simulation session")
    run_parser.add_argument("session")
    run_parser.add_argument("--report", help="optional report output path")
    run_parser.add_argument(
        "--event-delay-ms",
        type=int,
        default=0,
        help="wall-clock delay at event boundaries (test hook; never changes simulation state)",
    )
    run_parser.add_argument(
        "--skip-prefetch",
        action="store_true",
        help="skip the concurrent prefetch validation layer",
    )
    run_parser.set_defaults(func=command_run)

    generate_parser = subparsers.add_parser("generate-samples", help="write deterministic local samples")
    generate_parser.add_argument("destination")
    generate_parser.add_argument("--seed", type=int, default=20261119)
    generate_parser.add_argument("--clean", action="store_true")
    generate_parser.set_defaults(func=command_generate)

    verify_parser = subparsers.add_parser("verify-snapshots", help="validate persisted snapshots")
    verify_parser.add_argument("session")
    verify_parser.set_defaults(func=command_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except BusinessError as exc:
        print(canonical_json(error_response(exc)))
        return _exit_code_for(exc)
    except OSError as exc:
        print(
            canonical_json(
                error_response(BusinessError("E_IO", str(exc), details={"type": type(exc).__name__}))
            )
        )
        return 3
    except KeyboardInterrupt:
        print(canonical_json({"ok": False, "error": {"code": "E_INTERRUPTED", "message": "keyboard interrupt"}}))
        return 130


if __name__ == "__main__":
    sys.exit(main())
