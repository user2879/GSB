"""Command-line entry point; all fault injection happens through arguments."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from .errors import BusinessError
from .runner import replay_pipeline, run_pipeline
from .util import canonical_json, monotonic_ms

EXIT_CODES = {
    "E_SPEC": 2,
    "E_EVENT": 3,
    "E_LATE": 4,
    "E_LEDGER": 5,
    "E_BUDGET": 6,
    "E_REPLAY": 7,
}


def _write_report(path: str | None, payload: dict[str, Any]) -> None:
    text = canonical_json(payload)
    if path:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
    else:
        sys.stdout.write(text + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="swa", description="Deterministic streaming window aggregation middleware")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="generate events, aggregate windows and commit the ledger")
    run.add_argument("--spec", required=True)
    run.add_argument("--ledger", required=True)
    run.add_argument("--result", required=True)
    run.add_argument("--log", required=True)
    run.add_argument("--global-limit", type=int, default=3)
    run.add_argument("--samples")
    run.add_argument("--report")
    run.add_argument(
        "--inject",
        choices=["corrupt-spec", "unsupported-ledger-version", "budget-exhausted"],
    )
    run.add_argument("--ledger-format-version", type=int, default=1)
    run.add_argument("--budget-wait-threshold-ms", type=float, default=5000.0)
    run.add_argument("--crash-after-commits", type=int)
    run.add_argument("--reorder", choices=["reverse"])
    run.add_argument("--schedule-jitter", action="store_true")

    replay = sub.add_parser("replay", help="recompute windows and verify them against the ledger")
    replay.add_argument("--spec", required=True)
    replay.add_argument("--ledger", required=True)
    replay.add_argument("--log", required=True)
    replay.add_argument("--global-limit", type=int, default=3)
    replay.add_argument("--samples")
    replay.add_argument("--report")
    replay.add_argument("--ledger-format-version", type=int, default=1)

    view = sub.add_parser("view", help="print the current committed view of a ledger")
    view.add_argument("--ledger", required=True)
    view.add_argument("--report")
    return parser


def _run(args: argparse.Namespace) -> dict[str, Any]:
    injection = args.inject
    threshold = args.budget_wait_threshold_ms
    ledger_version = args.ledger_format_version
    if injection == "unsupported-ledger-version":
        ledger_version = 999
    if injection == "budget-exhausted":
        threshold = 0.0
    return run_pipeline(
        args.spec,
        args.ledger,
        args.result,
        log_path=args.log,
        global_limit=args.global_limit,
        sample_path=args.samples,
        injection=injection,
        ledger_format_version=ledger_version,
        budget_wait_threshold_ms=threshold,
        crash_after_commits=args.crash_after_commits,
        reorder=args.reorder,
        schedule_jitter=args.schedule_jitter,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    triggered = monotonic_ms()
    try:
        if args.command == "run":
            payload = _run(args)
            _write_report(args.report, payload)
            return 0
        if args.command == "replay":
            payload = replay_pipeline(
                args.spec,
                args.ledger,
                log_path=args.log,
                global_limit=args.global_limit,
                sample_path=args.samples,
                ledger_format_version=args.ledger_format_version,
            )
            _write_report(args.report, payload)
            return 0
        # view
        from .ledger import Ledger

        ledger = Ledger(args.ledger)
        payload = {"status": "ok", "view": ledger.current_view(), "stale": ledger.stale_view()}
        ledger.close()
        _write_report(args.report, payload)
        return 0
    except BusinessError as exc:
        payload = exc.to_payload(triggered_monotonic_ms=triggered)
        _write_report(getattr(args, "report", None), payload)
        return EXIT_CODES.get(exc.code, 1)


if __name__ == "__main__":
    raise SystemExit(main())
