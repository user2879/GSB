"""Command line entry point for the local event bus system.

Usage examples:
  python main.py run --sources samples/sources.json --subscribers samples/subscribers.json --ledger var/ledger.db
  python main.py run --kill-after 5 ...          # simulate forced termination
  python main.py recover ...                     # recover + report + replay check
  python main.py inject corrupt-source --sources var/sources.json
  python main.py inject corrupt-ledger --ledger var/ledger.db
  python main.py inject progress-overflow --ledger var/ledger.db
"""

import argparse
import json
import os
import sqlite3
import sys

from eventbus.bus import EventBus
from eventbus.errors import BusError, EXIT_CODES
from eventbus.ledger import open_ledger
from eventbus.logutil import StructuredLogger
from eventbus.spec import load_specs

DEFAULT_FORMAT_VERSION = 1


def _print_json(payload):
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))


def _build(args):
    logger = StructuredLogger(args.log)
    spec = load_specs(args.sources, args.subscribers)
    ledger, replacement = open_ledger(
        args.ledger, args.format_version, spec.set_id, spec.set_version,
        spec.subscriber_snapshot, logger)
    if replacement:
        _print_json({"ledger_replacement": replacement.replaced})
    ledger.check_progress_consistency()
    bus = EventBus(spec, ledger, logger)
    return spec, ledger, bus, logger


def cmd_run(args):
    spec, ledger, bus, logger = _build(args)
    report = bus.recover()
    _print_json({"recovery_report": report.to_dict()})
    bus.replay_check()
    commits = bus.process_all(kill_after=args.kill_after)
    summary = {
        "run_summary": {
            "commits_this_run": commits,
            "progress": {s.subscriber_id: ledger.get_progress(s.subscriber_id) or 0
                         for s in sorted(spec.subscribers,
                                         key=lambda item: item.subscriber_id)},
            "final_states": {sid: bus.machines[sid].state
                             for sid in sorted(bus.machines)},
        }
    }
    _print_json(summary)
    ledger.close()
    return 0


def cmd_recover(args):
    spec, ledger, bus, logger = _build(args)
    report = bus.recover()
    _print_json({"recovery_report": report.to_dict()})
    bus.replay_check()
    _print_json({"replay_check": "PASS"})
    ledger.close()
    return 0


def cmd_replay(args):
    spec, ledger, bus, logger = _build(args)
    bus.recover()
    bus.replay_check()
    _print_json({"replay_check": "PASS"})
    ledger.close()
    return 0


def cmd_inject(args):
    if args.kind == "corrupt-source":
        with open(args.sources, "r", encoding="utf-8") as handle:
            content = handle.read()
        with open(args.sources, "w", encoding="utf-8") as handle:
            handle.write(content[: max(1, len(content) // 2)])
        print("injected: event source file truncated (expect E_SPEC on next run)")
        return 0
    if args.kind == "corrupt-ledger":
        connection = sqlite3.connect(args.ledger)
        row = connection.execute("SELECT MIN(id) FROM transitions").fetchone()
        if row and row[0] is not None:
            connection.execute("UPDATE transitions SET to_state='corrupted' WHERE id=?",
                                (row[0],))
        else:
            connection.execute(
                "INSERT INTO transitions(subscriber_id, source_id, event_seq,"
                " from_state, to_state, clock_ms, ledger_version, complete)"
                " VALUES('injected', 'injected', 0, 'corrupted', 'corrupted', 0, 0, 1)")
        connection.commit()
        connection.close()
        print("injected: ledger transition corrupted (expect E_LEDGER on next run)")
        return 0
    if args.kind == "progress-overflow":
        connection = sqlite3.connect(args.ledger)
        row = connection.execute(
            "SELECT subscriber_id, committed_seq FROM progress ORDER BY subscriber_id"
            " LIMIT 1").fetchone()
        if row is None:
            connection.execute(
                "INSERT INTO progress(subscriber_id, committed_seq) VALUES('injected', 999)")
        else:
            connection.execute(
                "UPDATE progress SET committed_seq=? WHERE subscriber_id=?",
                (row[1] + 100, row[0]))
        connection.commit()
        connection.close()
        print("injected: progress moved beyond ledger records"
              " (expect E_PROGRESS on next run)")
        return 0
    raise BusError("E_SPEC", "unknown injection kind %r" % args.kind)


def build_parser():
    parser = argparse.ArgumentParser(description="Local event bus streaming system")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(target):
        target.add_argument("--sources", default="samples/sources.json")
        target.add_argument("--subscribers", default="samples/subscribers.json")
        target.add_argument("--ledger", default="var/ledger.db")
        target.add_argument("--log", default="var/eventbus.log")
        target.add_argument("--format-version", type=int, default=DEFAULT_FORMAT_VERSION)

    run = sub.add_parser("run", help="recover then process all remaining events")
    add_common(run)
    run.add_argument("--kill-after", type=int, default=None,
                     help="simulate forced termination after N commits")
    run.set_defaults(func=cmd_run)

    recover = sub.add_parser("recover", help="recover from ledger and print report")
    add_common(recover)
    recover.set_defaults(func=cmd_recover)

    replay = sub.add_parser("replay", help="run replay consistency check only")
    add_common(replay)
    replay.set_defaults(func=cmd_replay)

    inject = sub.add_parser("inject", help="inject a fault into input files")
    inject.add_argument("kind",
                        choices=["corrupt-source", "corrupt-ledger", "progress-overflow"])
    inject.add_argument("--sources", default="samples/sources.json")
    inject.add_argument("--ledger", default="var/ledger.db")
    inject.set_defaults(func=cmd_inject)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    log_path = getattr(args, "log", "var/eventbus.log")
    try:
        return args.func(args)
    except BusError as error:
        try:
            StructuredLogger(log_path).log(
                error.code, error.message, subscriber_id=error.subscriber_id,
                event_seq=error.event_seq, field=error.field)
        except OSError:
            pass
        _print_json({"error": error.to_dict()})
        return EXIT_CODES.get(error.code, 1)


if __name__ == "__main__":
    sys.exit(main())
