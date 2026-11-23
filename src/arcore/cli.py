"""Command line entry point for local Windows / cross-platform runs."""

import argparse
import json
import sys

from .clock import MonotonicClock
from .engine import run_round
from .ledger import (
    commit_round,
    last_committed_round_id,
    open_ledger,
    replay_rounds,
    unknown_round_ids,
)
from .spec import load_declaration
from .stream import generate_stream


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Deterministic alert correlation and root-cause ledger")
    parser.add_argument("--declaration", required=True,
                        help="Path to the local JSON declaration")
    parser.add_argument("--ledger", required=True,
                        help="Path to the SQLite ledger file")
    parser.add_argument("--rounds", type=int, default=3,
                        help="Number of deterministic rounds to commit")
    args = parser.parse_args(argv)

    declaration = load_declaration(args.declaration)
    stream = generate_stream(declaration)
    connection, open_notes = open_ledger(args.ledger)
    clock = MonotonicClock()
    next_round = last_committed_round_id(connection) + 1
    print(json.dumps({
        "open_notes": open_notes,
        "unknown_rounds": unknown_round_ids(connection),
        "global_digest": stream.global_digest,
        "alert_count": len(stream.alerts),
        "resume_from_round": next_round,
    }, indent=2))
    replayed_payloads = []
    for round_id in range(next_round, next_round + args.rounds):
        result = run_round(declaration, round_id, stream=stream, clock=clock)
        digest = commit_round(
            connection, round_id, result.payload, result.stats,
            clock.now_ms())
        replayed_payloads.append(
            run_round(declaration, round_id, stream=stream).payload)
        print(json.dumps({
            "committed_round": round_id,
            "path": result.payload["path_decision"]["selected_path"],
            "digest": digest,
            "stats": result.stats,
        }, indent=2))
    diffs = replay_rounds(connection, replayed_payloads)
    print(json.dumps({"replay_diffs": diffs}, indent=2))
    connection.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
