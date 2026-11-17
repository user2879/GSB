"""Standalone verification tool (stdlib only, independent of eventbus package).

Reads the SQLite ledger and the event-source spec, then checks per subscriber:
  1. the recorded migration chain is legal (idle/processing/committed/failed);
  2. event seqs are strictly increasing per (subscriber, source);
  3. no duplicate committed migration per (subscriber, source, event seq);
  4. progress equals the max committed event seq and never exceeds it;
  5. committed events exist in the declared source event sequence.

Usage: python tools/verify.py --ledger var/ledger.db --sources samples/sources.json
Exit code 0 on PASS, 1 on FAIL.
"""

import argparse
import json
import sqlite3
import sys

ALLOWED = {
    "idle": {"processing"},
    "processing": {"committed", "failed"},
    "committed": {"processing"},
    "failed": {"processing"},
}


def load_source_seqs(path):
    with open(path, "r", encoding="utf-8") as handle:
        doc = json.load(handle)
    seqs = {}
    for source in doc.get("sources", []):
        seqs[source["source_id"]] = set(source["event_seqs"])
    return seqs


def fail(code, message, subscriber_id=None, event_seq=None):
    print(json.dumps({
        "result": "FAIL",
        "code": code,
        "message": message,
        "subscriber_id": subscriber_id,
        "event_seq": event_seq,
    }, ensure_ascii=False, sort_keys=True))
    return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="Verify ledger consistency")
    parser.add_argument("--ledger", default="var/ledger.db")
    parser.add_argument("--sources", default="samples/sources.json")
    args = parser.parse_args(argv)

    try:
        source_seqs = load_source_seqs(args.sources)
    except (OSError, json.JSONDecodeError) as exc:
        return fail("E_SPEC", "cannot load sources: %s" % exc)

    try:
        connection = sqlite3.connect("file:%s?mode=ro" % args.ledger, uri=True)
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT id, subscriber_id, source_id, event_seq, from_state, to_state,"
            " complete FROM transitions ORDER BY id").fetchall()
        progress_rows = connection.execute(
            "SELECT subscriber_id, committed_seq FROM progress").fetchall()
    except sqlite3.DatabaseError as exc:
        return fail("E_LEDGER", "ledger unreadable: %s" % exc)

    progress = {row["subscriber_id"]: row["committed_seq"] for row in progress_rows}
    per_subscriber = {}
    ignored_incomplete = 0
    for row in rows:
        if row["complete"] != 1:
            ignored_incomplete += 1
            continue
        per_subscriber.setdefault(row["subscriber_id"], []).append(row)

    summaries = []
    for subscriber_id in sorted(per_subscriber):
        chain = per_subscriber[subscriber_id]
        state = "idle"
        committed_keys = set()
        last_seq_per_source = {}
        last_committed_per_source = {}
        committed_count = 0
        for row in chain:
            if row["from_state"] not in ALLOWED or row["to_state"] not in ALLOWED:
                return fail("E_LEDGER", "unknown state in ledger row %d" % row["id"],
                            subscriber_id, row["event_seq"])
            if row["from_state"] != state:
                return fail("E_STATE", "chain broken: expected from_state %r got %r"
                            % (state, row["from_state"]), subscriber_id, row["event_seq"])
            if row["to_state"] not in ALLOWED[state]:
                return fail("E_STATE", "illegal transition %s -> %s"
                            % (state, row["to_state"]), subscriber_id, row["event_seq"])
            source_id = row["source_id"]
            if source_id in source_seqs and row["event_seq"] not in source_seqs[source_id]:
                return fail("E_SPEC", "event seq %d not declared by source %r"
                            % (row["event_seq"], source_id), subscriber_id,
                            row["event_seq"])
            previous = last_seq_per_source.get(source_id)
            if previous is not None and row["event_seq"] < previous:
                return fail("E_PROGRESS", "event seq decreases per source",
                            subscriber_id, row["event_seq"])
            last_seq_per_source[source_id] = row["event_seq"]
            if row["to_state"] == "committed":
                key = (source_id, row["event_seq"])
                if key in committed_keys:
                    return fail("E_STATE", "duplicate committed event", subscriber_id,
                                row["event_seq"])
                last_committed = last_committed_per_source.get(source_id)
                if last_committed is not None and row["event_seq"] <= last_committed:
                    return fail("E_PROGRESS",
                                "committed event seq not strictly increasing per source",
                                subscriber_id, row["event_seq"])
                last_committed_per_source[source_id] = row["event_seq"]
                committed_keys.add(key)
                committed_count += 1
            state = row["to_state"]
        max_committed = max((seq for _, seq in committed_keys), default=None)
        recorded = progress.get(subscriber_id)
        if max_committed is not None:
            if recorded is None:
                return fail("E_PROGRESS", "progress row missing", subscriber_id,
                            max_committed)
            if recorded != max_committed:
                return fail("E_PROGRESS", "progress %d does not match max committed %d"
                            % (recorded, max_committed), subscriber_id, recorded)
        summaries.append({
            "subscriber_id": subscriber_id,
            "transitions": len(chain),
            "committed_events": committed_count,
            "progress": recorded if recorded is not None else 0,
            "final_state": state,
            "status": "OK",
        })

    for subscriber_id in sorted(progress):
        if subscriber_id not in per_subscriber:
            return fail("E_PROGRESS", "progress exists without any committed transition",
                        subscriber_id, progress[subscriber_id])

    print(json.dumps({
        "result": "PASS",
        "ledger": args.ledger,
        "subscribers": summaries,
        "ignored_incomplete_rows": ignored_incomplete,
    }, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
