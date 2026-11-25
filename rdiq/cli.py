"""Command line entry points for deterministic local verification."""

import argparse
import json
import os

from .canonical import canonical_text, digest
from .engine import Engine, recover_and_replay
from .errors import RDIQError
from .invalidation import propagate_invalidation
from .ledger import Ledger, expected_from_config
from .models import load_dataset
from .synth import generate_sample_set


def _write(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(canonical_text(payload))
        handle.write("\n")


def cmd_generate(args):
    generate_sample_set(args.output_dir, seed=args.seed)
    print(f"generated deterministic samples in {args.output_dir}")


def cmd_run(args):
    dataset = load_dataset(args.sample_dir)
    engine = Engine(
        dataset,
        args.ledger_db,
        args.backup_dir,
        global_limit=args.global_limit,
        per_session_limit=args.per_session_limit,
        batch_size=args.batch_size,
    )
    result = engine.run()
    output_path, summary_path, output_digest = engine.write_outputs(args.output)
    engine.close()
    print(json.dumps({
        "output": output_path,
        "summary": summary_path,
        "output_digest": output_digest,
        "graph_digest": result["graph"]["graph_digest"],
    }, indent=2, sort_keys=True))


def cmd_edit_recompute(args):
    old_dataset = load_dataset(args.base_dir)
    new_dataset = load_dataset(args.edited_dir)
    baseline_engine = Engine(
        old_dataset,
        args.ledger_db,
        args.backup_dir,
        global_limit=args.global_limit,
        per_session_limit=args.per_session_limit,
    )
    baseline_engine.run()
    baseline_graph = baseline_engine.graph
    before_digests = {
        session_id: {
            round_index: record["round_digest"]
            for round_index, record in rounds.items()
        }
        for session_id, rounds in baseline_engine.round_records.items()
    }
    before_evidence = {
        session_id: {
            round_index: digest(record["evidence_digests"])
            for round_index, record in rounds.items()
        }
        for session_id, rounds in baseline_engine.round_records.items()
    }
    baseline_engine.close()

    invalidation = propagate_invalidation(
        old_dataset, new_dataset, baseline_graph
    )
    edited_engine = Engine(
        new_dataset,
        args.ledger_db,
        args.backup_dir,
        global_limit=args.global_limit,
        per_session_limit=args.per_session_limit,
    )
    recompute = edited_engine.rerun_invalid(old_dataset, invalidation)
    unchanged_checks = []
    invalid_keys = {
        (item["session_id"], item["round"])
        for item in invalidation["invalid_rounds"]
    }
    for session_id in sorted(before_digests):
        for round_index in sorted(before_digests[session_id]):
            key = (session_id, round_index)
            if key in invalid_keys:
                continue
            current = edited_engine.round_records[session_id][round_index]
            unchanged_checks.append(
                {
                    "session_id": session_id,
                    "round": round_index,
                    "round_digest_unchanged": current["round_digest"]
                    == before_digests[session_id][round_index],
                    "evidence_chain_digest_unchanged": digest(
                        current["evidence_digests"]
                    )
                    == before_evidence[session_id][round_index],
                }
            )
    result = edited_engine.snapshot_result(
        recompute["schedule_reports"],
        recompute["limits"],
        recompute["batches"],
    )
    payload = {
        "invalidation": invalidation,
        "unchanged_checks": unchanged_checks,
        "ledger_replacement": edited_engine.ledger.replacement,
        "result": result,
    }
    _write(args.output, payload)
    edited_engine.close()
    print(canonical_text({
        "output": args.output,
        "invalid_round_count": len(invalidation["invalid_rounds"]),
        "unchanged_check_count": len(unchanged_checks),
        "all_unchanged_byte_identical": all(
            item["round_digest_unchanged"]
            and item["evidence_chain_digest_unchanged"]
            for item in unchanged_checks
        ),
    }))


def cmd_crash(args):
    dataset = load_dataset(args.sample_dir)
    if os.path.exists(args.ledger_db):
        os.remove(args.ledger_db)
    ledger = Ledger(
        args.ledger_db, args.backup_dir, expected_from_config(dataset.config)
    )
    from .error_bounds import round_error_attribution
    from .evidence import arbitrate_round

    committed = []
    entries = []
    for session in dataset.sessions[:2]:
        for request in session["rounds"][:2]:
            record = arbitrate_round(
                dataset, session["session_id"], request
            )
            entries.append(
                {
                    "session_id": session["session_id"],
                    "round": request["round"],
                    "round_record": record,
                    "error_record": round_error_attribution(record),
                }
            )
    first = entries[:2]
    ledger.write_batch(
        first,
        [
            {
                "session_id": entry["session_id"],
                "last_committed_round": entry["round"],
                "total_rounds": 6,
            }
            for entry in first
        ],
        first,
    )
    committed.extend(first)
    interrupted = entries[2:]
    ledger.begin_interrupted_batch(interrupted, [], interrupted)
    ledger.close()
    _write(
        args.output,
        {
            "committed_keys": [
                {"session_id": item["session_id"], "round": item["round"]}
                for item in committed
            ],
            "interrupted_uncommitted_keys": [
                {"session_id": item["session_id"], "round": item["round"]}
                for item in interrupted
            ],
            "note": "process is terminated with an open transaction; no exit report is fabricated",
        },
    )
    print(canonical_text({
        "ledger_db": args.ledger_db,
        "committed": len(committed),
        "interrupted": len(interrupted),
    }))


def cmd_recover(args):
    dataset = load_dataset(args.sample_dir)
    engine, result = recover_and_replay(
        dataset,
        args.ledger_db,
        args.backup_dir,
        global_limit=args.global_limit,
        per_session_limit=args.per_session_limit,
        batch_size=args.batch_size,
    )
    output_path, summary_path, _ = engine.write_outputs(args.output)
    _write(args.recovery_output, result["recovery"])
    engine.close()
    print(canonical_text({
        "output": output_path,
        "recovery": result["recovery"],
    }))


def build_parser():
    parser = argparse.ArgumentParser(description="deterministic RDIQ pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate")
    gen.add_argument("--output-dir", required=True)
    gen.add_argument("--seed", type=int, default=20261126)
    gen.set_defaults(func=cmd_generate)

    run = sub.add_parser("run")
    run.add_argument("--sample-dir", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--ledger-db", required=True)
    run.add_argument("--backup-dir", required=True)
    run.add_argument("--global-limit", type=int, default=3)
    run.add_argument("--per-session-limit", type=int, default=2)
    run.add_argument("--batch-size", type=int, default=8)
    run.set_defaults(func=cmd_run)

    edit = sub.add_parser("edit-recompute")
    edit.add_argument("--base-dir", required=True)
    edit.add_argument("--edited-dir", required=True)
    edit.add_argument("--output", required=True)
    edit.add_argument("--ledger-db", required=True)
    edit.add_argument("--backup-dir", required=True)
    edit.add_argument("--global-limit", type=int, default=3)
    edit.add_argument("--per-session-limit", type=int, default=2)
    edit.set_defaults(func=cmd_edit_recompute)

    crash = sub.add_parser("crash")
    crash.add_argument("--sample-dir", required=True)
    crash.add_argument("--output", required=True)
    crash.add_argument("--ledger-db", required=True)
    crash.add_argument("--backup-dir", required=True)
    crash.set_defaults(func=cmd_crash)

    recover = sub.add_parser("recover")
    recover.add_argument("--sample-dir", required=True)
    recover.add_argument("--output", required=True)
    recover.add_argument("--recovery-output", required=True)
    recover.add_argument("--ledger-db", required=True)
    recover.add_argument("--backup-dir", required=True)
    recover.add_argument("--global-limit", type=int, default=3)
    recover.add_argument("--per-session-limit", type=int, default=2)
    recover.add_argument("--batch-size", type=int, default=8)
    recover.set_defaults(func=cmd_recover)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except RDIQError as exc:
        print(canonical_text(exc.to_dict()))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
