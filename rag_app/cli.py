"""Local command-line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .engine import Engine
from .generator import generate
from .loader import load_documents, load_query_stream
from .report import write_report
from .replay import load_inputs, run_replay
from .snapshot import ConfigMismatchError, SnapshotStore


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLE = PROJECT_ROOT / "sample"
DEFAULT_SNAPSHOTS = PROJECT_ROOT / "artifacts" / "snapshots"
DEFAULT_REPORTS = PROJECT_ROOT / "artifacts" / "reports"


def _print_json(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def cmd_generate(args: argparse.Namespace) -> int:
    paths = generate(args.sample_dir)
    _print_json(
        {
            "generated": {
                key: str(value) for key, value in paths.items()
            }
        }
    )
    return 0


def cmd_full(args: argparse.Namespace) -> int:
    config, rules, faults, docs, turns, _commands, failures = load_inputs(
        args.sample_dir
    )
    store = SnapshotStore(args.snapshot_dir)
    if not args.no_clean:
        store.reset()
    engine = Engine(
        docs=docs,
        turns=turns,
        config=config,
        rules=rules,
        store=store,
        base_failures=failures,
        faults=faults,
    )
    summary = engine.run_full()
    report = {
        "mode": "full",
        "summary": summary,
        "turns": engine.turn_results(),
        "failures": [item.to_dict() for item in engine.failure_list()],
        "stats": {
            "cache_hits": dict(engine.stats.cache_hits),
            "cache_misses": dict(engine.stats.cache_misses),
            "invalidations": dict(engine.stats.invalidations),
            "stage_skips": dict(engine.stats.stage_skips),
            "retrievals_executed": engine.stats.retrievals_executed,
        },
        "manifest": engine.manifest(),
    }
    path = write_report(args.report, report)
    engine.shutdown()
    _print_json({"report": str(path), "summary": summary})
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    result = run_replay(
        args.sample_dir, args.snapshot_dir, clean=not args.no_clean
    )
    path = write_report(args.report, result)
    _print_json(
        {
            "report": str(path),
            "consistent": result["consistent"],
            "diffs": result["diffs"],
            "budget_pending_seen": result["budget_pending_seen"],
            "turn_count": result["turn_count"],
        }
    )
    return 0 if result["consistent"] else 2


def cmd_resume(args: argparse.Namespace) -> int:
    config, rules, faults, docs, turns, _commands, failures = load_inputs(
        args.sample_dir
    )
    store = SnapshotStore(args.snapshot_dir)
    try:
        prior = store.check_resume(
            config.fingerprint(), rules.fingerprint()
        )
    except ConfigMismatchError as exc:
        _print_json({"code": exc.code, "detail": exc.detail})
        return 3
    engine = Engine(
        docs=docs,
        turns=turns,
        config=config,
        rules=rules,
        store=store,
        base_failures=failures,
        faults=faults,
    )
    resumed_manifest = engine.resume_from_manifest()
    summary = engine.run_full()
    _print_json(
        {
            "resumed": prior is not None,
            "prior_manifest": prior,
            "post_resume_manifest": resumed_manifest,
            "summary": summary,
            "cache_hits": dict(engine.stats.cache_hits),
            "retrievals_executed": engine.stats.retrievals_executed,
        }
    )
    engine.shutdown()
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    docs, doc_failures = load_documents(
        Path(args.sample_dir) / "documents.jsonl"
    )
    turns, query_failures = load_query_stream(
        Path(args.sample_dir) / "queries.jsonl"
    )
    payload = {
        "documents": len(docs),
        "turns": len(turns),
        "failures": [
            failure.to_dict()
            for failure in [*doc_failures, *query_failures]
        ],
    }
    _print_json(payload)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-app",
        description=(
            "Local conversational retrieval and generative answer application"
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="regenerate deterministic samples")
    gen.add_argument("--sample-dir", default=str(DEFAULT_SAMPLE))
    gen.set_defaults(func=cmd_generate)

    full = sub.add_parser("full", help="one-shot full execution")
    full.add_argument("--sample-dir", default=str(DEFAULT_SAMPLE))
    full.add_argument("--snapshot-dir", default=str(DEFAULT_SNAPSHOTS))
    full.add_argument(
        "--report", default=str(DEFAULT_REPORTS / "full_report.json")
    )
    full.add_argument("--no-clean", action="store_true")
    full.set_defaults(func=cmd_full)

    replay = sub.add_parser(
        "replay", help="incremental replay vs blank full run"
    )
    replay.add_argument("--sample-dir", default=str(DEFAULT_SAMPLE))
    replay.add_argument("--snapshot-dir", default=str(DEFAULT_SNAPSHOTS))
    replay.add_argument(
        "--report", default=str(DEFAULT_REPORTS / "replay_report.json")
    )
    replay.add_argument("--no-clean", action="store_true")
    replay.set_defaults(func=cmd_replay)

    resume = sub.add_parser("resume", help="resume from existing snapshot")
    resume.add_argument("--sample-dir", default=str(DEFAULT_SAMPLE))
    resume.add_argument("--snapshot-dir", default=str(DEFAULT_SNAPSHOTS))
    resume.set_defaults(func=cmd_resume)

    validate = sub.add_parser("validate", help="validate input JSONL only")
    validate.add_argument("--sample-dir", default=str(DEFAULT_SAMPLE))
    validate.set_defaults(func=cmd_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
