"""Command line interface for the intent router."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .commands import parse_commands
from .engine import EngineResult
from .fixedpoint import QuantGear
from .loader import load_inputs, read_yaml
from .pipeline import Pipeline, full_rebuild
from .snapshot import SnapshotStore


def _load(data_dir: Path):
    corpus, chain, config, failures = load_inputs(
        data_dir / "corpus.yaml",
        data_dir / "revisions.yaml",
        data_dir / "config.yaml",
    )
    if corpus is None or chain is None or config is None:
        raise SystemExit(
            "fatal validation failures: "
            + json.dumps(failures.count_by_code(), ensure_ascii=False)
        )
    return corpus, chain, config, failures


def cmd_run(args: argparse.Namespace) -> None:
    data_dir = Path(args.data)
    corpus, chain, config, failures = _load(data_dir)
    store = SnapshotStore(args.snapshot_dir) if args.snapshot_dir else None
    pipeline = Pipeline(
        config, chain, corpus.by_id(), corpus.vocab, store=store
    )
    pipeline.run_to_completion()
    report = pipeline.report()
    report["load_failures"] = failures.to_list()
    _write(args.out, report)


def cmd_replay(args: argparse.Namespace) -> None:
    data_dir = Path(args.data)
    corpus, chain, config, failures = _load(data_dir)
    raw = read_yaml(data_dir / "commands.yaml")
    commands = parse_commands(raw["commands"], failures)
    store = SnapshotStore(args.snapshot_dir) if args.snapshot_dir else None
    pipeline = Pipeline(
        config, chain, corpus.by_id(), corpus.vocab, store=store
    )
    # Baseline full run, then apply edits one at a time incrementally.
    pipeline.run_to_completion()
    checkpoints: list[dict] = []
    for command in commands:
        pipeline.apply_commands([command])
        pipeline.run_to_completion()
        checkpoints.append(
            {
                "seq": command.seq,
                "kind": command.kind,
                "report": pipeline.report(),
            }
        )
    final_report = pipeline.report()
    payload = {
        "final": final_report,
        "checkpoints": checkpoints,
        "load_failures": failures.to_list(),
    }
    _write(args.out, payload)


def cmd_consistency(args: argparse.Namespace) -> None:
    """Incremental replay vs from-scratch rebuild must match field by field."""
    data_dir = Path(args.data)
    corpus, chain, config, failures = _load(data_dir)
    raw = read_yaml(data_dir / "commands.yaml")
    commands = parse_commands(raw["commands"], failures)

    # Incremental path.
    inc = Pipeline(config, chain, corpus.by_id(), corpus.vocab)
    inc.run_to_completion()
    for command in commands:
        inc.apply_commands([command])
        inc.run_to_completion()
    inc_result = inc.previous

    # Full rebuild path on the exact final state, with an empty ledger.
    full_result = full_rebuild(
        inc.config, inc.chain, inc.texts, inc.vocab
    )

    comparison = compare_results(inc_result, full_result)
    payload = {
        "match": comparison["match"],
        "diffs": comparison["diffs"],
        "incremental_summary": inc.report()["summary"],
        "full_summary": _summary(full_result, inc.config),
    }
    _write(args.out, payload)
    if not comparison["match"]:
        raise SystemExit("incremental/full mismatch")


def _summary(result: EngineResult, config) -> dict:
    return {
        "texts_total": len(result.scores),
        "clusters": len(result.clusters),
        "duplicates_removed": len(result.removed),
        "pending": len(result.pending),
    }


def compare_results(inc: EngineResult, full: EngineResult) -> dict:
    diffs: list[str] = []
    if set(inc.scores) != set(full.scores):
        diffs.append("score text-id set differs")
    for text_id in sorted(set(inc.scores) & set(full.scores)):
        left = inc.scores[text_id].to_dict(
            QuantGear(8, 1)
        )
        right = full.scores[text_id].to_dict(QuantGear(8, 1))
        if left != right:
            diffs.append(f"score mismatch {text_id}")
        if inc.matches[text_id].to_dict() != full.matches[text_id].to_dict():
            diffs.append(f"match mismatch {text_id}")
    inc_clusters = [c.to_dict() for c in inc.clusters]
    full_clusters = [c.to_dict() for c in full.clusters]
    if inc_clusters != full_clusters:
        diffs.append("cluster assignment differs")
    inc_removed = {k: v.to_dict() for k, v in inc.removed.items()}
    full_removed = {k: v.to_dict() for k, v in full.removed.items()}
    if inc_removed != full_removed:
        diffs.append("dedup removal differs")
    return {"match": not diffs, "diffs": diffs}


def _write(out_path: str | None, payload: dict) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(text)
        print(f"wrote {out_path}")
    else:
        print(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="intent-router")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="full run with snapshot + report")
    run.add_argument("--data", default="data")
    run.add_argument("--snapshot-dir", default="artifacts/snapshot")
    run.add_argument("--out", default="artifacts/report.json")
    run.set_defaults(func=cmd_run)

    replay = sub.add_parser("replay", help="incremental edit replay")
    replay.add_argument("--data", default="data")
    replay.add_argument("--snapshot-dir", default="artifacts/snapshot")
    replay.add_argument("--out", default="artifacts/replay.json")
    replay.set_defaults(func=cmd_replay)

    consistency = sub.add_parser(
        "consistency", help="incremental vs full rebuild field comparison"
    )
    consistency.add_argument("--data", default="data")
    consistency.add_argument("--out", default="artifacts/consistency.json")
    consistency.set_defaults(func=cmd_consistency)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
