"""Command line entry point for the fixed-point pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .errors import IssueLog
from .loader import load_edits, load_pipeline
from .orchestrator import Orchestrator
from .samplegen import generate
from .exchange import parse_e1_file


def cmd_acceptance(args) -> int:
    from .demo import run_demo
    summary = run_demo(args.workspace)
    print(json.dumps({
        "incremental_invalidated": summary["incremental"]["invalidated"],
        "incremental_reused": summary["incremental"]["reused"],
        "incremental_equals_clean_rebuild":
            summary["incremental_vs_clean_rebuild"]["consistent"],
        "e1_e2_integer_outputs_equal":
            summary["e2_vs_e1"]["identical_integer_outputs"],
        "version_distinct_cache_keys":
            summary["e2_vs_e1"]["cache_keys_that_differ_by_version"],
        "acceptance_report": str(Path(args.workspace) / "run" / "acceptance.json"),
    }, indent=2, sort_keys=True))
    return 0


def _resolve_exchange(args) -> str:
    return getattr(args, "version", None) or ("e1" if args.e1 else "e2")


def cmd_generate(args) -> int:
    paths = generate(args.sample_root)
    print(json.dumps({name: str(path) for name, path in paths.items()}, indent=2, sort_keys=True))
    return 0


def _build_model(args, log: IssueLog):
    version = _resolve_exchange(args)
    e1_path = args.e1_file if version == "e1" else None
    e2_path = args.e2_file if version == "e2" else None
    bypass = args.bypass if version == "e2" else None
    model = load_pipeline(
        args.config, args.tokens, args.segments, args.graph,
        e1_path=e1_path, e2_path=e2_path, bypass_path=bypass,
    )
    log.extend(model.log)
    return model, version


def cmd_validate(args) -> int:
    log = IssueLog()
    model, version = _build_model(args, log)
    payload = {
        "exchange_version": version,
        "chains": {cid: [op.node_id() for op in chain.ordered_ops()]
                   for cid, chain in sorted(model.chains.items())},
        "token_count": len(model.tokens),
        "segments": [list(pair) for pair in model.segments],
        "issues": log.to_dict(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def cmd_run(args) -> int:
    log = IssueLog()
    model, version = _build_model(args, log)
    orchestrator = Orchestrator(model, args.snapshot_root, exchange_version=version)
    orchestrator.log.extend(log)
    edits = load_edits(args.edits, log) if args.edits else []
    manifest = orchestrator.start()
    edit_invalidated = 0
    if edits:
        boundary = len(orchestrator.engine.ledger.events)
        orchestrator.apply_edits(edits)
        edit_invalidated = sum(
            1 for event in orchestrator.engine.ledger.events[boundary:]
            if event.kind == "invalidated"
        )
    manifest, diagnostics, history = orchestrator.run_to_completion(manifest)
    diagnostics.invalidated_count += edit_invalidated
    orchestrator.write_report(diagnostics, args.report)
    print(json.dumps({
        "global_version": manifest.global_version,
        "batches": len(history),
        "report": str(args.report),
        "snapshot": str(orchestrator.store.current_dir()),
        "code_counts": diagnostics.code_counts,
        "reused": diagnostics.reused_count,
        "invalidated": diagnostics.invalidated_count,
    }, indent=2, sort_keys=True))
    return 0


def cmd_report(args) -> int:
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    summary = {
        "run_id": report["run_id"],
        "stage_points": report["stage_points"],
        "invalidated_count": report["invalidated_count"],
        "pending": report["pending"],
        "code_counts": report["code_counts"],
        "completed_ops": report["completed_ops"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fxpipe", description="Fixed-point embedding chain engine")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="generate deterministic samples")
    gen.add_argument("sample_root")
    gen.set_defaults(func=cmd_generate)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", required=True)
    common.add_argument("--tokens", required=True)
    common.add_argument("--segments", required=True)
    common.add_argument("--graph", required=True)
    common.add_argument("--e1-file", dest="e1_file")
    common.add_argument("--e2-file", dest="e2_file")
    common.add_argument("--bypass")
    common.add_argument("--version", choices=["e1", "e2"])
    common.add_argument("--e1", action="store_true", help="short-hand for --version e1")

    val = sub.add_parser("validate", parents=[common])
    val.set_defaults(func=cmd_validate)

    run = sub.add_parser("run", parents=[common])
    run.add_argument("--edits")
    run.add_argument("--snapshot-root", required=True)
    run.add_argument("--report", required=True)
    run.set_defaults(func=cmd_run)

    rep = sub.add_parser("report")
    rep.add_argument("report")
    rep.set_defaults(func=cmd_report)

    acc = sub.add_parser("acceptance", help="run the full deterministic acceptance demo")
    acc.add_argument("workspace")
    acc.set_defaults(func=cmd_acceptance)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
