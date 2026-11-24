"""Command line interface for sample generation, runs, conversion, and drills."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import known_parameter_versions, load_config
from .conversion import ConversionJournal, SidecarStore, convert_receipt, write_conversion_bundle
from .demo import generate_workspace
from .jsonutil import canonical_dumps, hash_json, pretty_dumps
from .pipeline import PipelineConfigError, run_pipeline
from .receipts import parse_receipt_paths


def _root(value: str) -> Path:
    return Path(value).resolve()


def _print_json(payload: Any) -> None:
    print(pretty_dumps(payload), end="")


def _config_path(args: argparse.Namespace) -> Path:
    return Path(args.config).resolve() if args.config else _root(args.root) / "samples" / "config.json"


def cmd_generate(args: argparse.Namespace) -> int:
    summary = generate_workspace(
        _root(args.root),
        use_subprocess_workers=not args.no_workers,
        clean_outputs=not args.keep_outputs,
        jobs=args.jobs,
    )
    _print_json(summary)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    try:
        result = run_pipeline(
            _config_path(args),
            jobs=args.jobs,
            full_rebuild=args.full_rebuild,
            normalization_budget=args.budget,
            worker_quota_points=args.quota,
        )
    except PipelineConfigError as exc:
        _print_json({"ok": False, "failures": [item.to_dict() for item in exc.failures]})
        return 2
    _print_json(
        {
            "ok": True,
            "ledger_digest": result.ledger["ledger_digest"],
            "counts": result.ledger["counts"],
            "pending_batches": result.ledger["scheduler"]["pending_queue"],
            "quota_blocked_batches": result.ledger["scheduler"]["quota_blocked"],
            "snapshot_version": result.snapshot_version,
            "snapshot_path": result.snapshot_path,
            "ledger_path": result.ledger_path,
            "report_path": result.report_json_path,
        }
    )
    return 0


def cmd_fault_drill(args: argparse.Namespace) -> int:
    try:
        result = run_pipeline(
            _config_path(args),
            jobs=args.jobs,
            fault_drill=True,
            commit_snapshot=False,
            normalization_budget=args.budget,
            worker_quota_points=args.quota,
        )
    except PipelineConfigError as exc:
        _print_json({"ok": False, "failures": [item.to_dict() for item in exc.failures]})
        return 2
    hits = result.report.get("injection_hits", [])
    _print_json(
        {
            "ok": all(hit.get("hit") for hit in hits) if hits else False,
            "injection_hits": hits,
            "failure_codes": sorted({item["code"] for item in result.report.get("failures", [])}),
            "report_path": result.report_json_path,
            "ledger_path": result.ledger_path,
        }
    )
    return 0 if hits and all(hit.get("hit") for hit in hits) else 1


def cmd_demo(args: argparse.Namespace) -> int:
    root = _root(args.root)
    summary = generate_workspace(
        root,
        use_subprocess_workers=not args.no_workers,
        clean_outputs=True,
        jobs=args.jobs,
    )
    config_path = root / "samples" / "config.json"
    run_result = run_pipeline(
        config_path,
        jobs=args.jobs,
        full_rebuild=True,
    )
    fault_result = run_pipeline(
        config_path,
        jobs=args.jobs,
        fault_drill=True,
        commit_snapshot=False,
    )
    _print_json(
        {
            "ok": True,
            "generation": summary,
            "ledger_digest": run_result.ledger["ledger_digest"],
            "counts": run_result.ledger["counts"],
            "quality": run_result.ledger["quality"],
            "snapshot_version": run_result.snapshot_version,
            "reports": {
                "ledger": run_result.ledger_path,
                "report": run_result.report_json_path,
                "fault_report": fault_result.report_json_path,
            },
            "injection_hits": fault_result.report.get("injection_hits", []),
        }
    )
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    config_result = load_config(_config_path(args))
    if config_result.bundle is None:
        _print_json({"ok": False, "failures": [item.to_dict() for item in config_result.failures]})
        return 2
    bundle = config_result.bundle
    source_path = Path(args.input).resolve()
    base_dir = Path(args.base_dir).resolve() if args.base_dir else Path(bundle.config_path).parent
    parsed = parse_receipt_paths(
        [source_path],
        base_dir=base_dir,
        known_versions=known_parameter_versions(bundle),
    )
    if parsed.failures or not parsed.receipts:
        _print_json(
            {
                "ok": False,
                "failures": [item.to_dict() for item in parsed.failures],
            }
        )
        return 2
    sidecar = SidecarStore()
    if args.sidecar_in and Path(args.sidecar_in).exists():
        sidecar = SidecarStore.from_dict(json.loads(Path(args.sidecar_in).read_text(encoding="utf-8")))
    output = Path(args.output).resolve()
    journal_path = Path(args.journal).resolve() if args.journal else output.with_suffix(".journal.json")
    journal = ConversionJournal()
    if journal_path.exists():
        try:
            journal = ConversionJournal.from_dict(json.loads(journal_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            journal = ConversionJournal()
    next_sequence = max((event.sequence for event in journal.events), default=0) + 1
    converted, event, failure = convert_receipt(
        parsed.receipts[0],
        args.to,
        sidecar,
        sequence=next_sequence,
        prune_fields=tuple(args.prune_field or ()),
    )
    if failure is not None or converted is None or event is None:
        _print_json({"ok": False, "failures": [failure.to_dict()] if failure else []})
        return 2
    journal.add(event)
    sidecar_path = Path(args.sidecar_out).resolve() if args.sidecar_out else output.with_suffix(".sidecar.json")
    write_conversion_bundle(
        converted,
        receipt_path=output,
        sidecar_path=sidecar_path,
        sidecar=sidecar,
        journal=journal,
        journal_path=journal_path,
    )
    _print_json(
        {
            "ok": True,
            "output": str(output),
            "sidecar": str(sidecar_path),
            "journal": str(journal_path),
            "event": event.as_dict(),
            "converted": converted.as_ledger_dict(),
        }
    )
    return 0


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default=".", help="workspace root")
    parser.add_argument("--config", default=None, help="path to config.json")
    parser.add_argument("--jobs", type=int, default=4, help="thread-pool size")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m mmledger", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="generate deterministic sample inputs")
    _add_common(generate)
    generate.add_argument("--no-workers", action="store_true", help="skip subprocess workers")
    generate.add_argument("--keep-outputs", action="store_true", help="do not clean snapshots/reports")
    generate.set_defaults(func=cmd_generate)

    run = subparsers.add_parser("run", help="run the staged incremental pipeline")
    _add_common(run)
    run.add_argument("--full-rebuild", action="store_true", help="ignore existing snapshots")
    run.add_argument("--budget", type=int, default=None, help="normalization point budget override")
    run.add_argument("--quota", type=int, default=None, help="per-worker point quota override")
    run.set_defaults(func=cmd_run)

    drill = subparsers.add_parser("fault-drill", help="run isolated fault injection diagnostics")
    _add_common(drill)
    drill.add_argument("--budget", type=int, default=None)
    drill.add_argument("--quota", type=int, default=None)
    drill.set_defaults(func=cmd_fault_drill)

    demo = subparsers.add_parser("demo", help="generate samples, run pipeline, and run fault drill")
    _add_common(demo)
    demo.add_argument("--no-workers", action="store_true")
    demo.set_defaults(func=cmd_demo)

    convert = subparsers.add_parser("convert", help="convert one receipt between r1/r2/r3")
    _add_common(convert)
    convert.add_argument("--input", required=True)
    convert.add_argument("--to", required=True, choices=("r1", "r2", "r3"))
    convert.add_argument("--output", required=True)
    convert.add_argument("--base-dir", default=None)
    convert.add_argument("--sidecar-in", default=None)
    convert.add_argument("--sidecar-out", default=None)
    convert.add_argument("--journal", default=None)
    convert.add_argument("--prune-field", action="append", default=None)
    convert.set_defaults(func=cmd_convert)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, RuntimeError, ValueError) as exc:
        print(canonical_dumps({"ok": False, "error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
