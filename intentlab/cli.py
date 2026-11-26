"""Command-line interface for IntentLab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import PipelinePaths, run_pipeline
from .snapshot import SnapshotError, SnapshotStore
from .validation import load_inputs


DEFAULT_INPUT = Path("input")
DEFAULT_SNAPSHOTS = Path("snapshots")
DEFAULT_REPORTS = Path("reports")


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_INPUT / "config.json"))
    parser.add_argument("--vocabulary", default=str(DEFAULT_INPUT / "vocabulary.json"))
    parser.add_argument("--queries", default=str(DEFAULT_INPUT / "queries.json"))
    parser.add_argument("--rules", default=str(DEFAULT_INPUT / "rules.json"))
    parser.add_argument("--edits", default=str(DEFAULT_INPUT / "edits.json"))
    parser.add_argument("--snapshots", default=str(DEFAULT_SNAPSHOTS))
    parser.add_argument("--reports", default=str(DEFAULT_REPORTS))


def _paths(args: argparse.Namespace) -> PipelinePaths:
    edits = None if args.edits in {"", "none", "None"} else args.edits
    return PipelinePaths(
        config_path=args.config,
        vocabulary_path=args.vocabulary,
        queries_path=args.queries,
        rules_path=args.rules,
        edits_path=edits,
        snapshot_root=args.snapshots,
        report_root=args.reports,
    )


def _print_summary(report: dict) -> None:
    summary = report.get("summary", {})
    print(
        "processed={processed}/{loaded} pending={pending} segments={segments} "
        "boundary={boundary} reuse={reuse} codes={codes}".format(
            processed=summary.get("queries_processed", 0),
            loaded=summary.get("queries_loaded", 0),
            pending=summary.get("queries_pending", 0),
            segments=summary.get("segments", 0),
            boundary=summary.get("boundary_intents", 0),
            reuse=summary.get("reuse_hits", 0),
            codes=json.dumps(summary.get("business_code_counts", {}), sort_keys=True, ensure_ascii=False),
        )
    )


def command_run(args: argparse.Namespace) -> int:
    result = run_pipeline(_paths(args))
    _print_summary(dict(result.report))
    print(f"json={result.json_path}")
    print(f"markdown={result.markdown_path}")
    snapshot = result.report.get("snapshot", {})
    print(f"snapshot={snapshot.get('generation')} committed={snapshot.get('committed')}")
    return 0 if snapshot.get("committed") else 2


def command_validate(args: argparse.Namespace) -> int:
    loaded, issues = load_inputs(
        config_path=args.config,
        vocabulary_path=args.vocabulary,
        queries_path=args.queries,
        rules_path=args.rules,
        edits_path=None if args.edits in {"", "none", "None"} else args.edits,
    )
    print(
        json.dumps(
            {
                "queries": len(loaded.queries),
                "revisions": len(loaded.revisions),
                "active_revision_ids": list(loaded.active_revision_ids),
                "issues": issues.to_dicts(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if len(issues) == 0 else 1


def command_recover(args: argparse.Namespace) -> int:
    store = SnapshotStore(args.snapshots)
    try:
        current = store.load_current()
    except SnapshotError as exc:
        print(json.dumps({"code": exc.code, "message": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(store.recovery_info(current), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def command_gui(args: argparse.Namespace) -> int:
    from .gui import main as gui_main

    return gui_main(
        report_path=Path(args.report) if args.report else None,
        paths=_paths(args) if args.enable_run else None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="intentlab", description="Deterministic search intent diagnostics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run the diagnostic pipeline")
    _add_run_arguments(run_parser)
    run_parser.set_defaults(func=command_run)

    validate_parser = subparsers.add_parser("validate", help="validate inputs and apply edits without running")
    _add_run_arguments(validate_parser)
    validate_parser.set_defaults(func=command_validate)

    recover_parser = subparsers.add_parser("recover", help="inspect committed snapshot and staging recovery state")
    recover_parser.add_argument("--snapshots", default=str(DEFAULT_SNAPSHOTS))
    recover_parser.set_defaults(func=command_recover)

    gui_parser = subparsers.add_parser("gui", help="open the local Tkinter diagnostics viewer")
    _add_run_arguments(gui_parser)
    gui_parser.add_argument("--report", default=None, help="report JSON path; defaults to reports/latest.json")
    gui_parser.add_argument("--enable-run", action="store_true", help="enable the Run button")
    gui_parser.set_defaults(func=command_gui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
