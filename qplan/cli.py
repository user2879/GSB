from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .canonical import normalize
from .errors import QPlanError
from .ledger import Ledger
from .pipeline import run_pipeline
from .spec import apply_overrides, load_spec, validate_spec_file
from .verify import verify_ledger


def _emit(value: Any, out: str | None = None) -> None:
    text = json.dumps(normalize(value), ensure_ascii=False, indent=2, sort_keys=True)
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


def _workspace_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    ledger = Path(args.ledger) if args.ledger else workspace / "qplan_ledger.sqlite"
    database = Path(args.db) if args.db else workspace / "sample.sqlite"
    return ledger, database


def _load_weights(text: str | None, path: str | None) -> dict[str, int] | None:
    if text is None and path is None:
        return None
    try:
        raw = text if text is not None else Path(path).read_text(encoding="utf-8")
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise QPlanError("E_SPEC", f"invalid cost weight JSON: {exc}", "--weights") from exc
    if not isinstance(value, dict):
        raise QPlanError("E_SPEC", "cost weights must be a JSON object", "--weights")
    if "cost_weights" in value and isinstance(value["cost_weights"], dict):
        value = value["cost_weights"]
    return value


def _cmd_spec_check(args: argparse.Namespace) -> int:
    result = validate_spec_file(args.spec)
    _emit(result, args.out)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    weights = _load_weights(args.weights, args.weights_file)
    spec = apply_overrides(
        spec,
        weights=weights,
        stats_row_threshold=args.row_threshold,
        sample_size=args.sample_size,
        exhaustive_max_candidates=args.exhaustive_max_candidates,
        repetitions=args.repetitions,
    )
    ledger, database = _workspace_paths(args)
    report = run_pipeline(
        spec,
        spec_path=args.spec,
        ledger_path=ledger,
        db_path=database,
        batch_size=args.batch_size,
        force_db=args.force_db,
        expected_snapshot_version=args.expect_snapshot_version,
        expected_snapshot_digest=args.expect_snapshot_digest,
        force_new_version=args.force_new_version,
        crash_after_batches=args.crash_after_batches,
    )
    _emit(report, args.out)
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    ledger = Ledger.read_only(args.ledger)
    try:
        view = ledger.load_current_view()
    finally:
        ledger.close()
    if args.full:
        _emit(view, args.out)
        return 0
    snapshot = view.get("snapshot")
    session = view.get("session")
    batches = []
    queries = []
    for batch in view.get("batches", []):
        batch_queries = batch["payload"].get("queries", [])
        batches.append(
            {
                "batch_no": batch["batch_no"],
                "content_digest": batch["content_digest"],
                "query_ids": [item["query_id"] for item in batch_queries],
                "status": batch["status"],
            }
        )
        for query in batch_queries:
            queries.append(
                {
                    "query_id": query["query_id"],
                    "selected_path_id": query["selected_path_id"],
                    "estimated_cost": query["selected_estimate"]["cost"],
                    "result_match": query["result_match"],
                    "bias_triggered": query["bias"]["bias_triggered"],
                    "bias_category": (query["bias"].get("attribution") or {}).get("category"),
                }
            )
    summary = {
        "code": view.get("code"),
        "snapshot": {
            key: snapshot.get(key)
            for key in (
                "snapshot_version",
                "plan_version",
                "content_digest",
                "spec_digest",
                "stats_digest",
                "status",
                "invalidation",
                "replacement_reason",
                "created_at",
            )
        }
        if snapshot
        else None,
        "session": {
            key: session.get(key)
            for key in (
                "session_id",
                "status",
                "expected_batches",
                "started_at",
                "ended_at",
                "unknown_batches",
            )
        }
        if session
        else None,
        "batches": batches,
        "queries": queries,
        "view_semantics": view.get("view_semantics"),
    }
    _emit(summary, args.out)
    return 0


def _cmd_recover(args: argparse.Namespace) -> int:
    ledger = Ledger(args.ledger)
    try:
        result = ledger.recover()
    finally:
        ledger.close()
    _emit(result, args.out)
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    result = verify_ledger(
        spec_path=args.spec,
        ledger_path=args.ledger,
        expected_snapshot_version=args.expect_snapshot_version,
        expected_snapshot_digest=args.expect_snapshot_digest,
        allow_partial=args.allow_partial,
    )
    _emit(result, args.out)
    return 0 if result.get("code") == "OK" else 2


def _cmd_gui(args: argparse.Namespace) -> int:
    try:
        from .gui import run_gui
    except ImportError as exc:
        raise QPlanError(
            "E_GUI",
            f"Tkinter is unavailable in this Python environment: {exc}. Install the OS Tk libraries and retry.",
            "tkinter",
        ) from exc
    try:
        run_gui(
            default_spec=args.spec,
            workspace=args.workspace,
            ledger=args.ledger,
            database=args.db,
        )
    except Exception as exc:
        raise QPlanError(
            "E_GUI",
            f"Tkinter window could not be started: {exc}",
            "tkinter",
        ) from exc
    return 0


def _add_workspace_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", default="workspace", help="workspace directory for ledger and sample database")
    parser.add_argument("--ledger", default=None, help="SQLite ledger path (default: WORKSPACE/qplan_ledger.sqlite)")
    parser.add_argument("--db", default=None, help="sample SQLite path (default: WORKSPACE/sample.sqlite)")
    parser.add_argument("--out", default=None, help="also write the JSON result to this path")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qplan",
        description="Local SQLite query-plan and index-selection diagnostics",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    spec_check = subparsers.add_parser("spec-check", help="validate a declaration file and exit")
    spec_check.add_argument("--spec", required=True)
    spec_check.add_argument("--out", default=None)
    spec_check.set_defaults(func=_cmd_spec_check)

    run = subparsers.add_parser("run", help="build/reuse data, estimate, search, execute and commit a batch session")
    run.add_argument("--spec", required=True)
    _add_workspace_options(run)
    run.add_argument("--batch-size", type=int, default=2, help="queries per committed batch; <=0 means one batch")
    run.add_argument("--force-db", action="store_true", help="rebuild deterministic sample database")
    run.add_argument("--weights", default=None, help="session integer weights as JSON, e.g. '{\"rows\":1,...}'")
    run.add_argument("--weights-file", default=None, help="JSON file with integer weights")
    run.add_argument("--row-threshold", type=int, default=None, help="override exact/sampling row threshold")
    run.add_argument("--sample-size", type=int, default=None, help="override deterministic sample size")
    run.add_argument("--exhaustive-max-candidates", type=int, default=None, help="override exhaustive-search switch")
    run.add_argument("--repetitions", type=int, default=None, help="override measured timing repetitions")
    run.add_argument("--expect-snapshot-version", type=int, default=None)
    run.add_argument("--expect-snapshot-digest", default=None)
    run.add_argument("--force-new-version", action="store_true")
    run.add_argument(
        "--crash-after-batches",
        type=int,
        default=None,
        help="test hook: force-terminate the process after N committed batches",
    )
    run.set_defaults(func=_cmd_run)

    show = subparsers.add_parser("show", help="show the current ledger conclusion view")
    show.add_argument("--ledger", required=True)
    show.add_argument("--full", action="store_true")
    show.add_argument("--out", default=None)
    show.set_defaults(func=_cmd_show)

    recover = subparsers.add_parser("recover", help="recover committed batches after forced termination")
    recover.add_argument("--ledger", required=True)
    recover.add_argument("--out", default=None)
    recover.set_defaults(func=_cmd_recover)

    verify = subparsers.add_parser("verify", help="read-only independent estimate re-computation")
    verify.add_argument("--spec", required=True)
    verify.add_argument("--ledger", required=True)
    verify.add_argument("--expect-snapshot-version", type=int, default=None)
    verify.add_argument("--expect-snapshot-digest", default=None)
    verify.add_argument("--allow-partial", action="store_true")
    verify.add_argument("--out", default=None)
    verify.set_defaults(func=_cmd_verify)

    gui = subparsers.add_parser("gui", help="launch the Tkinter desktop window")
    gui.add_argument("--spec", default=None)
    gui.add_argument("--workspace", default="workspace")
    gui.add_argument("--ledger", default=None)
    gui.add_argument("--db", default=None)
    gui.set_defaults(func=_cmd_gui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except QPlanError as exc:
        _emit({"code": exc.code, "status": "REJECTED", **exc.to_dict()}, getattr(args, "out", None))
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
