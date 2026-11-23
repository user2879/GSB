"""Command-line interface for validation, execution, recovery and GUI launch."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from .errors import DiagnosticError
from .ledger import Ledger
from .runtime import InferenceRuntime, recover_and_resume
from .schema import load_plan, memory_profile, plan_diff
from .snapshot import list_snapshots, load_snapshot, save_snapshot
from .util import atomic_write_json


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _load_report(path: str | None) -> None:
    if path is None:
        return
    previous = None
    # The caller writes the rich report separately; this helper only validates
    # that the parent directory exists.
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)


def cmd_validate(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    result = {
        "ok": True,
        "code": None,
        "plan_version": plan.version,
        "plan_content_digest": plan.content_digest,
        "layer_count": len(plan.layers),
        "layers": [layer.id for layer in plan.layers],
        "block_size_bytes": plan.block_size_bytes,
        "memory_budget_bytes": plan.memory_budget_bytes,
        "memory_profile_batch_max": memory_profile(plan, plan.batch_size_max),
        "lease_ttl_ms": plan.lease_ttl_ms,
        "batch_size_max": plan.batch_size_max,
    }
    _print(result)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    ledger = Ledger(args.db)
    runtime = InferenceRuntime(
        plan,
        args.samples,
        batch_size=args.batch_size,
        ledger=ledger,
    )
    report = runtime.run(
        worker_count=args.workers,
        crash_after_committed_layers=args.crash_after_committed_layers,
    )
    cursor = ledger.recovery_view()["cursor"]
    snapshot = runtime.snapshot_payload(cursor=cursor)
    snapshot_path = save_snapshot(args.snapshots, snapshot)
    report["snapshot_path"] = snapshot_path
    report["ledger_integrity"] = ledger.verify_integrity()
    if args.report:
        atomic_write_json(args.report, report)
    ledger.close()
    summary = {
        "ok": report["ok"],
        "plan_version": report["plan_version"],
        "batches": len(report["schedule"]["batches"]),
        "failed_batches": report["schedule"]["failed_batches"],
        "dataset_digest": report["dataset"]["dataset_digest"],
        "layer_summaries": {
            key: {
                "float_digest": value["float_digest"],
                "quant_digest": value["quant_digest"],
                "max_abs_error": value["errors"]["max_abs_error"],
            }
            for key, value in report["layer_summaries"].items()
        },
        "max_deviation_layer": report["attribution"]["max_deviation_layer"],
        "dominant_component": report["attribution"]["dominant_component"],
        "snapshot_path": snapshot_path,
        "ledger_integrity": report["ledger_integrity"],
        "report_path": args.report,
    }
    _print(summary)
    return 0 if report["ok"] else 4


def cmd_invalidate(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    ledger = Ledger(args.db)
    runtime = InferenceRuntime(plan, args.samples, batch_size=args.batch_size, ledger=ledger)
    runtime.run(worker_count=args.workers)
    runtime.hold_version_leases(runtime.batches[0].batch_id)
    before = runtime.layer_summaries()
    updates = json.loads(args.updates)
    change = runtime.apply_change(args.layer, updates)
    after = runtime.layer_summaries()
    verification = runtime.verify_incremental()
    unchanged_digest = {
        layer: before[layer]["combined_digest"] == after[layer]["combined_digest"]
        for layer in change["unaffected"]
    }
    unchanged_ranges = {
        layer: before[layer]["lease_range"] == after[layer]["lease_range"]
        for layer in change["unaffected"]
    }
    snapshot = runtime.snapshot_payload(cursor=ledger.recovery_view()["cursor"])
    snapshot_path = save_snapshot(args.snapshots, snapshot)
    result = {
        "ok": all(unchanged_digest.values()) and all(unchanged_ranges.values()) and verification["ok"],
        "change": change,
        "verification": verification,
        "unchanged_digest": unchanged_digest,
        "unchanged_ranges": unchanged_ranges,
        "snapshot_path": snapshot_path,
        "ledger_integrity": ledger.verify_integrity(),
    }
    if args.report:
        atomic_write_json(args.report, result)
    if args.plan_out:
        atomic_write_json(args.plan_out, runtime.plan.to_dict())
    ledger.close()
    _print(result)
    return 0 if result["ok"] else 4


def cmd_recover(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    ledger = Ledger(args.db)
    result = recover_and_resume(
        plan=plan,
        sample_count=args.samples,
        batch_size=args.batch_size,
        ledger=ledger,
        worker_count=args.workers,
    )
    result["ledger_integrity_after_resume"] = ledger.verify_integrity()
    if args.report:
        atomic_write_json(args.report, result)
    ledger.close()
    _print(result)
    return 0 if result["ok"] else 4


def cmd_snapshot_list(args: argparse.Namespace) -> int:
    _print({"ok": True, "snapshots": list_snapshots(args.snapshots)})
    return 0


def cmd_snapshot_load(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    ledger = Ledger(args.db) if args.db else None
    result = load_snapshot(
        args.snapshot,
        current_plan=plan,
        current_ledger_digest=None if ledger is None else ledger.content_digest(),
        current_lease_records=None if ledger is None else ledger.load_leases(),
    )
    if ledger is not None:
        ledger.close()
    _print(result.to_dict())
    return 0


def cmd_snapshot_save(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    ledger = Ledger(args.db)
    cursor = ledger.recovery_view()["cursor"]
    from .snapshot import build_snapshot

    payload = build_snapshot(
        plan,
        sample_count=args.samples,
        dataset_digest=None,
        lease_records=[],
        layer_summaries={},
        processed_batch_cursor=cursor,
        ledger_content_digest=ledger.content_digest(),
    )
    path = save_snapshot(args.snapshots, payload)
    ledger.close()
    _print({"ok": True, "snapshot_path": path, "content_digest": payload["content_digest"]})
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    from .demo import run_demo

    result = run_demo(Path(args.workspace), Path(args.plan), workers=args.workers)
    _print(result)
    return 0 if result.get("ok") else 4


def cmd_gui(args: argparse.Namespace) -> int:
    from .gui import launch_gui

    launch_gui(
        plan_path=args.plan,
        db_path=args.db,
        snapshot_dir=args.snapshots,
        sample_count=args.samples,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mixedmem",
        description="Mixed-precision block-pool and lease diagnostics workbench",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="strictly validate a local JSON plan")
    validate.add_argument("--plan", required=True)
    validate.set_defaults(func=cmd_validate)

    run = sub.add_parser("run", help="run scheduled dual-path inference")
    run.add_argument("--plan", required=True)
    run.add_argument("--db", required=True)
    run.add_argument("--snapshots", required=True)
    run.add_argument("--samples", type=int, default=8)
    run.add_argument("--batch-size", type=int, default=4)
    run.add_argument("--workers", type=int, default=2)
    run.add_argument("--report")
    run.add_argument("--crash-after-committed-layers", type=int, default=None)
    run.set_defaults(func=cmd_run)

    invalidate = sub.add_parser("invalidate", help="run then apply an incremental layer change")
    invalidate.add_argument("--plan", required=True)
    invalidate.add_argument("--db", required=True)
    invalidate.add_argument("--snapshots", required=True)
    invalidate.add_argument("--samples", type=int, default=8)
    invalidate.add_argument("--batch-size", type=int, default=4)
    invalidate.add_argument("--workers", type=int, default=2)
    invalidate.add_argument("--layer", required=True)
    invalidate.add_argument("--updates", required=True, help="JSON object of layer field updates")
    invalidate.add_argument("--plan-out", help="atomically write the new plan declaration")
    invalidate.add_argument("--report")
    invalidate.set_defaults(func=cmd_invalidate)

    recover = sub.add_parser("recover", help="verify committed ledger rows and resume missing batches")
    recover.add_argument("--plan", required=True)
    recover.add_argument("--db", required=True)
    recover.add_argument("--samples", type=int, default=8)
    recover.add_argument("--batch-size", type=int, default=4)
    recover.add_argument("--workers", type=int, default=2)
    recover.add_argument("--report")
    recover.set_defaults(func=cmd_recover)

    snapshots = sub.add_parser("snapshots", help="list versioned snapshots")
    snapshots.add_argument("--snapshots", required=True)
    snapshots.set_defaults(func=cmd_snapshot_list)

    snapshot_load = sub.add_parser("snapshot-load", help="validate and describe whole-state replacement")
    snapshot_load.add_argument("--snapshot", required=True)
    snapshot_load.add_argument("--plan", required=True)
    snapshot_load.add_argument("--db")
    snapshot_load.set_defaults(func=cmd_snapshot_load)

    snapshot_save = sub.add_parser("snapshot-save", help="save a minimal run-state snapshot")
    snapshot_save.add_argument("--plan", required=True)
    snapshot_save.add_argument("--db", required=True)
    snapshot_save.add_argument("--snapshots", required=True)
    snapshot_save.add_argument("--samples", type=int, default=8)
    snapshot_save.set_defaults(func=cmd_snapshot_save)

    demo = sub.add_parser("demo", help="run the complete local acceptance scenario")
    demo.add_argument("--workspace", default="artifacts/demo")
    demo.add_argument("--plan", default="sample_plan.json")
    demo.add_argument("--workers", type=int, default=2)
    demo.set_defaults(func=cmd_demo)

    gui = sub.add_parser("gui", help="open the Tkinter diagnostic workbench")
    gui.add_argument("--plan", default="sample_plan.json")
    gui.add_argument("--db", default="artifacts/demo/ledger.db")
    gui.add_argument("--snapshots", default="artifacts/demo/snapshots")
    gui.add_argument("--samples", type=int, default=8)
    gui.add_argument("--batch-size", type=int, default=4)
    gui.add_argument("--workers", type=int, default=2)
    gui.set_defaults(func=cmd_gui)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except DiagnosticError as exc:
        _print(exc.to_dict())
        return 2
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        error = DiagnosticError("E_IO", f"{type(exc).__name__}: {exc}")
        _print(error.to_dict())
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
