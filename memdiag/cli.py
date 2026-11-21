"""Command-line entry points for validation, diagnostics, snapshots and recovery."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .engine import run_initial
from .errors import MemDiagError, RecoveryRequired
from .ledger import Ledger
from .models import DiagnosticRun
from .plan import apply_plan_changes, load_plan, load_samples
from .services import (
    check_incremental,
    recover_session,
    replay_session,
    resume_session,
    run_full_session,
    run_incremental_session,
)
from .snapshots import SnapshotStore
from .util import atomic_write_json, read_json
from .viewmodel import build_report


DEFAULT_PLAN = "data/plan.json"
DEFAULT_SAMPLES = "data/samples.json"
DEFAULT_DB = "workbench.sqlite3"
DEFAULT_SNAPSHOTS = "snapshots"


def _emit(payload: dict[str, Any], *, report_path: str | None = None) -> None:
    if report_path:
        atomic_write_json(report_path, payload)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False)
    print(text)


def _success_report(run: DiagnosticRun, snapshots: list[Any] | None = None) -> dict[str, Any]:
    return build_report(run, snapshots=snapshots or [])


def _add_common(parser: argparse.ArgumentParser, *, plan_required: bool = False) -> None:
    parser.add_argument("--plan", default=None if plan_required else DEFAULT_PLAN)
    parser.add_argument("--samples", default=None if plan_required else DEFAULT_SAMPLES)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--snapshots", default=DEFAULT_SNAPSHOTS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memdiag",
        description="混合精度推理显存预算诊断工作台（Windows 本地、SQLite、Tkinter）",
    )
    parser.add_argument("--version", action="version", version="memdiag 1.0.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="严格校验计划与确定性样本")
    _add_common(validate)

    run = subparsers.add_parser("run", help="执行全量诊断并逐层提交账本")
    _add_common(run)
    run.add_argument("--session-id", default=None)
    run.add_argument("--crash-after", type=int, default=None)
    run.add_argument("--no-snapshot", action="store_true")
    run.add_argument("--report", default="artifacts/latest_report.json")

    incremental = subparsers.add_parser("incremental", help="应用本地 JSON 增量修改并仅重算失效集")
    _add_common(incremental)
    incremental.add_argument("--changes", required=True)
    incremental.add_argument("--session-id", default=None)
    incremental.add_argument("--crash-after", type=int, default=None)
    incremental.add_argument("--no-snapshot", action="store_true")
    incremental.add_argument("--report", default="artifacts/latest_incremental.json")

    check = subparsers.add_parser("check-incremental", help="用全量数值参考校验增量结果")
    check.add_argument("--plan", default=DEFAULT_PLAN)
    check.add_argument("--samples", default=DEFAULT_SAMPLES)
    check.add_argument("--changes", required=True)
    check.add_argument("--report", default="artifacts/incremental_check.json")

    snapshot = subparsers.add_parser("snapshot", help="将当前账本状态保存为只读快照")
    snapshot.add_argument("--db", default=DEFAULT_DB)
    snapshot.add_argument("--snapshots", default=DEFAULT_SNAPSHOTS)
    snapshot.add_argument("--label", default="manual")

    restore = subparsers.add_parser("restore", help="校验并整体替换恢复快照")
    restore.add_argument("--db", default=DEFAULT_DB)
    restore.add_argument("--snapshots", default=DEFAULT_SNAPSHOTS)
    restore.add_argument("--snapshot-id", required=True)

    recover = subparsers.add_parser("recover", help="扫描强制终止会话，仅信任已提交层批次")
    recover.add_argument("--db", default=DEFAULT_DB)
    recover.add_argument("--session-id", required=True)
    recover.add_argument("--report", default=None)

    resume = subparsers.add_parser("resume", help="重放已提交层并补跑未知层")
    resume.add_argument("--db", default=DEFAULT_DB)
    resume.add_argument("--snapshots", default=DEFAULT_SNAPSHOTS)
    resume.add_argument("--session-id", required=True)
    resume.add_argument("--no-snapshot", action="store_true")
    resume.add_argument("--report", default="artifacts/resume_report.json")

    replay = subparsers.add_parser("replay", help="按账本计划版本与缩放因子重放校验")
    replay.add_argument("--db", default=DEFAULT_DB)
    replay.add_argument("--session-id", required=True)
    replay.add_argument("--report", default=None)

    current = subparsers.add_parser("report", help="输出当前账本诊断报告")
    current.add_argument("--db", default=DEFAULT_DB)
    current.add_argument("--snapshots", default=DEFAULT_SNAPSHOTS)
    current.add_argument("--report", default=None)

    snapshots = subparsers.add_parser("snapshots", help="列出只读快照与账本会话")
    snapshots.add_argument("--db", default=DEFAULT_DB)
    snapshots.add_argument("--snapshots", default=DEFAULT_SNAPSHOTS)

    gui = subparsers.add_parser("gui", help="启动 Tkinter 诊断界面")
    _add_common(gui)

    demo = subparsers.add_parser("demo", help="运行本地确定性端到端演示")
    demo.add_argument("--plan", default=DEFAULT_PLAN)
    demo.add_argument("--samples", default=DEFAULT_SAMPLES)
    demo.add_argument("--changes", default="data/changes_scale.json")
    demo.add_argument("--db", default="artifacts/demo.sqlite3")
    demo.add_argument("--snapshots", default="artifacts/demo_snapshots")

    return parser


def _handle(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if args.command == "validate":
        plan = load_plan(args.plan)
        samples = load_samples(args.samples)
        payload = {
            "ok": True,
            "plan_version": plan.plan_version,
            "plan_digest": plan.digest,
            "layers": [layer.layer_id for layer in plan.layers],
            "samples": len(samples.samples),
            "samples_digest": samples.digest,
            "capacity_blocks": plan.capacity_blocks,
        }
        _emit(payload)
        return payload, 0

    if args.command == "run":
        plan = load_plan(args.plan)
        samples = load_samples(args.samples)
        ledger = Ledger(args.db)
        store = SnapshotStore(args.snapshots)
        try:
            result = run_full_session(
                plan,
                samples,
                ledger,
                snapshot_store=store,
                session_id=args.session_id,
                crash_after=args.crash_after,
                save_snapshot=not args.no_snapshot,
                snapshot_label="cli-full",
            )
            report = _success_report(result["run"], store.list())
            report["session_id"] = result["session_id"]
            report["snapshot"] = result["snapshot"]
            _emit(report, report_path=args.report)
            return report, 0
        finally:
            ledger.close()

    if args.command == "incremental":
        base_plan = load_plan(args.plan)
        samples = load_samples(args.samples)
        changes = read_json(args.changes)
        new_plan, changed_sources, details = apply_plan_changes(base_plan, changes)
        ledger = Ledger(args.db)
        store = SnapshotStore(args.snapshots)
        try:
            current = ledger.load_current_run()
            if (
                current is not None
                and current.plan.digest == base_plan.digest
                and current.samples.digest == samples.digest
            ):
                base_run = current
                base_source = "ledger"
            else:
                base_run = run_initial(base_plan, samples)
                base_source = "cold"
            result = run_incremental_session(
                new_plan,
                samples,
                base_run,
                changed_sources,
                ledger,
                snapshot_store=store,
                session_id=args.session_id,
                crash_after=args.crash_after,
                save_snapshot=not args.no_snapshot,
                snapshot_label="cli-incremental",
            )
            report = _success_report(result["run"], store.list())
            report["session_id"] = result["session_id"]
            report["snapshot"] = result["snapshot"]
            report["invalidated_layers"] = result["invalidated"]
            report["base_source"] = base_source
            report["plan_changes"] = details
            _emit(report, report_path=args.report)
            return report, 0
        finally:
            ledger.close()

    if args.command == "check-incremental":
        base_plan = load_plan(args.plan)
        samples = load_samples(args.samples)
        changes = read_json(args.changes)
        new_plan, changed_sources, details = apply_plan_changes(base_plan, changes)
        base = run_initial(base_plan, samples)
        result = check_incremental(new_plan, samples, base, changed_sources)
        payload = {
            key: value
            for key, value in result.items()
            if key not in {"incremental_run", "full_run"}
        }
        payload["plan_version"] = new_plan.plan_version
        payload["plan_changes"] = details
        payload["comparison_mode"] = "full_numeric_recompute_with_anchored_block_layout"
        _emit(payload, report_path=args.report)
        return payload, 0

    if args.command == "snapshot":
        ledger = Ledger(args.db)
        store = SnapshotStore(args.snapshots)
        try:
            run = ledger.load_current_run()
            if run is None:
                raise MemDiagError("ledger has no current diagnostic state", field="current")
            info = store.save(run, label=args.label, ledger=ledger)
            payload = {"ok": True, "snapshot": info.to_dict(), "snapshots": [item.to_dict() for item in store.list()]}
            _emit(payload)
            return payload, 0
        finally:
            ledger.close()

    if args.command == "restore":
        ledger = Ledger(args.db)
        store = SnapshotStore(args.snapshots)
        try:
            payload = store.restore(args.snapshot_id, ledger)
            _emit(payload)
            return payload, 0
        finally:
            ledger.close()

    if args.command == "recover":
        ledger = Ledger(args.db)
        try:
            payload = recover_session(ledger, args.session_id)
            _emit(payload, report_path=args.report)
            return payload, 0 if payload["exit_report_available"] else 8
        finally:
            ledger.close()

    if args.command == "resume":
        ledger = Ledger(args.db)
        store = SnapshotStore(args.snapshots)
        try:
            result = resume_session(
                ledger,
                args.session_id,
                snapshot_store=store,
                save_snapshot=not args.no_snapshot,
            )
            payload = {key: value for key, value in result.items() if key != "run"}
            if "run" in result:
                payload["report"] = _success_report(result["run"], store.list())
            _emit(payload, report_path=args.report)
            return payload, 0
        finally:
            ledger.close()

    if args.command == "replay":
        ledger = Ledger(args.db)
        try:
            payload = replay_session(ledger, args.session_id)
            _emit(payload, report_path=args.report)
            return payload, 0
        finally:
            ledger.close()

    if args.command == "report":
        ledger = Ledger(args.db)
        store = SnapshotStore(args.snapshots)
        try:
            run = ledger.load_current_run()
            if run is None:
                raise MemDiagError("ledger has no current diagnostic state", field="current")
            payload = _success_report(run, store.list())
            payload["ledger_meta"] = ledger.current_meta()
            _emit(payload, report_path=args.report)
            return payload, 0
        finally:
            ledger.close()

    if args.command == "snapshots":
        store = SnapshotStore(args.snapshots)
        ledger = Ledger(args.db)
        try:
            payload = {
                "ok": True,
                "snapshots": [item.to_dict() for item in store.list()],
                "sessions": ledger.list_sessions(),
                "ledger_meta": ledger.current_meta(),
            }
            _emit(payload)
            return payload, 0
        finally:
            ledger.close()

    if args.command == "gui":
        from .gui import GuiUnavailable, run_gui

        try:
            run_gui(args.plan, args.samples, args.db, args.snapshots)
        except GuiUnavailable as error:
            raise MemDiagError(str(error), field="tkinter") from error
        return {"ok": True, "ui": "closed"}, 0

    if args.command == "demo":
        base_plan = load_plan(args.plan)
        samples = load_samples(args.samples)
        changes = read_json(args.changes)
        new_plan, changed_sources, details = apply_plan_changes(base_plan, changes)
        base = run_initial(base_plan, samples)
        check = check_incremental(new_plan, samples, base, changed_sources)
        ledger = Ledger(args.db)
        store = SnapshotStore(args.snapshots)
        try:
            session = run_incremental_session(
                new_plan,
                samples,
                base,
                changed_sources,
                ledger,
                snapshot_store=store,
                save_snapshot=True,
                snapshot_label="demo",
            )
            replay = replay_session(ledger, session["session_id"])
            payload = {
                "ok": True,
                "invalidated_layers": check["invalidated_layers"],
                "checked_layers": check["checked_layers"],
                "session_id": session["session_id"],
                "replay": replay,
                "content_digest": session["run"].content_digest,
                "snapshot": session["snapshot"],
            }
            _emit(payload)
            return payload, 0
        finally:
            ledger.close()
    raise MemDiagError(f"unsupported command: {args.command}", field="command")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _, exit_code = _handle(args)
        return exit_code
    except MemDiagError as error:
        print(json.dumps(error.to_dict(), ensure_ascii=False, indent=2), file=sys.stderr)
        return error.exit_code
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        payload = {
            "ok": False,
            "code": "E_DOMAIN",
            "message": str(error),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
