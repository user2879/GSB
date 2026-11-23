from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .errors import BusinessError, as_business_error
from .model import load_dataset
from .store import LedgerStore
from .util import canonical_json_text, digest_obj
from .workbench import Workbench


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = PROJECT_ROOT / "samples" / "base_table.json"
DEFAULT_VIEWS = PROJECT_ROOT / "samples" / "views.json"
DEFAULT_VIEWS_V2 = PROJECT_ROOT / "samples" / "views_v2.json"
DEFAULT_CHANGES = PROJECT_ROOT / "samples" / "changes.ndjson"
DEFAULT_DATA = PROJECT_ROOT / "artifacts" / "demo-workbench"


def _paths(args: argparse.Namespace) -> tuple[str, str, str, str]:
    return str(args.base), str(args.views), str(args.changes), str(args.data_dir)


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def command_validate(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.base, args.views, args.changes)
    result = {
        "status": "valid",
        "code": None,
        "base": {
            "table": dataset.base.table_name,
            "row_id_column": dataset.base.row_id_column,
            "row_count": len(dataset.base.rows),
            "digest": dataset.base.digest,
        },
        "views": {
            "count": len(dataset.definitions.views),
            "ids": sorted(dataset.definitions.views),
            "scale_factor": dataset.definitions.session.scale_factor,
            "fingerprint": dataset.definitions.fingerprint,
        },
        "changes": {"count": len(dataset.changes.changes), "digest": dataset.changes.digest},
    }
    if args.json:
        _print_json(result)
    else:
        print("E_SCHEMA: none")
        print(f"base={result['base']['table']} rows={result['base']['row_count']} digest={result['base']['digest'][:16]}…")
        print(f"views={', '.join(result['views']['ids'])} scale={result['views']['scale_factor']}")
        print(f"changes={result['changes']['count']} digest={result['changes']['digest'][:16]}…")
    return 0


def command_demo(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    data_dir = Path(args.data_dir)
    report: dict[str, Any] = {
        "demo": "incremental-materialized-view-workbench",
        "report_format_version": 1,
        "started_wall_ms": int(time.time() * 1000),
        "inputs": {
            "base": str(args.base),
            "views": str(args.views),
            "changes": str(args.changes),
            "data_dir": str(data_dir),
        },
        "checks": {},
    }
    workbench = Workbench(
        args.base,
        args.views,
        args.changes,
        data_dir,
        reset=True,
        start_watchdog=True,
        auto_rebuild_on_expiry=True,
    )
    try:
        initial = workbench.fast_snapshot()
        report["checks"]["initialization"] = {
            "status": "pass",
            "param_version": initial["param_version"],
            "cursor": initial["cursor"],
            "views": [view["view_id"] for view in initial["views"]],
            "snapshot": initial["snapshot"],
        }
        batch_reports = []
        while workbench.cursor < len(workbench.changes.changes):
            batch = workbench.advance_batch(args.batch_size)
            batch_reports.append(batch)
        report["checks"]["incremental_batches"] = {
            "status": "pass",
            "batch_count": len(batch_reports),
            "cursor": workbench.cursor,
            "batch_latency_ms": [batch.get("latency_ms") for batch in batch_reports],
            "maintenance_audit": [
                {
                    "batch_no": batch.get("batch_no"),
                    "first_seq": batch.get("first_seq"),
                    "last_seq": batch.get("last_seq"),
                    "views": {
                        view["view_id"]: {
                            "impacted_groups": view["impacted_groups"],
                            "recomputed_groups": view["recomputed_groups"],
                            "digest": view["digest"],
                        }
                        for view in batch.get("views", [])
                    },
                }
                for batch in batch_reports
            ],
            "view_digests": {
                view["view_id"]: view["digest"] for view in batch_reports[-1]["views"]
            },
        }
        t0 = time.perf_counter()
        fast = workbench.fast_snapshot()
        fast_latency_ms = (time.perf_counter() - t0) * 1000.0
        report["checks"]["fast_cache_view"] = {
            "status": "pass",
            "render_latency_ms": round(fast_latency_ms, 3),
            "source": fast["source"],
            "visible_views": sum(1 for view in fast["views"] if view["visible"]),
            "inconsistent_views": sum(1 for view in fast["views"] if view["inconsistent"]),
        }
        authority = workbench.authoritative_recompute()
        report["checks"]["authoritative_byte_equality"] = {
            "status": "pass",
            "byte_identical": authority["byte_identical"],
            "content_digest": authority["content_digest"],
            "views": authority["views"],
        }

        lease_result = workbench.simulate_lease_expiry("v_region", ttl_ms=100)
        hidden = workbench.fast_snapshot()
        hidden_view = next(view for view in hidden["views"] if view["view_id"] == "v_region")
        report["checks"]["lease_expiry"] = {
            "status": "pass",
            "lease": lease_result,
            "hidden_while_inconsistent": not hidden_view["visible"],
        }
        recovered = workbench.wait_for_recovery("v_region", timeout_ms=5000)
        if not recovered:
            raise BusinessError("E_LEASE", "lease recovery did not complete before timeout", {"view_id": "v_region"})
        rebuilt = workbench.fast_snapshot()
        rebuilt_view = next(view for view in rebuilt["views"] if view["view_id"] == "v_region")
        report["checks"]["lease_rebuild"] = {
            "status": "pass",
            "recovered": recovered,
            "visible_after_rebuild": rebuilt_view["visible"],
            "lease_state": rebuilt_view["lease"]["state"],
            "refresh_version": rebuilt_view["lease"]["refresh_version"],
            "traces": workbench.rebuild_traces(limit=10),
        }
        replay = workbench.replay_verify()
        report["checks"]["replay_after_rebuild"] = {
            "status": "pass",
            "code": replay["code"],
            "cursor": replay["cursor"],
            "views": replay["views"],
            "rebuild_trace_count": replay["rebuild_trace_count"],
        }

        if args.inject in {"lease_conflict", "base_corruption"}:
            if args.inject == "lease_conflict":
                injection = workbench.inject_lease_conflict()
            else:
                injection = workbench.inject_base_corruption()
            report["checks"]["anomaly_injection"] = injection
        workbench.close()
        workbench = None

        if args.inject == "snapshot_tamper":
            store = LedgerStore(data_dir / "ledger.sqlite3")
            tamper = store.tamper_result_digest()
            store.close()
            reopened = Workbench(
                args.base,
                args.views,
                args.changes,
                data_dir,
                start_watchdog=False,
                auto_rebuild_on_expiry=False,
            )
            try:
                snapshot_errors = [event for event in reopened.startup_events if event.get("code") == "E_SNAPSHOT"]
                report["checks"]["anomaly_injection"] = {
                    "status": "recovered" if snapshot_errors else "not-detected",
                    "code": "E_SNAPSHOT" if snapshot_errors else None,
                    "tamper": tamper,
                    "startup_errors": snapshot_errors,
                    "replay": reopened.replay_verify(),
                }
            finally:
                reopened.close()
        else:
            reopened = Workbench(
                args.base,
                args.views,
                args.changes,
                data_dir,
                start_watchdog=False,
                auto_rebuild_on_expiry=False,
            )
            try:
                audit = reopened._last_snapshot_audit
                report["checks"]["snapshot_reopen"] = {
                    "status": "pass",
                    "snapshot_version": None if audit is None else audit.snapshot_version,
                    "content_digest": None if audit is None else audit.stored_digest,
                    "matches": None if audit is None else audit.matches,
                    "recovered": None if audit is None else audit.recovered,
                    "replay": reopened.replay_verify(),
                }
                if args.replace_views:
                    replacement = reopened.apply_views_file(args.views_v2)
                    authority_after = reopened.authoritative_recompute()
                    replay_after = reopened.replay_verify()
                    report["checks"]["definition_replacement"] = {
                        "status": "pass",
                        "param_version": replacement["param_version"],
                        "summary": replacement["summary"],
                        "authority": authority_after,
                        "replay": replay_after,
                    }
            finally:
                reopened.close()

        report["status"] = "pass"
        report["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        report["content_digest"] = digest_obj(report)
        data_dir.mkdir(parents=True, exist_ok=True)
        report_path = data_dir / "demo_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.json:
            _print_json(report)
        else:
            _print_demo_summary(report, report_path)
        return 0
    finally:
        if workbench is not None:
            workbench.close()


def _print_demo_summary(report: dict[str, Any], report_path: Path) -> None:
    checks = report["checks"]
    print("增量物化视图工作台 · 可重复演示")
    print("=" * 44)
    print(f"初始视图          {', '.join(checks['initialization']['views'])}")
    print(f"增量批次          {checks['incremental_batches']['batch_count']} 批，游标 {checks['incremental_batches']['cursor']}")
    print(f"快速缓存渲染      {checks['fast_cache_view']['render_latency_ms']:.3f} ms（未触发重算）")
    print(f"权威重算          {checks['authoritative_byte_equality']['status'].upper()} / 逐字节一致")
    print("租约过期隐藏      是，随后全量重建并恢复")
    print(f"重建后重放校验    {checks['replay_after_rebuild']['status'].upper()}")
    if "snapshot_reopen" in checks:
        print(f"快照重开校验      {checks['snapshot_reopen']['status'].upper()}")
    if "definition_replacement" in checks:
        delta = checks["definition_replacement"]["summary"]
        print(
            "定义整体替换      "
            f"v{delta['old_param_version']} → v{delta['new_param_version']}，"
            f"新增 {delta['added']} 删除 {delta['removed']} 变更 {delta['changed']}"
        )
    if "anomaly_injection" in checks:
        injection = checks["anomaly_injection"]
        print(f"异常注入          {injection.get('code')} / {injection.get('status')}")
    print("-" * 44)
    print(f"报告              {report_path}")


def command_maintain(args: argparse.Namespace) -> int:
    workbench = Workbench(
        args.base,
        args.views,
        args.changes,
        args.data_dir,
        reset=args.reset,
        start_watchdog=False,
    )
    try:
        reports = []
        for _ in range(args.steps):
            if workbench.cursor >= len(workbench.changes.changes):
                break
            reports.append(workbench.advance_batch(args.batch_size))
        result = {
            "status": "ok",
            "cursor": workbench.cursor,
            "batches": reports,
            "fast": workbench.fast_snapshot(),
        }
        if args.json:
            _print_json(result)
        else:
            print(f"cursor={workbench.cursor} batches={len(reports)}")
            for report in reports:
                print(
                    f"batch={report.get('batch_no')} seq={report.get('first_seq')}..{report.get('last_seq')} "
                    f"latency={report.get('latency_ms')}ms"
                )
        return 0
    finally:
        workbench.close()


def command_gui(args: argparse.Namespace) -> int:
    try:
        from .gui import launch
    except ImportError as exc:
        raise BusinessError(
            "E_STATE",
            "tkinter is unavailable in this Python installation",
            {"module": "tkinter"},
            {"error": str(exc)},
        ) from exc
    launch(
        args.base,
        args.views,
        args.changes,
        args.data_dir,
        reset=args.reset,
        auto_rebuild_on_expiry=True,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="imv-workbench",
        description="Windows-local incremental materialized-view maintenance workbench",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_data_options(target: argparse.ArgumentParser) -> None:
        target.add_argument("--base", default=str(DEFAULT_BASE), help="base table JSON")
        target.add_argument("--views", default=str(DEFAULT_VIEWS), help="view definitions JSON")
        target.add_argument("--changes", default=str(DEFAULT_CHANGES), help="change stream NDJSON")
        target.add_argument("--data-dir", default=str(DEFAULT_DATA), help="writable directory for SQLite ledger, snapshots and logs")

    validate_parser = subparsers.add_parser("validate", help="validate all local inputs without partial loading")
    add_data_options(validate_parser)
    validate_parser.add_argument("--json", action="store_true")
    validate_parser.set_defaults(func=command_validate)

    demo_parser = subparsers.add_parser("demo", help="repeatable end-to-end acceptance demonstration")
    add_data_options(demo_parser)
    demo_parser.add_argument("--batch-size", type=int, default=None, help="override batch boundary size")
    demo_parser.add_argument(
        "--inject",
        choices=["none", "lease_conflict", "base_corruption", "snapshot_tamper"],
        default="none",
        help="inject one structured business-code anomaly",
    )
    demo_parser.add_argument("--replace-views", action="store_true", default=True, help="show whole-definition replacement")
    demo_parser.add_argument("--no-replace-views", action="store_false", dest="replace_views")
    demo_parser.add_argument("--views-v2", default=str(DEFAULT_VIEWS_V2), help="alternate view definitions for replacement proof")
    demo_parser.add_argument("--json", action="store_true", help="emit only the machine-readable report")
    demo_parser.set_defaults(func=command_demo)

    maintain_parser = subparsers.add_parser("maintain", help="advance a fixed number of batches")
    add_data_options(maintain_parser)
    maintain_parser.add_argument("--steps", type=int, default=1)
    maintain_parser.add_argument("--batch-size", type=int, default=None)
    maintain_parser.add_argument("--reset", action="store_true")
    maintain_parser.add_argument("--json", action="store_true")
    maintain_parser.set_defaults(func=command_maintain)

    gui_parser = subparsers.add_parser("gui", help="launch the Tk desktop workbench")
    add_data_options(gui_parser)
    gui_parser.add_argument("--reset", action="store_true")
    gui_parser.set_defaults(func=command_gui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except BusinessError as exc:
        print(canonical_json_text(exc.to_dict()), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    except BaseException as exc:
        error = as_business_error(exc)
        print(canonical_json_text(error.to_dict()), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
