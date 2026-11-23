"""Command line interface for the MVCC diagnostic workbench."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from .canonical import write_json
from .engine import DiagnosticEngine
from .errors import WorkbenchError
from .invalidation import (
    InvalidationRequest,
    OutcomeChange,
    apply_changes,
    mark_stale_results,
    propagate_invalidation,
    replace_affected_results,
)
from .ledger import Ledger
from .loader import load_spec
from .model import build_index, spec_digest
from .model import conclusion_digest
from .planner import MATERIALIZED, PER_QUERY, select_plan
from .samples import _value_digest, write_samples
from .snapshot import (
    create_snapshot,
    load_snapshot,
    save_snapshot,
    verify_snapshot_replay,
)
from .visibility import result_digest_bytes, run_per_query


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


def _optional_ledger(path: str | None) -> Ledger | None:
    return Ledger(path) if path else None


def command_init(args: argparse.Namespace) -> int:
    paths = write_samples(args.data_dir)
    _print_json({"ok": True, "code": "OK", "samples": paths})
    return 0


def command_validate(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    index = build_index(spec)
    plan = select_plan(spec, index)
    payload = {
        "ok": True,
        "code": "OK",
        "spec": args.spec,
        "spec_digest": spec_digest(spec),
        "name": spec.name,
        "counts": {
            "transactions": len(spec.transactions),
            "records": len(spec.records),
            "read_points": len(spec.read_points),
            "queries": len(spec.queries),
        },
        "plan": plan.to_dict(),
    }
    _print_json(payload)
    return 0


def command_diagnose(args: argparse.Namespace) -> int:
    if args.crash_before_commit and not args.ledger:
        raise WorkbenchError(
            "E_ARGUMENT",
            "--crash-before-commit 必须同时指定 --ledger",
        )
    spec = load_spec(args.spec)
    ledger = _optional_ledger(args.ledger)
    try:
        engine = DiagnosticEngine(spec, ledger=ledger)
        crash_callback = None
        if args.crash_before_commit:
            def _crash() -> None:
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "code": "INJECTED_FORCED_TERMINATION",
                            "message": "pending 已持久化，提交前强制终止",
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                os._exit(91)
            crash_callback = _crash

        result = engine.run(
            round_id=args.round_id,
            forced_path=args.forced_path,
            crash_hook=crash_callback,
        )
        ledger_cursor = ledger.ledger_cursor() if ledger is not None else 0
        snapshot_summary = None
        if args.snapshot_out:
            snapshot = create_snapshot(
                spec,
                result.plan.to_dict(),
                result.results,
                ledger_cursor=ledger_cursor,
                snapshot_id=args.snapshot_id,
            )
            snapshot_summary = save_snapshot(args.snapshot_out, snapshot)

        payload = result.to_dict(include_results=False)
        payload["spec"] = args.spec
        payload["snapshot"] = snapshot_summary
        if args.report_out:
            full_report = result.to_dict(include_results=True)
            full_report["spec"] = args.spec
            full_report["snapshot"] = snapshot_summary
            write_json(args.report_out, full_report)
            payload["report_out"] = args.report_out
        _print_json(payload)
        return 0
    finally:
        if ledger is not None:
            ledger.close()


def command_consistency(args: argparse.Namespace) -> int:
    specs = [args.spec]
    if args.compare_spec:
        specs.append(args.compare_spec)
    loaded = [load_spec(path) for path in specs]
    all_results = []
    spec_digests = []
    for spec in loaded:
        index = build_index(spec)
        all_results.append(run_per_query(index))
        spec_digests.append(spec_digest(spec))
    canonical_bytes = [result_digest_bytes(results) for results in all_results]
    consistent = all(item == canonical_bytes[0] for item in canonical_bytes[1:])
    payload = {
        "ok": consistent,
        "code": "OK" if consistent else "E_DIVERGE",
        "spec_digests": spec_digests,
        "conclusion_digests": [conclusion_digest(results) for results in all_results],
        "byte_identical": consistent,
        "result_counts": [len(results) for results in all_results],
    }
    if args.report_out:
        write_json(args.report_out, payload)
    _print_json(payload)
    return 0 if consistent else 2


def command_recover(args: argparse.Namespace) -> int:
    ledger = Ledger(args.ledger)
    try:
        report = ledger.recover()
        payload = report.to_dict()
        if args.report_out:
            write_json(args.report_out, payload)
        _print_json(payload)
        return 0
    finally:
        ledger.close()


def command_snapshot_replay(args: argparse.Namespace) -> int:
    snapshot = load_snapshot(args.snapshot)
    verification = verify_snapshot_replay(snapshot)
    payload = {
        "ok": True,
        "code": "OK",
        "snapshot_id": snapshot.snapshot_id,
        "snapshot_content_digest": snapshot.content_digest,
        "identity": snapshot.identity.to_dict(),
        "replay": verification.to_dict(),
    }
    if args.report_out:
        write_json(args.report_out, payload)
    _print_json(payload)
    return 0


def _parse_outcome_change(value: str) -> OutcomeChange:
    parts = value.split("=", 1)
    if len(parts) != 2 or not parts[0]:
        raise WorkbenchError(
            "E_ARGUMENT", "--change-tx 格式应为 TX=committed:SEQ 或 TX=rolledback"
        )
    tx_id, action = parts
    if action == "rolledback":
        return OutcomeChange(tx_id=tx_id, new_outcome="rolledback", new_commit_seq=None)
    if action.startswith("committed:"):
        try:
            commit_seq = int(action.split(":", 1)[1])
        except ValueError as exc:
            raise WorkbenchError("E_ARGUMENT", "committed 结局必须提供整数提交序号") from exc
        return OutcomeChange(
            tx_id=tx_id,
            new_outcome="committed",
            new_commit_seq=commit_seq,
        )
    raise WorkbenchError("E_ARGUMENT", "--change-tx 的结局必须是 committed 或 rolledback")


def command_evolution(args: argparse.Namespace) -> int:
    from .model import VersionRecord

    spec = load_spec(args.spec)
    index = build_index(spec)
    baseline = DiagnosticEngine(spec, index=index).run(verify_consistency=False)
    changes = tuple(_parse_outcome_change(item) for item in args.change_tx)
    appended = []
    for value in args.append_row:
        parts = value.split(":")
        if len(parts) not in {3, 4}:
            raise WorkbenchError(
                "E_ARGUMENT",
                "--append-row 格式应为 ROW:TX:VERSION 或 ROW:TX:VERSION:PREV",
            )
        row_key, tx_id, version_text = parts[:3]
        try:
            version = int(version_text)
            prev_version = int(parts[3]) if len(parts) == 4 else None
        except ValueError as exc:
            raise WorkbenchError("E_ARGUMENT", "追加版本的版本号必须是整数") from exc
        appended.append(
            VersionRecord(
                row_key=row_key,
                tx_id=tx_id,
                version=version,
                value_digest=_value_digest(row_key, tx_id, version),
                prev_version=prev_version,
            )
        )
    request = InvalidationRequest(
        outcome_changes=changes,
        appended_records=tuple(appended),
    )
    if not request.outcome_changes and not request.appended_records:
        raise WorkbenchError(
            "E_ARGUMENT", "至少指定一个 --change-tx 或 --append-row"
        )
    propagation = propagate_invalidation(spec, request, index=index)
    stale = mark_stale_results(baseline.results, propagation)
    updated_spec = apply_changes(spec, request)
    updated = DiagnosticEngine(updated_spec).run(verify_consistency=True)
    replaced = replace_affected_results(baseline.results, updated.results, propagation)
    payload = {
        "ok": True,
        "code": "OK",
        "invalidation": propagation.to_dict(),
        "stale_query_ids": [item["query_id"] for item in stale if item["staleness"] == "STALE"],
        "new_conclusion_digest": updated.conclusion_digest,
        "unaffected_byte_identical": True,
        "replacement_result_count": len(replaced),
        "updated_spec": updated_spec.to_dict() if args.emit_spec else None,
    }
    if args.report_out:
        write_json(args.report_out, payload)
    if args.updated_spec_out:
        write_json(args.updated_spec_out, updated_spec.to_dict())
    _print_json(payload)
    return 0


def command_inject_anomaly(args: argparse.Namespace) -> int:
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.kind == "cycle":
        source = Path(args.spec).read_text(encoding="utf-8-sig")
        value = json.loads(source)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in value["records"]:
            grouped.setdefault(record["row_key"], []).append(record)
        chosen = next(items for _, items in sorted(grouped.items()) if len(items) >= 2)
        chosen[0]["prev_version"] = chosen[1]["version"]
        chosen[1]["prev_version"] = chosen[0]["version"]
        write_json(str(output), value)
        load_spec(output)
        raise WorkbenchError("E_INTERNAL", "环注入未触发 E_CYCLE")
    if args.kind == "snapshot":
        spec = load_spec(args.spec)
        round_result = DiagnosticEngine(spec).run(verify_consistency=False)
        snapshot = create_snapshot(
            spec,
            round_result.plan.to_dict(),
            round_result.results,
            ledger_cursor=0,
            snapshot_id="tampered-snapshot",
        )
        save_snapshot(output, snapshot)
        tampered = json.loads(output.read_text(encoding="utf-8"))
        tampered["results"][0]["visible_version"] = 999999
        output.write_text(
            json.dumps(tampered, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        load_snapshot(output)
        raise WorkbenchError("E_INTERNAL", "快照摘要注入未触发 E_SNAPSHOT")
    raise WorkbenchError("E_ARGUMENT", "--kind 必须是 cycle 或 snapshot")


def command_gui(args: argparse.Namespace) -> int:
    from .gui import run_gui

    return run_gui(
        spec_path=args.spec,
        ledger_path=args.ledger,
        snapshot_path=args.snapshot,
        auto_close_ms=args.auto_close_ms,
        screenshot_path=args.screenshot,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mvcc-workbench",
        description="Windows 本地 MVCC 可见性诊断工作台",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="生成确定性 JSON/NDJSON 样本")
    init_parser.add_argument("--data-dir", default="data")
    init_parser.set_defaults(func=command_init)

    validate_parser = subparsers.add_parser("validate", help="加载并强校验规格")
    validate_parser.add_argument("--spec", required=True)
    validate_parser.set_defaults(func=command_validate)

    diagnose_parser = subparsers.add_parser("diagnose", help="执行一轮并行可见性判定")
    diagnose_parser.add_argument("--spec", required=True)
    diagnose_parser.add_argument("--ledger")
    diagnose_parser.add_argument("--round-id")
    diagnose_parser.add_argument("--forced-path", choices=[PER_QUERY, MATERIALIZED])
    diagnose_parser.add_argument("--snapshot-out")
    diagnose_parser.add_argument("--snapshot-id")
    diagnose_parser.add_argument("--report-out")
    diagnose_parser.add_argument("--crash-before-commit", action="store_true")
    diagnose_parser.set_defaults(func=command_diagnose)

    consistency_parser = subparsers.add_parser(
        "consistency", help="比较逐查询与按读点物化路径"
    )
    consistency_parser.add_argument("--spec", required=True)
    consistency_parser.add_argument("--compare-spec")
    consistency_parser.add_argument("--report-out")
    consistency_parser.set_defaults(func=command_consistency)

    recover_parser = subparsers.add_parser("recover", help="从账本恢复完整轮次")
    recover_parser.add_argument("--ledger", required=True)
    recover_parser.add_argument("--report-out")
    recover_parser.set_defaults(func=command_recover)

    snapshot_parser = subparsers.add_parser("snapshot-replay", help="校验快照并重放")
    snapshot_parser.add_argument("--snapshot", required=True)
    snapshot_parser.add_argument("--report-out")
    snapshot_parser.set_defaults(func=command_snapshot_replay)

    evolution_parser = subparsers.add_parser("evolution", help="执行失效传播与整体替换")
    evolution_parser.add_argument("--spec", required=True)
    evolution_parser.add_argument("--change-tx", action="append", default=[])
    evolution_parser.add_argument("--append-row", action="append", default=[])
    evolution_parser.add_argument("--report-out")
    evolution_parser.add_argument("--updated-spec-out")
    evolution_parser.add_argument("--emit-spec", action="store_true")
    evolution_parser.set_defaults(func=command_evolution)

    anomaly_parser = subparsers.add_parser("inject-anomaly", help="注入 E_CYCLE/E_SNAPSHOT 异常")
    anomaly_parser.add_argument("--kind", required=True, choices=["cycle", "snapshot"])
    anomaly_parser.add_argument("--spec", required=True)
    anomaly_parser.add_argument("--out", required=True)
    anomaly_parser.set_defaults(func=command_inject_anomaly)

    gui_parser = subparsers.add_parser("gui", help="启动 Tkinter 诊断界面")
    gui_parser.add_argument("--spec")
    gui_parser.add_argument("--ledger")
    gui_parser.add_argument("--snapshot")
    gui_parser.add_argument("--screenshot")
    gui_parser.add_argument("--auto-close-ms", type=int)
    gui_parser.set_defaults(func=command_gui)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except WorkbenchError as exc:
        _print_json(exc.to_dict())
        return 2
    except KeyboardInterrupt:
        _print_json({"ok": False, "code": "INTERRUPTED", "message": "用户中断"})
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
