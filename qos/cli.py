"""命令行入口：覆盖加载、增量、编译、整形、恢复、快照、重放与异常注入。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .applog import StructuredLogger
from .changeset import ChangeSet
from .errors import QosError
from .workbench import Workbench

DEFAULT_DB = "data/qos_ledger.db"
DEFAULT_LOG = "logs/qos.jsonl"


def _print(obj: Any) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))


def _load_changeset(path: str) -> ChangeSet:
    return ChangeSet.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def cmd_load(args: argparse.Namespace) -> int:
    wb = Workbench(args.db, args.log)
    try:
        result = wb.load(args.rules)
        _print({"ok": True, **result})
        return 0
    finally:
        wb.close()


def cmd_change(args: argparse.Namespace) -> int:
    wb = Workbench(args.db, args.log)
    try:
        wb.load(args.rules)
        result = wb.apply_change(_load_changeset(args.changeset))
        if args.verify:
            result["incremental_check"] = wb.verify_incremental()
        _print({"ok": True, **result})
        return 0
    finally:
        wb.close()


def cmd_verify_incr(args: argparse.Namespace) -> int:
    wb = Workbench(args.db, args.log)
    try:
        wb.load(args.rules)
        wb.apply_change(_load_changeset(args.changeset))
        result = wb.verify_incremental()
        _print({"ok": True, **result})
        return 0
    finally:
        wb.close()


def cmd_simulate(args: argparse.Namespace) -> int:
    wb = Workbench(args.db, args.log)
    try:
        wb.load(args.rules)
        result = wb.simulate(count=args.count, size=args.size,
                             dst_ports=args.ports or None,
                             arrival_gap_us=args.gap_us)
        _print({"ok": True, **result})
        return 0
    finally:
        wb.close()


def cmd_crash_recover(args: argparse.Namespace) -> int:
    wb = Workbench(args.db, args.log)
    try:
        wb.load(args.rules)
        op_id = wb.simulate_crash_during_compile()
        # 模拟强制终止：丢弃内存状态后重新打开账本恢复。
        wb.close()
        wb2 = Workbench(args.db, args.log)
        report = wb2.recover()
        _print({"ok": True, "crashed_op_id": op_id, "recovery": report})
        return 0
    finally:
        wb.close()


def cmd_snapshot_create(args: argparse.Namespace) -> int:
    wb = Workbench(args.db, args.log)
    try:
        wb.load(args.rules)
        if args.changeset:
            wb.apply_change(_load_changeset(args.changeset))
        wb.simulate(count=args.count, size=args.size)
        snap = wb.create_snapshot(args.out)
        _print({"ok": True, "snapshot": snap})
        return 0
    finally:
        wb.close()


def cmd_snapshot_restore(args: argparse.Namespace) -> int:
    wb = Workbench(args.db, args.log)
    try:
        wb.load(args.rules)
        result = wb.restore_snapshot(args.snapshot, require_versions=True)
        _print({"ok": True, **result})
        return 0
    finally:
        wb.close()


def cmd_replay(args: argparse.Namespace) -> int:
    wb = Workbench(args.db, args.log)
    try:
        wb.load(args.rules)
        baseline = list(wb.rules)
        if args.changeset:
            wb.apply_change(_load_changeset(args.changeset))
        result = wb.replay(args.snapshot, baseline_rules=baseline)
        _print({"ok": True, **result})
        return 0
    finally:
        wb.close()


def cmd_inject(args: argparse.Namespace) -> int:
    """异常注入：corrupt -> E_RULE，cycle -> E_DEPEND，snapshot -> E_SNAP。

    注入仅改变输入文件选择或参数，不改动业务代码路径。
    """
    logger = StructuredLogger(args.log)
    wb = Workbench(args.db, args.log)
    try:
        if args.fault == "corrupt":
            try:
                wb.load(args.rules)
            except QosError as exc:
                logger.error("inject_corrupt", exc.code,
                             details=exc.details, file=args.rules)
                _print({"ok": False, **exc.to_dict()})
                return 2
        elif args.fault == "cycle":
            try:
                wb.load(args.rules)
            except QosError as exc:
                logger.error("inject_cycle", exc.code,
                             details=exc.details, file=args.rules)
                _print({"ok": False, **exc.to_dict()})
                return 2
        elif args.fault == "snapshot":
            try:
                wb.load("samples/rules.json")
                # 注入仅改变参数：强制版本识别，缺失版本字段须以 E_SNAP 拒绝。
                wb.restore_snapshot(args.snapshot, require_versions=True)
            except QosError as exc:
                logger.error("inject_snapshot", exc.code,
                             details=exc.details, file=args.snapshot)
                _print({"ok": False, **exc.to_dict()})
                return 2
        _print({"ok": False, "code": "E_UNKNOWN",
                "message": "注入未触发预期错误"})
        return 3
    finally:
        wb.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qos", description="QoS 流量整形规则诊断工作台")
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--log", default=DEFAULT_LOG)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_rules(p: argparse.ArgumentParser) -> None:
        p.add_argument("--rules", default="samples/rules.json")

    p = sub.add_parser("load"); add_rules(p)
    p.set_defaults(func=cmd_load)

    p = sub.add_parser("change"); add_rules(p)
    p.add_argument("--changeset", required=True)
    p.add_argument("--verify", action="store_true")
    p.set_defaults(func=cmd_change)

    p = sub.add_parser("verify-incr"); add_rules(p)
    p.add_argument("--changeset", required=True)
    p.set_defaults(func=cmd_verify_incr)

    p = sub.add_parser("simulate"); add_rules(p)
    p.add_argument("--count", type=int, default=40)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--ports", type=int, nargs="*")
    p.add_argument("--gap-us", type=int, default=50_000)
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("crash-recover"); add_rules(p)
    p.set_defaults(func=cmd_crash_recover)

    p = sub.add_parser("snapshot-create"); add_rules(p)
    p.add_argument("--changeset")
    p.add_argument("--out", default="data/snapshot.json")
    p.add_argument("--count", type=int, default=20)
    p.add_argument("--size", type=int, default=512)
    p.set_defaults(func=cmd_snapshot_create)

    p = sub.add_parser("snapshot-restore"); add_rules(p)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--inject-missing-version", action="store_true",
                   help="注入快照版本缺失（E_SNAP）")
    p.set_defaults(func=cmd_snapshot_restore)

    p = sub.add_parser("replay"); add_rules(p)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--changeset")
    p.set_defaults(func=cmd_replay)

    p = sub.add_parser("inject")
    p.add_argument("--fault", choices=["corrupt", "cycle", "snapshot"],
                   required=True)
    p.add_argument("--rules")
    p.add_argument("--snapshot")
    p.set_defaults(func=cmd_inject)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except QosError as exc:
        StructuredLogger(args.log).error(
            args.cmd, exc.code, message=exc.message, details=exc.details)
        _print({"ok": False, **exc.to_dict()})
        return 2
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        _print({"ok": False, "code": "E_UNKNOWN", "message": str(exc)})
        return 2


if __name__ == "__main__":
    sys.exit(main())
