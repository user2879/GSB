"""命令行诊断入口。

示例:
  python -m dwb_diag.cli demo --root data/demo
  python -m dwb_diag.cli inject --root data/demo --kind main_corrupt
  python -m dwb_diag.cli recover --root data/demo
  python -m dwb_diag.cli view-verify --root data/demo
  python -m dwb_diag.cli snapshot-open --root data/demo --session s1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import DEFAULT_CAPACITY, DEFAULT_PAGE_SIZE
from .engine import DiagnosticEngine, build_demo_views, deterministic_changes
from .errors import DiagError
from .storage import CrashPlan
from . import fault_injection as fi
from .buffer import DoubleWriteBuffer
from .config import DataLayout
from .versions import VersionMap
from .util import StructuredLogger


def _print(obj: Any) -> None:
    print(json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2))


def _engine(root: Path, page_size: int, capacity: int) -> DiagnosticEngine:
    return DiagnosticEngine(root, page_size=page_size, capacity=capacity)


def cmd_init(args: argparse.Namespace) -> int:
    eng = _engine(args.root, args.page_size, args.capacity)
    build_demo_views(eng.catalog)
    eng.catalog.save()
    _print({"status": "initialized", "root": str(args.root), "views": eng.catalog.names_sorted()})
    return 0


def cmd_write(args: argparse.Namespace) -> int:
    eng = _engine(args.root, args.page_size, args.capacity)
    changes: dict[str, int | None] = {}
    for item in args.kv:
        key, value = item.split("=", 1)
        changes[key] = None if value == "" else int(value)
    result = eng.submit(changes)
    _print(
        {
            "status": "committed",
            "commit_id": result.commit_id,
            "pages": result.page_nos,
            "seqs": result.seqs,
        }
    )
    return 0


def cmd_crash(args: argparse.Namespace) -> int:
    eng = _engine(args.root, args.page_size, args.capacity)
    changes: dict[str, int | None] = {}
    for item in args.kv:
        key, value = item.split("=", 1)
        changes[key] = None if value == "" else int(value)
    plan = CrashPlan(commit_id=args.commit, phase=args.phase)
    try:
        eng.submit(changes, crash_plan=plan)
    except Exception as exc:  # noqa: BLE001 - 模拟崩溃是预期控制流
        _print({"status": "crashed", "type": type(exc).__name__, "detail": str(exc)})
        return 0
    _print({"status": "no_crash"})
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    eng = _engine(args.root, args.page_size, args.capacity)
    if not eng.layout.views_file.exists():
        build_demo_views(eng.catalog)
        eng.views = type(eng.views)(eng.catalog, eng.storage.keyspace)
        eng.catalog.save()
    out: dict[str, Any] = {"batches": []}
    for idx, changes in enumerate(deterministic_changes()):
        # 第 3 批在主文件覆盖阶段模拟一次崩溃
        plan = CrashPlan(commit_id=idx + 1, phase="main_overwrite") if idx == 2 else None
        try:
            result = eng.submit(changes, crash_plan=plan)
            out["batches"].append(
                {"commit_id": result.commit_id, "pages": result.page_nos, "crashed": False}
            )
        except Exception as exc:  # noqa: BLE001
            out["batches"].append(
                {"commit_id": idx + 1, "crashed": True, "detail": str(exc)}
            )
            break
    report = eng.recover()
    out["recovery"] = report.to_dict()
    out["view_verify"] = eng.verify_views()
    snap = eng.save_snapshot("s1", report)
    out["snapshot"] = snap.to_dict()
    replay = eng.replay(snap, args.root.parent / f"{args.root.name}-replay")
    out["replay"] = replay
    _print(out)
    return 0


def cmd_recover(args: argparse.Namespace) -> int:
    eng = _engine(args.root, args.page_size, args.capacity)
    report = eng.recover(raise_on_lost=not args.tolerate_lost)
    _print(report.to_dict())
    return 0


def cmd_view_rebuild(args: argparse.Namespace) -> int:
    eng = _engine(args.root, args.page_size, args.capacity)
    eng.views.rebuild_graph()
    rebuilt = eng.views.rebuild_all()
    _print({"status": "rebuilt", "views": rebuilt})
    return 0


def cmd_view_verify(args: argparse.Namespace) -> int:
    eng = _engine(args.root, args.page_size, args.capacity)
    _print({"status": "ok", "views": eng.verify_views()})
    return 0


def cmd_validate_buffer(args: argparse.Namespace) -> int:
    layout = DataLayout(args.root)
    dwb = DoubleWriteBuffer(layout.buffer_file, args.capacity)
    result = dwb.validate_strict()
    _print(
        {
            "status": "ok",
            "mirrors": [m.to_row() for m in result.mirrors],
            "max_seq": result.max_seq,
            "buffer_digest": dwb.file_digest_hex(),
        }
    )
    return 0


def cmd_validate_main(args: argparse.Namespace) -> int:
    layout = DataLayout(args.root)
    fi.assert_main_intact(layout, args.page_size)
    _print({"status": "ok", "digests": VersionMap.load(layout.version_map_file).as_dict()})
    return 0


def cmd_snapshot_list(args: argparse.Namespace) -> int:
    eng = _engine(args.root, args.page_size, args.capacity)
    _print({"snapshots": eng.snapshots.list_snapshots()})
    return 0


def cmd_snapshot_open(args: argparse.Namespace) -> int:
    eng = _engine(args.root, args.page_size, args.capacity)
    try:
        result = eng.open_snapshot(args.session)
        _print(result)
        return 0
    except DiagError as exc:
        _print(exc.to_dict())
        return 0  # 业务码作为结构化结果输出, 进程仍以 0 结束便于脚本采集


def cmd_inject(args: argparse.Namespace) -> int:
    layout = DataLayout(args.root).ensure()
    if args.kind == "main_corrupt":
        versions = VersionMap.load(layout.version_map_file)
        target = args.page
        if target not in versions.pages and versions.pages:
            # 指定页未写入时回退到首个已写页, 保证损坏可被摘要比对捕获
            target = sorted(versions.pages)[0]
        result = fi.inject_main_corrupt(layout, target, args.page_size)
        # 注入后立即校验主文件, 应报 E_CORRUPT
        try:
            fi.assert_main_intact(layout, args.page_size)
        except DiagError as exc:
            eng = _engine(args.root, args.page_size, args.capacity)
            eng.logger.log_error("inject_assert", exc)
            result["asserted"] = exc.to_dict()
            _print(result)
            return 0
        raise DiagError(
            "main_corrupt injection did not produce E_CORRUPT",
            code="E_CORRUPT",
            detail={"page_no": target},
        )
    elif args.kind == "buffer_truncate":
        result = fi.inject_buffer_truncate(layout)
        dwb = DoubleWriteBuffer(layout.buffer_file, args.capacity)
        try:
            dwb.validate_strict()
        except DiagError as exc:
            logger = StructuredLogger(layout.log_file)
            logger.log_error("inject_assert", exc)
            result["asserted"] = exc.to_dict()
            _print(result)
            return 0
    elif args.kind == "depend_cycle":
        result = fi.inject_dependency_cycle(layout, ("rollup_all", "cnt_user"))
        from .views import ViewCatalog, ViewManager
        from .keyspace import Keyspace
        from .paging import MainDataFile

        catalog = ViewCatalog.load(layout.views_file)
        main = MainDataFile(layout.main_file, args.page_size, args.capacity)
        vm = ViewManager(catalog, Keyspace(main))
        try:
            vm.propagate_invalidation({0})
        except DiagError as exc:
            logger = StructuredLogger(layout.log_file)
            logger.log_error("inject_assert", exc)
            result["asserted"] = exc.to_dict()
            _print(result)
            return 0
    elif args.kind == "snapshot_stale":
        result = fi.inject_snapshot_stale(layout)
        eng = _engine(args.root, args.page_size, args.capacity)
        snap = eng.snapshots.latest()
        try:
            eng.open_snapshot(snap.session_id)
        except DiagError as exc:
            result["asserted"] = exc.to_dict()
            _print(result)
            return 0
    else:
        raise SystemExit(f"unknown injection kind {args.kind}")
    _print(result)
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    eng = _engine(args.root, args.page_size, args.capacity)
    snap = eng.snapshots.latest()
    if snap is None:
        raise SystemExit("no snapshot available")
    scratch = args.root.parent / f"{args.root.name}-replay-manual"
    if args.tamper_page >= 0:
        # 先克隆再篡改, 验证 E_REPLAY 链路
        cloned = fi.clone_layout(eng.layout, scratch)
        fi.inject_replay_tamper(cloned, args.tamper_page, args.page_size)
        from .storage import Storage
        from .views import ViewCatalog, ViewManager
        from .recovery import RecoveryEngine

        logger = StructuredLogger(cloned.log_file, eng.clock)
        store = Storage(cloned, args.page_size, args.capacity, eng.clock, logger)
        vman = ViewManager(ViewCatalog.load(cloned.views_file), store.keyspace)
        report = RecoveryEngine(
            store.main, store.dwb, store.versions, store.keyspace, logger
        ).run()
        try:
            fi.replay_compare(
                report.final_digests,
                eng._snapshot_main_digests(snap),  # noqa: SLF001
                [],
                snap.invalidation,
            )
        except DiagError as exc:
            eng.logger.log_error("replay_assert", exc)
            _print(exc.to_dict())
            return 0
    result = eng.replay(snap, scratch)
    _print(result)
    return 0


def cmd_keyspace(args: argparse.Namespace) -> int:
    eng = _engine(args.root, args.page_size, args.capacity)
    if not eng.storage.keyspace.ready:
        eng.recover()
    _print({"keyspace": eng.storage.keyspace.scan()})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dwb_diag", description="页级双写缓冲恢复诊断")
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--capacity", type=int, default=DEFAULT_CAPACITY)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init"); p.set_defaults(func=cmd_init)
    p.add_argument("--root", type=Path, required=True)

    p = sub.add_parser("write"); p.set_defaults(func=cmd_write)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--kv", action="append", required=True, help="key=value, 空 value 表示删除")

    p = sub.add_parser("crash"); p.set_defaults(func=cmd_crash)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--commit", type=int, required=True)
    p.add_argument("--phase", choices=list(("buffer_write", "main_overwrite", "commit_mark")), required=True)
    p.add_argument("--kv", action="append", required=True)

    p = sub.add_parser("demo"); p.set_defaults(func=cmd_demo)
    p.add_argument("--root", type=Path, required=True)

    p = sub.add_parser("recover"); p.set_defaults(func=cmd_recover)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--tolerate-lost", action="store_true")

    p = sub.add_parser("view-rebuild"); p.set_defaults(func=cmd_view_rebuild)
    p.add_argument("--root", type=Path, required=True)

    p = sub.add_parser("view-verify"); p.set_defaults(func=cmd_view_verify)
    p.add_argument("--root", type=Path, required=True)

    p = sub.add_parser("validate-buffer"); p.set_defaults(func=cmd_validate_buffer)
    p.add_argument("--root", type=Path, required=True)

    p = sub.add_parser("validate-main"); p.set_defaults(func=cmd_validate_main)
    p.add_argument("--root", type=Path, required=True)

    p = sub.add_parser("snapshot-list"); p.set_defaults(func=cmd_snapshot_list)
    p.add_argument("--root", type=Path, required=True)

    p = sub.add_parser("snapshot-open"); p.set_defaults(func=cmd_snapshot_open)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--session", required=True)

    p = sub.add_parser("inject"); p.set_defaults(func=cmd_inject)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument(
        "--kind",
        required=True,
        choices=["main_corrupt", "buffer_truncate", "depend_cycle", "snapshot_stale"],
    )
    p.add_argument("--page", type=int, default=0)

    p = sub.add_parser("replay"); p.set_defaults(func=cmd_replay)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--tamper-page", type=int, default=-1)

    p = sub.add_parser("keyspace"); p.set_defaults(func=cmd_keyspace)
    p.add_argument("--root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except DiagError as exc:
        _print(exc.to_dict())
        return 2


if __name__ == "__main__":
    sys.exit(main())
