"""命令行入口：运行轨迹、交互查询、权威校验、快照恢复与异常注入。

示例：
  python -m kvstore.cli run --db results/demo.db --trace data/sample_trace.jsonl \
      --initial data/initial_records.json --snapshot results/snap.json
  python -m kvstore.cli query --db results/demo.db --prefix user/1 --clock 1200
  python -m kvstore.cli verify --db results/demo.db --trace data/sample_trace.jsonl \
      --initial data/initial_records.json --clock 1200
  python -m kvstore.cli inject --kind trace_corrupt --trace data/sample_trace.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from . import inject as inject_mod
from .engine import Engine
from .errors import KvError
from .replay import verify_replay_after_restore, verify_visible
from .session import Session
from .snapshot import open_snapshot
from .storage import Storage
from .structlog import StructuredLogger
from .trace import load_trace, operations_digest


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _print(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))


def cmd_run(args: argparse.Namespace) -> int:
    logger = StructuredLogger(args.log)
    operations = load_trace(args.trace)
    initial = _load_json(args.initial) if args.initial else []
    session = Session(args.db, logger, initial)
    try:
        conclusions = session.run(operations)
        consistency = session.engine.verify_index_consistency()
        snapshot_info = None
        if args.snapshot:
            snapshot_info = session.snapshot(args.snapshot)
        result = {
            "trace_digest": operations_digest(operations),
            "ops_executed": len(conclusions),
            "keyspace_version": session.engine.keyspace_version,
            "consistency": consistency,
            "snapshot": snapshot_info,
            "last_conclusions": conclusions[-3:],
        }
        _print(result)
        return 0
    finally:
        session.close()


def cmd_query(args: argparse.Namespace) -> int:
    storage = Storage(args.db)
    try:
        engine = Engine(storage)
        _print(engine.query_prefix(args.prefix, args.clock))
        return 0
    finally:
        storage.close()


def cmd_get(args: argparse.Namespace) -> int:
    storage = Storage(args.db)
    try:
        engine = Engine(storage)
        _print(engine.get(args.key, args.clock))
        return 0
    finally:
        storage.close()


def cmd_verify(args: argparse.Namespace) -> int:
    storage = Storage(args.db)
    logger = StructuredLogger(args.log)
    try:
        engine = Engine(storage)
        operations = load_trace(args.trace)
        initial = _load_json(args.initial) if args.initial else []
        prefixes = args.prefixes.split(",") if args.prefixes else _all_prefixes(engine)
        replay_db = args.db + ".replay"
        _remove(replay_db)
        visible = verify_visible(
            engine, operations, initial, replay_db,
            clock=args.clock, prefixes=prefixes,
        )
        consistency = engine.verify_index_consistency(clock=args.clock)
        _print({"visible_check": visible, "index_consistency": consistency})
        _remove(replay_db)
        return 0
    finally:
        storage.close()


def cmd_restore(args: argparse.Namespace) -> int:
    logger = StructuredLogger(args.log)
    storage = Storage(args.db)
    try:
        engine = Engine(storage)
        opened = open_snapshot(
            args.snapshot, engine,
            readonly_copy_dir=args.readonly_dir,
            mismatch_digest=args.force_digest_mismatch,
        )
        operations = load_trace(args.trace)
        initial = _load_json(args.initial) if args.initial else []
        replay_db = args.db + ".restore_replay"
        _remove(replay_db)
        replay_result = verify_replay_after_restore(
            engine, operations, initial, opened["conclusions"], replay_db
        )
        _remove(replay_db)
        _print({"open": {k: v for k, v in opened.items() if k != "conclusions"},
                "replay_check": replay_result})
        return 0
    finally:
        storage.close()


def cmd_inject(args: argparse.Namespace) -> int:
    os.makedirs(args.out_dir, exist_ok=True)
    if args.kind in ("trace_corrupt", "trace_bad_json"):
        path = inject_mod.INJECTORS[args.kind](args.trace, args.out_dir)
    else:
        path = inject_mod.INJECTORS[args.kind](args.snapshot, args.out_dir)
    _print({"injected": args.kind, "path": path})
    return 0


def cmd_trace_digest(args: argparse.Namespace) -> int:
    _print(operations_digest(load_trace(args.trace)))
    return 0


def _all_prefixes(engine: Engine) -> list[str]:
    return sorted(engine.storage.all_counts().keys())


def _remove(path: str) -> None:
    for suffix in ("", "-wal", "-shm"):
        target = path + suffix
        if os.path.exists(target):
            os.remove(target)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kvstore", description="嵌入式键值索引与过期组件")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="加载初始记录并顺序执行轨迹")
    run.add_argument("--db", required=True)
    run.add_argument("--trace", required=True)
    run.add_argument("--initial")
    run.add_argument("--snapshot")
    run.add_argument("--log", default="results/events.jsonl")
    run.set_defaults(func=cmd_run)

    query = sub.add_parser("query", help="按前缀快速查询键集合与计数")
    query.add_argument("--db", required=True)
    query.add_argument("--prefix", default="")
    query.add_argument("--clock", type=int, required=True)
    query.set_defaults(func=cmd_query)

    get = sub.add_parser("get", help="按键 GET 并输出过期判定依据")
    get.add_argument("--db", required=True)
    get.add_argument("--key", required=True)
    get.add_argument("--clock", type=int, required=True)
    get.set_defaults(func=cmd_get)

    verify = sub.add_parser("verify", help="快速结论 vs 权威重放一致性校验")
    verify.add_argument("--db", required=True)
    verify.add_argument("--trace", required=True)
    verify.add_argument("--initial")
    verify.add_argument("--clock", type=int, required=True)
    verify.add_argument("--prefixes", default="")
    verify.add_argument("--log", default="results/events.jsonl")
    verify.set_defaults(func=cmd_verify)

    restore = sub.add_parser("restore", help="打开快照整体恢复并重放校验")
    restore.add_argument("--db", required=True)
    restore.add_argument("--snapshot", required=True)
    restore.add_argument("--trace", required=True)
    restore.add_argument("--initial")
    restore.add_argument("--readonly-dir", default="results/readonly")
    restore.add_argument("--log", default="results/events.jsonl")
    restore.add_argument("--force-digest-mismatch", default=None)
    restore.set_defaults(func=cmd_restore)

    inject = sub.add_parser("inject", help="生成被注入异常的输入文件（不改源码）")
    inject.add_argument("--kind", required=True,
                        choices=list(inject_mod.INJECTORS.keys()))
    inject.add_argument("--trace")
    inject.add_argument("--snapshot")
    inject.add_argument("--out-dir", default="results/injected")
    inject.set_defaults(func=cmd_inject)

    digest = sub.add_parser("digest", help="输出轨迹确定性摘要")
    digest.add_argument("--trace", required=True)
    digest.set_defaults(func=cmd_trace_digest)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KvError as exc:
        payload = exc.to_dict()
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
              file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(json.dumps({"code": "E_IO", "message": str(exc)},
                         ensure_ascii=False), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
