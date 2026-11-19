"""命令行入口。

示例：
python -m cache_invalidation.cli validate --namespaces ... --entries ...
python -m cache_invalidation.cli demo
python -m cache_invalidation.cli inject --fault annotation_cycle ...
python -m cache_invalidation.cli loopback
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from .demo import run_demo
from .engine import Engine
from .envcheck import check_environment
from .errors import BusinessError
from .fault_injection import (
    inject_annotation_cycle,
    inject_missing_inheritance,
    inject_truncated_snapshot,
)
from .loader import load_declarations
from .logging_json import configure_logger, log_business_error
from .model import EntryID
from .netloop import LoopbackClient, LoopbackServer
from .snapshot import decode_snapshot_file
from .store import Store


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cache_invalidation",
        description="本地缓存失效传播中间件")
    parser.add_argument("--log-file", default=None,
                        help="结构化 JSON Lines 日志文件")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("env", help="检查 Python/标准库/回环端口环境")
    sub.add_parser("demo", help="运行三个确定性演示场景")

    validate = sub.add_parser("validate", help="强校验声明文件")
    validate.add_argument("--namespaces", required=True)
    validate.add_argument("--entries", required=True)

    propagate = sub.add_parser("propagate", help="执行一轮失效传播")
    propagate.add_argument("--namespaces", required=True)
    propagate.add_argument("--entries", required=True)
    propagate.add_argument("--db", required=True)
    propagate.add_argument("--reason",
                           choices=("explicit", "expired", "refresh"),
                           required=True)
    propagate.add_argument("--namespace")
    propagate.add_argument("--key", type=int)
    propagate.add_argument("--now-ms", type=int, required=True)
    propagate.add_argument("--snapshot", default=None)
    propagate.add_argument("--format-version", type=int, default=3)

    snapshot = sub.add_parser("snapshot", help="编解码并检查快照")
    snapshot.add_argument("--file", required=True)

    inject = sub.add_parser("inject", help="注入异常（只改输入副本）")
    inject.add_argument(
        "--fault",
        choices=("annotation_cycle", "truncated_snapshot",
                 "missing_inherit"),
        required=True)
    inject.add_argument("--namespaces")
    inject.add_argument("--entries")
    inject.add_argument("--snapshot")

    loopback = sub.add_parser("loopback", help="回环端口端到端自检")
    loopback.add_argument("--namespaces")
    loopback.add_argument("--entries")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logger = configure_logger(log_path=args.log_file)

    try:
        if args.command == "env":
            print(json.dumps(check_environment(), ensure_ascii=False,
                             indent=2, sort_keys=True))
            return 0
        if args.command == "demo":
            run_demo(
                _root() / "artifacts" / "demo",
                _root() / "samples" / "namespaces.json",
                _root() / "samples" / "entries.json")
            return 0
        if args.command == "validate":
            graph = load_declarations(args.namespaces, args.entries)
            print(json.dumps({
                "code": "OK",
                "namespaces": sorted(graph.namespaces),
                "entries": [eid.to_str()
                            for eid in graph.entry_ids_sorted()],
                "edges": [
                    [source.to_str(), target.to_str()]
                    for source in graph.entry_ids_sorted()
                    for target in sorted(graph.annotation_edges[source])
                ],
            }, ensure_ascii=False, indent=2))
            return 0
        if args.command == "propagate":
            return _run_propagate(args)
        if args.command == "snapshot":
            data = decode_snapshot_file(args.file)
            print(json.dumps({
                "code": "OK",
                "format_version": data.format_version,
                "state_version": data.state_version,
                "entries": [eid.to_str()
                            for eid in data.graph.entry_ids_sorted()],
                "rounds": len(data.rounds),
            }, ensure_ascii=False, indent=2))
            return 0
        if args.command == "inject":
            return _run_inject(args, logger)
        if args.command == "loopback":
            return _run_loopback(args)
    except BusinessError as exc:
        log_business_error(logger, exc)
        print(json.dumps(exc.to_dict(), ensure_ascii=False, indent=2,
                         sort_keys=True), file=sys.stderr)
        return 1
    return 0


def _run_propagate(args: argparse.Namespace) -> int:
    graph = load_declarations(args.namespaces, args.entries)
    store = Store(args.db)
    engine = Engine(graph, store=store,
                    format_version=args.format_version)
    if args.reason == "explicit":
        if args.namespace is None or args.key is None:
            raise ValueError("explicit 需要 --namespace 与 --key")
        result = engine.explicit_invalidate(
            EntryID(args.namespace, args.key), args.now_ms)
    elif args.reason == "expired":
        result = engine.reap_expired(args.now_ms)
    else:
        if args.namespace is None:
            raise ValueError("refresh 需要 --namespace")
        result = engine.refresh_namespace(args.namespace, args.now_ms)
    payload = {
        "code": "OK",
        "affected": list(result.affected),
        "round": result.round_record.to_dict()
        if result.round_record else None,
    }
    if args.snapshot:
        engine.save_snapshot(args.snapshot, args.format_version)
        payload["snapshot"] = args.snapshot
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    engine.close()
    return 0


def _run_inject(args: argparse.Namespace, logger) -> int:
    with tempfile.TemporaryDirectory(prefix="cache_fault_") as tmp:
        tmp_path = Path(tmp)
        if args.fault == "truncated_snapshot":
            if not args.snapshot:
                raise ValueError("truncated_snapshot 需要 --snapshot")
            target = inject_truncated_snapshot(args.snapshot, tmp_path)
            decode_snapshot_file(target)
            return 0

        if not args.namespaces or not args.entries:
            raise ValueError(
                "annotation_cycle/missing_inherit 需要声明文件参数")
        if args.fault == "annotation_cycle":
            paths = inject_annotation_cycle(
                args.namespaces, args.entries, tmp_path)
        else:
            paths = inject_missing_inheritance(
                args.namespaces, args.entries, tmp_path)
        load_declarations(paths.namespace_path, paths.entry_path)
        return 0


def _run_loopback(args: argparse.Namespace) -> int:
    namespace_path = (args.namespaces
                      or _root() / "samples" / "namespaces.json")
    entry_path = args.entries or _root() / "samples" / "entries.json"
    graph = load_declarations(namespace_path, entry_path)
    engine = Engine(graph, store=Store(), format_version=3)
    server = LoopbackServer(engine)
    port = server.start()
    try:
        client = LoopbackClient(port)
        ping = client.call({"command": "ping"})
        invalidated = client.call({
            "command": "invalidate",
            "namespace": "catalog",
            "key": 100,
            "now_ms": 1000,
        })
        state = client.call({"command": "state"})
    finally:
        server.stop()
        engine.close()
    print(json.dumps({
        "code": "OK",
        "port": port,
        "ping": ping,
        "invalidate": invalidated,
        "state": state,
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
