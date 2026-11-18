"""命令行演示入口。

典型流程
--------
    python -m ccdplatform.cli clean
    python -m ccdplatform.cli run --exponent 3 --concurrency 2
    python -m ccdplatform.cli restore
    python -m ccdplatform.cli replay
    python -m ccdplatform.cli inject --kind range|cycle|grid
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from .errors import PlatformError
from .injection import inject_cycle, inject_range, strip_grid_from_snapshot
from .snapshot import SnapshotStore

WORKSPACE = Path(__file__).resolve().parent.parent
SAMPLES = WORKSPACE / "samples"
RUNTIME = WORKSPACE / "runtime"
DEFAULT_MANIFEST = SAMPLES / "manifest.json"
DEFAULT_NODES = SAMPLES / "nodes.json"
DEFAULT_DB = RUNTIME / "session.sqlite"
DEFAULT_LOG = RUNTIME / "events.jsonl"


def _emit_error(exc: PlatformError, logger_store: SnapshotStore | None = None,
                version: int | None = None) -> int:
    payload = exc.to_dict()
    print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
    if logger_store is not None:
        try:
            logger_store.append_error(exc.code, payload, version)
        except Exception:
            pass
    return 1


def cmd_clean(_args: argparse.Namespace) -> int:
    if RUNTIME.exists():
        shutil.rmtree(RUNTIME)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    print(f"已清空运行时目录: {RUNTIME}")
    return 0


def _build_engine(args: argparse.Namespace):
    from .engine import SessionEngine

    return SessionEngine(
        workdir=WORKSPACE,
        manifest_path=args.manifest,
        nodes_path=args.nodes,
        db_path=args.db,
        log_path=args.log,
        exponent=args.exponent,
        int_bits=args.int_bits,
        max_concurrency=args.concurrency,
    )


def cmd_run(args: argparse.Namespace) -> int:
    engine = None
    try:
        engine = _build_engine(args)
        summaries = engine.process_versions(up_to=args.up_to)
        print(json.dumps({"ok": True, "summaries": summaries}, ensure_ascii=False, indent=2))
        return 0
    except PlatformError as exc:
        store = engine.store if engine else None
        return _emit_error(exc, store)
    finally:
        if engine:
            engine.close()


def cmd_restore(args: argparse.Namespace) -> int:
    from .engine import SessionEngine

    engine = None
    try:
        engine = _build_engine(args)
        info = engine.restore_latest()
        if not info.get("restored"):
            print(json.dumps(info, ensure_ascii=False, indent=2))
            return 0
        snapshot = info.pop("snapshot")
        info["keys"] = sorted(snapshot.compiled)
        info["error_bounds"] = {
            k: snapshot.compiled[k].error_bound for k in sorted(snapshot.compiled)
        }
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0
    except PlatformError as exc:
        return _emit_error(exc, engine.store if engine else None)
    finally:
        if engine:
            engine.close()


def cmd_replay(args: argparse.Namespace) -> int:
    from .engine import SessionEngine

    engine = None
    try:
        engine = _build_engine(args)
        engine.restore_latest()
        result = engine.replay()
        print(json.dumps({"ok": True, "replay": result}, ensure_ascii=False, indent=2))
        return 0
    except PlatformError as exc:
        return _emit_error(exc, engine.store if engine else None)
    finally:
        if engine:
            engine.close()


def cmd_inject(args: argparse.Namespace) -> int:
    """异常注入：仅改输入文件或参数，随后立即运行以触发对应业务码。"""
    manifest = Path(args.manifest)
    if args.kind == "range":
        info = inject_range(manifest, args.target_key, args.exponent, args.int_bits)
    elif args.kind == "cycle":
        info = inject_cycle(manifest, args.cycle_a, args.cycle_b)
    elif args.kind == "grid":
        if not Path(args.db).exists():
            print("grid 注入需要先存在快照库，请先执行 run", file=sys.stderr)
            return 2
        info = strip_grid_from_snapshot(args.db)
    else:  # pragma: no cover
        raise SystemExit(2)
    print(json.dumps({"injected": info}, ensure_ascii=False, indent=2))

    if args.kind == "grid":
        # 直接以被破坏的快照打开，触发 E_GRID
        store = SnapshotStore(args.db)
        try:
            store.load_grid()
        except PlatformError as exc:
            return _emit_error(exc, store)
        finally:
            store.close()
        return 0

    engine = None
    try:
        engine = _build_engine(args)
        engine.process_versions()
    except PlatformError as exc:
        expected = {"range": "E_RANGE", "cycle": "E_CYCLE"}[args.kind]
        store = engine.store if engine else None
        rc = _emit_error(exc, store)
        if exc.code != expected:
            print(f"预期 {expected} 实际 {exc.code}", file=sys.stderr)
            return 3
        return rc
    finally:
        if engine:
            engine.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ccdplatform",
        description="跨组件配置分发与一致性审计平台",
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--nodes", default=str(DEFAULT_NODES))
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--log", default=str(DEFAULT_LOG))
    parser.add_argument("--exponent", type=int, default=3)
    parser.add_argument("--int-bits", type=int, default=63, dest="int_bits")
    parser.add_argument("--concurrency", type=int, default=2)

    sub = parser.add_subparsers(dest="command", required=True)

    p_clean = sub.add_parser("clean", help="清空 SQLite 会话库与运行时快照")
    p_clean.set_defaults(func=cmd_clean)

    p_run = sub.add_parser("run", help="编译并分发全部清单版本")
    p_run.add_argument("--up-to", type=int, default=None)
    p_run.set_defaults(func=cmd_run)

    p_restore = sub.add_parser("restore", help="打开最新快照并整体替换恢复")
    p_restore.set_defaults(func=cmd_restore)

    p_replay = sub.add_parser("replay", help="恢复后重放校验")
    p_replay.set_defaults(func=cmd_replay)

    p_inject = sub.add_parser("inject", help="异常注入（range/cycle/grid）")
    p_inject.add_argument("--kind", choices=["range", "cycle", "grid"], required=True)
    p_inject.add_argument("--target-key", default="alpha")
    p_inject.add_argument("--cycle-a", default="alpha")
    p_inject.add_argument("--cycle-b", default="beta")
    p_inject.set_defaults(func=cmd_inject)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
