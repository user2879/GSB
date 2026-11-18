"""layerbench 命令行入口（纯标准库 argparse）。

子命令：
  build       执行一次多阶段并行构建并产出报告与快照
  verify      连续构建两次，比对 canonical 报告逐字节一致
  validate    仅做强校验
  snapshots   列出快照
  restore     恢复快照（版本/摘要不一致整体替换）
  replay      恢复后重放校验
  inject      异常注入（corrupt-manifest / cache-mismatch / zero-slots）
  gui         启动 Tkinter 诊断窗口（无显示环境会明确提示）
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

from . import __version__
from .app import Workbench
from .errors import LayerBenchError, ManifestError
from .manifest import load_manifest
from . import replay as replay_mod
from . import snapshot as snapshot_mod


def _print(obj) -> None:
    print(json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2))


def _cmd_build(args) -> int:
    bench = Workbench(args.workdir)
    result = bench.build(args.manifest, crash_at=args.crash_at,
                         source_root=args.source_root)
    summary = {
        "build_id": result["build_id"],
        "build_code": result["report"]["build_code"],
        "canonical_digest": result["canonical_digest"],
        "canonical_path": result["canonical_path"],
        "full_path": result["full_path"],
        "snapshot_path": result["snapshot_path"],
        "peak_global_running":
            result["report"]["budget"]["peak_global_running"],
        "cache_fingerprint": result["cache_fingerprint"],
    }
    _print(summary)
    return 0 if result["report"]["build_code"] == "OK" else 2


def _cmd_verify(args) -> int:
    bench = Workbench(args.workdir)
    first = bench.build(args.manifest, source_root=args.source_root)
    digest_a = first["canonical_digest"]
    second = bench.build(args.manifest, source_root=args.source_root)
    digest_b = second["canonical_digest"]
    _print({
        "first_build": first["build_id"],
        "second_build": second["build_id"],
        "canonical_digest_a": digest_a,
        "canonical_digest_b": digest_b,
        "byte_identical": digest_a == digest_b,
    })
    return 0 if digest_a == digest_b else 3


def _cmd_validate(args) -> int:
    manifest = load_manifest(args.manifest, workspace=args.source_root)
    _print({
        "accepted": True,
        "version": manifest.version,
        "stages": manifest.topo_order,
        "global_slots": manifest.global_slots,
        "instruction_count": len(manifest.all_instructions()),
    })
    return 0



def _cmd_resume(args) -> int:
    bench = Workbench(args.workdir)
    result = bench.resume_build(
        args.manifest, args.build_id, source_root=args.source_root
    )
    _print({
        "build_id": result["build_id"],
        "resumed_unknown": result["resumed_unknown"],
        "build_code": result["report"]["build_code"],
        "canonical_digest": result["canonical_digest"],
        "snapshot_path": result["snapshot_path"],
    })
    return 0 if result["report"]["build_code"] == "OK" else 2


def _cmd_snapshots(args) -> int:
    root = os.path.join(args.workdir, "snapshots")
    items = []
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            meta_path = os.path.join(root, name, "meta.json")
            if os.path.isfile(meta_path):
                with open(meta_path, encoding="utf-8") as handle:
                    meta = json.load(handle)
                items.append({
                    "snapshot_id": meta.get("snapshot_id"),
                    "manifest_version": meta.get("manifest_version"),
                    "completed": len(meta.get("completed_node_uids", [])),
                    "path": os.path.join(root, name),
                })
    _print({"snapshots": items})
    return 0


def _cmd_restore(args) -> int:
    bench = Workbench(args.workdir)
    # 确保工作区有活动清单副本，恢复只替换该副本而非外部原始清单
    active = bench.stage_manifest(args.manifest)
    manifest = load_manifest(active, workspace=args.source_root)
    current_manifest_fp = snapshot_mod.manifest_fingerprint(manifest)
    decision = snapshot_mod.restore_snapshot(
        args.snapshot, args.workdir, active,
        bench.cache_dir, bench.ledger_path,
        current_manifest_fp, bench.current_cache_fingerprint(),
    )
    # 恢复后识别强制终止未提交指令
    from .ledger import Ledger
    from .scheduler import build_dag
    ledger = Ledger(args.workdir)
    try:
        restored_manifest = load_manifest(active, workspace=args.source_root)
        dag_uids = set(build_dag(restored_manifest))
        inflight = snapshot_mod.detect_in_flight(
            ledger, decision["meta"]["build_id"], dag_uids
        )
    finally:
        ledger.close()
    _print({
        "replaced": decision["replaced"],
        "mismatch": decision["mismatch"],
        "archive_dir": decision["archive_dir"],
        "in_flight": inflight,
    })
    return 0


def _cmd_replay(args) -> int:
    result = replay_mod.replay_snapshot(args.snapshot, args.source_root)
    _print(result)
    return 0


def _cmd_inject(args) -> int:
    bench = Workbench(args.workdir)
    if args.kind in ("corrupt-manifest", "zero-slots"):
        inject_dir = os.path.join(args.workdir, "injections")
        os.makedirs(inject_dir, exist_ok=True)
        target = os.path.join(inject_dir, f"_inject_{args.kind}.json")
        if os.path.exists(target):
            os.chmod(target, 0o644)
        shutil.copy2(args.manifest, target)
        outcome = replay_mod.assert_injection_rejected(
            args.kind, target, args.source_root
        )
    elif args.kind == "cache-mismatch":
        from .cache import INDEX_FILENAME
        cache_path = os.path.join(bench.cache_dir, INDEX_FILENAME)
        before_fp = bench.current_cache_fingerprint()
        replay_mod.inject_cache_mismatch(cache_path)
        after_fp = bench.current_cache_fingerprint()
        # 用最近快照的摘要比对，触发整体替换恢复
        latest = _latest_snapshot(bench.snapshot_dir)
        active = bench.stage_manifest(args.manifest)
        decision = snapshot_mod.restore_snapshot(
            latest, args.workdir, active, bench.cache_dir,
            bench.ledger_path,
            snapshot_mod.manifest_fingerprint(
                load_manifest(active, workspace=args.source_root)),
            after_fp,
        )
        if not decision["replaced"] or "cache_index" not in decision["mismatch"]:
            raise LayerBenchError("E_SNAPSHOT", "缓存摘要注入未触发整体替换")
        outcome = {
            "injection": args.kind, "rejected": True, "code": "E_SNAPSHOT",
            "fingerprint_before": before_fp,
            "fingerprint_after_tamper": after_fp,
            "replaced": True,
            "mismatch": decision["mismatch"],
            "archive_dir": decision["archive_dir"],
        }
    else:
        raise LayerBenchError("E_BUDGET", f"未知注入类型 {args.kind}")
    _print(outcome)
    return 0


def _latest_snapshot(root: str) -> str:
    candidates = [
        os.path.join(root, name) for name in sorted(os.listdir(root))
        if os.path.isfile(os.path.join(root, name, "meta.json"))
    ]
    if not candidates:
        raise LayerBenchError("E_SNAPSHOT", "尚无快照可用于注入验证", root)
    return candidates[-1]


def _cmd_gui(args) -> int:
    try:
        from .gui import launch_gui
        launch_gui(args.workdir, args.manifest, args.source_root)
    except Exception as exc:  # 无显示/未带 Tkinter 的环境
        print(f"当前环境无法启动 Tkinter 窗口：{exc}", file=sys.stderr)
        print("Windows 本地 Python 3.11 标准库自带 Tkinter，可直接运行。",
              file=sys.stderr)
        return 4
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="layerbench",
                                     description="本地分层构建与缓存诊断工作台")
    parser.add_argument("--version", action="version",
                        version=f"layerbench {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="执行一次并行构建")
    p_build.add_argument("--manifest", required=True)
    p_build.add_argument("--workdir", required=True)
    p_build.add_argument("--source-root", required=True)
    p_build.add_argument("--crash-at", default=None,
                         help="强制终止注入点，形如 stage/seq")
    p_build.set_defaults(func=_cmd_build)

    p_verify = sub.add_parser("verify", help="两次构建比对 canonical 报告")
    p_verify.add_argument("--manifest", required=True)
    p_verify.add_argument("--workdir", required=True)
    p_verify.add_argument("--source-root", required=True)
    p_verify.set_defaults(func=_cmd_verify)

    p_val = sub.add_parser("validate", help="仅强校验清单")
    p_val.add_argument("--manifest", required=True)
    p_val.add_argument("--workdir", required=True)
    p_val.add_argument("--source-root", required=True)
    p_val.set_defaults(func=_cmd_validate)

    p_resume = sub.add_parser("resume", help="强制终止后续建（重跑 unknown）")
    p_resume.add_argument("--build-id", required=True)
    p_resume.add_argument("--manifest", required=True)
    p_resume.add_argument("--workdir", required=True)
    p_resume.add_argument("--source-root", required=True)
    p_resume.set_defaults(func=_cmd_resume)

    p_snap = sub.add_parser("snapshots", help="列出快照")
    p_snap.add_argument("--workdir", required=True)
    p_snap.set_defaults(func=_cmd_snapshots)

    p_restore = sub.add_parser("restore", help="恢复快照（整体替换）")
    p_restore.add_argument("--snapshot", required=True)
    p_restore.add_argument("--manifest", required=True)
    p_restore.add_argument("--workdir", required=True)
    p_restore.add_argument("--source-root", required=True)
    p_restore.set_defaults(func=_cmd_restore)

    p_replay = sub.add_parser("replay", help="恢复后重放校验")
    p_replay.add_argument("--snapshot", required=True)
    p_replay.add_argument("--workdir", required=True)
    p_replay.add_argument("--source-root", required=True)
    p_replay.set_defaults(func=_cmd_replay)

    p_inject = sub.add_parser("inject", help="异常注入")
    p_inject.add_argument("--kind", required=True,
                          choices=["corrupt-manifest", "cache-mismatch",
                                   "zero-slots"])
    p_inject.add_argument("--manifest", required=True)
    p_inject.add_argument("--workdir", required=True)
    p_inject.add_argument("--source-root", required=True)
    p_inject.set_defaults(func=_cmd_inject)

    p_gui = sub.add_parser("gui", help="启动 Tkinter 诊断窗口")
    p_gui.add_argument("--manifest", required=True)
    p_gui.add_argument("--workdir", required=True)
    p_gui.add_argument("--source-root", required=True)
    p_gui.set_defaults(func=_cmd_gui)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except LayerBenchError as exc:
        print(json.dumps({"error": True, "code": exc.code,
                          "message": str(exc), "details": exc.details},
                         sort_keys=True, ensure_ascii=False, indent=2),
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
