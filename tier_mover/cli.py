"""命令行入口。

子命令：
  init       初始化样例工作区（清单 + 确定性对象 + 账本）
  validate   强校验三份清单
  plan       推导并打印确定性迁移计划
  run        按批次预算执行真实搬运并写账本
  status     打印当前账本视图（排除 stale）
  recover    强制终止后恢复（--mode rollback|rebuild）
  replay     恢复后一致性重放校验
  inject     故障注入 capacity|integrity|version
  demo       一键可重复演示全流程
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .applog import StructuredLogger
from .engine import TierEngine
from .errors import TierMoverError
from .faults import (inject_capacity_overflow, inject_truncated_object,
                     inject_version_conflict)
from .ledger import Ledger
from .manifest import load_manifest
from .planner import derive_plan
from .recovery import RecoveryService
from .replay import ReplayService
from .workspace import bootstrap_workspace, workspace_paths

EXIT_OK = 0
EXIT_BUSINESS = 2


def _print(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def _open(workspace: str):
    paths = workspace_paths(workspace)
    logger = StructuredLogger(paths["logs"])
    manifest = load_manifest(paths["tiers"], paths["objects"], paths["seed"])
    engine = TierEngine(workspace, logger=logger)
    ledger = Ledger(paths["ledger"])
    return paths, logger, manifest, engine, ledger


def cmd_init(args) -> int:
    paths = workspace_paths(args.workspace)
    logger = StructuredLogger(paths["logs"])
    engine, ledger, manifest = bootstrap_workspace(
        args.workspace, seed=args.seed, backend=args.backend,
        hot_capacity=args.hot_capacity, logger=logger)
    _print({"event": "initialized", "workspace": str(paths["root"]),
            "spec_version": manifest.spec_version,
            "objects": len(manifest.objects),
            "tiers": [t.tier_id for t in manifest.tiers],
            "ledger": str(paths["ledger"])})
    ledger.close()
    return EXIT_OK


def cmd_validate(args) -> int:
    paths = workspace_paths(args.workspace)
    manifest = load_manifest(paths["tiers"], paths["objects"], paths["seed"])
    _print({"event": "validated", "spec_version": manifest.spec_version,
            "spec_digest": manifest.spec_digest,
            "tier_count": len(manifest.tiers),
            "object_count": len(manifest.objects)})
    return EXIT_OK


def cmd_plan(args) -> int:
    _paths, _logger, manifest, _engine, ledger = _open(args.workspace)
    plan = derive_plan(manifest, ledger.placement())
    _print(json.loads(plan.to_json()))
    ledger.close()
    return EXIT_OK


def cmd_run(args) -> int:
    _paths, logger, manifest, engine, ledger = _open(args.workspace)
    if ledger.active_spec_version() == 0 or ledger.object_count() == 0:
        ledger.register_spec(manifest, _now())
        engine.prepare_dirs(manifest)
        engine.seed_objects(manifest)
        ledger.initialize_objects(manifest, _now())
    plan = engine.plan(ledger, manifest)
    conclusion = engine.execute(
        ledger, manifest, plan=plan, crash_after_key=args.crash_after_key)
    engine.reconcile_bytes(manifest, ledger)
    _print(conclusion)
    ledger.close()
    return EXIT_OK


def cmd_status(args) -> int:
    _paths, _logger, _manifest, _engine, ledger = _open(args.workspace)
    view = ledger.current_view()
    view["superseded_plans"] = ledger.superseded_plan_fragments()
    view["active_conclusion"] = ledger.active_conclusion()
    view["chain_digest"] = ledger.chain_digest()
    _print(view)
    ledger.close()
    return EXIT_OK


def cmd_recover(args) -> int:
    _paths, logger, manifest, engine, ledger = _open(args.workspace)
    service = RecoveryService(engine, ledger, manifest, logger)
    if args.inspect:
        report = service.inspect()
    else:
        report = service.recover(mode=args.mode)
    _print(report.to_dict())
    ledger.close()
    return EXIT_OK


def cmd_replay(args) -> int:
    _paths, _logger, manifest, _engine, ledger = _open(args.workspace)
    result = ReplayService(ledger, manifest).run()
    _print(result.to_dict())
    ledger.close()
    return EXIT_OK


def cmd_inject(args) -> int:
    paths = workspace_paths(args.workspace)
    logger = StructuredLogger(paths["logs"])
    if args.fault == "capacity":
        path = inject_capacity_overflow(args.workspace, logger)
        _paths, _l, manifest, engine, ledger = _open(args.workspace)
        plan = engine.plan(ledger, manifest)
        engine.execute(ledger, manifest, plan=plan)
    elif args.fault == "integrity":
        path = inject_truncated_object(args.workspace, logger, args.object_key)
        _paths, _l, manifest, engine, ledger = _open(args.workspace)
        engine.reconcile_bytes(manifest, ledger)
    else:
        path = inject_version_conflict(args.workspace, logger)
        Ledger(path)  # 打开即抛 E_VERSION
    return EXIT_OK


def cmd_demo(args) -> int:
    from .demo import run_demo
    result = run_demo(Path(args.workspace) / "demo_run", seed=args.seed)
    _print(result)
    return EXIT_OK


def _now() -> int:
    import time
    return int(time.time() * 1000)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tier-mover", description="冷热分层与迁移组件")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="初始化样例工作区")
    p_init.add_argument("workspace")
    p_init.add_argument("--seed", type=int, default=20261122)
    p_init.add_argument("--backend", default="python",
                        choices=["python", "numpy"])
    p_init.add_argument("--hot-capacity", type=int, default=None)
    p_init.set_defaults(func=cmd_init)

    p_val = sub.add_parser("validate", help="强校验清单")
    p_val.add_argument("workspace")
    p_val.set_defaults(func=cmd_validate)

    p_plan = sub.add_parser("plan", help="推导迁移计划")
    p_plan.add_argument("workspace")
    p_plan.set_defaults(func=cmd_plan)

    p_run = sub.add_parser("run", help="执行迁移")
    p_run.add_argument("workspace")
    p_run.add_argument("--crash-after-key", default=None)
    p_run.set_defaults(func=cmd_run)

    p_status = sub.add_parser("status", help="查看账本视图")
    p_status.add_argument("workspace")
    p_status.set_defaults(func=cmd_status)

    p_rec = sub.add_parser("recover", help="崩溃恢复")
    p_rec.add_argument("workspace")
    p_rec.add_argument("--mode", choices=["rollback", "rebuild"],
                       default="rollback")
    p_rec.add_argument("--inspect", action="store_true",
                       help="只标注未知对象，不做处置")
    p_rec.set_defaults(func=cmd_recover)

    p_rep = sub.add_parser("replay", help="重放校验")
    p_rep.add_argument("workspace")
    p_rep.set_defaults(func=cmd_replay)

    p_inj = sub.add_parser("inject", help="故障注入")
    p_inj.add_argument("workspace")
    p_inj.add_argument("fault", choices=["capacity", "integrity", "version"])
    p_inj.add_argument("--object-key", default=None)
    p_inj.set_defaults(func=cmd_inject)

    p_demo = sub.add_parser("demo", help="一键全流程演示")
    p_demo.add_argument("workspace")
    p_demo.add_argument("--seed", type=int, default=20261122)
    p_demo.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except TierMoverError as exc:
        payload = exc.to_dict()
        print(json.dumps(payload, ensure_ascii=False, indent=2,
                         sort_keys=True), file=sys.stderr)
        return EXIT_BUSINESS
    except SystemExit as exc:
        # 模拟强制终止：确保信息落日志后以非零码退出
        print(json.dumps({"code": "E_CRASH", "message": str(exc)},
                         ensure_ascii=False), file=sys.stderr)
        return 137


if __name__ == "__main__":
    raise SystemExit(main())
