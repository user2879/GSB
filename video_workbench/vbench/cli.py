"""本地确定性文生视频/图生视频渲染工作台命令行入口。"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any, Mapping

from .config import apply_command, dump_json, load_json, validate_config
from .engine import RenderEngine
from .errors import WorkbenchError
from .persistence import SnapshotStore
from .signals import SignalGate
from .verify import assert_snapshots_equal, render_full


def _base_dir(config_path: str) -> str:
    # 样本与工作目录均以项目根（config 的父目录）为基准解析。
    return os.path.dirname(os.path.dirname(os.path.abspath(config_path)))


def _resolve_noise_for_steps(cfg: dict[str, Any], base_dir: str) -> None:
    steps = int(cfg["steps"])
    current = cfg.get("noise_table", "")
    if current.endswith(f"noise_{steps}.npy"):
        return
    candidate = os.path.join(base_dir, "samples", "noise", f"noise_{steps}.npy")
    if os.path.isfile(candidate):
        cfg["noise_table"] = os.path.relpath(candidate, base_dir)


def _load_stage_injections(path: str | None) -> list[Mapping[str, Any]]:
    if not path:
        return []
    payload = load_json(path)
    return list(payload.get("stage_injections", []))


def cmd_run(args: argparse.Namespace) -> int:
    raw = load_json(args.config)
    base_dir = _base_dir(args.config)
    cfg = validate_config(raw, base_dir)
    store = SnapshotStore(args.workdir)
    store.cleanup_orphans()
    injections = _load_stage_injections(args.injections)
    gate = SignalGate(injections) if args.handle_signals else SignalGate(
        injections, install_handlers=False
    )
    engine = RenderEngine(cfg, store, injections=injections, signal_gate=gate)
    report = engine.run(serial=args.serial)
    report["recovery_point"] = store.recovery_point()
    dump_json(report, args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    gate.restore()
    return 0 if report["status"] == "ok" else 2


def cmd_apply_commands(args: argparse.Namespace) -> int:
    base_dir = _base_dir(args.config)
    current = load_json(args.config)
    commands = load_json(args.commands)
    reports: list[dict[str, Any]] = []
    store = SnapshotStore(args.workdir)
    store.cleanup_orphans()

    for command in commands:
        current = apply_command(current, command)
        _resolve_noise_for_steps(current, base_dir)
        cfg = validate_config(current, base_dir)
        engine = RenderEngine(cfg, store)
        report = engine.run()
        reports.append(
            {
                "command_id": command.get("id"),
                "op": command.get("op"),
                "status": report["status"],
                "global_version": report["global_version"],
                "reused": report["reused"],
                "recomputed": report["recomputed"],
                "stats": report["stats"],
            }
        )

    final_config = copy.deepcopy(current)
    full_workdir = os.path.join(args.workdir + "_full")
    full_cfg, full_report = render_full(final_config, base_dir, full_workdir)
    divergence: dict[str, Any]
    try:
        assert_snapshots_equal(args.workdir, full_workdir, full_cfg)
        divergence = {"equal": True}
    except WorkbenchError as error:
        divergence = {
            "equal": False,
            "code": error.code,
            "message": error.message,
            "stage": error.stage,
            "frame": error.frame,
        }

    summary = {
        "status": "ok" if divergence.get("equal") else "E_STATE_DIVERGENCE",
        "commands_applied": len(commands),
        "final_config_hash": fingerprint_config(final_config, base_dir),
        "runs": reports,
        "full_render": {
            "recomputed": full_report["recomputed"],
            "stats": full_report["stats"],
        },
        "consistency": divergence,
    }
    dump_json(summary, args.report)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if divergence.get("equal") else 3


def fingerprint_config(raw: Mapping[str, Any], base_dir: str) -> str:
    cfg = validate_config(raw, base_dir)
    from .config import param_fingerprint_inputs
    from .hashing import fingerprint

    return fingerprint(param_fingerprint_inputs(cfg))


def cmd_recover(args: argparse.Namespace) -> int:
    store = SnapshotStore(args.workdir)
    orphans = store.cleanup_orphans()
    recovery = store.recovery_point()
    recovery["cleaned_orphan_staging"] = orphans
    if not args.no_resume and recovery["recoverable"]:
        raw = load_json(args.config)
        base_dir = _base_dir(args.config)
        cfg = validate_config(raw, base_dir)
        engine = RenderEngine(cfg, store)
        report = engine.run()
        recovery["resume_report"] = {
            "status": report["status"],
            "reused": report["reused"],
            "recomputed": report["recomputed"],
            "failures": report["failures"],
            "stats": report["stats"],
        }
    dump_json(recovery, args.report)
    print(json.dumps(recovery, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def cmd_validate_config(args: argparse.Namespace) -> int:
    base_dir = _base_dir(args.config)
    raw = load_json(args.config)
    injections_path = args.injections
    cases: list[dict[str, Any]] = [
        {"id": "baseline", "raw": copy.deepcopy(raw), "expect": None}
    ]
    if injections_path and os.path.isfile(injections_path):
        payload = load_json(injections_path)
        for case in payload.get("config_injections", []):
            mutated = copy.deepcopy(raw)
            for key in case.get("remove", "").split(","):
                if key:
                    mutated.pop(key, None)
            for key, value in case.get("set", {}).items():
                if key == "guidance_nan":
                    mutated["guidance"] = float("nan")
                else:
                    mutated[key] = value
            cases.append(
                {"id": case["id"], "raw": mutated, "expect": case.get("expect")}
            )

    results = []
    all_pass = True
    for case in cases:
        try:
            validate_config(case["raw"], base_dir)
            actual = None
        except WorkbenchError as error:
            actual = error.code
        passed = actual == case["expect"]
        all_pass = all_pass and passed
        results.append(
            {
                "id": case["id"],
                "expected": case["expect"],
                "actual": actual,
                "passed": passed,
            }
        )
    output = {"status": "ok" if all_pass else "failed", "cases": results}
    dump_json(output, args.report)
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if all_pass else 4


def cmd_concurrency_check(args: argparse.Namespace) -> int:
    import numpy as np

    base_dir = _base_dir(args.config)
    raw = load_json(args.config)
    serial_workdir = args.workdir + "_serial"
    parallel_workdir = args.workdir + "_parallel"
    import shutil

    for path in (serial_workdir, parallel_workdir):
        if os.path.exists(path):
            shutil.rmtree(path)

    cfg = validate_config(raw, base_dir)
    serial_engine = RenderEngine(cfg, SnapshotStore(serial_workdir))
    serial_report = serial_engine.run(serial=True)

    parallel_cfg = copy.deepcopy(cfg)
    parallel_engine = RenderEngine(parallel_cfg, SnapshotStore(parallel_workdir))
    parallel_report = parallel_engine.run(serial=False)

    comparison = assert_snapshots_equal(parallel_workdir, serial_workdir, cfg)
    output = {
        "status": "ok" if comparison["equal"] else "failed",
        "serial_stats": serial_report["stats"],
        "parallel_stats": parallel_report["stats"],
        "comparison": comparison,
        "ordering": {
            "serial_reused": serial_report["reused"],
            "parallel_reused": parallel_report["reused"],
            "serial_recomputed": serial_report["recomputed"],
            "parallel_recomputed": parallel_report["recomputed"],
        },
    }
    dump_json(output, args.report)
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if comparison["equal"] else 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vbench.cli",
        description="本地确定性多阶段文生视频/图生视频渲染工作台",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="执行一次增量渲染")
    run_parser.add_argument("--config", required=True)
    run_parser.add_argument("--workdir", required=True)
    run_parser.add_argument("--report", required=True)
    run_parser.add_argument("--injections")
    run_parser.add_argument("--serial", action="store_true")
    run_parser.add_argument("--no-signal-handlers", dest="handle_signals",
                            action="store_false")
    run_parser.set_defaults(func=cmd_run, handle_signals=True)

    apply_parser = sub.add_parser(
        "apply-commands", help="应用交互命令流并与全量渲染逐帧比对"
    )
    apply_parser.add_argument("--config", required=True)
    apply_parser.add_argument("--commands", required=True)
    apply_parser.add_argument("--workdir", required=True)
    apply_parser.add_argument("--report", required=True)
    apply_parser.set_defaults(func=cmd_apply_commands)

    recover_parser = sub.add_parser("recover", help="输出恢复点并可续跑")
    recover_parser.add_argument("--config", required=True)
    recover_parser.add_argument("--workdir", required=True)
    recover_parser.add_argument("--report", required=True)
    recover_parser.add_argument("--no-resume", action="store_true")
    recover_parser.set_defaults(func=cmd_recover)

    validate_parser = sub.add_parser(
        "validate-config", help="参数与结构注入校验矩阵"
    )
    validate_parser.add_argument("--config", required=True)
    validate_parser.add_argument("--injections", required=True)
    validate_parser.add_argument("--report", required=True)
    validate_parser.set_defaults(func=cmd_validate_config)

    concurrency_parser = sub.add_parser(
        "concurrency-check", help="并发与串行结果逐字段一致性"
    )
    concurrency_parser.add_argument("--config", required=True)
    concurrency_parser.add_argument("--workdir", required=True)
    concurrency_parser.add_argument("--report", required=True)
    concurrency_parser.set_defaults(func=cmd_concurrency_check)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except WorkbenchError as error:
        payload = {"status": "error", "code": error.code, "message": error.message,
                   "stage": error.stage, "step": error.step, "frame": error.frame}
        print(json.dumps(payload, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
