"""命令行入口：generate / run / edit / inject。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, List, Optional

from .config import RebuildConfig
from .editing import apply_edit_sequence
from .generate_sample import generate
from .injection import apply_injections, load_injection_config
from .loader import load_sample
from .runner import GracefulShutdown, ReconstructionRunner
from .utils import atomic_write_json, read_json
from .visualization import render_all


def _attach_injection_markers(sample_dir: str, load_result: Any) -> None:
    for view_data in load_result.views:
        camera_path = os.path.join(
            sample_dir, f"view_{view_data.view:02d}.camera.json"
        )
        if not os.path.exists(camera_path):
            continue
        raw = read_json(camera_path)
        if "injection" in raw:
            setattr(
                view_data.camera, "injection_marker", raw["injection"]
            )


def cmd_generate(args: argparse.Namespace) -> int:
    generate(args.sample_dir)
    if args.config:
        atomic_write_json(args.config, RebuildConfig().to_dict())
    print(f"样本已生成: {args.sample_dir}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    config = RebuildConfig.load(args.config)
    load_result = load_sample(args.sample_dir)
    _attach_injection_markers(args.sample_dir, load_result)

    injection_hits: List[Any] = []
    if args.injection:
        items = load_injection_config(args.injection)
        injection_hits = apply_injections(
            args.sample_dir, args.injected_sample, items
        )
        # 重新加载被注入的工作副本
        load_result = load_sample(args.injected_sample)
        _attach_injection_markers(args.injected_sample, load_result)
        for item in items:
            if item["type"] == "zero_voxel_size":
                object.__setattr__(config, "voxel_size", 0.0)

    runner = ReconstructionRunner(
        load_result, config, args.work_dir,
        injection_hits=injection_hits, concurrency=args.threads,
    )
    shutdown = GracefulShutdown()
    if args.handle_signals:
        shutdown.install()
    report = runner.execute(
        shutdown=shutdown, report_path=args.report,
    )
    if args.viz_dir and "float32" in runner.last_results:
        render_all(
            load_result.views, runner.last_results, args.viz_dir
        )
    print(json.dumps(report.report, ensure_ascii=False, indent=2))
    return report.exit_code


def cmd_edit(args: argparse.Namespace) -> int:
    commands = read_json(args.commands)
    config_path = args.config if os.path.exists(args.config) else None
    outcomes = apply_edit_sequence(
        args.sample_dir, args.edited_sample, commands,
        config_path=config_path,
    )
    payload = [
        {"command": o.command, "target": o.target,
         "changed_fingerprints": o.changed_fingerprints,
         "expected_invalidation": o.expected_invalidation}
        for o in outcomes
    ]
    atomic_write_json(args.plan_out, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recon-workbench",
        description="多视图三维几何重建可视化与诊断工作台",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="生成确定性多视图样本")
    gen.add_argument("sample_dir")
    gen.add_argument("--config", default=None)
    gen.set_defaults(func=cmd_generate)

    run = sub.add_parser("run", help="执行重建与诊断")
    run.add_argument("--sample-dir", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--work-dir", required=True)
    run.add_argument("--report", required=True)
    run.add_argument("--injection", default=None)
    run.add_argument("--injected-sample", default=None)
    run.add_argument("--viz-dir", default=None)
    run.add_argument("--threads", type=int, default=None)
    run.add_argument("--handle-signals", action="store_true")
    run.set_defaults(func=cmd_run)

    edit = sub.add_parser("edit", help="应用编辑命令序列")
    edit.add_argument("--sample-dir", required=True)
    edit.add_argument("--edited-sample", required=True)
    edit.add_argument("--commands", required=True)
    edit.add_argument("--config", default="")
    edit.add_argument("--plan-out", required=True)
    edit.set_defaults(func=cmd_edit)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
