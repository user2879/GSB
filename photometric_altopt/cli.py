"""命令行入口（Windows / POSIX 通用，仅标准库 + NumPy）。

示例见 ``examples/run_example.bat`` 与 ``examples/run_example.sh``。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .differentiable import SurfaceParams, finite_difference_check
from .faults import FaultConfig
from .forward import build_view_table
from .generator import MANIFEST_FILENAME, generate_observations, load_observations
from .persistence import directory_manifest, read_json
from .photometric import solve_photometric
from .pipeline import resume_run, run_alternating
from .verification import compare_with_truth


def _default_config() -> PipelineConfig:
    return PipelineConfig()


def _load_inputs(obs_dir: Path, config: PipelineConfig, fault: FaultConfig | None = None):
    fault = fault or FaultConfig()
    data = load_observations(obs_dir)
    obs_cfg = config.observation
    if fault.zero_light:
        from .faults import inject_zero_light

        obs_cfg = PipelineConfig.from_dict(config.to_dict()).observation
        obs_cfg.light_directions = inject_zero_light(obs_cfg.light_directions, fault)
    views = build_view_table(obs_cfg, skip_views=set(fault.skipped_views))
    return data["observations"], views, obs_cfg


def cmd_generate(args: argparse.Namespace) -> int:
    config = _default_config()
    if args.size:
        config.observation.size = args.size
    if args.seed is not None:
        config.observation.seed = args.seed
    out = generate_observations(config.observation, Path(args.workspace) / "observations")
    manifest = read_json(out / MANIFEST_FILENAME)
    print(f"[generate] 观测集目录: {out}")
    print(f"[generate] 视点数: {manifest['config']['num_views']}  观测哈希: {manifest['observation_array_hash']}")
    return 0


def _fault_from_args(args: argparse.Namespace) -> FaultConfig:
    fault = FaultConfig()
    if getattr(args, "inject_nonfinite", False):
        fault.nonfinite_albedo = True
        fault.nonfinite_pixels = [[10, 20], [40, 5]]
    if getattr(args, "inject_zero_light", False):
        fault.zero_light = True
    if getattr(args, "inject_degenerate", False):
        fault.degenerate_views = True
        fault.degenerate_skip_views = list(range(9))  # 12 视点中跳过 9 个
    if getattr(args, "inject_ill", False):
        fault.ill_conditioned = True
    if getattr(args, "skip_view", None) is not None:
        fault.skipped_views = [int(x) for x in args.skip_view.split(",") if x != ""]
    return fault


def cmd_run(args: argparse.Namespace) -> int:
    try:
        config = _default_config()
        if args.max_iterations:
            config.opt.max_iterations = args.max_iterations
        fault = _fault_from_args(args)
        workspace = Path(args.workspace)
        obs_dir = _resolve_obs_dir(workspace, args.obs_dir)
        observations, views, obs_cfg = _load_inputs(obs_dir, config, fault)
        config.observation = obs_cfg

        valid_mask = None
        if fault.skipped_views:
            # 观测与视点表按同一全局视点集合对齐：跳过视点从两者中同步移除
            keep = [i for i in range(observations.shape[0]) if i not in set(fault.skipped_views)]
            observations = observations[keep]
        if fault.nonfinite_albedo:
            # 参数层注入：污染反照率副本，首轮守卫即分类为 E_NONFINITE_ALBEDO
            from .faults import inject_nonfinite_albedo

            seed_albedo = inject_nonfinite_albedo(
                np.zeros(observations.shape[1:], dtype=np.float64), fault
            )
        else:
            seed_albedo = None
        if fault.degenerate_views:
            from .faults import build_valid_mask

            valid_mask = build_valid_mask(
                views.size,
                observations.shape[1],
                observations.shape[2],
                fault.degenerate_skip_views,
            )
        if fault.ill_conditioned:
            config.solve.condition_threshold = fault.ill_conditioned_threshold

        run_root = workspace / "trajectories" / args.run_name
        summary = run_alternating(
            config,
            observations,
            views,
            run_root,
            fault=fault,
            crash_after_round=args.crash_after,
            valid_mask=valid_mask,
            seed_albedo=seed_albedo,
        )
    except Exception as exc:  # 业务码分类打印
        return _print_error(exc)

    _print_summary(summary, run_root)
    if args.check_truth:
        from .persistence import load_npy
        from .pipeline import RunPaths, _round_paths

        last = summary["rounds_committed"] - 1
        rp = _round_paths(RunPaths(run_root), last)
        metrics = compare_with_truth(load_npy(rp["normal"]), load_npy(rp["albedo"]), obs_dir)
        print("[truth-check]", metrics)
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    config = _default_config()
    if args.max_iterations:
        config.opt.max_iterations = args.max_iterations
    workspace = Path(args.workspace)
    obs_dir = _resolve_obs_dir(workspace, args.obs_dir)
    observations, views, _ = _load_inputs(obs_dir, config)
    run_root = workspace / "trajectories" / args.run_name
    try:
        summary = resume_run(config, observations, views, run_root)
    except Exception as exc:
        return _print_error(exc)
    _print_summary(summary, run_root)
    return 0


def cmd_gradcheck(args: argparse.Namespace) -> int:
    config = _default_config()
    workspace = Path(args.workspace)
    obs_dir = _resolve_obs_dir(workspace, args.obs_dir)
    observations, views, _ = _load_inputs(obs_dir, config)
    ps = solve_photometric(observations, views, config.observation, config.solve)
    params = SurfaceParams(normals=ps.normals, albedo=ps.albedo)
    report = finite_difference_check(
        params, observations, views, config.observation, config.opt, sample_count=args.samples
    )
    ok = report["max_abs_error"] <= config.opt.gradcheck_tol
    print(f"[gradcheck] {report}  容差={config.opt.gradcheck_tol}  结论={'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


def cmd_manifest(args: argparse.Namespace) -> int:
    target = Path(args.path)
    if target.is_dir():
        manifest = directory_manifest(target)
    else:
        manifest = read_json(target)
    print(target)
    for key, value in manifest.items():
        print(f"  {key}: {value}")
    return 0


def _resolve_obs_dir(workspace: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    root = workspace / "observations"
    candidates = sorted(p for p in root.glob("obs_*") if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"未找到观测集，请先 generate: {root}")
    return candidates[-1]


def _print_summary(summary: dict, run_root: Path) -> None:
    print(f"[run] 工作目录: {run_root}")
    print(f"[run] 已提交轮次: {summary['rounds_committed']}  收敛原因: {summary['convergence_reason']}")
    if summary.get("crashed"):
        print("[run] 已模拟强制终止，等待 resume")
    for item in summary["trajectory"]:
        print(
            f"  round={item['round']:02d} pre={item['pre_dr_loss']:.10f} "
            f"post={item['post_dr_loss']:.10f} |g_n|={item['grad_normal_norm']:.6e} "
            f"|g_r|={item['grad_albedo_norm']:.6e} 冲突病态像素={item['ill_conditioned_pixels']}"
        )


def _print_error(exc: Exception) -> int:
    code = getattr(exc, "code", type(exc).__name__)
    details = getattr(exc, "details", {})
    layer = getattr(exc, "layer", "unknown")
    print(f"[error] 业务码={code} 层级={layer} 说明={exc}", file=sys.stderr)
    if details:
        print(f"[error] 诊断={details}", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="altopt", description="光度立体/可微渲染交替优化组件")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="生成确定性观测集")
    p_gen.add_argument("--workspace", default="workspace")
    p_gen.add_argument("--size", type=int, default=256)
    p_gen.add_argument("--seed", type=int, default=None)
    p_gen.set_defaults(func=cmd_generate)

    p_run = sub.add_parser("run", help="执行交替优化（可带异常注入/崩溃模拟）")
    p_run.add_argument("--workspace", default="workspace")
    p_run.add_argument("--obs-dir", default=None)
    p_run.add_argument("--run-name", default="baseline")
    p_run.add_argument("--max-iterations", type=int, default=None)
    p_run.add_argument("--crash-after", type=int, default=None)
    p_run.add_argument("--check-truth", action="store_true")
    p_run.add_argument("--inject-nonfinite", action="store_true")
    p_run.add_argument("--inject-zero-light", action="store_true")
    p_run.add_argument("--inject-degenerate", action="store_true")
    p_run.add_argument("--inject-ill", action="store_true")
    p_run.add_argument("--skip-view", default=None, help="逗号分隔的视点编号，如 3,7")
    p_run.set_defaults(func=cmd_run)

    p_res = sub.add_parser("resume", help="从最近已提交状态恢复并续写")
    p_res.add_argument("--workspace", default="workspace")
    p_res.add_argument("--obs-dir", default=None)
    p_res.add_argument("--run-name", default="baseline")
    p_res.add_argument("--max-iterations", type=int, default=None)
    p_res.set_defaults(func=cmd_resume)

    p_grad = sub.add_parser("gradcheck", help="解析梯度有限差分核对")
    p_grad.add_argument("--workspace", default="workspace")
    p_grad.add_argument("--obs-dir", default=None)
    p_grad.add_argument("--samples", type=int, default=32)
    p_grad.set_defaults(func=cmd_gradcheck)

    p_man = sub.add_parser("manifest", help="打印目录内容哈希清单或 JSON 清单")
    p_man.add_argument("path")
    p_man.set_defaults(func=cmd_manifest)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
