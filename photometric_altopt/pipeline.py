"""交替优化管线（业务要求 4、6、8）。

每轮固定顺序：
1. 光度立体（PS）独立求解一轮，得到 PS 法向/反照率；
2. 阶段间语义对照，输出冲突清单（不靠整体损失判定）；
3. 可微渲染（DR）独立解析梯度 + 增量参数更新；
4. 瓦片增量重渲染与全量重渲染核对，分歧即 ``E_RENDER_DIVERGE``；
5. 轨迹、冲突、参数版本戳全部原子提交；
6. 满足损失下降阈值或迭代上限时受控收敛。

强制终止恢复：只承认带 ``.committed`` 标记的最近一轮快照，
未完成写入的临时文件一律不构成状态，重启后从最近已提交轮次继续，
由于全程无随机数，续写轨迹与未中断运行逐字节一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import ObservationConfig, PipelineConfig, SolveConfig
from .conflict import ConflictReport, compare_stages
from .differentiable import (
    SurfaceParams,
    apply_gradient_step,
    compute_gradients,
    normalize_normals,
    tile_ranges,
    verify_incremental_vs_full,
)
from .faults import FaultConfig, build_valid_mask, validate_skipped_views
from .forward import ViewTable
from .persistence import (
    atomic_commit,
    atomic_save_npy,
    atomic_write_json,
    cleanup_temp_files,
    is_committed,
    load_npy,
    read_json,
    sha256_array,
)
from .photometric import photometric_blend, solve_photometric

CONVERGED_DELTA = "converged_loss_delta"
CONVERGED_MAX_ITER = "reached_max_iterations"
CRASH_REASON = "forced_termination_simulated"


@dataclass
class RunPaths:
    root: Path

    @property
    def traj_dir(self) -> Path:
        return self.root / "trajectories"

    @property
    def conflict_dir(self) -> Path:
        return self.root / "conflicts"

    @property
    def snapshot_dir(self) -> Path:
        return self.root / "snapshots"

    @property
    def run_manifest(self) -> Path:
        return self.root / "run_manifest.json"

    @property
    def status_file(self) -> Path:
        return self.root / "status.json"


def _round_paths(paths: RunPaths, round_id: int) -> dict[str, Path]:
    tag = f"round_{round_id:03d}"
    return {
        "record": paths.traj_dir / f"{tag}.json",
        "conflict": paths.conflict_dir / f"{tag}.json",
        "normal": paths.snapshot_dir / f"{tag}_normals.npy",
        "albedo": paths.snapshot_dir / f"{tag}_albedo.npy",
        "ps_normal": paths.snapshot_dir / f"{tag}_ps_normals.npy",
        "ps_albedo": paths.snapshot_dir / f"{tag}_ps_albedo.npy",
        "images": paths.snapshot_dir / f"{tag}_images.npy",
    }


def _write_round_snapshot(
    paths: RunPaths,
    round_id: int,
    params: SurfaceParams,
    ps_normals: np.ndarray,
    ps_albedo: np.ndarray,
    images: np.ndarray,
) -> dict[str, str]:
    rp = _round_paths(paths, round_id)
    hashes: dict[str, str] = {}
    for key, array in {
        "normal": params.normals,
        "albedo": params.albedo,
        "ps_normal": ps_normals,
        "ps_albedo": ps_albedo,
        "images": images,
    }.items():
        hashes[key] = atomic_save_npy(rp[key], np.ascontiguousarray(array))
        atomic_commit(rp[key], {"sha256": hashes[key], "round": round_id})
    return hashes


def _load_round_snapshot(paths: RunPaths, round_id: int) -> tuple[SurfaceParams, dict[str, str]]:
    rp = _round_paths(paths, round_id)
    for key in ("normal", "albedo"):
        if not is_committed(rp[key]):
            raise FileNotFoundError(f"轮次 {round_id} 快照未完整提交: {rp[key]}")
    params = SurfaceParams(normals=load_npy(rp["normal"]), albedo=load_npy(rp["albedo"]))
    hashes = {
        "normal": sha256_array(params.normals),
        "albedo": sha256_array(params.albedo),
    }
    return params, hashes


def _initial_params(
    observations: np.ndarray,
    views: ViewTable,
    obs_cfg: ObservationConfig,
    solve_cfg: SolveConfig,
    valid_mask: np.ndarray,
    seed_albedo: np.ndarray | None = None,
) -> tuple[SurfaceParams, Any]:
    first = solve_photometric(
        observations,
        views,
        obs_cfg,
        solve_cfg,
        valid_mask=valid_mask,
        albedo_seed=seed_albedo,
    )
    return SurfaceParams(normals=first.normals.copy(), albedo=first.albedo.copy()), first


def run_alternating(
    config: PipelineConfig,
    observations: np.ndarray,
    views: ViewTable,
    run_root: Path,
    fault: FaultConfig | None = None,
    crash_after_round: int | None = None,
    valid_mask: np.ndarray | None = None,
    seed_albedo: np.ndarray | None = None,
) -> dict[str, Any]:
    """执行交替优化并原子落盘，返回运行摘要。"""
    fault = fault or FaultConfig()
    paths = RunPaths(Path(run_root))
    for directory in (paths.traj_dir, paths.conflict_dir, paths.snapshot_dir):
        directory.mkdir(parents=True, exist_ok=True)
    cleanup_temp_files(paths.root)

    h, w = observations.shape[1:]
    if valid_mask is None:
        valid_mask = build_valid_mask(views.size, h, w, fault.skipped_views)
    validate_skipped_views(fault.skipped_views, views.size)

    atomic_write_json(
        paths.run_manifest,
        {
            "config": config.to_dict(),
            "fault": fault.to_dict(),
            "num_views": views.size,
            "view_index": views.view_index.tolist(),
        },
    )
    atomic_commit(paths.run_manifest)

    start_round = 0
    params: SurfaceParams | None = None
    previous_loss: float | None = None
    trajectory: list[dict[str, Any]] = []

    committed = _committed_rounds(paths)
    if committed:
        start_round = committed[-1]
        params, _ = _load_round_snapshot(paths, start_round)
        trajectory = _load_trajectory(paths, committed)
        previous_loss = trajectory[-1]["post_dr_loss"]
        start_round += 1

    if params is None:
        params, _ = _initial_params(
            observations,
            views,
            config.observation,
            config.solve,
            valid_mask,
            seed_albedo=seed_albedo,
        )

    convergence_reason: str | None = None
    crash_recorded = False
    # 本次 run 调用实际计算的首轮（全新运行=0；恢复运行=续写起始轮），
    # 该轮不做损失阈值收敛判定，与全新运行首轮语义保持一致
    first_computed_round = start_round

    for round_id in range(start_round, config.opt.max_iterations):
        # --- 阶段 1：光度立体独立求解（不读真值，不读 DR 缓存） ---
        # 首轮纯 PS 初始化；后续轮次在 DR 当前参数上做独立 PS 对照并融合，
        # PS 的原始独立解仍保存在 ps_raw 中用于语义对照。
        ps_raw = solve_photometric(
            observations,
            views,
            config.observation,
            config.solve,
            valid_mask=valid_mask,
            albedo_seed=params.albedo,
        )
        if round_id == 0:
            ps = ps_raw
            ps_params = SurfaceParams(normals=ps.normals.copy(), albedo=ps.albedo.copy())
        else:
            ps = photometric_blend(
                params,
                ps_raw,
                config.opt.ps_blend,
            )
            ps_params = SurfaceParams(normals=ps.normals.copy(), albedo=ps.albedo.copy())
        ps_update_n = ps_raw.normals - normalize_normals(params.normals)

        # --- 阶段 2：DR 独立解析梯度（在 PS 对照后的参数上） ---
        bundle2 = compute_gradients(
            ps_params, observations, views, config.observation, valid=valid_mask.astype(bool)
        )
        pre_loss = bundle2.loss

        # --- 阶段间对照（PS 独立解 vs DR 当前参数），不靠整体损失判定 ---
        report: ConflictReport = compare_stages(
            ps_raw.normals,
            ps_raw.albedo,
            normalize_normals(params.normals),
            params.albedo,
            config.opt,
            ps_update_normal=ps_update_n,
            dr_grad_normal=bundle2.grad_normal,
        )

        # DR 增量参数更新（独立梯度副本，归一化/非负裁剪）
        params = apply_gradient_step(ps_params, bundle2, config.opt)

        # --- 瓦片增量 vs 全量重渲染核对 ---
        all_tiles = [
            tuple(t)
            for t in tile_ranges(h, w, config.opt.tile_rows, config.opt.tile_cols)
        ]
        render_check = verify_incremental_vs_full(
            params, observations, views, config.observation, config.opt, all_tiles
        )

        post_loss = render_check["full_loss"]
        loss_delta = (
            pre_loss - post_loss
            if previous_loss is None
            else previous_loss - post_loss
        )
        previous_loss = post_loss

        hashes = _write_round_snapshot(
            paths, round_id, params, ps.normals, ps.albedo, bundle2.images
        )
        record = {
            "round": round_id,
            "pre_dr_loss": float(pre_loss),
            "post_dr_loss": float(post_loss),
            "loss_delta_vs_previous": float(loss_delta),
            "grad_normal_norm": float(bundle2.grad_normal_norm),
            "grad_albedo_norm": float(bundle2.grad_albedo_norm),
            "updated_parameters": ["normals", "albedo"],
            "ill_conditioned_pixels": int(ps.ill_conditioned.sum()),
            "max_condition_number": float(ps.condition_number.max()),
            "render_check": render_check,
            "snapshot_hashes": hashes,
            "ps_residual_rmse_max": float(ps.residual.max()),
        }
        rp = _round_paths(paths, round_id)
        atomic_write_json(rp["record"], record)
        atomic_commit(rp["record"])

        conflict_payload = {
            "round": round_id,
            "first_conflict": list(report.first_conflict) if report.first_conflict else None,
            "conflict_count": len(report.pixels),
            "tolerance": {
                "angle_deg": config.opt.conflict_angle_deg,
                "albedo_abs": config.opt.conflict_albedo_abs,
            },
            "pixels": report.pixels,
            "array_hash": sha256_array(report.conflict_mask),
        }
        atomic_write_json(rp["conflict"], conflict_payload)
        atomic_commit(rp["conflict"])
        trajectory.append(record)

        if crash_after_round is not None and round_id == crash_after_round:
            _write_status(paths, {"state": "crashed", "last_committed_round": round_id, "reason": CRASH_REASON})
            crash_recorded = True
            break

        if (
            round_id != first_computed_round
            and round_id < config.opt.max_iterations - 1
            and abs(loss_delta) < config.opt.loss_delta_tol
        ):
            convergence_reason = CONVERGED_DELTA
            break
    else:
        convergence_reason = CONVERGED_MAX_ITER

    if not crash_recorded:
        _write_status(
            paths,
            {
                "state": "completed",
                "convergence_reason": convergence_reason,
                "rounds_committed": len(_committed_rounds(paths)),
            },
        )

    return {
        "run_root": str(paths.root),
        "rounds_committed": len(_committed_rounds(paths)),
        "convergence_reason": convergence_reason,
        "crashed": crash_recorded,
        "trajectory": trajectory,
    }


def _committed_rounds(paths: RunPaths) -> list[int]:
    rounds: list[int] = []
    if not paths.snapshot_dir.exists():
        return rounds
    for record in sorted(paths.traj_dir.glob("round_*.json")):
        if not is_committed(record):
            continue
        round_id = int(record.stem.split("_")[1])
        rp = _round_paths(paths, round_id)
        if is_committed(rp["normal"]) and is_committed(rp["albedo"]) and is_committed(rp["conflict"]):
            rounds.append(round_id)
    return sorted(rounds)


def _load_trajectory(paths: RunPaths, rounds: list[int]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for round_id in rounds:
        out.append(read_json(_round_paths(paths, round_id)["record"]))
    return out


def _write_status(paths: RunPaths, payload: dict[str, Any]) -> None:
    atomic_write_json(paths.status_file, payload)
    atomic_commit(paths.status_file)


def resume_run(
    config: PipelineConfig,
    observations: np.ndarray,
    views: ViewTable,
    run_root: Path,
    fault: FaultConfig | None = None,
    valid_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """从最近一次已提交状态恢复并继续迭代至收敛或上限。"""
    paths = RunPaths(Path(run_root))
    removed = cleanup_temp_files(paths.root)
    committed = _committed_rounds(paths)
    if not committed:
        from .errors import E_NO_CHECKPOINT, BusinessError

        raise BusinessError(
            E_NO_CHECKPOINT,
            "没有任何已提交快照可用于恢复",
            layer="pipeline",
            details={"run_root": str(paths.root), "cleaned": removed},
        )
    # 以已提交状态为准：把崩溃标记覆盖为续写前状态，再走标准 run 路径
    result = run_alternating(config, observations, views, run_root, fault=fault, valid_mask=valid_mask)
    result["resumed_from_round"] = committed[-1]
    result["cleaned_temp_files"] = removed
    return result
