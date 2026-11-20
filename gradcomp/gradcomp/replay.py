"""恢复后的重放校验与差异定位。

从快照恢复后续训若干步，与不中断训练在同区间逐步比较参数、参数版本号
与全局梯度摘要；任何差异定位到首个不一致的训练步、工作进程与元素序号。
缩放因子或分片划分不一致导致不可复现时以 E_REPLAY 报明依据。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import Config
from .data import initial_parameters
from .digest import array_digest, first_diff_index
from .engine import run_step
from .errors import E_REPLAY, GradCompError
from .reference import analyze_step
from .snapshot import Snapshot, assert_replay_compatible, restore_state


@dataclass
class ReplayCheck:
    matched: bool
    start_step: int
    end_step: int
    per_step: list[dict[str, Any]] = field(default_factory=list)
    first_mismatch: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched": self.matched,
            "start_step": self.start_step,
            "end_step": self.end_step,
            "per_step": self.per_step,
            "first_mismatch": self.first_mismatch,
        }


def _step_state(cfg: Config, step: int, parameters: np.ndarray
                ) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    payloads, updated = run_step(cfg, step, parameters)
    global_grad = payloads[0]["global_gradient"]
    analysis = analyze_step(cfg, step,
                            [p["global_gradient"] for p in payloads],
                            parameters)
    if not analysis.consistent:
        raise GradCompError(E_REPLAY, "重放过程中全局梯度不一致")
    return payloads, updated, global_grad


def replay_resume(cfg: Config, snap: Snapshot, extra_steps: int,
                  baseline_cfg: Config | None = None) -> ReplayCheck:
    """恢复快照后续训 extra_steps 步，并与不中断基线逐位比较。"""
    assert_replay_compatible(cfg, snap)
    decision = restore_state(cfg, snap)
    start_step = snap.completed_steps
    end_step = start_step + extra_steps

    # 恢复路径：以快照参数为起点整体替换后继续训练。
    resumed_params = snap.parameters.copy()
    # 基线路径：从相同初始参数不中断跑到 end_step。
    baseline_params = initial_parameters(cfg)

    check = ReplayCheck(matched=True, start_step=start_step, end_step=end_step)
    check.per_step.append({"restore": decision, "step": start_step,
                           "resumed_from_snapshot": True})

    # 先把基线推进到快照点（start_step），保证两条路径同区间比较。
    for step in range(start_step):
        _, baseline_params, _ = _step_state(cfg, step, baseline_params)

    for step in range(start_step, end_step):
        resumed_payloads, resumed_params, resumed_grad = _step_state(
            cfg, step, resumed_params)
        base_payloads, baseline_params, baseline_grad = _step_state(
            cfg, step, baseline_params)

        mismatch = _locate(step, resumed_payloads, base_payloads,
                           resumed_params, baseline_params,
                           resumed_grad, baseline_grad)
        step_report = {
            "step": step,
            "parameter_version": step + 1,
            "resumed_parameters_digest": array_digest(resumed_params),
            "baseline_parameters_digest": array_digest(baseline_params),
            "resumed_global_digest": array_digest(resumed_grad),
            "baseline_global_digest": array_digest(baseline_grad),
            "matched": mismatch is None,
        }
        check.per_step.append(step_report)
        if mismatch is not None:
            check.matched = False
            check.first_mismatch = mismatch
            break
    return check


def _locate(step: int, resumed_payloads: list[dict[str, Any]],
            base_payloads: list[dict[str, Any]],
            resumed_params: np.ndarray, baseline_params: np.ndarray,
            resumed_grad: np.ndarray, baseline_grad: np.ndarray
            ) -> dict[str, Any] | None:
    """定位首个差异到训练步、工作进程与元素序号。"""
    grad_idx = first_diff_index(resumed_grad, baseline_grad)
    param_idx = first_diff_index(resumed_params, baseline_params)
    worker_mismatch = None
    for rank, (a, b) in enumerate(zip(resumed_payloads, base_payloads)):
        idx = first_diff_index(a["global_gradient"], b["global_gradient"])
        if idx is not None:
            worker_mismatch = {"worker": rank, "element": idx}
            break
    if grad_idx is None and param_idx is None and worker_mismatch is None:
        return None
    return {
        "step": step,
        "gradient_element": grad_idx,
        "parameter_element": param_idx,
        "worker": worker_mismatch["worker"] if worker_mismatch else None,
        "worker_element": worker_mismatch["element"]
        if worker_mismatch else None,
        "resumed_digest": array_digest(resumed_grad),
        "baseline_digest": array_digest(baseline_grad),
    }
