"""三种正则化线性回归的确定性小批量梯度下降。

正则项（截距 w[0] 不正则）：
- ridge:      lambda * sum_{j>=1} w_j^2
- lasso:      lambda * sum_{j>=1} |w_j|
- elasticnet: lambda * (rho * ||w||_1 + 0.5 * (1-rho) * ||w||_2^2)

L1 次梯度在 w_j = 0 处采用固定取值规则 0（确定性约定，写入文档与报告）。
批量划分与逐轮批量顺序均由任务键派生的 RandomState 决定，与线程调度无关。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .config import ExperimentConfig
from .data import PreparedData
from .util import derive_seed

STAGE_NAME = "train"
L1_ZERO_SUBGRADIENT = 0.0


@dataclass(frozen=True)
class TaskSpec:
    scheme: str
    penalty: float
    l1_ratio: float
    learning_rate: float
    max_epochs: int
    tol: float
    batch_size: int

    def param_dict(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme,
            "penalty": self.penalty,
            "l1_ratio": self.l1_ratio,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "tol": self.tol,
            "batch_size": self.batch_size,
        }


@dataclass
class Checkpoint:
    epoch: int
    weights: np.ndarray
    rng_state: Any
    loss_history: list[float]
    weight_summaries: list[dict[str, float]]
    converged: bool
    convergence_epoch: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "weights": np.asarray(self.weights, dtype=np.float64).tolist(),
            "rng_state": _rng_state_to_json(self.rng_state),
            "loss_history": list(self.loss_history),
            "weight_summaries": list(self.weight_summaries),
            "converged": self.converged,
            "convergence_epoch": self.convergence_epoch,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Checkpoint":
        return cls(
            epoch=int(payload["epoch"]),
            weights=np.asarray(payload["weights"], dtype=np.float64),
            rng_state=_rng_state_from_json(payload["rng_state"]),
            loss_history=list(payload["loss_history"]),
            weight_summaries=list(payload["weight_summaries"]),
            converged=bool(payload["converged"]),
            convergence_epoch=payload["convergence_epoch"],
        )


@dataclass
class TaskResult:
    task_key: str
    cache_key: str
    scheme: str
    penalty: float
    l1_ratio: float
    status: str  # completed / converged / quota_exceeded / budget_truncated / failed
    epochs_executed: int
    loss_history: list[float]
    weights: np.ndarray
    weight_summaries: list[dict[str, float]]
    final_weight_summary: dict[str, float]
    converged: bool
    convergence_epoch: int | None
    validation: dict[str, float]
    quota_touched: bool
    seed: int
    failure: dict[str, Any] | None = None
    started_monotonic: float | None = None
    ended_monotonic: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_key": self.task_key,
            "cache_key": self.cache_key,
            "scheme": self.scheme,
            "penalty": self.penalty,
            "l1_ratio": self.l1_ratio,
            "status": self.status,
            "epochs_executed": self.epochs_executed,
            "loss_history": list(self.loss_history),
            "weights": np.asarray(self.weights, dtype=np.float64).tolist(),
            "weight_summaries": list(self.weight_summaries),
            "final_weight_summary": dict(self.final_weight_summary),
            "converged": self.converged,
            "convergence_epoch": self.convergence_epoch,
            "validation": dict(self.validation),
            "quota_touched": self.quota_touched,
            "seed": self.seed,
            "failure": self.failure,
            "started_monotonic": self.started_monotonic,
            "ended_monotonic": self.ended_monotonic,
        }


def _rng_state_to_json(state: Any) -> list[Any]:
    # NumPy 1.x 为 4 元组，2.x 为 5 元组；统一按索引访问。
    first = state[0]
    if isinstance(first, str):  # NumPy 2.x: ('MT19937', keys, pos, ...)
        keys = np.asarray(state[1], dtype=np.uint32).tolist()
        pos, has_gauss, cached = int(state[2]), bool(state[3]), float(state[4])
        return ["MT19937", keys, pos, has_gauss, cached]
    keys = np.asarray(first, dtype=np.uint32).tolist()
    return [keys, int(state[1]), bool(state[2]), float(state[3])]


def _rng_state_from_json(payload: list[Any]) -> Any:
    if payload and payload[0] == "MT19937":
        return (
            "MT19937",
            np.asarray(payload[1], dtype=np.uint32),
            int(payload[2]),
            bool(payload[3]),
            float(payload[4]),
        )
    return (
        np.asarray(payload[0], dtype=np.uint32),
        int(payload[1]),
        bool(payload[2]),
        float(payload[3]),
    )


def _regularized_loss(
    scheme: str,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    penalty: float,
    l1_ratio: float,
) -> float:
    residual = x.dot(weights) - y
    data_loss = 0.5 * float(np.mean(residual**2))
    body = weights[1:]
    if scheme == "ridge":
        reg = penalty * float(np.sum(body**2))
    elif scheme == "lasso":
        reg = penalty * float(np.sum(np.abs(body)))
    elif scheme == "elasticnet":
        l1_part = l1_ratio * float(np.sum(np.abs(body)))
        l2_part = 0.5 * (1.0 - l1_ratio) * float(np.sum(body**2))
        reg = penalty * (l1_part + l2_part)
    else:  # pragma: no cover - 配置层已拦截未知方案
        raise ValueError(f"未知正则化方案: {scheme}")
    return data_loss + reg


def _regularized_gradient(
    scheme: str,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    weights: np.ndarray,
    penalty: float,
    l1_ratio: float,
) -> np.ndarray:
    residual = x_batch.dot(weights) - y_batch
    n_rows = x_batch.shape[0]
    grad_data = x_batch.T.dot(residual) / n_rows

    body = weights[1:]
    grad_reg = np.zeros_like(weights)
    if scheme == "ridge":
        grad_reg[1:] = 2.0 * penalty * body
    elif scheme == "lasso":
        grad_reg[1:] = penalty * _subgradient_l1(body)
    elif scheme == "elasticnet":
        sub = _subgradient_l1(body)
        grad_reg[1:] = penalty * (
            l1_ratio * sub + (1.0 - l1_ratio) * body
        )
    else:  # pragma: no cover
        raise ValueError(f"未知正则化方案: {scheme}")
    return grad_data + grad_reg


def _subgradient_l1(body: np.ndarray) -> np.ndarray:
    # 固定取值规则：零系数处次梯度取 0，其余取 sign(w)。
    grad = np.sign(body)
    grad[body == 0.0] = L1_ZERO_SUBGRADIENT
    return grad


def _weight_summary(weights: np.ndarray) -> dict[str, float]:
    body = weights[1:]
    return {
        "bias": float(weights[0]),
        "l1_norm": float(np.sum(np.abs(body))),
        "l2_norm": float(np.sqrt(np.sum(body**2))),
        "max_abs": float(np.max(np.abs(body))) if body.size else 0.0,
        "zero_count": int(np.sum(body == 0.0)),
    }


def _validation_metrics(
    x_val: np.ndarray, y_val: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    prediction = x_val.dot(weights)
    residual = prediction - y_val
    mse = float(np.mean(residual**2))
    denominator = float(np.sum((y_val - np.mean(y_val)) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / denominator if denominator > 0 else 0.0
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(residual))),
        "r2": r2,
    }


def make_task_spec(
    scheme: str,
    penalty: float,
    l1_ratio: float,
    config: ExperimentConfig,
) -> TaskSpec:
    optimizer = config.optimizer
    return TaskSpec(
        scheme=scheme,
        penalty=float(penalty),
        l1_ratio=float(l1_ratio),
        learning_rate=optimizer.learning_rate,
        max_epochs=optimizer.max_epochs,
        tol=optimizer.tol,
        batch_size=optimizer.batch_size,
    )


def make_cache_key(spec: TaskSpec, data_fingerprint: str) -> str:
    from .util import fingerprint

    return fingerprint({"task": spec.param_dict(), "data": data_fingerprint})


def make_task_key(spec: TaskSpec, seed: int) -> str:
    return (
        f"{spec.scheme}|pen={spec.penalty:.8g}|rho={spec.l1_ratio:.8g}"
        f"|lr={spec.learning_rate:.8g}|seed={seed}"
    )


def initial_checkpoint(n_features: int, seed: int, spec: TaskSpec) -> Checkpoint:
    rng = np.random.RandomState(derive_seed("batch", spec.param_dict(), seed))
    return Checkpoint(
        epoch=0,
        weights=np.zeros(n_features, dtype=np.float64),
        rng_state=rng.get_state(),
        loss_history=[],
        weight_summaries=[],
        converged=False,
        convergence_epoch=None,
    )


def train_task(
    spec: TaskSpec,
    data: PreparedData,
    task_key: str,
    cache_key: str,
    seed: int,
    epoch_limit: int,
    checkpoint: Checkpoint | None = None,
    stop_requested: Callable[[], bool] | None = None,
    failpoint: Callable[[str, int], None] | None = None,
    on_epoch: Callable[[int, "Checkpoint"], None] | None = None,
) -> tuple[TaskResult, Checkpoint]:
    """执行单个训练任务。

    epoch_limit 为该任务允许执行的最大轮次（已扣除历史轮次），由调度器按
    全局预算与单方案配额取 min 得到；本函数只关心确定性数值过程。
    """

    x, y = data.x_train, data.y_train
    n_rows = x.shape[0]
    if checkpoint is None:
        checkpoint = initial_checkpoint(x.shape[1], seed, spec)
    weights = np.array(checkpoint.weights, dtype=np.float64)
    rng = np.random.RandomState()
    rng.set_state(checkpoint.rng_state)
    batch_size = min(spec.batch_size, n_rows)

    quota_touched = False
    converged = checkpoint.converged
    convergence_epoch = checkpoint.convergence_epoch
    loss_history = list(checkpoint.loss_history)
    weight_summaries = list(checkpoint.weight_summaries)
    executed = 0

    while executed < epoch_limit:
        if converged:
            break
        if stop_requested is not None and stop_requested():
            break
        if failpoint is not None:
            failpoint("train", checkpoint.epoch + executed + 1)

        permutation = np.arange(n_rows)
        rng.shuffle(permutation)
        for start in range(0, n_rows, batch_size):
            batch_index = permutation[start : start + batch_size]
            x_batch = x[batch_index]
            y_batch = y[batch_index]
            gradient = _regularized_gradient(
                spec.scheme,
                x_batch,
                y_batch,
                weights,
                spec.penalty,
                spec.l1_ratio,
            )
            weights = weights - spec.learning_rate * gradient

        epoch_no = checkpoint.epoch + executed + 1
        loss = _regularized_loss(
            spec.scheme, x, y, weights, spec.penalty, spec.l1_ratio
        )
        loss_history.append(float(loss))
        summary = _weight_summary(weights)
        weight_summaries.append(summary)
        executed += 1

        if len(loss_history) >= 2:
            delta = abs(loss_history[-2] - loss_history[-1])
            if delta <= spec.tol:
                converged = True
                convergence_epoch = epoch_no
        if on_epoch is not None:
            on_epoch(
                epoch_no,
                Checkpoint(
                    epoch=epoch_no,
                    weights=weights,
                    rng_state=rng.get_state(),
                    loss_history=list(loss_history),
                    weight_summaries=list(weight_summaries),
                    converged=converged,
                    convergence_epoch=convergence_epoch,
                ),
            )

    total_epochs = checkpoint.epoch + executed
    if total_epochs >= spec.max_epochs and not converged:
        quota_touched = True
    new_checkpoint = Checkpoint(
        epoch=total_epochs,
        weights=weights,
        rng_state=rng.get_state(),
        loss_history=loss_history,
        weight_summaries=weight_summaries,
        converged=converged,
        convergence_epoch=convergence_epoch,
    )

    if converged:
        status = "converged"
    elif total_epochs >= spec.max_epochs:
        status = "completed"
    else:
        status = "paused"

    result = TaskResult(
        task_key=task_key,
        cache_key=cache_key,
        scheme=spec.scheme,
        penalty=spec.penalty,
        l1_ratio=spec.l1_ratio,
        status=status,
        epochs_executed=total_epochs,
        loss_history=loss_history,
        weights=weights,
        weight_summaries=weight_summaries,
        final_weight_summary=_weight_summary(weights),
        converged=converged,
        convergence_epoch=convergence_epoch,
        validation=_validation_metrics(data.x_val, data.y_val, weights),
        quota_touched=quota_touched,
        seed=seed,
    )
    return result, new_checkpoint
